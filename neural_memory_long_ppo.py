#!/usr/bin/env python3
"""
neural_memory_long_ppo.py

Long-horizon Neural Memory benchmark using recurrent PPO.

Controller:
    - GRU or Mamba2 (via MemActorCritic in mem_actor_critic_mamba.py)
Memory:
    - External PseudoModeMemory (pseudomodes) inside MemActorCritic

Tasks (assuming neural_memory_final.make_env supports these):
    delayed_cue   -> original delayed cue task
    copy_memory   -> NTM-style copy task
    assoc_recall  -> key->value associative recall
    tmaze         -> classic T-maze

Features:
    - Full-sequence evaluation each PPO epoch (recurrent PPO)
    - GAE-Lambda advantages
    - Gate sparsity penalty (target ~= 1 / horizon)
    - Optional benchmark suite over all tasks with shared hyperparams
"""

from __future__ import annotations

import argparse
import random
from copy import deepcopy
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from neural_memory_final import make_env
from mem_actor_critic_mamba import MemActorCritic
import wandb_integration as wlog


# ============================================================================
# Utilities
# ============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# PPO Trainer
# ============================================================================


class PPOTrainer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Auto-detect device if "cuda" is requested but not available
        if args.device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                print("Warning: CUDA not available, switching to MPS.")
                self.device = torch.device("mps")
            else:
                print("Warning: CUDA not available, switching to CPU.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(args.device)

        set_seed(args.seed)

        # Environment
        self.env = make_env(
            task=args.task,
            horizon=args.horizon,
            num_envs=args.num_envs,
            cue_bits=args.cue_bits,
            distractor_prob=args.distractor_prob,
            device=str(self.device),  # Pass string representation
            copy_seq_len=args.copy_seq_len,
            copy_alphabet_size=args.copy_alphabet_size,
            copy_delay=args.copy_delay,
            assoc_num_pairs=args.assoc_num_pairs,
            assoc_alphabet_size=args.assoc_alphabet_size,
            assoc_delay=args.assoc_delay,
            tmaze_corridor_len=args.tmaze_corridor_len,
        )

        # Agent
        self.agent = MemActorCritic(
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            controller_type=args.controller,
            hidden_size=args.hidden_size,
            memory_slots=args.memory_slots,
            memory_dim=args.memory_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=args.learning_rate,
            eps=1e-5,
        )

        # Logging
        self.global_step = 0
        self.run = wlog.init_wandb(args)

    # ------------------------------------------------------------------ #

    def _mask_state_on_done(
        self,
        h: torch.Tensor,
        mem_state: Any,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Zero recurrent + pseudomode state where `done` is True.

        h:        (B, H)
        mem_state.modes: (B, K, D)
        mem_state.usage: (B, K)
        done:     (B,) bool
        """
        if done is None:
            return h, mem_state

        mask = (~done).float().unsqueeze(-1)  # (B, 1)

        h = h * mask

        # Broadcast mask to modes: (B, 1, 1)
        mw = mask.unsqueeze(-1)
        modes = mem_state.modes * mw
        usage = mem_state.usage * mask  # broadcast over K

        mem_state = type(mem_state)(modes=modes, usage=usage)
        return h, mem_state

    # ------------------------------------------------------------------ #

    def collect_rollout(
        self,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Collect a single rollout of length T across B environments.

        Returns:
            rollout_data: dict of tensors, each shaped (T, B, ...)
            info: dict with episode stats and initial recurrent/memory state
        """
        args = self.args
        T = args.rollout_length
        B = args.num_envs

        obs = self.env.reset()  # (B, obs_dim), already on device

        # Recurrent + memory state
        state = self.agent.initial_state(B, self.device)
        done_prev = torch.zeros(B, dtype=torch.bool, device=self.device)

        # Buffers: (T, B, ...)
        obs_buf = torch.zeros(T, B, self.env.obs_dim, device=self.device)
        act_buf = torch.zeros(T, B, dtype=torch.long, device=self.device)
        logp_buf = torch.zeros(T, B, device=self.device)
        rew_buf = torch.zeros(T, B, device=self.device)
        done_buf = torch.zeros(T, B, dtype=torch.bool, device=self.device)
        val_buf = torch.zeros(T, B, device=self.device)
        gate_buf = torch.zeros(T, B, device=self.device)

        # Episode stats
        ep_returns = torch.zeros(B, device=self.device)
        ep_lengths = torch.zeros(B, device=self.device)
        completed_returns: List[float] = []
        completed_lengths: List[float] = []

        # Save initial recurrent + memory state
        init_state = {
            "h": state["h"].detach().clone(),                     # (B, H)
            "mem_modes": state["mem"].modes.detach().clone(),     # (B, K, D)
            "mem_usage": state["mem"].usage.detach().clone(),     # (B, K)
        }

        for t in range(T):
            # Mask recurrent + memory state where previous step was done
            h_masked, mem_masked = self._mask_state_on_done(
                state["h"], state["mem"], done_prev
            )
            state["h"] = h_masked
            state["mem"] = mem_masked

            with torch.no_grad():
                logits, value, next_state, gate, extras = self.agent(obs, state)
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)

            next_obs, rew, done = self.env.step(action)

            # Store rollout data
            obs_buf[t] = obs
            act_buf[t] = action
            logp_buf[t] = logp
            rew_buf[t] = rew
            done_buf[t] = done
            val_buf[t] = value
            gate_buf[t] = gate

            # Episode stats
            ep_returns += rew
            ep_lengths += 1.0

            if done.any():
                idx = torch.where(done)[0]
                completed_returns.extend(ep_returns[idx].tolist())
                completed_lengths.extend(ep_lengths[idx].tolist())
                ep_returns[idx] = 0.0
                ep_lengths[idx] = 0.0

            # Advance
            obs = next_obs
            state = next_state
            done_prev = done

        # Bootstrap value for GAE
        with torch.no_grad():
            logits, next_value, _, _, _ = self.agent(obs, state)

        rollout_data = {
            "obs": obs_buf,               # (T, B, obs_dim)
            "actions": act_buf,           # (T, B)
            "logprobs": logp_buf,         # (T, B)
            "rewards": rew_buf,           # (T, B)
            "dones": done_buf,            # (T, B)
            "values": val_buf,            # (T, B)
            "gates": gate_buf,            # (T, B)
            "next_value": next_value,     # (B,)
        }

        info = {
            "ep_returns": completed_returns,
            "ep_lengths": completed_lengths,
            "init_state": init_state,
        }
        return rollout_data, info

    # ------------------------------------------------------------------ #

    def compute_gae(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute GAE-Lambda advantages and returns."""
        gamma = self.args.gamma
        lam = self.args.gae_lambda

        rewards = rollout["rewards"]            # (T, B)
        values = rollout["values"]              # (T, B)
        dones = rollout["dones"]                # (T, B)
        next_value = rollout["next_value"]      # (B,)

        T, B = rewards.shape
        device = rewards.device

        advantages = torch.zeros(T, B, device=device)
        last_gae = torch.zeros(B, device=device)
        last_value = next_value

        for t in reversed(range(T)):
            mask = ~dones[t]                   # (B,)
            delta = rewards[t] + gamma * last_value * mask.float() - values[t]
            last_gae = delta + gamma * lam * mask.float() * last_gae
            advantages[t] = last_gae
            last_value = values[t]

        returns = advantages + values
        rollout["advantages"] = advantages
        rollout["returns"] = returns
        return rollout

    # ------------------------------------------------------------------ #

    def _unpack_init_state(
        self,
        init_state: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Rebuild the initial recurrent + memory state from stored tensors.
        """
        h0 = init_state["h"]                      # (B, H)
        modes0 = init_state["mem_modes"]          # (B, K, D)
        usage0 = init_state["mem_usage"]          # (B, K)

        # PseudoModeState structure is defined in mem_actor_critic_mamba.PseudoModeState
        from mem_actor_critic_mamba import PseudoModeState

        mem0 = PseudoModeState(modes=modes0, usage=usage0)
        return {"h": h0, "mem": mem0}

    # ------------------------------------------------------------------ #

    def evaluate_sequence(
        self,
        obs: torch.Tensor,          # (T, B, obs_dim)
        actions: torch.Tensor,      # (T, B)
        dones: torch.Tensor,        # (T, B)
        init_state: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-run the agent over the entire rollout with gradients
        starting from the stored initial state.
        """
        T, B, _ = obs.shape
        device = obs.device

        state = self._unpack_init_state(init_state)
        done_prev = torch.zeros(B, dtype=torch.bool, device=device)

        logp_seq = torch.zeros(T, B, device=device)
        val_seq = torch.zeros(T, B, device=device)
        ent_seq = torch.zeros(T, B, device=device)
        gate_seq = torch.zeros(T, B, device=device)

        for t in range(T):
            # Mask on previous done
            state["h"], state["mem"] = self._mask_state_on_done(
                state["h"], state["mem"], done_prev
            )

            logits, value, state, gate, extras = self.agent(obs[t], state)
            dist = Categorical(logits=logits)

            logp = dist.log_prob(actions[t])
            ent = dist.entropy()

            logp_seq[t] = logp
            val_seq[t] = value
            ent_seq[t] = ent
            gate_seq[t] = gate

            done_prev = dones[t]

        return logp_seq, val_seq, ent_seq, gate_seq

    # ------------------------------------------------------------------ #

    def update(
        self,
        rollout: Dict[str, torch.Tensor],
        init_state: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Single PPO update over one rollout using recurrent evaluation.
        """
        args = self.args

        obs = rollout["obs"]              # (T, B, obs_dim)
        actions = rollout["actions"]      # (T, B)
        old_logp = rollout["logprobs"]    # (T, B)
        old_values = rollout["values"]    # (T, B)
        dones = rollout["dones"]          # (T, B)
        advantages = rollout["advantages"]
        returns = rollout["returns"]

        T, B, _ = obs.shape

        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

        def flat(x: torch.Tensor) -> torch.Tensor:
            if x.ndim == 3:
                return x.reshape(T * B, -1)
            return x.reshape(T * B)

        old_logp_flat = flat(old_logp).detach()
        old_values_flat = flat(old_values).detach()
        adv_flat = flat(advantages).detach()
        ret_flat = flat(returns).detach()

        clip_coef = args.clip_coef
        ent_coef = args.ent_coef
        vf_coef = args.vf_coef
        gate_target = 1.0 / max(1, args.horizon)
        gate_coef = args.gate_coef

        approx_kl = 0.0
        value_loss = 0.0
        policy_loss = 0.0
        gate_loss = 0.0
        entropy_loss = 0.0

        for epoch in range(args.ppo_epochs):
            # Recompute sequence with gradients
            logp_seq, val_seq, ent_seq, gate_seq = self.evaluate_sequence(
                obs=obs,
                actions=actions,
                dones=dones,
                init_state=init_state,
            )

            logp_new = flat(logp_seq)
            values_new = flat(val_seq)
            entropy = flat(ent_seq)
            gates = flat(gate_seq)

            # PPO ratio
            log_ratio = logp_new - old_logp_flat
            ratio = log_ratio.exp()

            # Surrogate losses
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * adv_flat
            policy_loss_term = -torch.min(surr1, surr2).mean()

            # Value loss (clipped)
            v_clipped = old_values_flat + (values_new - old_values_flat).clamp(
                -clip_coef, clip_coef
            )
            v_loss_unclipped = (values_new - ret_flat).pow(2)
            v_loss_clipped = (v_clipped - ret_flat).pow(2)
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            # Entropy
            entropy_term = entropy.mean()

            # Gate sparsity penalty
            gate_mean = gates.mean()
            gate_penalty = (gate_mean - gate_target).pow(2)

            loss = (
                policy_loss_term
                + vf_coef * v_loss
                - ent_coef * entropy_term
                + gate_coef * gate_penalty
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
            self.optimizer.step()

            approx_kl = 0.5 * (log_ratio.pow(2).mean()).item()
            policy_loss = policy_loss_term.item()
            value_loss = v_loss.item()
            entropy_loss = entropy_term.item()
            gate_loss = gate_penalty.item()

        metrics = {
            "loss/policy": policy_loss,
            "loss/value": value_loss,
            "loss/entropy": entropy_loss,
            "loss/gate": gate_loss,
            "stats/approx_kl": approx_kl,
        }
        return metrics

    # ------------------------------------------------------------------ #

    def train(self, return_summary: bool = False) -> Optional[Dict[str, float]]:
        """
        Main training loop.

        If return_summary=True, returns a dict with final metrics.
        """
        args = self.args
        total_steps = args.total_updates * args.rollout_length * args.num_envs

        last_mean_return = 0.0
        last_mean_length = 0.0
        last_gate_mean = 0.0

        for update in range(1, args.total_updates + 1):
            rollout, info = self.collect_rollout()
            rollout = self.compute_gae(rollout)

            # PPO update
            metrics = self.update(rollout, info["init_state"])

            # Global step (count environment steps)
            self.global_step += args.rollout_length * args.num_envs

            # Episode stats
            ep_returns = info["ep_returns"]
            ep_lengths = info["ep_lengths"]
            mean_return = float(np.mean(ep_returns)) if ep_returns else 0.0
            mean_length = float(np.mean(ep_lengths)) if ep_lengths else 0.0

            gate_mean = rollout["gates"].mean().item()

            last_mean_return = mean_return
            last_mean_length = mean_length
            last_gate_mean = gate_mean

            if update % args.log_interval == 0 or update == 1:
                print(
                    f"[{args.task} | {args.controller}] "
                    f"Update {update}/{args.total_updates} "
                    f"Step={self.global_step}/{total_steps} "
                    f"Return={mean_return:.3f} "
                    f"Len={mean_length:.1f} "
                    f"GateMean={gate_mean:.4f} "
                    f"KL={metrics['stats/approx_kl']:.4e}"
                )

            # WandB logging
            log_data = {
                "train/return_mean": mean_return,
                "train/episode_length_mean": mean_length,
                "train/gate_mean": gate_mean,
                "train/step": self.global_step,
            }
            log_data.update(metrics)
            wlog.log_metrics(self.global_step, log_data, run=self.run)

        wlog.finish_wandb(self.run)

        if return_summary:
            return {
                "final_return_mean": last_mean_return,
                "final_episode_length_mean": last_mean_length,
                "final_gate_mean": last_gate_mean,
                "total_steps": float(self.global_step),
            }
        return None


# ============================================================================
# CLI
# ============================================================================


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neural Memory Long-Horizon PPO Benchmark (multi-task)",
        fromfile_prefix_chars="@",
    )

    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        default="delayed_cue",
        choices=["delayed_cue", "copy_memory", "assoc_recall", "tmaze"],
        help="Which task to run.",
    )

    # Benchmark suite mode
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help="Run all tasks sequentially and print summary table.",
    )

    # Base environment params
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--cue-bits", type=int, default=1)
    parser.add_argument("--distractor-prob", type=float, default=0.0)

    # Copy-Memory params
    parser.add_argument("--copy-seq-len", type=int, default=20)
    parser.add_argument("--copy-alphabet-size", type=int, default=8)
    parser.add_argument("--copy-delay", type=int, default=200)

    # Associative Recall params
    parser.add_argument("--assoc-num-pairs", type=int, default=8)
    parser.add_argument("--assoc-alphabet-size", type=int, default=16)
    parser.add_argument("--assoc-delay", type=int, default=200)

    # T-Maze params
    parser.add_argument("--tmaze-corridor-len", type=int, default=1000)

    # Agent
    parser.add_argument(
        "--controller",
        type=str,
        default="gru",
        choices=["gru", "mamba"],  # add "bamba" later if you wire it
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-dim", type=int, default=64)

    # PPO
    parser.add_argument("--rollout-length", type=int, default=256)
    parser.add_argument("--total-updates", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--gate-coef", type=float, default=1.0)

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)

    # WandB
    parser.add_argument("--track", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb-project", type=str, default="neural-memory-suite")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default="neural-memory-ppo")

    return parser


def main() -> None:
    parser = make_argparser()
    args = parser.parse_args()

    if args.benchmark_suite:
        tasks = ["delayed_cue", "copy_memory", "assoc_recall", "tmaze"]
        summaries: List[Tuple[str, Dict[str, float]]] = []

        print("Running benchmark suite on tasks:", ", ".join(tasks))
        print(
            f"Hyperparameters: total_updates={args.total_updates}, "
            f"rollout_length={args.rollout_length}, num_envs={args.num_envs}, "
            f"controller={args.controller}"
        )

        for task in tasks:
            task_args = deepcopy(args)
            task_args.task = task
            task_args.run_name = f"{args.run_name}-{task}"

            trainer = PPOTrainer(task_args)
            summary = trainer.train(return_summary=True)
            assert summary is not None
            summaries.append((task, summary))

        # Pretty summary table
        print("\n================ Benchmark Suite Summary ================")
        header = f"{'Task':<14} {'Return':>8} {'GateMean':>10} {'EpLen':>8} {'Steps':>10}"
        print(header)
        print("-" * len(header))
        for task, s in summaries:
            print(
                f"{task:<14} "
                f"{s['final_return_mean']:>8.3f} "
                f"{s['final_gate_mean']:>10.4f} "
                f"{s['final_episode_length_mean']:>8.1f} "
                f"{int(s['total_steps']):>10}"
            )
        print("========================================================")
    else:
        trainer = PPOTrainer(args)
        trainer.train()


if __name__ == "__main__":
    main()
