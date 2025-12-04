#!/usr/bin/env python3
"""
neural_memory_final.py

Vectorized long-horizon environments for Neural Memory benchmarks.

Exposes:
    - make_env(...): factory for tasks:
        * delayed_cue
        * copy_memory
        * assoc_recall
        * tmaze

Each environment is:
    - Fully vectorized (num_envs parallel)
    - Torch-based (returns tensors on the requested device)
    - Implements .reset() and .step(actions)

Interface:
    obs = env.reset()                         # (B, obs_dim)
    obs, reward, done = env.step(actions)     # (B, obs_dim), (B,), (B,)
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Base vectorized env
# ============================================================


class BaseVecEnv:
    def reset(self) -> torch.Tensor:
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError


# ============================================================
# 1) Delayed Cue Environment
# ============================================================


class DelayedCueEnv(BaseVecEnv):
    """
    Vectorized delayed-cue environment.

    Episode:
        - Step t=0: show cue bit (0 or 1) in observation.
        - Steps t=1..(horizon-2): blank/distractor observations.
        - Step t=horizon-1: query step; agent must output original cue.

    Reward:
        - +1 if action == cue_bit at query step, else 0.
        - 0 at all other steps.

    Observation:
        obs_dim = cue_bits + 2:
            [cue_bits..., is_cue_step, is_query_step]

        - At cue step: cue bits set, is_cue_step=1, is_query_step=0.
        - At blank steps: cue_bits=0, is_cue_step=0, is_query_step=0.
        - At query step: cue_bits=0, is_cue_step=0, is_query_step=1.

    Action:
        - Discrete, action_dim = 2 ** cue_bits.
        - For cue_bits=1, we expect actions in {0, 1}, matching the single cue bit.
    """

    def __init__(
        self,
        horizon: int,
        num_envs: int,
        cue_bits: int,
        device: str,
    ):
        assert horizon >= 2, "Horizon must be at least 2."
        self.horizon = horizon
        self.num_envs = num_envs
        self.cue_bits = cue_bits
        self.device = torch.device(device)

        self.obs_dim = cue_bits + 2
        self.action_dim = 2 ** cue_bits  # can be 2 for cue_bits=1

        # Per-env state
        self.cues = torch.zeros(num_envs, cue_bits, device=self.device)  # (B, cue_bits)
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)  # step index per env

    # ------------------------------ #

    def _sample_cues(self, mask: torch.Tensor) -> None:
        """
        Sample new cue bits for envs where mask[b] == True.
        """
        num = mask.sum().item()
        if num == 0:
            return
        # Bernoulli(0.5) per bit
        new_cues = torch.randint(
            low=0,
            high=2,
            size=(num, self.cue_bits),
            device=self.device,
            dtype=torch.float32,
        )
        self.cues[mask] = new_cues
        self.t[mask] = 0

    # ------------------------------ #

    def _build_obs(self) -> torch.Tensor:
        """
        Build observation for all envs based on current t and cues.
        """
        B = self.num_envs
        obs = torch.zeros(B, self.obs_dim, device=self.device)

        # Flags
        is_cue_step = self.t == 0
        is_query_step = self.t == (self.horizon - 1)

        # Cue step: show cue bits
        if is_cue_step.any():
            obs[is_cue_step, : self.cue_bits] = self.cues[is_cue_step]

        # Flags
        obs[:, self.cue_bits] = is_cue_step.float()
        obs[:, self.cue_bits + 1] = is_query_step.float()

        return obs

    # ------------------------------ #

    def reset(self) -> torch.Tensor:
        """
        Reset all envs and return initial observation.
        """
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sample_cues(mask)
        obs = self._build_obs()
        return obs

    # ------------------------------ #

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        actions: (B,) int64

        Returns:
            obs:    (B, obs_dim)
            reward: (B,)
            done:   (B,) bool
        """
        B = self.num_envs
        device = self.device

        actions = actions.to(device)
        reward = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        # Query step: reward based on correct recall
        is_query_step = self.t == (self.horizon - 1)

        if is_query_step.any():
            # Use only the first cue bit if cue_bits > 1 (simplification)
            cue_vals = self.cues[is_query_step, 0].long()  # (N,)
            act_vals = actions[is_query_step].long()
            correct = (act_vals == cue_vals)
            reward[is_query_step] = correct.float()
            done[is_query_step] = True

        # Advance time / reset finished envs
        self.t += 1
        # envs that just finished: resample cues and restart t[b]=0
        finished = done
        if finished.any():
            self._sample_cues(finished)

        # Any env that didn't just finish but reached horizon anyway: wrap (safety)
        overshoot = self.t >= self.horizon
        if overshoot.any():
            self._sample_cues(overshoot)

        # Build next observation
        obs = self._build_obs()
        return obs, reward, done


# ============================================================
# 2) Copy-Memory Environment (simplified)
# ============================================================


class CopyMemoryEnv(BaseVecEnv):
    """
    Simplified copy-memory task.

    Parameters:
        seq_len          = copy_seq_len
        alphabet_size    = copy_alphabet_size
        delay            = copy_delay

    Episode structure (per env):
        Phase 0: present sequence (length seq_len)
            - obs: one-hot symbol, plus flags [is_input, is_query]
        Phase 1: delay (length delay)
            - obs: all zeros, flags indicate delay
        Phase 2: query (length seq_len)
            - obs: all zeros except is_query flag
            - agent must output symbols in original order

    Reward:
        - During query phase, +1 for each correct symbol at each step.
        - 0 otherwise.

    Observation:
        obs_dim = alphabet_size + 3:
            [symbol_one_hot..., is_input, is_delay, is_query]

    Action:
        action_dim = alphabet_size
    """

    def __init__(
        self,
        num_envs: int,
        seq_len: int,
        alphabet_size: int,
        delay: int,
        device: str,
    ):
        self.num_envs = num_envs
        self.seq_len = seq_len
        self.alphabet_size = alphabet_size
        self.delay = delay
        self.device = torch.device(device)

        self.obs_dim = alphabet_size + 3
        self.action_dim = alphabet_size

        # Per-env state
        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        # Stored sequence for each env: (B, seq_len)
        self.seq = torch.zeros(num_envs, seq_len, dtype=torch.long, device=self.device)

        self.total_len = seq_len + delay + seq_len

    # ------------------------------ #

    def _sample_sequences(self, mask: torch.Tensor) -> None:
        num = mask.sum().item()
        if num == 0:
            return

        new_seq = torch.randint(
            low=0,
            high=self.alphabet_size,
            size=(num, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        self.seq[mask] = new_seq
        self.t[mask] = 0

    # ------------------------------ #

    def _build_obs(self) -> torch.Tensor:
        B = self.num_envs
        obs = torch.zeros(B, self.obs_dim, device=self.device)

        phase0 = self.t < self.seq_len
        phase1 = (self.t >= self.seq_len) & (self.t < self.seq_len + self.delay)
        phase2 = self.t >= (self.seq_len + self.delay)

        # Phase 0: show input symbols
        if phase0.any():
            idx = self.t[phase0]  # (N,)
            sym = self.seq[phase0, idx]  # (N,)
            obs[phase0, sym] = 1.0
            obs[phase0, self.alphabet_size] = 1.0  # is_input

        # Phase 1: delay
        if phase1.any():
            obs[phase1, self.alphabet_size + 1] = 1.0  # is_delay

        # Phase 2: query
        if phase2.any():
            obs[phase2, self.alphabet_size + 2] = 1.0  # is_query

        return obs

    # ------------------------------ #

    def reset(self) -> torch.Tensor:
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sample_sequences(mask)
        obs = self._build_obs()
        return obs

    # ------------------------------ #

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.num_envs
        device = self.device

        actions = actions.to(device)
        reward = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        phase2 = self.t >= (self.seq_len + self.delay)
        if phase2.any():
            query_pos = self.t[phase2] - (self.seq_len + self.delay)  # (N,)
            target_sym = self.seq[phase2, query_pos]  # (N,)
            act_sym = actions[phase2].long()
            correct = (act_sym == target_sym)
            reward[phase2] = correct.float()

        # Increase time
        self.t += 1

        # Episode done if t == total_len
        finished = self.t >= self.total_len
        done[finished] = True

        # Reset finished envs
        if finished.any():
            self._sample_sequences(finished)

        # Safety wrap (shouldn't be needed)
        overshoot = self.t >= self.total_len
        if overshoot.any():
            self._sample_sequences(overshoot)

        obs = self._build_obs()
        return obs, reward, done


# ============================================================
# 3) Associative Recall Environment (simplified)
# ============================================================


class AssocRecallEnv(BaseVecEnv):
    """
    Simplified associative recall:

    - We generate num_pairs key→value pairs over alphabet.
    - Phase 0: show all (key, value) pairs to the agent.
    - Phase 1: delay phase.
    - Phase 2: query: show a key; agent must output correct value.

    For simplicity, we show one (key,value) per step in Phase 0,
    then a delay, then one query per step (keys only), in fixed order.

    Observation layout:
        obs_dim = 2*alphabet_size + 3:
            [key_one_hot..., value_one_hot..., is_pair, is_delay, is_query]

    Action:
        - action_dim = alphabet_size
        - output is predicted value symbol
    """

    def __init__(
        self,
        num_envs: int,
        num_pairs: int,
        alphabet_size: int,
        delay: int,
        device: str,
    ):
        self.num_envs = num_envs
        self.num_pairs = num_pairs
        self.alphabet_size = alphabet_size
        self.delay = delay
        self.device = torch.device(device)

        self.obs_dim = 2 * alphabet_size + 3
        self.action_dim = alphabet_size

        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        # keys, values: (B, num_pairs)
        self.keys = torch.zeros(num_envs, num_pairs, dtype=torch.long, device=self.device)
        self.values = torch.zeros(num_envs, num_pairs, dtype=torch.long, device=self.device)

        self.total_len = num_pairs + delay + num_pairs

    # ------------------------------ #

    def _sample_pairs(self, mask: torch.Tensor) -> None:
        num = mask.sum().item()
        if num == 0:
            return

        new_keys = torch.randint(
            low=0,
            high=self.alphabet_size,
            size=(num, self.num_pairs),
            device=self.device,
            dtype=torch.long,
        )
        new_values = torch.randint(
            low=0,
            high=self.alphabet_size,
            size=(num, self.num_pairs),
            device=self.device,
            dtype=torch.long,
        )
        self.keys[mask] = new_keys
        self.values[mask] = new_values
        self.t[mask] = 0

    # ------------------------------ #

    def _build_obs(self) -> torch.Tensor:
        B = self.num_envs
        obs = torch.zeros(B, self.obs_dim, device=self.device)

        phase0 = self.t < self.num_pairs
        phase1 = (self.t >= self.num_pairs) & (self.t < self.num_pairs + self.delay)
        phase2 = self.t >= (self.num_pairs + self.delay)

        # Phase 0: show (key, value) pair
        if phase0.any():
            idx = self.t[phase0]  # (N,)
            key = self.keys[phase0, idx]     # (N,)
            val = self.values[phase0, idx]   # (N,)

            obs[phase0, key] = 1.0
            obs[phase0, self.alphabet_size + val] = 1.0
            obs[phase0, 2 * self.alphabet_size] = 1.0  # is_pair

        # Phase 1: delay
        if phase1.any():
            obs[phase1, 2 * self.alphabet_size + 1] = 1.0  # is_delay

        # Phase 2: query key only
        if phase2.any():
            idx = self.t[phase2] - (self.num_pairs + self.delay)  # (N,)
            key = self.keys[phase2, idx]  # (N,)
            obs[phase2, key] = 1.0
            obs[phase2, 2 * self.alphabet_size + 2] = 1.0  # is_query

        return obs

    # ------------------------------ #

    def reset(self) -> torch.Tensor:
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sample_pairs(mask)
        obs = self._build_obs()
        return obs

    # ------------------------------ #

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.num_envs
        device = self.device

        actions = actions.to(device)
        reward = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        phase2 = self.t >= (self.num_pairs + self.delay)
        if phase2.any():
            idx = self.t[phase2] - (self.num_pairs + self.delay)  # (N,)
            correct_val = self.values[phase2, idx]
            act_val = actions[phase2].long()
            correct = (act_val == correct_val)
            reward[phase2] = correct.float()

        self.t += 1
        finished = self.t >= self.total_len
        done[finished] = True

        if finished.any():
            self._sample_pairs(finished)

        overshoot = self.t >= self.total_len
        if overshoot.any():
            self._sample_pairs(overshoot)

        obs = self._build_obs()
        return obs, reward, done


# ============================================================
# 4) T-Maze Environment (simplified)
# ============================================================


class TMazeEnv(BaseVecEnv):
    """
    Simplified vectorized T-maze.

    Episode:
        - Step t=0: cue left/right (0 or 1).
        - Steps t=1..corridor_len: corridor steps (no info).
        - Final step (t=corridor_len+1): junction; agent chooses action.

    Reward:
        - +1 if action matches cue at junction, else 0.

    Observation:
        obs_dim = 3:
            [cue_bit (only at t=0), is_corridor, is_junction]

    Action:
        action_dim = 2   (0=left, 1=right)
    """

    def __init__(
        self,
        corridor_len: int,
        num_envs: int,
        device: str,
    ):
        assert corridor_len >= 1
        self.corridor_len = corridor_len
        self.num_envs = num_envs
        self.device = torch.device(device)

        self.obs_dim = 3
        self.action_dim = 2

        self.t = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cues = torch.zeros(num_envs, dtype=torch.long, device=self.device)

        self.total_len = corridor_len + 2  # cue + corridor_len + junction

    # ------------------------------ #

    def _sample_cues(self, mask: torch.Tensor) -> None:
        num = mask.sum().item()
        if num == 0:
            return

        new_cues = torch.randint(
            low=0,
            high=2,
            size=(num,),
            device=self.device,
            dtype=torch.long,
        )
        self.cues[mask] = new_cues
        self.t[mask] = 0

    # ------------------------------ #

    def _build_obs(self) -> torch.Tensor:
        B = self.num_envs
        obs = torch.zeros(B, self.obs_dim, device=self.device)

        t0 = self.t == 0
        corridor = (self.t > 0) & (self.t <= self.corridor_len)
        junction = self.t == (self.corridor_len + 1)

        # Cue step
        if t0.any():
            obs[t0, 0] = self.cues[t0].float()

        # Corridor
        if corridor.any():
            obs[corridor, 1] = 1.0

        # Junction
        if junction.any():
            obs[junction, 2] = 1.0

        return obs

    # ------------------------------ #

    def reset(self) -> torch.Tensor:
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._sample_cues(mask)
        obs = self._build_obs()
        return obs

    # ------------------------------ #

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = self.num_envs
        device = self.device

        actions = actions.to(device)
        reward = torch.zeros(B, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)

        junction = self.t == (self.corridor_len + 1)
        if junction.any():
            correct = (actions[junction].long() == self.cues[junction])
            reward[junction] = correct.float()
            done[junction] = True

        self.t += 1

        finished = self.t >= self.total_len
        if finished.any():
            done[finished] = True
            self._sample_cues(finished)

        overshoot = self.t >= self.total_len
        if overshoot.any():
            self._sample_cues(overshoot)

        obs = self._build_obs()
        return obs, reward, done


# ============================================================
# make_env factory
# ============================================================


def make_env(
    task: str,
    horizon: int,
    num_envs: int,
    cue_bits: int,
    distractor_prob: float,  # not used in this minimal version
    device: str,
    copy_seq_len: int,
    copy_alphabet_size: int,
    copy_delay: int,
    assoc_num_pairs: int,
    assoc_alphabet_size: int,
    assoc_delay: int,
    tmaze_corridor_len: int,
) -> BaseVecEnv:
    """
    Factory to construct a vectorized environment for the given task.

    Parameters must match the calls from neural_memory_long_ppo.PPOTrainer.
    """
    task = task.lower()

    if task == "delayed_cue":
        env = DelayedCueEnv(
            horizon=horizon,
            num_envs=num_envs,
            cue_bits=cue_bits,
            device=device,
        )

    elif task == "copy_memory":
        env = CopyMemoryEnv(
            num_envs=num_envs,
            seq_len=copy_seq_len,
            alphabet_size=copy_alphabet_size,
            delay=copy_delay,
            device=device,
        )

    elif task == "assoc_recall":
        env = AssocRecallEnv(
            num_envs=num_envs,
            num_pairs=assoc_num_pairs,
            alphabet_size=assoc_alphabet_size,
            delay=assoc_delay,
            device=device,
        )

    elif task == "tmaze":
        env = TMazeEnv(
            corridor_len=tmaze_corridor_len,
            num_envs=num_envs,
            device=device,
        )

    else:
        raise ValueError(f"Unknown task: {task}")

    return env
