#!/usr/bin/env python3
"""
mem_actor_critic_mamba.py

MemActorCritic wired to:
  - Controller: GRU or Mamba2
  - External memory: PseudoModeMemory (your pseudomodes)

This is a drop-in replacement for the old MemActorCritic used in
`neural_memory_long_ppo.py`. PPO code does NOT need to change
except for the import.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_compat import patch_mamba_for_cpu
    patch_mamba_for_cpu()
    from mamba_ssm.modules.mamba2 import Mamba2
except ImportError:
    print("Error: mamba_ssm not found. Please install it or check your environment.")
    Mamba2 = None


# ============================================================
# Pseudomode Memory (external long-term store)
# ============================================================


@dataclass
class PseudoModeState:
    modes: torch.Tensor  # (B, K, D)
    usage: torch.Tensor  # (B, K)


class PseudoModeMemory(nn.Module):
    """
    Simple pseudomode-style memory:

    - K "modes" per batch: long-lived vectors of dimension D.
    - Each write excites one mode selected by low usage.
    - Read is content-based attention over modes.
    """

    def __init__(self, num_slots: int, slot_dim: int, in_dim: int, decay: float = 0.0):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.decay = decay

        # Project controller state to query / write vectors
        # in_dim is the controller hidden size
        # slot_dim is the memory vector size
        self.key_proj = nn.Linear(in_dim, slot_dim)
        self.write_proj = nn.Linear(in_dim, slot_dim)

    def initial_state(self, batch_size: int, device: torch.device) -> PseudoModeState:
        modes = torch.zeros(batch_size, self.num_slots, self.slot_dim, device=device)
        usage = torch.zeros(batch_size, self.num_slots, device=device)
        return PseudoModeState(modes=modes, usage=usage)

    def read(self, state: PseudoModeState, query: torch.Tensor) -> Tuple[torch.Tensor, PseudoModeState]:
        """
        query: (B, in_dim).
        """
        B, K, D = state.modes.shape
        assert query.shape[0] == B

        # Project query into same dimension as modes
        q = self.key_proj(query)                 # (B, D)
        q = q.unsqueeze(1)                       # (B, 1, D)

        # Scores: dot product between modes and query
        # scores = torch.einsum("bkd,b1d->bk", state.modes, q)  # (B, K) - einsum doesn't like '1'
        scores = (state.modes * q).sum(dim=-1)
        attn = F.softmax(scores, dim=-1).unsqueeze(-1)        # (B, K, 1)

        read_vec = (attn * state.modes).sum(dim=1)            # (B, D)
        return read_vec, state

    def write(self, state: PseudoModeState, h: torch.Tensor, gate: torch.Tensor) -> PseudoModeState:
        """
        h:    (B, in_dim)  — controller state
        gate: (B,)     — write strength in [0, 1]

        Strategy:
            - Pick least-used slot per batch.
            - Decay existing content.
            - Add new write vector proportional to gate.
        """
        modes = state.modes
        usage = state.usage

        B, K, D = modes.shape
        assert h.shape[0] == B

        # Project h to D
        w = self.write_proj(h)          # (B, D)

        # Slot to write = argmin usage (least excited mode)
        idx = usage.argmin(dim=-1)      # (B,)

        # Vectorized update to avoid in-place operations (which break autograd)
        # idx: (B,) -> mask: (B, K)
        mask = F.one_hot(idx, num_classes=K).float()
        
        # Expand gate to (B, K, 1) for broadcasting
        # gate is (B,), mask is (B, K)
        write_gate = (gate.unsqueeze(-1) * mask).unsqueeze(-1)  # (B, K, 1)
        
        # w is (B, D) -> (B, 1, D)
        w_expanded = w.unsqueeze(1)
        
        # Decay factor
        decay = 1.0 - self.decay
        
        # New mode content for selected slots:
        # mode_new = (1 - g) * mode_old * decay + g * w
        # We apply this only where mask=1. Where mask=0, we keep mode_old.
        
        # Calculate updated values for ALL slots (masked later)
        # We want:
        # if mask=1: new_val = (1-g)*old*decay + g*w
        # if mask=0: new_val = old
        
        # Term 1: old * decay * (1 - g)
        # But wait, unselected slots are NOT decayed in the original loop.
        # So we only apply this logic to selected slots.
        
        # Let's construct the update term for selected slots
        # selected_old = modes
        # selected_new = (1 - gate_expanded) * selected_old * decay + gate_expanded * w
        # But gate_expanded has 0 for unselected slots, so (1-0)=1.
        # If we just apply the formula globally:
        # new = (1 - write_gate) * modes * decay + write_gate * w
        # For unselected: write_gate=0 -> new = modes * decay.
        # BUT original code did NOT decay unselected slots.
        # Original: for b in range(B): k=idx[b]; ...
        
        # So we must ensure unselected slots are untouched.
        
        # Correct logic:
        # delta = new_val - old_val
        # new_modes = modes + delta * mask
        
        # target_val = (1 - g) * old * decay + g * w
        # delta = target_val - old
        #       = old * decay - g * old * decay + g * w - old
        #       = old * (decay - 1 - g * decay) + g * w
        
        # This seems complex. Simpler:
        # new_modes = modes * (1 - mask_expanded) + target_val * mask_expanded
        
        # target_val (for selected) = (1 - g_scalar) * old * decay + g_scalar * w
        # We can compute this using broadcasted gate.
        
        g_broad = gate.view(B, 1, 1)
        target = (1.0 - g_broad) * modes * decay + g_broad * w_expanded
        
        mask_broad = mask.unsqueeze(-1) # (B, K, 1)
        
        new_modes = modes * (1.0 - mask_broad) + target * mask_broad
        
        # Usage update
        # usage[b, k] += g if k==idx
        new_usage = usage + gate.unsqueeze(-1) * mask

        return PseudoModeState(modes=new_modes, usage=new_usage)


# ============================================================
# MemActorCritic: GRU / Mamba + Pseudomodes
# ============================================================


class MemActorCritic(nn.Module):
    """
    Actor-Critic with external pseudomode memory.

    Controller options:
        - "gru"   : GRUCell
        - "mamba" : Mamba2 (stateless per-step, recurrence is via pseudomodes)

    State dict used by PPO:
        state = {
            "h":   (B, hidden_size),
            "mem": PseudoModeState
        }
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        controller_type: str = "gru",
        hidden_size: int = 128,
        memory_slots: int = 16,
        memory_dim: int = 64,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.controller_type = controller_type
        self.hidden_size = hidden_size
        self.memory_slots = memory_slots
        self.memory_dim = memory_dim

        # Encode observation to hidden_size
        self.obs_encoder = nn.Linear(obs_dim, hidden_size)

        # External pseudomode memory (slot_dim = memory_dim)
        # Fix: Pass hidden_size as in_dim to allow projection
        self.memory = PseudoModeMemory(
            num_slots=memory_slots,
            slot_dim=memory_dim,
            in_dim=hidden_size,
            decay=0.0,
        )

        # Controller input is [obs_encoded, mem_read] -> size = hidden_size + memory_dim
        ctrl_input_dim = hidden_size + memory_dim

        if controller_type == "gru":
            self.controller = nn.GRUCell(ctrl_input_dim, hidden_size)

        elif controller_type == "mamba":
            if Mamba2 is None:
                raise ImportError("mamba_ssm is not installed. Cannot use mamba controller.")
            # Mamba2 processes sequences (B, T, D). We'll use T=1 per step.
            self.controller = Mamba2(
                d_model=ctrl_input_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )

        else:
            raise ValueError(f"Unknown controller_type: {controller_type}")

        # Gate projector for writes into pseudomodes (scalar per env)
        self.gate_proj = nn.Linear(hidden_size + memory_dim, 1)

        # Policy + value heads use [controller_output, mem_read] again
        self.policy_head = nn.Linear(hidden_size + memory_dim, action_dim)
        self.value_head = nn.Linear(hidden_size + memory_dim, 1)

    # -------------------------------------------------------- #

    def initial_state(self, batch_size: int, device: torch.device) -> Dict[str, object]:
        """
        Returns the initial recurrent + memory state for PPO.
        """
        h0 = torch.zeros(batch_size, self.hidden_size, device=device)
        mem0 = self.memory.initial_state(batch_size, device=device)
        return {"h": h0, "mem": mem0}

    # -------------------------------------------------------- #

    def forward(
        self,
        obs: torch.Tensor,              # (B, obs_dim)
        state: Dict[str, object],       # {"h": (B,H), "mem": PseudoModeState}
    ):
        """
        Single step forward.

        Returns:
            logits: (B, action_dim)
            value:  (B,)
            next_state: {"h": ..., "mem": ...}
            gate:   (B,)
            extras: dict (for debug/analysis)
        """
        B = obs.shape[0]
        assert state["h"].shape[0] == B

        # 1) Encode observation
        x = self.obs_encoder(obs)          # (B, H)

        # 2) Read from pseudomode memory using previous controller state as query
        read_vec, mem_state = self.memory.read(state["mem"], query=state["h"])  # (B, M), state

        # 3) Controller input = [encoded obs, last memory read]
        ctrl_in = torch.cat([x, read_vec], dim=-1)   # (B, H+M)

        # 4) Controller update
        if self.controller_type == "gru":
            h = self.controller(ctrl_in, state["h"])     # (B, H)

        elif self.controller_type == "mamba":
            # Mamba2 expects (B, T, D). We treat each step as T=1.
            ctrl_in_seq = ctrl_in.unsqueeze(1)           # (B, 1, H+M)
            h_seq = self.controller(ctrl_in_seq)         # (B, 1, H+M)
            h_full = h_seq.squeeze(1)                    # (B, H+M)

            # Split back into (H, M) chunks: first H dims = new h, last M dims reused for mem interface
            h = h_full[:, : self.hidden_size]
            read_vec = h_full[:, self.hidden_size : self.hidden_size + self.memory_dim]
        else:
            raise RuntimeError("Invalid controller_type at runtime")

        # 5) Compute gate for pseudomode write
        gate = torch.sigmoid(self.gate_proj(torch.cat([h, read_vec], dim=-1))).squeeze(-1)  # (B,)

        # 6) Write into pseudomodes
        mem_state = self.memory.write(mem_state, h=h, gate=gate)

        # 7) Policy + value from [h, read_vec]
        joint = torch.cat([h, read_vec], dim=-1)    # (B, H+M)
        logits = self.policy_head(joint)            # (B, action_dim)
        value = self.value_head(joint).squeeze(-1)  # (B,)

        next_state = {"h": h, "mem": mem_state}
        extras = {
            "read_vec": read_vec,
        }

        return logits, value, next_state, gate, extras
