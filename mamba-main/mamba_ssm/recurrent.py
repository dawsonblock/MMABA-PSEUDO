# mamba_ssm/recurrent.py
"""
CPU Reference Implementation for Recurrent Mamba Cell.

This is the gold-standard behavior for a kernel dev to match, not the final fast path.
It uses a for-loop + full Mamba2 passes, so it's O(T²) if you stream it in a training loop.

Use it for:
    - Unit tests
    - Numerical equivalence checks (full vs streaming)
    - Mac CPU/MPS debugging

The kernel dev will later replace RecurrentMambaCellRef with a proper RecurrentMambaCell
that uses conv/SSM state instead of history, achieving O(T) streaming.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MambaStateRef:
    """
    Reference recurrent state for Mamba.

    For the CPU reference implementation, we simply keep the full input
    history X_{0..t-1} as a tensor of shape (B, T_prev, D).

    This is NOT the final, efficient state (conv_state + ssm_state);
    it is just a mathematically trivial way to define streaming behavior
    using the existing Mamba2 module.
    """
    history: Optional[torch.Tensor]  # (B, T_prev, D) or None

    @classmethod
    def zeros(cls, batch_size: int, d_model: int, device: torch.device) -> "MambaStateRef":
        """Create initial state with no history."""
        # No actual data needed at t = 0; history is None.
        return cls(history=None)

    def mask_done(self, done: torch.Tensor) -> "MambaStateRef":
        """
        Zero input history for environments where done[b] is True.
        This is equivalent to resetting the recurrent state for those envs.

        Args:
            done: Bool tensor of shape (B,).
        
        Returns:
            New MambaStateRef with zeroed history for done envs.
        """
        if self.history is None:
            return self

        # self.history: (B, T_prev, D)
        mask = (~done).view(-1, 1, 1).to(self.history.dtype)  # (B,1,1)
        new_hist = self.history * mask
        return MambaStateRef(history=new_hist)


class RecurrentMambaCellRef(nn.Module):
    """
    Reference recurrent Mamba cell built on top of Mamba2.

    forward(x_t, state) -> (y_t, new_state),
    where state stores the full input history up to t-1.

    Behavior:
      - At step t:
          history: X_{0..t-1}  (possibly None at t=0)
          input:   x_t         (B, D)

        We build:
          seq = concat(history, x_t.unsqueeze(1))  # (B, t+1, D)

        Then run the existing full Mamba2:
          y_seq = core(seq)      # (B, t+1, D_out)

        And take:
          y_t = y_seq[:, -1, :]  # (B, D_out)

        new_state.history = seq.

    This is O(T^2) if you run it across a whole sequence step-by-step,
    but it exactly defines the streaming semantics we want:

        y_full = Mamba2(x)   # x: (B, T, D)

    must match:

        state = MambaStateRef.zeros(...)
        ys = []
        for t in range(T):
            y_t, state = RecurrentMambaCellRef(...)(x[:, t, :], state)
            ys.append(y_t.unsqueeze(1))
        y_stream = torch.cat(ys, dim=1)

    for all B, T, D (up to numerical tolerance).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        from mamba_ssm.modules.mamba2 import Mamba2  # lazy import

        self.core = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(
        self,
        x_t: torch.Tensor,               # (B, D)
        state: MambaStateRef,           # history or None
    ) -> Tuple[torch.Tensor, MambaStateRef]:
        """
        Process single timestep through Mamba.
        
        Args:
            x_t: Input tensor of shape (B, D)
            state: MambaStateRef containing history
            
        Returns:
            y_t: Output tensor of shape (B, D_out)
            new_state: Updated MambaStateRef with appended history
        """
        # x_t: (B, D)
        B, D = x_t.shape
        assert D == self.d_model, f"Expected d_model={self.d_model}, got {D}"

        x_t_seq = x_t.unsqueeze(1)  # (B, 1, D)

        if state.history is None:
            seq = x_t_seq                           # (B, 1, D)
        else:
            # Concatenate along time dimension (full history)
            # history: (B, T_prev, D)
            seq = torch.cat([state.history, x_t_seq], dim=1)  # (B, T_prev+1, D)

        # Run full sequence through existing Mamba2 implementation
        y_seq = self.core(seq)   # (B, T, D_out)
        y_t = y_seq[:, -1, :]    # (B, D_out)

        new_state = MambaStateRef(history=seq)
        return y_t, new_state


# =============================================================================
# Production State Design (for kernel dev to implement)
# =============================================================================

@dataclass
class MambaState:
    """
    Production recurrent state for Mamba.
    
    This is the efficient state representation that a kernel dev will implement.
    It consists of:
        - conv_state: Rolling window for the causal convolution
        - ssm_state: Recurrent SSM state from selective_scan
    
    Shapes:
        - conv_state: (B, d_model, k-1) where k = conv_kernel_size
        - ssm_state: (B, d_state, d_inner) from SSM internals
    """
    conv_state: torch.Tensor   # (B, d_model, k-1)
    ssm_state: torch.Tensor    # (B, d_state, d_inner)

    @classmethod
    def zeros(cls, batch_size: int, d_model: int, d_state: int, d_inner: int,
              k: int, device: torch.device, dtype: torch.dtype = torch.float32) -> "MambaState":
        """
        Create zero-initialized state.
        
        Args:
            batch_size: Batch size B
            d_model: Model dimension D
            d_state: SSM state dimension N
            d_inner: Internal hidden dimension
            k: Convolution kernel size
            device: Torch device
            dtype: Tensor dtype
        """
        conv_state = torch.zeros(batch_size, d_model, max(k - 1, 0), device=device, dtype=dtype)
        ssm_state = torch.zeros(batch_size, d_state, d_inner, device=device, dtype=dtype)
        return cls(conv_state=conv_state, ssm_state=ssm_state)

    def mask_done(self, done: torch.Tensor) -> "MambaState":
        """
        Zero state for environments where done[b] is True.
        
        Args:
            done: Bool tensor of shape (B,)
            
        Returns:
            New MambaState with zeroed state for done envs.
        """
        mask = (~done).view(-1, 1, 1).to(self.conv_state.dtype)   # (B,1,1)
        return MambaState(
            conv_state=self.conv_state * mask,
            ssm_state=self.ssm_state * mask,
        )


# =============================================================================
# Aliases for RL code compatibility
# =============================================================================

# For now, use the reference implementation. Once a kernel dev implements
# a proper RecurrentMambaCell with conv/SSM state, replace these aliases.
RecurrentMambaCell = RecurrentMambaCellRef
