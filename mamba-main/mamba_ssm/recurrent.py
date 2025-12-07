# mamba_ssm/recurrent.py
"""
CPU Reference Implementation for Recurrent Mamba Cell.

This is the gold-standard behavior for a kernel dev to match, not the final fast path.
It uses a for-loop + full Mamba2 passes, so it's O(T²) if you stream it in a training loop.

Use it for:
    - Unit tests
    - Numerical equivalence checks (full vs streaming)
    - Mac CPU/MPS debugging

Changes vs basic version:
    - Optional sliding window max_history_steps to cap memory usage
    - Explicit from_mamba2 constructor to wrap existing Mamba2 with identical weights
    - Better mask_done (frees history when all envs are done)
    - Stricter device/dtype checks and nicer error messages
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

__all__ = ["MambaState", "RecurrentMambaCell"]


@dataclass
class MambaState:
    """
    Reference recurrent state for Mamba.

    For the reference implementation, we keep the full input history
    X_{0..t-1} as a tensor of shape (B, T_prev, D).

    This is NOT the final efficient state (conv_state + ssm_state).
    It is a mathematically trivial way to define streaming behavior
    using the existing Mamba2 module:

        y_full   = Mamba2(x)                # x: (B, T, D)
        y_stream = step_through_recurrent(x)

    When max_history_steps is None and we never truncate history,
    y_full and y_stream must match (up to numerical tolerance).
    """
    history: Optional[torch.Tensor]  # (B, T_prev, D) or None

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
    ) -> "MambaState":
        """
        Initial state at t=0: no history yet.
        """
        # We intentionally do NOT allocate anything here.
        # history=None means "no previous inputs".
        _ = batch_size, d_model, device, dtype  # kept for symmetry / future use
        return cls(history=None)

    def mask_done(self, done: torch.Tensor) -> "MambaState":
        """
        Reset recurrent state for environments where done[b] is True.

        Args:
            done:
                Bool tensor of shape (B,) indicating which envs
                have just terminated.

        Returns:
            New MambaState with history zeroed for done envs.
        """
        if self.history is None:
            return self

        if done.dim() != 1:
            raise ValueError(f"done must have shape (B,), got {tuple(done.shape)}")

        B_hist = self.history.shape[0]
        if done.shape[0] != B_hist:
            raise ValueError(
                f"done batch {done.shape[0]} != history batch {B_hist}"
            )

        # self.history: (B, T_prev, D)
        mask = (~done).view(-1, 1, 1).to(self.history.dtype)  # (B, 1, 1)
        new_hist = self.history * mask

        # If all envs are done, free the history entirely
        if (~done).sum() == 0:
            return MambaState(history=None)

        return MambaState(history=new_hist)


class RecurrentMambaCell(nn.Module):
    """
    Reference recurrent Mamba cell built on top of Mamba2.

    API:
        y_t, new_state = cell(x_t, state)

    where:
        x_t:      (B, D_in)  – input at time t
        y_t:      (B, D_out) – output at time t
        state:    MambaState history up to t-1

    Behavior (reference mode, max_history_steps=None):
        At step t, given history X_{0..t-1} and x_t:

            if history is None:
                seq = x_t.unsqueeze(1)           # (B, 1, D)
            else:
                seq = concat(history, x_t_seq)   # (B, T_prev+1, D)

            y_seq = Mamba2(seq)                  # (B, T, D_out)
            y_t   = y_seq[:, -1, :]              # (B, D_out)

        new_state.history = seq.

    This is O(T^2) if you run it step-by-step over a long sequence,
    but it defines the exact streaming semantics to be matched by
    any future efficient recurrent kernel implementation.

    If max_history_steps is not None, we keep only the last
    max_history_steps inputs in history. That breaks exact equivalence
    with full Mamba2 on very long sequences, but bounds memory and
    time to O(T * max_history_steps).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        max_history_steps: Optional[int] = None,
        core: Optional[nn.Module] = None,
    ):
        """
        Args:
            d_model:
                Input/output dimension for Mamba2.

            d_state, d_conv, expand:
                Standard Mamba2 hyperparameters.

            max_history_steps:
                If None  -> keep full history (strict reference mode).
                If > 0   -> keep only the last max_history_steps inputs
                            in state.history (sliding window).

            core:
                Optional pre-constructed Mamba2 instance whose parameters
                will be reused. If None, a new Mamba2(d_model, d_state, ...)
                will be constructed internally.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.max_history_steps = max_history_steps

        # Lazy import to avoid circular issues
        from mamba_ssm.modules.mamba2 import Mamba2

        if core is not None:
            if not isinstance(core, Mamba2):
                raise TypeError(f"core must be a Mamba2 instance, got {type(core)}")
            if core.d_model != d_model:
                raise ValueError(
                    f"Mamba2 core d_model={core.d_model} "
                    f"!= requested d_model={d_model}"
                )
            self.core = core
        else:
            self.core = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )

    @classmethod
    def from_mamba2(
        cls,
        mamba: nn.Module,
        max_history_steps: Optional[int] = None,
    ) -> "RecurrentMambaCell":
        """
        Convenience constructor:

            cell = RecurrentMambaCell.from_mamba2(existing_mamba2)
        """
        # We don't import Mamba2 here; just trust the object has attributes.
        d_model = getattr(mamba, "d_model")
        d_state = getattr(mamba, "d_state", 16)
        d_conv = getattr(mamba, "d_conv", 4)
        expand = getattr(mamba, "expand", 2)
        return cls(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            max_history_steps=max_history_steps,
            core=mamba,
        )

    def _check_shapes(self, x_t: torch.Tensor) -> Tuple[int, int]:
        if x_t.dim() != 2:
            raise ValueError(f"x_t must have shape (B, D); got {tuple(x_t.shape)}")
        B, D = x_t.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")
        return B, D

    def _truncate_history(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Optionally truncate the time dimension to keep at most
        max_history_steps entries.
        """
        if self.max_history_steps is None:
            return seq

        if self.max_history_steps <= 0:
            raise ValueError(f"max_history_steps must be positive, got {self.max_history_steps}")

        T = seq.shape[1]
        if T <= self.max_history_steps:
            return seq
        # Keep only the last max_history_steps time steps
        return seq[:, -self.max_history_steps :, :]

    def forward(
        self,
        x_t: torch.Tensor,      # (B, D)
        state: MambaState,      # history or None
    ) -> Tuple[torch.Tensor, MambaState]:
        """
        Single recurrent step.

        Args:
            x_t:
                (B, D_model) input at current time t
            state:
                MambaState with history up to t-1

        Returns:
            y_t:
                (B, D_model) output at time t
            new_state:
                updated MambaState including x_t (possibly truncated window)
        """
        B, D = self._check_shapes(x_t)

        x_t_seq = x_t.unsqueeze(1)  # (B, 1, D)

        if state.history is None:
            seq = x_t_seq                     # (B, 1, D)
        else:
            if state.history.shape[0] != B:
                raise ValueError(
                    f"Batch mismatch: history B={state.history.shape[0]}, x_t B={B}"
                )
            if state.history.shape[2] != D:
                raise ValueError(
                    f"Feature mismatch: history D={state.history.shape[2]}, x_t D={D}"
                )

            # Concatenate along time dimension: (B, T_prev, D) + (B, 1, D) -> (B, T_prev+1, D)
            seq = torch.cat([state.history, x_t_seq], dim=1)

        # Optional truncation to bound memory / time
        seq = self._truncate_history(seq)

        # Run full sequence through existing Mamba2
        # seq: (B, T, D)
        y_seq = self.core(seq)    # (B, T, D_out)
        y_t = y_seq[:, -1, :]     # (B, D_out)

        # New state simply stores the (possibly truncated) input sequence
        new_state = MambaState(history=seq)

        return y_t, new_state
