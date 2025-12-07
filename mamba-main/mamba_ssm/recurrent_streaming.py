# mamba_ssm/recurrent_streaming.py
"""
True streaming recurrent Mamba cell with O(T) complexity.

This module provides:
- MambaState: Fixed-size state (conv_state + ssm_state, both (B, D, ...))
- RecurrentMambaCell: True O(T) streaming cell using kernel

Unlike the reference implementation in recurrent.py (which uses full history
and is O(T²)), this uses fixed-size state that updates in O(1) per step.

The kernel implements per-feature SSM+conv:
  u = sum_k W_conv[d,k] * w_k + b_conv[d]
  z = A[d] * s_prev + B[d] * u  
  y = C[d] * z + D_skip[d] * u
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch import Tensor

from mamba_ssm.ops.recurrent_step_torch import recurrent_mamba_step, has_cuda_kernel


@dataclass
class MambaState:
    """
    True recurrent state for streaming Mamba.
    
    This is fixed-size and updates in O(1) per step.
    
    Attributes:
        conv_state: (B, D, K-1) rolling window for causal convolution
        ssm_state: (B, D) SSM internal state (one scalar per feature)
    """
    conv_state: Tensor   # (B, D, K-1)
    ssm_state: Tensor    # (B, D)
    
    @classmethod
    def zeros(
        cls,
        batch_size: int,
        d_model: int,
        kernel_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "MambaState":
        """
        Create zero-initialized state.
        
        Args:
            batch_size: Batch size B
            d_model: Model dimension D
            kernel_size: Conv kernel size K (state holds K-1)
            device: Torch device
            dtype: Tensor dtype
        """
        conv_state = torch.zeros(
            batch_size, d_model, max(kernel_size - 1, 0),
            device=device, dtype=dtype
        )
        ssm_state = torch.zeros(
            batch_size, d_model,
            device=device, dtype=dtype
        )
        return cls(conv_state=conv_state, ssm_state=ssm_state)
    
    def mask_done(self, done: Tensor) -> "MambaState":
        """
        Reset state for finished environments.
        
        Args:
            done: (B,) bool tensor indicating which envs are done
            
        Returns:
            New MambaState with zeroed state for done envs
        """
        if done.dim() != 1:
            raise ValueError(f"done must be (B,), got {tuple(done.shape)}")
        
        B = self.conv_state.shape[0]
        if done.shape[0] != B:
            raise ValueError(f"done batch {done.shape[0]} != state batch {B}")
        
        # Create masks
        mask_2d = (~done).view(-1, 1).to(self.ssm_state.dtype)  # (B, 1)
        mask_3d = mask_2d.unsqueeze(-1)  # (B, 1, 1)
        
        conv_state = self.conv_state * mask_3d
        ssm_state = self.ssm_state * mask_2d
        
        return MambaState(conv_state=conv_state, ssm_state=ssm_state)
    
    def detach(self) -> "MambaState":
        """Detach state from computation graph."""
        return MambaState(
            conv_state=self.conv_state.detach(),
            ssm_state=self.ssm_state.detach(),
        )
    
    def clone(self) -> "MambaState":
        """Deep copy of state."""
        return MambaState(
            conv_state=self.conv_state.clone(),
            ssm_state=self.ssm_state.clone(),
        )


class RecurrentMambaCell(nn.Module):
    """
    True streaming recurrent Mamba cell with O(T) complexity.
    
    This wraps a Mamba2 module or creates standalone SSM+conv parameters,
    and provides a step API:
        y_t, new_state = cell(x_t, state)
    
    The state is fixed-size (doesn't grow with T), achieving O(T) streaming.
    
    Example:
        >>> cell = RecurrentMambaCell(d_model=64, kernel_size=4)
        >>> state = cell.initial_state(batch_size=4, device='cuda')
        >>> for t in range(seq_len):
        ...     y_t, state = cell(x[:, t, :], state)
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 4,
        core_mamba: Optional[nn.Module] = None,
    ):
        """
        Initialize streaming cell.
        
        Args:
            d_model: Feature dimension D
            kernel_size: Conv kernel size K
            core_mamba: Optional Mamba2 to extract weights from (not used yet)
        """
        super().__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Create learnable parameters
        # Conv: (D, K)
        self.W_conv = nn.Parameter(torch.randn(d_model, kernel_size) * 0.02)
        self.b_conv = nn.Parameter(torch.zeros(d_model))
        
        # SSM parameters: (D,)
        # A: decay (initialized near 1 so state persists)
        self.A = nn.Parameter(torch.ones(d_model) * 0.9)
        # B: input gain
        self.Bp = nn.Parameter(torch.ones(d_model) * 0.1)
        # C: output gain
        self.C = nn.Parameter(torch.ones(d_model))
        # D: skip connection
        self.D_skip = nn.Parameter(torch.zeros(d_model))
        
        # If core_mamba provided, try to extract weights
        if core_mamba is not None:
            self._extract_from_mamba(core_mamba)
    
    def _extract_from_mamba(self, core: nn.Module) -> None:
        """
        Extract parameters from Mamba2 core.
        
        This is approximate - real Mamba2 has more complex structure.
        For now, we just use our own parameters.
        """
        # TODO: Extract conv1d weights and SSM params from core
        # This is non-trivial due to Mamba2's data-dependent A/B/C
        pass
    
    def initial_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> MambaState:
        """
        Create zero-initialized state for streaming.
        
        Args:
            batch_size: Batch size
            device: Torch device
            dtype: Tensor dtype
            
        Returns:
            MambaState with zeroed conv_state and ssm_state
        """
        return MambaState.zeros(
            batch_size=batch_size,
            d_model=self.d_model,
            kernel_size=self.kernel_size,
            device=device,
            dtype=dtype,
        )
    
    def forward(
        self,
        x_t: Tensor,
        state: MambaState,
    ) -> Tuple[Tensor, MambaState]:
        """
        Single recurrent step.
        
        Args:
            x_t: (B, D) input at current timestep
            state: MambaState from previous step
            
        Returns:
            y_t: (B, D) output at current timestep
            new_state: Updated MambaState
        """
        if x_t.dim() != 2:
            raise ValueError(f"x_t must be (B, D), got {tuple(x_t.shape)}")
        
        B, D = x_t.shape
        if D != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got D={D}")
        
        # Call the recurrent step kernel
        y_t, new_conv_state, new_ssm_state = recurrent_mamba_step(
            x_t,
            state.conv_state,
            state.ssm_state,
            self.W_conv,
            self.b_conv,
            self.A,
            self.Bp,
            self.C,
            self.D_skip,
        )
        
        new_state = MambaState(
            conv_state=new_conv_state,
            ssm_state=new_ssm_state,
        )
        
        return y_t, new_state
    
    @classmethod
    def from_mamba2(
        cls,
        mamba: nn.Module,
        kernel_size: int = 4,
    ) -> "RecurrentMambaCell":
        """
        Create from existing Mamba2.
        
        Args:
            mamba: A Mamba2 instance
            kernel_size: Conv kernel size
            
        Returns:
            RecurrentMambaCell
        """
        d_model = getattr(mamba, 'd_model', 64)
        return cls(d_model=d_model, kernel_size=kernel_size, core_mamba=mamba)
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"d_model={self.d_model}, "
            f"kernel_size={self.kernel_size}, "
            f"kernel_available={has_cuda_kernel()})"
        )
