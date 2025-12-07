# mamba_ssm/ops/recurrent_step_torch.py
"""
Python autograd.Function wrapper for the recurrent Mamba step kernel.

This provides a differentiable interface to the C++/CUDA kernel, allowing
the streaming RecurrentMambaCell to be used in training with gradients.

The kernel implements a 1D Mamba-like SSM+conv cell per feature:
  - Conv: u = sum_k W_conv[d,k] * w_k + b_conv[d]
  - SSM:  z = A[d] * s_prev + B[d] * u
  - Out:  y = C[d] * z + D_skip[d] * u

When the kernel extension is not compiled, falls back to a pure PyTorch
reference implementation.
"""
from __future__ import annotations

from typing import Tuple, Optional, Any

import torch
from torch import Tensor
from torch.autograd import Function

# Try to import compiled extension
_HAS_CUDA_EXT = False
try:
    from . import recurrent_mamba_step_ext  # compiled .so
    _HAS_CUDA_EXT = True
except ImportError:
    recurrent_mamba_step_ext = None


# ============================================================================
# Pure PyTorch fallback (reference implementation)
# ============================================================================

def _recurrent_mamba_step_forward_torch(
    x_t: Tensor,          # (B, D)
    conv_state: Tensor,   # (B, D, K-1)
    ssm_state: Tensor,    # (B, D)
    W_conv: Tensor,       # (D, K)
    b_conv: Tensor,       # (D)
    A: Tensor,            # (D)
    Bp: Tensor,           # (D)
    C: Tensor,            # (D)
    D_skip: Tensor,       # (D)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Pure PyTorch reference implementation of recurrent Mamba step.
    
    Implements per-feature SSM+conv:
      u = sum_k W_conv[d,k] * w_k + b_conv[d]
      z = A[d] * s_prev + B[d] * u
      y = C[d] * z + D_skip[d] * u
    """
    B_size, D = x_t.shape
    K_minus_1 = conv_state.shape[2]
    K = K_minus_1 + 1
    
    # Allocate outputs
    y_t = torch.zeros_like(x_t)
    new_conv_state = torch.zeros_like(conv_state)
    new_ssm_state = torch.zeros_like(ssm_state)
    
    for d in range(D):
        # Build conv window: [conv_state[:, d, :], x_t[:, d]]
        # Shape: (B, K)
        window = torch.cat([conv_state[:, d, :], x_t[:, d:d+1]], dim=1)  # (B, K)
        
        # Convolution: u = sum_k W[d,k] * w_k + b[d]
        u = (window * W_conv[d:d+1, :]).sum(dim=1) + b_conv[d]  # (B,)
        
        # SSM: z = A[d] * s_prev + B[d] * u
        s_prev = ssm_state[:, d]  # (B,)
        z = A[d] * s_prev + Bp[d] * u  # (B,)
        
        # Output: y = C[d] * z + D_skip[d] * u
        y = C[d] * z + D_skip[d] * u  # (B,)
        
        # Update conv state: shift and append
        if K_minus_1 > 0:
            new_conv_state[:, d, :-1] = conv_state[:, d, 1:]
            new_conv_state[:, d, -1] = x_t[:, d]
        
        # Update SSM state
        new_ssm_state[:, d] = z
        
        # Write output
        y_t[:, d] = y
    
    return y_t, new_conv_state, new_ssm_state


def _recurrent_mamba_step_backward_torch(
    grad_y: Tensor,       # (B, D)
    x_t: Tensor,          # (B, D)
    conv_state: Tensor,   # (B, D, K-1)
    ssm_state: Tensor,    # (B, D)
    W_conv: Tensor,       # (D, K)
    b_conv: Tensor,       # (D)
    A: Tensor,            # (D)
    Bp: Tensor,           # (D)
    C: Tensor,            # (D)
    D_skip: Tensor,       # (D)
    y_t: Tensor,          # (B, D)
    new_conv_state: Tensor,  # (B, D, K-1)
    new_ssm_state: Tensor,   # (B, D)
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    PyTorch backward pass for recurrent Mamba step.
    """
    B_size, D = x_t.shape
    K_minus_1 = conv_state.shape[2]
    K = K_minus_1 + 1
    
    # Allocate gradients
    grad_x = torch.zeros_like(x_t)
    grad_conv_state = torch.zeros_like(conv_state)
    grad_ssm_state = torch.zeros_like(ssm_state)
    grad_W_conv = torch.zeros_like(W_conv)
    grad_b_conv = torch.zeros_like(b_conv)
    grad_A = torch.zeros_like(A)
    grad_Bp = torch.zeros_like(Bp)
    grad_C = torch.zeros_like(C)
    grad_D_skip = torch.zeros_like(D_skip)
    
    for d in range(D):
        # Recompute forward
        window = torch.cat([conv_state[:, d, :], x_t[:, d:d+1]], dim=1)  # (B, K)
        u = (window * W_conv[d:d+1, :]).sum(dim=1) + b_conv[d]  # (B,)
        s_prev = ssm_state[:, d]
        z = A[d] * s_prev + Bp[d] * u
        
        # Backward
        gy = grad_y[:, d]  # (B,)
        
        # y = C*z + D*u
        gu = gy * D_skip[d]
        gz = gy * C[d]
        
        grad_C[d] += (gy * z).sum()
        grad_D_skip[d] += (gy * u).sum()
        
        # z = A*s_prev + B*u
        grad_A[d] += (gz * s_prev).sum()
        grad_Bp[d] += (gz * u).sum()
        
        gs_prev = gz * A[d]
        gu = gu + gz * Bp[d]
        
        # u = sum W*w + b
        grad_b_conv[d] += gu.sum()
        
        for k in range(K_minus_1):
            w_k = conv_state[:, d, k]
            grad_W_conv[d, k] += (gu * w_k).sum()
            grad_conv_state[:, d, k] += gu * W_conv[d, k]
        
        # k = K-1 (x_t)
        grad_W_conv[d, K - 1] += (gu * x_t[:, d]).sum()
        grad_x[:, d] += gu * W_conv[d, K - 1]
        
        grad_ssm_state[:, d] += gs_prev
    
    return (grad_x, grad_conv_state, grad_ssm_state,
            grad_W_conv, grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip)


# ============================================================================
# Autograd Function
# ============================================================================

class _RecurrentMambaStepFn(Function):
    """
    Autograd-compatible wrapper for recurrent Mamba step.
    
    Uses CUDA kernel when available, otherwise falls back to PyTorch.
    """
    
    @staticmethod
    def forward(
        ctx: Any,
        x_t: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
        W_conv: Tensor,
        b_conv: Tensor,
        A: Tensor,
        Bp: Tensor,
        C: Tensor,
        D_skip: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass of recurrent Mamba step.
        
        Args:
            x_t: (B, D) input
            conv_state: (B, D, K-1) conv rolling window
            ssm_state: (B, D) SSM state (one scalar per feature)
            W_conv: (D, K) conv weights
            b_conv: (D) conv bias
            A: (D) SSM A parameter (decay)
            Bp: (D) SSM B parameter (input gain)
            C: (D) SSM C parameter (output gain)
            D_skip: (D) skip connection
            
        Returns:
            y_t: (B, D) output
            new_conv_state: (B, D, K-1)
            new_ssm_state: (B, D)
        """
        if _HAS_CUDA_EXT and x_t.is_cuda:
            y_t, new_conv_state, new_ssm_state = \
                recurrent_mamba_step_ext.recurrent_mamba_step_forward(
                    x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip
                )
        else:
            y_t, new_conv_state, new_ssm_state = \
                _recurrent_mamba_step_forward_torch(
                    x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip
                )
        
        # Save for backward
        ctx.save_for_backward(
            x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
            y_t, new_conv_state, new_ssm_state
        )
        ctx.use_cuda = _HAS_CUDA_EXT and x_t.is_cuda
        
        return y_t, new_conv_state, new_ssm_state
    
    @staticmethod
    def backward(
        ctx: Any,
        grad_y: Tensor,
        grad_conv_out: Tensor,
        grad_ssm_out: Tensor,
    ) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass of recurrent Mamba step.
        """
        (x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
         y_t, new_conv_state, new_ssm_state) = ctx.saved_tensors
        
        if ctx.use_cuda:
            (grad_x, grad_conv_state, grad_ssm_state,
             grad_W_conv, grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip) = \
                recurrent_mamba_step_ext.recurrent_mamba_step_backward(
                    grad_y, x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
                    y_t, new_conv_state, new_ssm_state
                )
        else:
            (grad_x, grad_conv_state, grad_ssm_state,
             grad_W_conv, grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip) = \
                _recurrent_mamba_step_backward_torch(
                    grad_y, x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip,
                    y_t, new_conv_state, new_ssm_state
                )
        
        return (grad_x, grad_conv_state, grad_ssm_state,
                grad_W_conv, grad_b_conv, grad_A, grad_Bp, grad_C, grad_D_skip)


# ============================================================================
# Public API
# ============================================================================

def recurrent_mamba_step(
    x_t: Tensor,
    conv_state: Tensor,
    ssm_state: Tensor,
    W_conv: Tensor,
    b_conv: Tensor,
    A: Tensor,
    Bp: Tensor,
    C: Tensor,
    D_skip: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Single recurrent step of Mamba-like SSM+conv cell.
    
    This is the O(T) streaming kernel API. For a sequence of length T,
    calling this T times is O(T) total (not O(T²) like history-based).
    
    Per-feature math:
        u = sum_k W_conv[d,k] * w_k + b_conv[d]   (conv)
        z = A[d] * s_prev + B[d] * u              (SSM)
        y = C[d] * z + D_skip[d] * u              (output)
    
    Args:
        x_t: (B, D) input at current timestep
        conv_state: (B, D, K-1) causal conv rolling window
        ssm_state: (B, D) SSM state (one scalar per feature)
        W_conv: (D, K) conv weights
        b_conv: (D) conv bias
        A: (D) SSM A (decay)
        Bp: (D) SSM B (input gain)
        C: (D) SSM C (output gain)
        D_skip: (D) skip connection
        
    Returns:
        y_t: (B, D) output at current timestep
        new_conv_state: (B, D, K-1) updated conv state
        new_ssm_state: (B, D) updated SSM state
    """
    return _RecurrentMambaStepFn.apply(
        x_t, conv_state, ssm_state, W_conv, b_conv, A, Bp, C, D_skip
    )


def has_cuda_kernel() -> bool:
    """Check if CUDA kernel extension is available."""
    return _HAS_CUDA_EXT
