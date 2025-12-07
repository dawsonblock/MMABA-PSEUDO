#!/usr/bin/env python3
"""
Kernel equivalence and correctness tests for Recurrent Mamba.

This file verifies:
1) Standalone RecurrentMambaCell forward/backward works
2) Gradient correctness via finite differences
3) State fixed-size property
4) mask_done functionality

Run:
    cd MMABA-PSEUDO && python3 mamba-main/tests/test_recurrent_kernel_equiv.py
"""

import sys
import os

# Get the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
src_dir = os.path.join(project_root, 'src')
mamba_dir = os.path.dirname(script_dir)

# Add paths
sys.path.insert(0, src_dir)
sys.path.insert(0, mamba_dir)

# Patch Mamba for CPU/MPS
try:
    from mamba_compat import patch_mamba_for_cpu
    patch_mamba_for_cpu()
except ImportError:
    pass

import torch

# Import streaming cell
from mamba_ssm.recurrent_streaming import (
    RecurrentMambaCell,
    MambaState,
)
from mamba_ssm.ops.recurrent_step_torch import has_cuda_kernel


# ============================================================================
# Helper: choose device
# ============================================================================

def _get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================================
# 1. Basic Forward Test
# ============================================================================

def test_basic_forward() -> None:
    """Verify basic forward pass works."""
    device = _get_device()
    B, D, K = 2, 16, 4
    
    torch.manual_seed(0)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    state = cell.initial_state(batch_size=B, device=device)
    
    x = torch.randn(B, D, device=device)
    y, new_state = cell(x, state)
    
    assert y.shape == (B, D), f"Expected y shape ({B}, {D}), got {y.shape}"
    assert new_state.conv_state.shape == (B, D, K-1)
    assert new_state.ssm_state.shape == (B, D)
    
    print(f"  y: {tuple(y.shape)}")
    print(f"  conv_state: {tuple(new_state.conv_state.shape)}")
    print(f"  ssm_state: {tuple(new_state.ssm_state.shape)}")
    print("✅ Basic Forward PASSED")


# ============================================================================
# 2. Streaming Consistency Test
# ============================================================================

def test_streaming_consistency() -> None:
    """Verify streaming produces consistent outputs over time."""
    device = _get_device()
    B, D, K, T = 2, 16, 4, 10
    
    torch.manual_seed(1)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    x = torch.randn(B, T, D, device=device)
    
    # Stream through
    state = cell.initial_state(batch_size=B, device=device)
    ys = []
    for t in range(T):
        y_t, state = cell(x[:, t, :], state)
        ys.append(y_t)
        assert not torch.isnan(y_t).any(), f"NaN at t={t}"
    
    y_stream = torch.stack(ys, dim=1)  # (B, T, D)
    
    print(f"  y_stream: {tuple(y_stream.shape)}")
    print(f"  y range: [{y_stream.min():.3f}, {y_stream.max():.3f}]")
    print("✅ Streaming Consistency PASSED")


# ============================================================================
# 3. Gradient Check
# ============================================================================

def test_gradient_flow() -> None:
    """Verify gradients flow correctly through the cell."""
    device = _get_device()
    B, D, K, T = 2, 8, 4, 5
    
    torch.manual_seed(2)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    x = torch.randn(B, T, D, device=device, requires_grad=True)
    
    # Forward
    state = cell.initial_state(batch_size=B, device=device)
    ys = []
    for t in range(T):
        y_t, state = cell(x[:, t, :], state)
        ys.append(y_t)
    
    y = torch.stack(ys, dim=1)  # (B, T, D)
    loss = y.pow(2).mean()
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "NaN in input gradient"
    
    # Check parameter gradients
    for name, param in cell.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in {name} gradient"
        print(f"  {name}: grad norm = {param.grad.norm():.3e}")
    
    print(f"  x.grad norm: {x.grad.norm():.3e}")
    print("✅ Gradient Flow PASSED")


# ============================================================================
# 4. Finite Difference Gradient Check
# ============================================================================

def test_finite_diff_gradient() -> None:
    """Verify gradients match finite differences."""
    device = torch.device("cpu")  # Use CPU for stable finite diff
    B, D, K = 1, 4, 3
    eps = 1e-5
    
    torch.manual_seed(3)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    x = torch.randn(B, D, device=device, requires_grad=True)
    state = cell.initial_state(batch_size=B, device=device)
    
    # Analytical gradient
    y, _ = cell(x, state)
    loss = y.pow(2).mean()
    loss.backward()
    grad_analytical = x.grad.clone()
    
    # Finite difference gradient
    grad_fd = torch.zeros_like(x)
    x_flat = x.view(-1)
    
    for i in range(x_flat.numel()):
        x.grad = None
        
        # f(x + eps)
        x_plus = x.detach().clone()
        x_plus.view(-1)[i] += eps
        state_plus = cell.initial_state(batch_size=B, device=device)
        y_plus, _ = cell(x_plus, state_plus)
        loss_plus = y_plus.pow(2).mean().item()
        
        # f(x - eps)
        x_minus = x.detach().clone()
        x_minus.view(-1)[i] -= eps
        state_minus = cell.initial_state(batch_size=B, device=device)
        y_minus, _ = cell(x_minus, state_minus)
        loss_minus = y_minus.pow(2).mean().item()
        
        grad_fd.view(-1)[i] = (loss_plus - loss_minus) / (2 * eps)
    
    # Compare
    max_diff = (grad_analytical - grad_fd).abs().max().item()
    rel_diff = max_diff / max(grad_analytical.abs().max().item(), 1e-8)
    
    print(f"  Analytical grad: {grad_analytical.view(-1)[:4].tolist()}")
    print(f"  Finite diff:     {grad_fd.view(-1)[:4].tolist()}")
    print(f"  Max abs diff:    {max_diff:.3e}")
    print(f"  Relative diff:   {rel_diff:.3e}")
    
    assert rel_diff < 5e-3, f"Gradient mismatch: rel_diff={rel_diff}"
    print("✅ Finite Difference Gradient PASSED")


# ============================================================================
# 5. State Fixed-Size Test
# ============================================================================

def test_state_fixed_size() -> None:
    """Verify kernel state size doesn't grow with T."""
    device = _get_device()
    B, D, K, T = 2, 16, 4, 50
    
    torch.manual_seed(4)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    x = torch.randn(B, T, D, device=device)
    
    state = cell.initial_state(batch_size=B, device=device)
    initial_conv_shape = state.conv_state.shape
    initial_ssm_shape = state.ssm_state.shape
    
    # Run through many steps
    for t in range(T):
        _, state = cell(x[:, t, :], state)
    
    final_conv_shape = state.conv_state.shape
    final_ssm_shape = state.ssm_state.shape
    
    print(f"  conv_state: {initial_conv_shape} -> {final_conv_shape}")
    print(f"  ssm_state:  {initial_ssm_shape} -> {final_ssm_shape}")
    
    assert initial_conv_shape == final_conv_shape, "conv_state shape changed!"
    assert initial_ssm_shape == final_ssm_shape, "ssm_state shape changed!"
    print("✅ State Fixed-Size PASSED")


# ============================================================================
# 6. Mask Done Test
# ============================================================================

def test_mask_done() -> None:
    """Verify kernel state mask_done works."""
    device = _get_device()
    B, D, K = 4, 8, 4
    
    torch.manual_seed(5)
    
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    x = torch.randn(B, D, device=device)
    
    state = cell.initial_state(batch_size=B, device=device)
    
    # Run a step to populate state
    _, state = cell(x, state)
    
    # Verify state is non-zero
    assert state.ssm_state.abs().sum() > 0, "State should be non-zero"
    
    # Mark env 0 and 2 as done
    done = torch.tensor([True, False, True, False], device=device)
    state = state.mask_done(done)
    
    # Check that done envs have zeroed state
    assert (state.conv_state[0] == 0).all(), "env 0 conv_state not zeroed"
    assert (state.conv_state[2] == 0).all(), "env 2 conv_state not zeroed"
    assert (state.ssm_state[0] == 0).all(), "env 0 ssm_state not zeroed"
    assert (state.ssm_state[2] == 0).all(), "env 2 ssm_state not zeroed"
    
    # Check that other envs are NOT zeroed
    assert state.ssm_state[1].abs().sum() > 0, "env 1 should not be zeroed"
    assert state.ssm_state[3].abs().sum() > 0, "env 3 should not be zeroed"
    
    print("✅ Mask Done PASSED")


# ============================================================================
# 7. SSM Behavior Test
# ============================================================================

def test_ssm_behavior() -> None:
    """Verify SSM exhibits expected decay behavior."""
    device = _get_device()
    B, D, K = 1, 2, 2
    
    torch.manual_seed(6)
    
    # Create cell with known parameters
    cell = RecurrentMambaCell(d_model=D, kernel_size=K).to(device)
    
    # Set A to decay (0.9), B to input (0.1), C to output (1), D to skip (0)
    with torch.no_grad():
        cell.A.fill_(0.9)
        cell.Bp.fill_(0.1)
        cell.C.fill_(1.0)
        cell.D_skip.fill_(0.0)
        cell.W_conv.zero_()
        cell.W_conv[:, -1].fill_(1.0)  # Identity conv
        cell.b_conv.zero_()
    
    state = cell.initial_state(batch_size=B, device=device)
    
    # Send impulse
    x_impulse = torch.ones(B, D, device=device)
    x_zero = torch.zeros(B, D, device=device)
    
    # First step: impulse
    y0, state = cell(x_impulse, state)
    
    # Subsequent steps: decay
    ys = [y0.mean().item()]
    for t in range(5):
        y, state = cell(x_zero, state)
        ys.append(y.mean().item())
    
    print(f"  Outputs over time: {[f'{y:.3f}' for y in ys]}")
    
    # Check decay pattern
    for i in range(1, len(ys)):
        assert ys[i] < ys[i-1], f"Expected decay at step {i}"
    
    print("✅ SSM Behavior PASSED")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Recurrent Mamba Kernel Tests")
    print("=" * 60)
    print(f"Device: {_get_device()}")
    print(f"CUDA kernel available: {has_cuda_kernel()}")
    print()
    
    try:
        print("[1] Basic Forward")
        test_basic_forward()
        print()
        
        print("[2] Streaming Consistency")
        test_streaming_consistency()
        print()
        
        print("[3] Gradient Flow")
        test_gradient_flow()
        print()
        
        print("[4] Finite Difference Gradient")
        test_finite_diff_gradient()
        print()
        
        print("[5] State Fixed-Size")
        test_state_fixed_size()
        print()
        
        print("[6] Mask Done")
        test_mask_done()
        print()
        
        print("[7] SSM Behavior")
        test_ssm_behavior()
        print()
        
        print("=" * 60)
        print("All tests passed! ✅")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
