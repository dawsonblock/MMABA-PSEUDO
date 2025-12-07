#!/usr/bin/env python3
"""
Quick sanity + equivalence test for RecurrentMambaCell and MambaState.

Run with:
    cd src && python test_recurrent_mamba_cell.py
"""

import sys
import os

# Add parent directory to path for mamba_compat import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch Mamba for CPU/MPS before any mamba_ssm imports
from mamba_compat import patch_mamba_for_cpu
patch_mamba_for_cpu()

import torch
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.recurrent import RecurrentMambaCell, MambaState


def test_basic_forward(device: torch.device) -> None:
    print(f"[basic_forward] device={device}")

    # D must be divisible by expand * headdim for Mamba2 compatibility
    # With expand=2 and headdim=64 (default), we need D >= 128
    B, D = 4, 128
    x_t = torch.randn(B, D, device=device)

    cell = RecurrentMambaCell(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
    state = MambaState.zeros(batch_size=B, d_model=D, device=device, dtype=x_t.dtype)

    y_t, state2 = cell(x_t, state)

    print("  x_t:", tuple(x_t.shape))
    print("  y_t:", tuple(y_t.shape))
    hist = state2.history
    print("  history:", None if hist is None else tuple(hist.shape))
    
    assert y_t.shape == (B, D), f"Expected y_t shape ({B}, {D}), got {tuple(y_t.shape)}"
    assert hist is not None and hist.shape == (B, 1, D), f"Expected history shape ({B}, 1, {D})"
    print("  ✅ PASSED")


def test_full_vs_stream_equiv(device: torch.device) -> None:
    print(f"[full_vs_stream_equiv] device={device}")

    torch.manual_seed(0)

    # D must be >= 128 for Mamba2 with expand=2
    B, T, D = 2, 8, 128
    x = torch.randn(B, T, D, device=device)

    # Full Mamba2
    m_full = Mamba2(d_model=D, d_state=16, d_conv=4, expand=2).to(device)

    # Recurrent cell, sharing the exact same core
    cell = RecurrentMambaCell.from_mamba2(m_full, max_history_steps=None).to(device)

    # Full sequence pass
    y_full = m_full(x)  # (B, T, D)

    # Streaming pass
    state = MambaState.zeros(batch_size=B, d_model=D, device=device, dtype=x.dtype)

    ys = []
    for t in range(T):
        y_t, state = cell(x[:, t, :], state)  # (B,D), state
        ys.append(y_t.unsqueeze(1))
    y_stream = torch.cat(ys, dim=1)  # (B, T, D)

    max_abs = (y_full - y_stream).abs().max().item()
    mean_abs = (y_full - y_stream).abs().mean().item()

    print("  y_full:", tuple(y_full.shape))
    print("  y_stream:", tuple(y_stream.shape))
    print(f"  max_abs diff:  {max_abs:.3e}")
    print(f"  mean_abs diff: {mean_abs:.3e}")

    assert max_abs < 1e-5, f"max_abs={max_abs}"
    assert mean_abs < 1e-6, f"mean_abs={mean_abs}"
    print("  ✅ PASSED")


def test_mask_done(device: torch.device) -> None:
    print(f"[mask_done] device={device}")

    torch.manual_seed(1)

    # D must be >= 128 for Mamba2 with expand=2
    B, T, D = 3, 5, 128
    x = torch.randn(B, T, D, device=device)

    cell = RecurrentMambaCell(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
    state = MambaState.zeros(batch_size=B, d_model=D, device=device, dtype=x.dtype)

    # All envs start as not done
    done = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(T):
        y_t, state = cell(x[:, t, :], state)
        print(f"  step {t}: y_t shape={tuple(y_t.shape)}")

        # Mark env 0 done at t == 2
        if t == 2:
            done[0] = True
            state = state.mask_done(done)
            done[0] = False  # Reset for next iteration
            print("    marked env 0 done, applied mask_done")

        if state.history is not None:
            print("    history shape:", tuple(state.history.shape))
        else:
            print("    history: None")

    print("  ✅ PASSED")


def test_sliding_window(device: torch.device) -> None:
    print(f"[sliding_window] device={device}")

    torch.manual_seed(2)

    # D must be >= 128 for Mamba2 with expand=2
    B, T, D = 2, 10, 128
    max_history = 4
    x = torch.randn(B, T, D, device=device)

    cell = RecurrentMambaCell(d_model=D, d_state=16, d_conv=4, expand=2, max_history_steps=max_history).to(device)
    state = MambaState.zeros(batch_size=B, d_model=D, device=device, dtype=x.dtype)

    for t in range(T):
        y_t, state = cell(x[:, t, :], state)
        hist_len = state.history.shape[1] if state.history is not None else 0
        print(f"  step {t}: history_len={hist_len}")
        
        # Verify history never exceeds max_history_steps
        assert hist_len <= max_history, f"History length {hist_len} > max {max_history}"

    print("  ✅ PASSED")


def main() -> None:
    # Prefer MPS if available on Mac, else CPU.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("RecurrentMambaCell Test Suite")
    print("=" * 60)
    print(f"Using device: {device}")
    print()

    test_basic_forward(device)
    print()
    test_full_vs_stream_equiv(device)
    print()
    test_mask_done(device)
    print()
    test_sliding_window(device)
    print()

    print("=" * 60)
    print("All tests passed! ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
