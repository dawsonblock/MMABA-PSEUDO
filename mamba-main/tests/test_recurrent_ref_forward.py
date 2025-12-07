# tests/test_recurrent_ref_forward.py
"""
Forward equivalence tests for recurrent Mamba reference implementation.

Verifies that full Mamba2(x) produces identical outputs to streaming
RecurrentMambaCellRef step-by-step.
"""
import pytest
import torch

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.recurrent import RecurrentMambaCellRef, MambaStateRef


def _run_forward_equiv(device: torch.device, B: int, T: int, D: int) -> None:
    """Run forward equivalence test for given config."""
    torch.manual_seed(0)

    x = torch.randn(B, T, D, device=device)

    m_full = Mamba2(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
    m_ref = RecurrentMambaCellRef(d_model=D, d_state=16, d_conv=4, expand=2).to(device)

    # Ensure parameter equality
    m_ref.core.load_state_dict(m_full.state_dict())

    # Full sequence pass
    y_full = m_full(x)  # (B, T, D_out)

    # Streaming via recurrent reference cell
    state = MambaStateRef.zeros(batch_size=B, d_model=D, device=device)
    ys = []
    for t in range(T):
        y_t, state = m_ref(x[:, t, :], state)
        ys.append(y_t.unsqueeze(1))
    y_stream = torch.cat(ys, dim=1)  # (B, T, D_out)

    max_abs = (y_full - y_stream).abs().max().item()
    mean_abs = (y_full - y_stream).abs().mean().item()

    # Tight tolerances; this is meant to be the gold standard
    assert max_abs < 1e-5, f"max_abs={max_abs}"
    assert mean_abs < 1e-6, f"mean_abs={mean_abs}"


@pytest.mark.parametrize("B,T,D", [(1, 4, 8), (2, 8, 16), (4, 12, 32)])
def test_forward_equiv_cpu(B: int, T: int, D: int) -> None:
    """Test forward equivalence on CPU."""
    device = torch.device("cpu")
    _run_forward_equiv(device, B, T, D)


@pytest.mark.cuda
@pytest.mark.parametrize("B,T,D", [(1, 4, 8), (2, 8, 16)])
def test_forward_equiv_cuda(B: int, T: int, D: int) -> None:
    """Test forward equivalence on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    _run_forward_equiv(device, B, T, D)


def test_recurrent_ref_respects_mask_done():
    """Test that mask_done properly resets state for finished envs."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    B, T, D = 3, 5, 8

    x = torch.randn(B, T, D, device=device)

    m_ref = RecurrentMambaCellRef(d_model=D).to(device)
    state = MambaStateRef.zeros(batch_size=B, d_model=D, device=device)

    done = torch.zeros(B, dtype=torch.bool, device=device)

    for t in range(T):
        y_t, state = m_ref(x[:, t, :], state)

        # Artificially mark env 0 as done at t == 2
        if t == 2:
            done[0] = True
            state = state.mask_done(done)
            done[0] = False  # Reset for next iteration

        # From now on, env 0's history is zeroed out, equivalent to
        # having restarted its recurrent state.
        assert state.history is None or state.history.shape[0] == B
