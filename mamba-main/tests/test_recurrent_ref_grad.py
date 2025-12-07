# tests/test_recurrent_ref_grad.py
"""
Gradient equivalence tests for recurrent Mamba reference implementation.

Verifies that gradients from full Mamba2(x) match gradients from streaming
RecurrentMambaCellRef, both for inputs and parameters.
"""
import pytest
import torch

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.recurrent import RecurrentMambaCellRef, MambaStateRef


def _clone_mamba2(module: Mamba2, device: torch.device) -> Mamba2:
    """Returns a cloned Mamba2 module with identical parameters."""
    clone = Mamba2(
        d_model=module.d_model,
        d_state=module.d_state,
        d_conv=module.d_conv,
        expand=module.expand
    ).to(device)
    clone.load_state_dict(module.state_dict())
    return clone


def _run_grad_equiv(device: torch.device, B: int, T: int, D: int) -> None:
    """Run gradient equivalence test for given config."""
    torch.manual_seed(1)

    x_full = torch.randn(B, T, D, device=device, requires_grad=True)
    x_stream = x_full.clone().detach().requires_grad_(True)

    # Base full Mamba2
    m_full = Mamba2(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
    # Reference recurrent wrapper
    m_ref = RecurrentMambaCellRef(d_model=D, d_state=16, d_conv=4, expand=2).to(device)

    # Sync parameters exactly
    m_ref.core.load_state_dict(m_full.state_dict())

    # ----- 1) Gradient w.r.t input x -----

    # Full sequence pass
    y_full = m_full(x_full)            # (B, T, D_out)
    loss_full = y_full.pow(2).mean()
    loss_full.backward()
    grad_x_full = x_full.grad.detach().clone()

    # Streaming via recurrent ref
    state = MambaStateRef.zeros(batch_size=B, d_model=D, device=device)
    ys_stream = []
    for t in range(T):
        y_t, state = m_ref(x_stream[:, t, :], state)
        ys_stream.append(y_t.unsqueeze(1))
    y_stream = torch.cat(ys_stream, dim=1)
    loss_stream = y_stream.pow(2).mean()
    loss_stream.backward()
    grad_x_stream = x_stream.grad.detach().clone()

    max_abs_x = (grad_x_full - grad_x_stream).abs().max().item()
    mean_abs_x = (grad_x_full - grad_x_stream).abs().mean().item()

    assert max_abs_x < 1e-4, f"grad_x max_abs={max_abs_x}"
    assert mean_abs_x < 1e-5, f"grad_x mean_abs={mean_abs_x}"

    # ----- 2) Gradient w.r.t parameters -----
    #
    # Use separate models so gradients don't mix. We compare
    # parameter grads of Mamba2(full) vs core inside RecurrentMambaCellRef(stream).

    x_full2 = torch.randn(B, T, D, device=device, requires_grad=True)
    x_stream2 = x_full2.clone().detach().requires_grad_(True)

    # Clone models for gradient comparison
    m_full_for_grad = _clone_mamba2(m_full, device)
    m_ref_for_grad = RecurrentMambaCellRef(d_model=D, d_state=16, d_conv=4, expand=2).to(device)
    m_ref_for_grad.core.load_state_dict(m_full.state_dict())

    # Full path with m_full_for_grad
    y_full2 = m_full_for_grad(x_full2)
    loss_full2 = y_full2.pow(2).mean()
    loss_full2.backward()
    full_params = dict(m_full_for_grad.named_parameters())
    full_grads = {k: v.grad.detach().clone() for k, v in full_params.items() if v.grad is not None}

    # Streaming path with m_ref_for_grad
    state2 = MambaStateRef.zeros(batch_size=B, d_model=D, device=device)
    ys2 = []
    for t in range(T):
        y_t2, state2 = m_ref_for_grad(x_stream2[:, t, :], state2)
        ys2.append(y_t2.unsqueeze(1))
    y_stream2 = torch.cat(ys2, dim=1)
    loss_stream2 = y_stream2.pow(2).mean()
    loss_stream2.backward()
    ref_core_params = dict(m_ref_for_grad.core.named_parameters())
    ref_core_grads = {k: v.grad.detach().clone() for k, v in ref_core_params.items() if v.grad is not None}

    # Compare grads parameter by parameter
    for name, g_full in full_grads.items():
        if name not in ref_core_grads:
            continue
        g_ref = ref_core_grads[name]
        max_abs_p = (g_full - g_ref).abs().max().item()
        mean_abs_p = (g_full - g_ref).abs().mean().item()

        assert max_abs_p < 1e-4, f"param {name} grad max_abs={max_abs_p}"
        assert mean_abs_p < 1e-5, f"param {name} grad mean_abs={mean_abs_p}"


@pytest.mark.parametrize("B,T,D", [(1, 4, 8), (2, 6, 16)])
def test_grad_equiv_cpu(B: int, T: int, D: int) -> None:
    """Test gradient equivalence on CPU."""
    device = torch.device("cpu")
    _run_grad_equiv(device, B, T, D)


@pytest.mark.cuda
@pytest.mark.parametrize("B,T,D", [(1, 4, 8)])
def test_grad_equiv_cuda(B: int, T: int, D: int) -> None:
    """Test gradient equivalence on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    _run_grad_equiv(device, B, T, D)
