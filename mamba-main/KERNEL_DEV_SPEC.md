# Recurrent Mamba Kernel Developer Specification

## Overview

This document specifies the requirements for implementing an efficient recurrent Mamba cell that can be used in RL training loops. The reference implementation (`RecurrentMambaCellRef`) is provided as the gold standard; any kernel implementation must produce numerically equivalent outputs.

---

## Target Python API (Non-Negotiable)

```python
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MambaState:
    conv_state: torch.Tensor   # (B, d_model, k-1)
    ssm_state: torch.Tensor    # (B, d_state, d_inner)

    @classmethod
    def zeros(cls, batch_size, d_model, d_state, d_inner, k, device, dtype):
        ...

    def mask_done(self, done: torch.Tensor) -> "MambaState":
        ...

class RecurrentMambaCell(nn.Module):
    def __init__(self, mamba_module: Mamba2):
        ...

    def forward(self, x_t: torch.Tensor, state: MambaState) -> Tuple[torch.Tensor, MambaState]:
        """
        x_t:    (B, d_model)
        state:  MambaState
        Returns:
            y_t:        (B, d_model)
            new_state:  MambaState
        """
```

---

## State Design

### 1. conv_state (Convolution Recurrence)

**Purpose**: Store the rolling window for the causal 1D convolution.

**Shape**: `(B, D, k-1)` where:
- `B` = batch size
- `D` = d_model (channel count)
- `k` = kernel size (e.g., 4)

**Update Rule**:
```python
# At step t:
window_t = concat(conv_state, x_t.unsqueeze(-1))  # (B, D, k)
conv_out_t = dot(window_t, W_conv)                 # depthwise conv
conv_state_new = window_t[:, :, 1:]                # left-shift
```

**Reset Rule**: `conv_state[done] = 0`

---

### 2. ssm_state (SSM Recurrence)

**Purpose**: Store the internal state of the selective scan.

**Shape**: `(B, N_state, D_inner)` where:
- `N_state` = d_state (from Mamba config)
- `D_inner` = internal dimension (derived from A/B/C shapes)

**Update Rule**:
```python
# Conceptually (exact impl depends on selective_scan):
z_t = A_t * z_{t-1} + B_t * u_t
y_t = C_t * z_t
ssm_state_new = z_t
```

**Reset Rule**: `ssm_state[done] = 0`

---

## Equivalence Requirements

For any input `x ∈ ℝ^{B×T×D}`:

```python
# Full Mamba2
m_full = Mamba2(d_model=D, ...).to(device)
y_full = m_full(x)  # (B, T, D)

# Recurrent Mamba
m_step = RecurrentMambaCell(m_full).to(device)
state = MambaState.zeros(...)
ys = []
for t in range(T):
    y_t, state = m_step(x[:, t, :], state)
    ys.append(y_t.unsqueeze(1))
y_stream = torch.cat(ys, dim=1)  # (B, T, D)

# MUST HOLD:
assert torch.allclose(y_full, y_stream, atol=1e-5, rtol=1e-5)
```

---

## Test Checklist

Run these tests to verify your implementation:

```bash
# Forward equivalence
pytest tests/test_recurrent_ref_forward.py -v

# Gradient equivalence
pytest tests/test_recurrent_ref_grad.py -v

# CUDA tests (on GPU box)
CUDA_VISIBLE_DEVICES=0 pytest tests/ -m cuda -v
```

### Required Tolerances

| Test | max_abs | mean_abs |
|:--|:--|:--|
| Forward output | < 1e-5 | < 1e-6 |
| Input gradient | < 1e-4 | < 1e-5 |
| Param gradient | < 1e-4 | < 1e-5 |

---

## RL Integration

In the RL training loop:

```python
# Reset state for finished envs
if done_prev.any():
    state = state.mask_done(done_prev)

# Step through env
y_t, state = m_step(x_t, state)
```

---

## Files to Implement

1. **`mamba_ssm/recurrent.py`** — `RecurrentMambaCell` class
2. **`mamba_ssm/ops/step_kernel.py`** — CUDA/Triton step kernel (optional but recommended)

---

## Acceptance Criteria

- [ ] `MambaState` holds `conv_state` and `ssm_state` with correct shapes
- [ ] `RecurrentMambaCell.forward(x_t, state)` returns `(y_t, new_state)`
- [ ] Forward equivalence tests pass on CPU
- [ ] Forward equivalence tests pass on CUDA
- [ ] Gradient equivalence tests pass on CPU
- [ ] Gradient equivalence tests pass on CUDA
- [ ] `mask_done()` correctly zeros state per env
- [ ] No memory leaks in streaming mode

---

## Reference Implementation

For debugging, compare against `RecurrentMambaCellRef` in `mamba_ssm/recurrent.py`.
This is O(T²) but mathematically correct.
