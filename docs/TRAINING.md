# Training Guide

This document explains how the MMABA-PSEUDO training system works and how to use it effectively.

## System Overview

MMABA-PSEUDO trains reinforcement learning agents with **long-term memory capabilities** using a combination of:

1. **Mamba 2 Controller** — A state-space model (SSM) that processes sequential observations efficiently
2. **PseudoMode Memory** — An external memory bank with K slots for storing and retrieving information
3. **PPO (Proximal Policy Optimization)** — The training algorithm that teaches the agent to maximize reward

```
Observation → [Mamba2] ←→ [PseudoMode Memory] → Policy → Action
                                              → Value  → Critic
```

## Tasks

The benchmark includes 4 memory-intensive tasks:

| Task | Description | Difficulty |
|:--|:--|:--|
| **Delayed Cue** | Remember a signal, act on it 200 steps later | Medium |
| **Copy Memory** | Memorize a sequence, reproduce it after delay | Hard |
| **Associative Recall** | Learn key→value mappings, answer queries | Hard |
| **T-Maze** | Navigate using a hint from the start | Very Hard |

## Quick Start

### Training on Mac (MPS)

```bash
python3 neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --device mps \
    --num-envs 4 \
    --total-updates 2000
```

> **Note**: Use `--num-envs 4` on Mac to avoid memory issues. Training takes ~20 hours.

### Training on CUDA GPU

```bash
python3 neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --device cuda \
    --num-envs 64 \
    --total-updates 2000
```

> **Note**: CUDA training is ~10-50x faster than MPS due to optimized Triton kernels.

### Running All Tasks (Benchmark Suite)

```bash
python3 neural_memory_long_ppo.py --benchmark-suite --controller mamba
```

## Understanding Training Output

```
[delayed_cue | mamba] Update 120/2000 Step=122880/2048000 Return=1.000 Len=200.0 GateMean=0.0152 KL=1.4081e-04
```

| Field | Meaning |
|:--|:--|
| `Update X/Y` | Current training iteration / total iterations |
| `Step` | Total environment steps taken |
| `Return` | Average episode reward (1.0 = perfect) |
| `Len` | Average episode length |
| `GateMean` | Memory write gate activation (lower = sparser writes) |
| `KL` | Policy divergence (should stay small) |

### What Good Training Looks Like

- **Return trending toward 1.0** — Agent is learning
- **GateMean around 0.01-0.05** — Sparse but active memory usage
- **KL staying below 0.01** — Stable policy updates
- **Periodic 1.0 returns** — Agent can solve the task

## Architecture Details

### Mamba 2 Controller

The Mamba 2 is a structured state-space model that processes sequences efficiently:

```python
self.controller = Mamba2(
    d_model=192,      # Input dimension (hidden + memory)
    d_state=16,       # Internal state size
    d_conv=4,         # Convolution window
    expand=2,         # Expansion factor
)
```

**MPS Compatibility**: On Mac, the system automatically uses pure PyTorch fallback implementations instead of Triton kernels. This is handled by `mamba_compat.py`.

### PseudoMode Memory

External memory with K slots that the agent can read/write:

```python
self.memory = PseudoModeMemory(
    num_slots=16,     # Number of memory slots (K)
    slot_dim=64,      # Dimension of each slot
    in_dim=128,       # Controller hidden size
    decay=0.0,        # Memory decay rate
)
```

**How it works:**
1. **Read**: Content-based attention over memory slots
2. **Write**: Store to least-used slot, controlled by learned gate
3. **Gate**: Sigmoid activation determines write strength (sparsity)

### PPO Hyperparameters

| Parameter | Default | Description |
|:--|:--|:--|
| `--learning-rate` | 3e-4 | Adam learning rate |
| `--gamma` | 0.99 | Discount factor |
| `--gae-lambda` | 0.95 | GAE lambda for advantage estimation |
| `--clip-coef` | 0.2 | PPO clipping coefficient |
| `--ent-coef` | 0.01 | Entropy bonus coefficient |
| `--vf-coef` | 0.5 | Value function loss coefficient |
| `--ppo-epochs` | 4 | PPO epochs per update |
| `--gate-coef` | 1.0 | Memory gate sparsity penalty |

## Performance Comparison

| Device | Time per Update | Full Training (2000 updates) | Batch Size |
|:--|:--|:--|:--|
| **CUDA (RTX 3090)** | ~2-5 seconds | ~1-2 hours | 64 envs |
| **MPS (M1/M2/M3)** | ~1-2 minutes | ~17-33 hours | 4 envs |
| **CPU** | ~5-10 minutes | ~7-14 days | 1 env |

The speedup on CUDA comes from:
- Fused Triton kernels (vs pure PyTorch ops)
- Higher parallelism (more environments)
- Better memory bandwidth utilization

## Weights & Biases Logging

Enable experiment tracking with:

```bash
python3 neural_memory_long_ppo.py \
    --track \
    --wandb-project neural-memory-suite \
    --run-name mamba_delayedcue_run1
```

Logged metrics:
- `train/return_mean` — Episode rewards
- `train/gate_mean` — Memory gate activation
- `loss/policy`, `loss/value`, `loss/entropy` — Training losses
- `stats/approx_kl`, `stats/grad_norm` — Training stability

## Troubleshooting

### MPS Out of Memory
Reduce batch size:
```bash
--num-envs 2  # or even 1
```

### Training Not Converging
Try:
- Increase `--total-updates` (more training time)
- Decrease `--horizon` (easier task)
- Increase `--ent-coef` (more exploration)

### Mamba Import Errors on Mac
Ensure `mamba_compat` is imported first:
```python
import mamba_compat  # Must be before mamba_ssm imports
```

## Citation

If you use this code, please cite:
```
@software{mmaba_pseudo,
  title={MMABA-PSEUDO: Neural Memory Mamba Benchmark},
  author={Dawson Block},
  year={2024},
  url={https://github.com/dawsonblock/MMABA-PSEUDO}
}
```
