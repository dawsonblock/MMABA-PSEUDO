# Neural Memory Mamba Integration (MMABA-PSEUDO)

This repository benchmarks **Neural Memory** architectures using **Mamba2** (or GRU) controllers paired with **Pseudomode Memory** for long-horizon Reinforcement Learning tasks.

It is designed to be a drop-in benchmark suite for testing memory capabilities in RL agents, supporting tasks like **Delayed Cue**, **Copy Memory**, **Associative Recall**, and **T-Maze**.

## Key Features

*   **Dual Controller Support**: Run with either `gru` (standard RNN) or `mamba` (State Space Model).
*   **Pseudomode Memory**: An external long-term memory module that allows the agent to store and retrieve information over long horizons.
*   **Full Mac/MPS Support**: Mamba 2 runs natively on Apple Silicon (M1/M2/M3) via a pure PyTorch compatibility layer.
    *   No CUDA or Triton required.
    *   Automatically patches `mamba_ssm` to use reference PyTorch implementations.
    *   Import `mamba_compat` at the top of your script to enable MPS support.
*   **Vectorized Environments**: Fast, pure-PyTorch implementations of memory tasks.
*   **Recurrent PPO**: A PPO implementation optimized for recurrent policies with full-sequence evaluation.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/dawsonblock/MMABA-PSEUDO.git
    cd MMABA-PSEUDO
    ```

2.  **Install Dependencies**:
    ```bash
    pip install torch numpy wandb einops
    ```

3.  **Install Mamba (editable mode)**:
    ```bash
    cd mamba-main
    pip install -e . --no-build-isolation
    cd ..
    ```
    > **Note**: The `mamba-main` package has been patched to remove Triton dependencies for macOS compatibility.

## Usage

### Training on Mac (MPS)

```bash
python3 neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --device mps \
    --num-envs 4 \
    --total-updates 2000
```

> **Recommended**: Use `--num-envs 4` on Mac to avoid MPS memory limits.

### Training on CUDA

```bash
python3 neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --device cuda \
    --num-envs 64 \
    --total-updates 2000
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--task` | Task: `delayed_cue`, `copy_memory`, `assoc_recall`, `tmaze` | `delayed_cue` |
| `--controller` | Controller type: `gru` or `mamba` | `gru` |
| `--device` | Device: `cuda`, `mps`, or `cpu` | `cuda` |
| `--horizon` | Task horizon (difficulty) | `200` |
| `--num-envs` | Number of parallel environments | `64` |
| `--rollout-length` | PPO rollout length (should be > horizon) | `256` |
| `--track` | Enable Weights & Biases logging | `False` |

## File Structure

| File | Description |
| :--- | :--- |
| `neural_memory_long_ppo.py` | Main training script with PPO loop |
| `mem_actor_critic_mamba.py` | Agent, PseudoModeMemory, and Mamba2 integration |
| `mamba_compat.py` | **MPS/CPU compatibility layer** for Mamba 2 |
| `neural_memory_final.py` | Vectorized environment implementations |
| `mamba-main/` | Patched Mamba SSM library (Triton-free) |

## Mamba MPS Compatibility

The `mamba_compat.py` module provides a compatibility layer for running Mamba 2 on non-CUDA devices:

- **Mocks Triton**: Prevents import errors on macOS.
- **Patches Kernels**: Replaces CUDA-specific functions with PyTorch equivalents.
- **Auto-Activates**: Automatically applies patches when CUDA is unavailable.

To use Mamba on MPS, ensure `mamba_compat` is imported **before** any Mamba modules:

```python
import mamba_compat  # Must be first!
from mamba_ssm.modules.mamba2 import Mamba2
```

## License

[MIT License](LICENSE)