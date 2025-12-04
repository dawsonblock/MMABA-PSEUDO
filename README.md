# Neural Memory Mamba Integration (MMABA-PSEUDO)

This repository benchmarks **Neural Memory** architectures using **Mamba2** (or GRU) controllers paired with **Pseudomode Memory** for long-horizon Reinforcement Learning tasks.

It is designed to be a drop-in benchmark suite for testing memory capabilities in RL agents, supporting tasks like **Delayed Cue**, **Copy Memory**, **Associative Recall**, and **T-Maze**.

## Key Features

*   **Dual Controller Support**: Run with either `gru` (standard RNN) or `mamba` (State Space Model).
*   **Pseudomode Memory**: An external long-term memory module that allows the agent to store and retrieve information over long horizons.
*   **Mac/MPS Support**: Fully compatible with macOS (Apple Silicon).
    *   **Auto-Device Detection**: Automatically uses `mps` (Metal Performance Shaders) on Mac, `cuda` on Linux/Windows, or `cpu` as a fallback.
    *   **MockMamba Fallback**: If the official `mamba_ssm` CUDA kernel is not installed (e.g., on a Mac), the system automatically falls back to a **MockMamba2** implementation (backed by a GRU) to ensure the pipeline remains functional for development and debugging.
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
    pip install torch numpy wandb
    ```

3.  **(Optional) Install Mamba**:
    For optimal performance on CUDA-enabled Linux systems, install the official Mamba kernels:
    ```bash
    pip install mamba-ssm
    ```
    *Note: On macOS, this step is optional. The code will use the MockMamba fallback if `mamba_ssm` is missing.*

## Usage

Run the training script `neural_memory_long_ppo.py`.

### Basic Example (Delayed Cue Task)

```bash
python3 neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --horizon 200 \
    --num-envs 64 \
    --rollout-length 256 \
    --total-updates 2000 \
    --hidden-size 128 \
    --memory-slots 16 \
    --memory-dim 64 \
    --gate-coef 1.0 \
    --track \
    --run-name mamba_pseudomode_delayedcue
```

### Command Line Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--task` | Task to run: `delayed_cue`, `copy_memory`, `assoc_recall`, `tmaze` | `delayed_cue` |
| `--controller` | Controller type: `gru` or `mamba` | `gru` |
| `--horizon` | Task horizon (difficulty) | `200` |
| `--num-envs` | Number of parallel environments | `64` |
| `--rollout-length` | PPO rollout length (should be > horizon) | `256` |
| `--track` | Enable Weights & Biases logging | `False` |

## File Structure

*   `neural_memory_long_ppo.py`: Main entry point. Contains the PPO training loop and agent initialization.
*   `mem_actor_critic_mamba.py`: Defines the `MemActorCritic` agent, the `PseudoModeMemory` module, and the `MockMamba2` fallback class.
*   `neural_memory_final.py`: Contains the vectorized environment implementations (`make_env`).
*   `wandb_integration.py`: A lightweight wrapper for WandB logging.

## License

[MIT License](LICENSE)