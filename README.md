# MMABA-PSEUDO: Mamba 2 Neural Memory Benchmark

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/MMABA-PSEUDO/blob/main/colab/MMABA_Colab.ipynb)

A benchmark suite for testing **long-horizon memory capabilities** in reinforcement learning agents using **Mamba 2** state-space models.

## Features

- 🧠 **Mamba 2 Controller** — State-space model for efficient sequence processing
- 💾 **PseudoMode Memory** — External memory bank with K slots for long-term storage
- 🎮 **4 Benchmark Tasks** — Delayed cue, copy memory, associative recall, T-maze
- 🍎 **MPS Compatible** — Runs on Apple Silicon via pure PyTorch fallbacks
- 🚀 **CUDA Optimized** — 10-50x faster on GPU with Triton kernels

## Project Structure

```
MMABA-PSEUDO/
├── src/                    # Core source code
│   ├── neural_memory_long_ppo.py  # Main training script
│   ├── mem_actor_critic_mamba.py  # Actor-critic with Mamba
│   ├── neural_memory_final.py     # Vectorized environments
│   ├── mamba_compat.py            # MPS/CPU compatibility layer
│   └── wandb_integration.py       # Weights & Biases logging
├── docs/                   # Documentation
│   ├── TRAINING.md         # Training guide
│   └── RESULTS.md          # Experiment results
├── colab/                  # Google Colab notebooks
│   └── MMABA_Colab.ipynb   # Interactive training notebook
├── scripts/                # Build and run scripts
│   ├── unified_build.sh    # Dependency installer
│   └── run_docker.sh       # Docker runner
├── mamba-main/             # Patched Mamba SSM library
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker compose config
└── requirements.txt        # Python dependencies
```

## Quick Start

### Option 1: Google Colab (Recommended)

Click the badge above to run on free GPU!

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/dawsonblock/MMABA-PSEUDO.git
cd MMABA-PSEUDO

# Install dependencies
pip install -r requirements.txt

# Install Mamba (skip CUDA build on Mac)
cd mamba-main && MAMBA_SKIP_CUDA_BUILD=TRUE pip install -e . && cd ..

# Run training
cd src
python neural_memory_long_ppo.py \
    --task delayed_cue \
    --controller mamba \
    --device mps \      # or cuda
    --num-envs 4 \      # 64 for CUDA
    --total-updates 2000
```

### Option 3: Docker

```bash
docker-compose up --build
```

## Tasks

| Task | Description | Horizon |
|:--|:--|:--|
| `delayed_cue` | Remember signal for N steps | 200 |
| `copy_memory` | Memorize and reproduce sequence | 240 |
| `assoc_recall` | Learn key→value pairs | 216 |
| `tmaze` | Navigate using start hint | 1002 |

## Performance

| Device | Time per 2000 updates | Batch Size |
|:--|:--|:--|
| **CUDA (T4)** | ~1-2 hours | 64 envs |
| **MPS (M1/M2)** | ~13-20 hours | 4 envs |
| **CPU** | ~7-14 days | 1 env |

## Documentation

- [Training Guide](docs/TRAINING.md) — Hyperparameters, usage, troubleshooting
- [Results](docs/RESULTS.md) — Experiment results and analysis

## License

MIT License