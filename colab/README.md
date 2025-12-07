# Google Colab Notebooks

This directory contains Jupyter notebooks optimized for Google Colab.

## Available Notebooks

| Notebook | Description |
|:--|:--|
| [MMABA_Colab.ipynb](MMABA_Colab.ipynb) | Main training notebook with interactive UI |

## Quick Start

1. Open notebook in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/MMABA-PSEUDO/blob/main/colab/MMABA_Colab.ipynb)

2. Select GPU runtime: **Runtime → Change runtime type → T4 GPU**

3. Run all cells in order

## Features

- Interactive parameter forms
- GPU availability checking
- Automatic repo cloning
- WandB logging support
- Benchmark suite option
- Mamba vs GRU comparison

## Expected Runtime

| Task | T4 GPU | V100 | A100 |
|:--|:--|:--|:--|
| Single task (2000 updates) | ~1-2 hours | ~30-45 min | ~15-20 min |
| Full benchmark (4 tasks) | ~4-6 hours | ~2-3 hours | ~1-1.5 hours |
