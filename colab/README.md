# Running MMABA-PSEUDO on Google Colab

This guide explains how to run the Mamba 2 Neural Memory Benchmark on Google Colab's free GPU.

## Quick Start (One Click)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dawsonblock/MMABA-PSEUDO/blob/main/colab/MMABA_Colab.ipynb)

Click the badge above to open the notebook directly in Google Colab!

---

## Step-by-Step Instructions

### Step 1: Open Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with your Google account

### Step 2: Open the Notebook

**Option A: Use the badge link (easiest)**
- Click the "Open in Colab" badge above

**Option B: Open from GitHub**
1. In Colab, click **File → Open notebook**
2. Select the **GitHub** tab
3. Enter the URL: `https://github.com/dawsonblock/MMABA-PSEUDO`
4. Select `colab/MMABA_Colab.ipynb`

**Option C: Upload manually**
1. Download `MMABA_Colab.ipynb` from this repository
2. In Colab, click **File → Upload notebook**
3. Select the downloaded file

### Step 3: Enable GPU Runtime

**This is critical for performance!**

1. Click **Runtime** in the top menu
2. Select **Change runtime type**
3. Under "Hardware accelerator", select **T4 GPU**
4. Click **Save**

![GPU Selection](https://i.imgur.com/placeholder.png)

### Step 4: Run the Notebook

1. Run cells in order from top to bottom
2. The first cell checks GPU availability - if it fails, go back to Step 3
3. Configure training parameters in the "Configuration" section
4. Click "Run Training" to start

---

## Available Tasks

| Task | Description | Difficulty | Est. Time (T4) |
|:--|:--|:--|:--|
| `delayed_cue` | Remember signal for 200 steps | Medium | ~1 hour |
| `copy_memory` | Memorize and reproduce sequence | Hard | ~1.5 hours |
| `assoc_recall` | Learn key→value mappings | Hard | ~1.5 hours |
| `tmaze` | Navigate using start hint | Very Hard | ~2 hours |

---

## Configuration Options

The notebook uses interactive forms. Key parameters:

### Task Settings
- **TASK**: Which benchmark task to run
- **CONTROLLER**: `mamba` (Mamba 2) or `gru` (baseline)
- **HORIZON**: Episode length (default: 200)

### Training Settings
- **TOTAL_UPDATES**: Number of PPO updates (default: 2000)
- **NUM_ENVS**: Parallel environments (default: 64 for GPU)
- **ROLLOUT_LENGTH**: Steps per rollout (default: 256)

### PPO Hyperparameters
- **LEARNING_RATE**: Adam learning rate (default: 3e-4)
- **ENT_COEF**: Entropy bonus for exploration (default: 0.05)
- **CLIP_COEF**: PPO clipping coefficient (default: 0.2)

---

## Expected Results

During training, you'll see output like:

```
[delayed_cue | mamba] Update 100/2000 Step=6553600 Return=0.750 GateMean=0.0142 KL=3.21e-04
```

**What to look for:**
- **Return**: Should trend toward 1.0 (perfect score)
- **GateMean**: Should be ~0.01-0.05 (sparse memory usage)
- **KL**: Should stay below 0.01 (stable training)

---

## Weights & Biases Logging (Optional)

To enable experiment tracking:

1. Set `USE_WANDB = True` in the Configuration section
2. Run the "WandB Login" cell
3. Enter your API key when prompted (get one at [wandb.ai](https://wandb.ai))

---

## Troubleshooting

### "No GPU detected" error
→ Go to **Runtime → Change runtime type → T4 GPU**

### "CUDA out of memory" error
→ Reduce `NUM_ENVS` to 32 or 16

### Training is slow
→ Make sure GPU is enabled (check first cell output)

### "Module not found" error
→ Run all setup cells in order before training

---

## Runtime Estimates

| GPU Type | Time for 2000 updates |
|:--|:--|
| T4 (free tier) | ~1-2 hours |
| V100 | ~30-45 minutes |
| A100 | ~15-20 minutes |

---

## Saving Results

Results are printed to the console. For persistent logging:
1. Enable WandB (see above)
2. Or copy/paste console output before session ends

**Note**: Colab sessions timeout after ~12 hours of inactivity.

---

## Need Help?

- Check the [main README](../README.md) for more details
- See [TRAINING.md](../docs/TRAINING.md) for hyperparameter guidance
- Open an issue on [GitHub](https://github.com/dawsonblock/MMABA-PSEUDO/issues)
