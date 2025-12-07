#!/bin/bash

# unified_build.sh
# A unified build script to install the Mamba integration.
# Handles directory navigation and installation for both Mac (Mock) and Linux (CUDA).

set -e

echo "========================================"
echo "   Neural Memory Mamba Build Script"
echo "========================================"

# Get the absolute path of the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
MAMBA_DIR="$PROJECT_ROOT/mamba-main"
SRC_DIR="$PROJECT_ROOT/src"

echo "[*] Project root: $PROJECT_ROOT"
echo "[*] Source directory: $SRC_DIR"
echo "[*] Mamba directory: $MAMBA_DIR"

# 1. Install root dependencies
echo ""
echo "[1/3] Installing root dependencies..."
pip3 install -r "$PROJECT_ROOT/requirements.txt"
echo "[+] Root dependencies installed."

# 2. Check for Mamba directory
if [ -d "$MAMBA_DIR" ]; then
    echo "[*] Found mamba-main at $MAMBA_DIR"
else
    echo "[-] Error: mamba-main directory not found at $MAMBA_DIR"
    exit 1
fi

# 3. Install Mamba
echo ""
echo "[2/3] Installing Mamba..."

# Detect OS
OS="$(uname -s)"
if [ "$OS" == "Darwin" ]; then
    echo "[*] macOS detected."
    echo "[*] Note: Official Mamba CUDA kernels are not supported on Mac."
    echo "[*] Installing mamba-ssm with MAMBA_SKIP_CUDA_BUILD=TRUE to enable CPU/MPS fallback..."
    
    cd "$MAMBA_DIR"
    export MAMBA_SKIP_CUDA_BUILD=TRUE
    pip3 install -e .
    cd "$PROJECT_ROOT"
    
    echo "[+] Setup complete for Mac."
else
    echo "[*] Linux/Other OS detected. Attempting to build Mamba CUDA kernels..."
    
    cd "$MAMBA_DIR"
    
    echo "[*] Running pip3 install -e . in $MAMBA_DIR"
    
    export MAX_JOBS=4
    
    if pip3 install -e .; then
        echo "[+] Mamba installed successfully."
    else
        echo "[-] Mamba build failed. This is expected if you don't have NVCC/CUDA."
        echo "[*] Falling back to MockMamba2 (GRU) mode."
    fi
    
    cd "$PROJECT_ROOT"
fi

echo ""
echo "[3/3] Verifying installation..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
cd "$SRC_DIR"
python3 -c "import neural_memory_long_ppo; print('Neural Memory PPO script imports successfully.')"
cd "$PROJECT_ROOT"

echo ""
echo "========================================"
echo "   Build Complete!"
echo "========================================"
echo "Run the training with:"
echo "cd src && python3 neural_memory_long_ppo.py --task delayed_cue --controller mamba --device cuda"
