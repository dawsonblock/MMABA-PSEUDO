#!/bin/bash

# run_docker.sh
# Build and run the Docker container for training

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo "[*] Building Docker image..."
docker build -t mmaba-pseudo:latest .

echo "[*] Running container..."
docker run --gpus all -v "$PROJECT_ROOT/wandb:/app/wandb" mmaba-pseudo:latest "$@"
