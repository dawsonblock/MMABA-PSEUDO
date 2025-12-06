FROM --platform=linux/arm64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir numpy wandb

RUN cd mamba-main && MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-cache-dir -e . || echo "Mamba install failed"

CMD ["python3", "neural_memory_long_ppo.py", "--task", "delayed_cue", "--controller", "gru", "--device", "cpu"]
