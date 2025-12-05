FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir numpy wandb && \
    pip install --no-cache-dir -e ./mamba-main || echo "Mamba install failed, will use MockMamba fallback"

CMD ["python3", "neural_memory_long_ppo.py", "--task", "delayed_cue", "--controller", "mamba", "--device", "cpu"]
