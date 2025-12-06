FROM --platform=linux/arm64 pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy application files
COPY --chown=appuser:appuser . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Mamba (skip CUDA build for compatibility)
RUN cd mamba-main && MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-cache-dir -e . || echo "Mamba install failed"

# Switch to non-root user
USER appuser

CMD ["python3", "neural_memory_long_ppo.py", "--task", "delayed_cue", "--controller", "gru", "--device", "cpu"]
