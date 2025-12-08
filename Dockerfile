FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies for Mamba CUDA build
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy application files
COPY --chown=appuser:appuser . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Mamba with CUDA support
# If build fails, the container build fails (no silent failures)
RUN cd mamba-main && pip install --no-cache-dir -e .

# Switch to non-root user
USER appuser

# Set working directory to src
WORKDIR /app/src

# Default: Mamba on CUDA with 64 envs
CMD ["python3", "neural_memory_long_ppo.py", \
    "--task", "delayed_cue", \
    "--controller", "mamba", \
    "--device", "cuda", \
    "--num-envs", "64"]
