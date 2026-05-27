# Dockerfile — Reproducible GPU environment for Tiny LLM
#
# Base: official PyTorch image with CUDA 12.1 and cuDNN 8
# Compatible with: T4, A10G, V100, A100 (all common Colab GPUs)
#
# Build:
#   docker build -t tiny-llm .
#
# Train:
#   docker run --gpus all \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     tiny-llm python train.py --steps 10000
#
# Inference (interactive):
#   docker run --gpus all -it \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     tiny-llm python inference.py --checkpoint checkpoints/best.pt

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config.py model.py data.py train_tokenizer.py train.py inference.py ./

# Create runtime directories
RUN mkdir -p data/raw data/tokenizer checkpoints

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Default command: smoke test (overridden at `docker run` time)
CMD ["python", "train.py", "--smoke-test", "--steps", "100"]

# Health check: verify imports work
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=2 \
    CMD python -c "import torch; import tokenizers; print('OK', torch.__version__)"
