# ACE-Step 1.5 - Music Generation Framework
# Following official installation: uv sync

# =============================================================================
# Stage 1: Build stage - use uv for dependency management as per README
# =============================================================================
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites and add deadsnakes PPA for Python 3.11
# Ubuntu 24.04 ships with Python 3.12, but this project requires Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    git \
    curl \
    wget \
    build-essential \
    ninja-build \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (as per README instructions)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Patch nano-vllm to use flash-attn wheel compatible with torch 2.7.x
# The upstream specifies torch2.10 wheel which has ABI incompatibility
RUN sed -i 's|flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3+cu128torch2.10|flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.7|g' \
    acestep/third_parts/nano-vllm/pyproject.toml

# Use uv sync as per README instructions
# This properly resolves all dependencies including flash-attn compatibility
ENV UV_PYTHON=python3.11
ENV MAX_JOBS=4
RUN uv sync

# =============================================================================
# Stage 2: Runtime stage
# =============================================================================
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04 AS runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites and add deadsnakes PPA for Python 3.11
# gcc/libc6-dev/python3.11-dev required by triton for JIT compilation at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    gcc \
    libc6-dev \
    libsndfile1 \
    ffmpeg \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install uv to system-wide location (accessible by non-root user)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv /usr/local/bin/ \
    && mv /root/.local/bin/uvx /usr/local/bin/ 2>/dev/null || true

# Create non-root user for security
RUN groupadd -r acestep && useradd -r -g acestep -d /app -s /sbin/nologin acestep

# Set working directory
WORKDIR /app

# Copy the entire project including .venv from builder
COPY --from=builder --chown=acestep:acestep /app /app

# Create directories for volumes with proper ownership
# Note: /app/.uv-cache is outside /app/.cache volume to avoid permission issues
# All huggingface subdirs must be pre-created for transformers dynamic modules
RUN mkdir -p /app/checkpoints /app/.cache/acestep/tmp /app/.cache/acestep/api_audio \
    /app/.cache/triton /app/.cache/torchinductor \
    /app/.cache/huggingface/hub /app/.cache/huggingface/modules \
    /app/.cache/modelscope \
    /app/.uv-cache \
    /app/outputs \
    && chown -R acestep:acestep /app

# =============================================================================
# Environment Configuration
# =============================================================================

# CUDA configuration
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# PyTorch / CUDA settings
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0"

# Application defaults
ENV ACESTEP_CONFIG_PATH=acestep-v15-turbo
ENV ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ENV ACESTEP_DEVICE=auto
ENV ACESTEP_LM_BACKEND=vllm

# API server configuration
ENV ACESTEP_API_HOST=0.0.0.0
ENV ACESTEP_API_PORT=8001

# Output directory for generated audio (mount as volume for persistence)
ENV ACESTEP_OUTPUT_DIR=/app/outputs

# Gradio configuration
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Cache directories (inside the mounted volume)
ENV TRITON_CACHE_DIR=/app/.cache/triton
ENV TORCHINDUCTOR_CACHE_DIR=/app/.cache/torchinductor
ENV HF_HOME=/app/.cache/huggingface
ENV MODELSCOPE_CACHE=/app/.cache/modelscope

# Disable tokenizer parallelism warning
ENV TOKENIZERS_PARALLELISM=false

# Python optimization
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# uv configuration (must match builder)
ENV UV_PYTHON=python3.11
# uv cache outside /app/.cache volume to avoid permission issues with mounted volumes
ENV UV_CACHE_DIR=/app/.uv-cache

# =============================================================================
# Ports and Volumes
# =============================================================================

# Expose ports for Gradio UI and REST API
EXPOSE 7860 8001

# Define volumes for persistent data
# - /app/checkpoints: Model checkpoints
# - /app/.cache: HuggingFace cache, triton cache, etc.
# - /app/outputs: Generated audio files (organized by date/session)
VOLUME ["/app/checkpoints", "/app/.cache", "/app/outputs"]

# =============================================================================
# Startup
# =============================================================================

# Copy entrypoint script from build context (not from builder, to get latest version)
COPY --chmod=755 docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user for security
USER acestep

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD curl -sf http://localhost:${GRADIO_SERVER_PORT:-7860}/ || curl -sf http://localhost:${ACESTEP_API_PORT:-8001}/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--", "docker-entrypoint.sh"]

# Default command: start Gradio UI with API enabled
CMD ["gradio"]
