# Qwen3-TTS OpenAI-Compatible API Server
# Multi-stage Dockerfile optimized for GPU/CUDA and CPU deployments

# =============================================================================
# Stage 1: Base image with system dependencies
# =============================================================================
ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE} AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# NVIDIA Container Runtime environment variables (required for PyTorch CUDA detection)
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# Install system dependencies (no build toolchain — all Python deps are wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# =============================================================================
# Stage 2: Heavy Python dependencies (stable layer — rebuilds only when deps change)
# =============================================================================
FROM base AS deps

ARG TORCH_VERSION=2.5.1
ARG FLASH_ATTN_WHEEL_URL=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Single RUN keeps one venv layer instead of stacking torch + deps + flash-attn separately
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    torch==${TORCH_VERSION} \
    torchaudio==${TORCH_VERSION} \
    --index-url https://download.pytorch.org/whl/cu121 \
    && pip install \
    transformers==4.57.3 \
    accelerate==1.12.0 \
    "PyYAML>=6.0" \
    librosa \
    soundfile \
    pydub \
    numpy \
    scipy \
    einops \
    onnxruntime-gpu==1.20.2 \
    sox \
    "fastapi>=0.109.0" \
    "uvicorn[standard]>=0.27.0" \
    python-multipart \
    "pydantic>=2.0.0" \
    inflect \
    aiofiles \
    "httpx>=0.24.0" \
    && pip install "${FLASH_ATTN_WHEEL_URL}" \
    && find /opt/venv -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Stage 3: Application package (small layer — rebuilds on code changes)
# =============================================================================
FROM deps AS app

WORKDIR /build

COPY pyproject.toml README.md MANIFEST.in LICENSE ./
COPY qwen_tts ./qwen_tts
COPY api ./api

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-deps . \
    && find /opt/venv -type d -name __pycache__ -exec rm -rf {} +

# =============================================================================
# Stage 4: Production image (official backend)
# =============================================================================
FROM base AS production

WORKDIR /app

# Stable ~3 GB layer: deps only (not invalidated by app code changes)
COPY --from=deps /opt/venv /opt/venv

# Small layers: app code only (~few MB), rebuilds when source changes
COPY --from=app /opt/venv/lib/python3.11/site-packages/qwen_tts /opt/venv/lib/python3.11/site-packages/qwen_tts
COPY --from=app /opt/venv/lib/python3.11/site-packages/api /opt/venv/lib/python3.11/site-packages/api
COPY --from=app /opt/venv/lib/python3.11/site-packages/qwen_tts-*.dist-info /opt/venv/lib/python3.11/site-packages/

ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV TTS_BACKEND=official

# Expose port
EXPOSE 8880

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

# Run the server
CMD ["python", "-m", "api.main"]

# =============================================================================
# Stage 4: vLLM-Omni backend (with vLLM dependencies)
# =============================================================================
FROM base AS vllm-builder

WORKDIR /build

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Install base dependencies first
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install vLLM (this may take a while)
RUN pip install --no-cache-dir vllm>=0.4.0

# Install the main package dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=1.0.0 \
    librosa \
    soundfile \
    pydub \
    numpy \
    scipy \
    einops \
    onnxruntime-gpu==1.20.2

# Install FastAPI and server dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    python-multipart \
    pydantic>=2.0.0 \
    inflect \
    aiofiles

# Optional: Install flash-attention for better performance
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# =============================================================================
# Stage 5: vLLM-Omni production image
# =============================================================================
FROM base AS vllm-production

WORKDIR /app

# Copy virtual environment from vllm-builder
COPY --from=vllm-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY . .

# Install the package in editable mode with vllm extras
RUN pip install --no-cache-dir -e ".[vllm]"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app
ENV TTS_BACKEND=vllm_omni

# Expose port
EXPOSE 8880

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

# Run the server
CMD ["python", "-m", "api.main"]

# =============================================================================
# CPU-only variant
# =============================================================================
FROM python:3.11-slim AS cpu-base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml README.md ./

# Install PyTorch (CPU version)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir \
    torch>=2.0.0 \
    torchaudio>=2.0.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    accelerate>=1.0.0 \
    librosa \
    soundfile \
    pydub \
    numpy \
    scipy \
    einops \
    onnxruntime \
    fastapi>=0.109.0 \
    uvicorn[standard]>=0.27.0 \
    python-multipart \
    pydantic>=2.0.0 \
    inflect \
    aiofiles

# Copy application code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /tmp/numba_cache \
    && chown -R appuser:appuser /app /tmp/numba_cache
USER appuser

# Environment variables
ENV HOST=0.0.0.0
ENV PORT=8880
ENV WORKERS=1
ENV PYTHONPATH=/app

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "api.main"]
