# syntax=docker/dockerfile:1
# Tichu inference serving image (ADR-0027). CPU-only, single uvicorn worker.
# Build via docker/build.ps1 — the model weights live OUTSIDE the repo and are
# supplied as a BuildKit named build-context ("models"), not copied into it.
#
#   docker build --build-context models=<export-root> --build-arg MODEL=<dir> ...
#
# The selected model dir's CONTENTS are flattened into /app/models/ so the
# baked config (configs/serve_docker.yaml) can use stable paths.

ARG PYTHON_VERSION=3.13
# Pin torch to the version that produced the TorchScript exports (2.11.0) so the
# artifacts load without version skew. CPU wheel only — see the index-url below.
ARG TORCH_VERSION=2.11.0

# ---- builder: install everything into a relocatable venv ---------------------
FROM python:${PYTHON_VERSION}-slim AS builder
ARG TORCH_VERSION
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /build
RUN python -m venv /opt/venv
# CPU-only torch first, from the dedicated index and pinned, so the package's
# `ml` extra (torch>=2.0) is already satisfied and pip never pulls a CUDA build.
RUN pip install "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
# Then the package + serving extra (fastapi, uvicorn[standard]). Core deps
# (numpy, pyarrow, pyyaml, tqdm, trueskill, zstandard) come from the package.
COPY pyproject.toml ./
COPY src ./src
RUN pip install ".[inference]"

# ---- runtime: clean slim image, non-root, weights baked in -------------------
FROM python:${PYTHON_VERSION}-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
RUN useradd --create-home --uid 10001 tichu
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs/serve_docker.yaml /app/configs/serve_docker.yaml
# Bake the selected model. The build-context "models" is the export root; MODEL
# names the subdir. Its contents (policy.pt + the 3 standalone nets) flatten
# into /app/models/ to match the stable paths in serve_docker.yaml.
ARG MODEL=bc_full_100k_v5
COPY --from=models ${MODEL}/ /app/models/

USER tichu
EXPOSE 8000
# Liveness via the app's own /health, using stdlib urllib (slim ships no curl).
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["serve_inference", "--config", "/app/configs/serve_docker.yaml", "--host", "0.0.0.0", "--port", "8000"]
