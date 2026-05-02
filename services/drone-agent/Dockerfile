FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Shared package layer (cache-friendly)
COPY packages/kamikaze_common /app/packages/kamikaze_common
COPY services/drone-agent/pyproject.toml /app/services/drone-agent/pyproject.toml

# Base deps only. KAM-9 will add `uv pip install -e ".[ppo]"` to pull torch+sb3,
# KAM-10 adds `[onnx]`, KAM-11 adds `[mesh]`. Keeping the KAM-5 image lean so
# `make up` is fast on the demo box.
RUN cd /app/services/drone-agent && \
    uv pip install --system --no-cache \
        -e /app/packages/kamikaze_common \
        -e .

COPY services/drone-agent/src /app/services/drone-agent/src

ENV PYTHONPATH=/app/services/drone-agent/src
WORKDIR /app/services/drone-agent

CMD ["python", "-m", "drone_agent"]
