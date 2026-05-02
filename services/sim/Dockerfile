# Multi-stage so the final image doesn't carry build-essential.
# Cache strategy: COPY full kamikaze_common (its subpackages must be visible
# at install time for hatchling's editable finder) but COPY services/sim/src
# AFTER the install — so edits to sim/main.py rebuild fast.

# ---------- builder ----------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    VIRTUAL_ENV=/opt/venv

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

RUN uv venv "$VIRTUAL_ENV"

WORKDIR /app

# Real kamikaze_common src (subpackages visible to hatchling) +
# sim's pyproject only (its src lands later for fast rebuilds).
COPY packages/kamikaze_common /app/packages/kamikaze_common
COPY services/sim/pyproject.toml /app/services/sim/pyproject.toml

# Stub sim's src so editable install resolves
RUN mkdir -p /app/services/sim/src/sim && \
    touch /app/services/sim/src/sim/__init__.py /app/services/sim/README.md

RUN uv pip install --python "$VIRTUAL_ENV/bin/python" --no-cache \
        "/app/packages/kamikaze_common[envs]" \
        -e /app/services/sim

# ---------- runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv
# Bring the real kamikaze_common src so editable-install path-pointers resolve
COPY --from=builder /app/packages/kamikaze_common /app/packages/kamikaze_common

WORKDIR /app
COPY services/sim /app/services/sim

WORKDIR /app/services/sim
CMD ["python", "-m", "sim.main"]
