FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Full kamikaze_common (subpackages visible to hatchling at install time)
# + drone-agent's pyproject only. Service src lands after install.
COPY packages/kamikaze_common /app/packages/kamikaze_common
COPY services/drone-agent/pyproject.toml /app/services/drone-agent/pyproject.toml

RUN mkdir -p /app/services/drone-agent/src/drone_agent && \
    touch /app/services/drone-agent/src/drone_agent/__init__.py \
          /app/services/drone-agent/README.md

RUN cd /app/services/drone-agent && \
    uv pip install --system --no-cache \
        -e /app/packages/kamikaze_common \
        -e .

# Real service src — invalidates only this layer + below.
COPY services/drone-agent /app/services/drone-agent

ENV PYTHONPATH=/app/services/drone-agent/src
WORKDIR /app/services/drone-agent

CMD ["python", "-m", "drone_agent"]
