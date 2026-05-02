FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app

# Bring in the shared package first so its layer caches independently
COPY packages/kamikaze_common /app/packages/kamikaze_common
COPY services/control-plane/pyproject.toml /app/services/control-plane/pyproject.toml

RUN cd /app/services/control-plane && \
    uv pip install --system --no-cache \
        -e /app/packages/kamikaze_common \
        -e .

COPY services/control-plane/src /app/services/control-plane/src

WORKDIR /app/services/control-plane
EXPOSE 8000

CMD ["uvicorn", "control_plane.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
