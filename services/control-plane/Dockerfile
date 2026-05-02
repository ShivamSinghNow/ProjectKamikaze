FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /uvx /usr/local/bin/

WORKDIR /app

COPY packages/kamikaze_common /app/packages/kamikaze_common
COPY services/control-plane/pyproject.toml /app/services/control-plane/pyproject.toml

RUN mkdir -p /app/services/control-plane/src/control_plane && \
    touch /app/services/control-plane/src/control_plane/__init__.py \
          /app/services/control-plane/README.md

RUN cd /app/services/control-plane && \
    uv pip install --system --no-cache \
        -e /app/packages/kamikaze_common \
        -e .

COPY services/control-plane /app/services/control-plane

WORKDIR /app/services/control-plane
EXPOSE 8000

CMD ["uvicorn", "control_plane.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
