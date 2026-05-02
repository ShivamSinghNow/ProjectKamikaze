.PHONY: up down logs ps build rebuild kill-drone roster fmt lint clean formation sim-logs world-state

up:
	docker compose up -d --build

down:
	docker compose down -v

logs:
	docker compose logs -f --tail=200

ps:
	docker compose ps

build:
	docker compose build

rebuild:
	docker compose build --no-cache

kill-drone:
	@echo "Killing drone-3 to demo graceful degradation (KAM-13)"
	docker compose kill drone-3

roster:
	curl -sS http://localhost:8000/roster | python -m json.tool

# KAM-8: host-side spike (PyBullet GUI, 5 drones in cardinal-sentry formation).
# Requires `uv pip install -e packages/kamikaze_common[envs]` in your host venv.
formation:
	uv run --extra envs python tools/formation_demo.py

sim-logs:
	docker compose logs -f --tail=200 sim

# Tail world:state pub events. msgpack payloads — pretty print via `python -c`.
world-state:
	docker compose exec redis redis-cli SUBSCRIBE world:state

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	docker compose down -v --rmi local
	rm -rf .venv services/*/.venv packages/*/.venv training/.venv
