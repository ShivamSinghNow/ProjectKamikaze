.PHONY: up down logs ps build rebuild kill-drone roster fmt lint clean

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

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

clean:
	docker compose down -v --rmi local
	rm -rf .venv services/*/.venv packages/*/.venv training/.venv
