.PHONY: install test lint format typecheck decider decider-check llm-check app demo eval eval-live db-up db-down docker-up docker-down

DECIDER_MODEL ?= StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
DECIDER_PORT  ?= 8099

install:
	uv pip install -e ".[strands,dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

typecheck:
	mypy src

# Run the Decider model server (Apple silicon / MLX). Blocks the terminal.
decider:
	strands-decider serve $(DECIDER_MODEL) --device mlx --port $(DECIDER_PORT)

# Check the Decider server is up and answering.
decider-check:
	python scripts/check_decider.py

# Check the host-agent LLM endpoint (reads .env).
llm-check:
	python scripts/check_llm.py

# One container on a new machine: backend + frontend + SQLite.
# Copy .env.example to .env first (LLM keys). Then: make docker-up
docker-up:
	docker compose up --build

docker-down:
	docker compose down

# Optional Postgres (not required for docker-up). Then set DATABASE_URL.
db-up:
	docker compose --profile postgres up -d postgres

db-down:
	docker compose --profile postgres down

# One process: desk UI + API. Starts the Decider too if :8099 is down.
# Persists to sqlite:///agentshield.db unless DATABASE_URL is set.
app:
	python app.py

# Interactive ShopHelp agent (needs .env; Decider at DECIDER_URL unless --fake-decider).
demo:
	python scripts/shophelp_demo.py --port 8080

# Evaluation harness. Oracle uses gold labels as Decider answers (no model server).
eval:
	python scripts/run_eval.py

# Same set against the live local Decider.
eval-live:
	python scripts/run_eval.py --live --modes full,policy,none
