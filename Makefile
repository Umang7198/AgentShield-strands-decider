.PHONY: install install-rag test lint format typecheck decider decider-check llm-check app demo eval eval-live eval-judge calibrate db-up db-down docker-up docker-down

DECIDER_MODEL ?= StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
DECIDER_PORT  ?= 8099

install:
	uv pip install -e ".[strands,dev]"

# MiniLM + Chroma for handbook RAG (python app.py auto-detects these).
install-rag:
	uv pip install -e ".[strands,dev,rag]"

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
	python scripts/run_eval.py --split all --write-splits

# Same set against the live local Decider. Writes git-tracked reports/.
eval-live:
	python scripts/run_eval.py --live --split all --modes full,policy,none --out reports/shophelp-live.md

# Host LLM answering the same Decider questions (Baseline B).
eval-judge:
	python scripts/run_judge.py --split all --out reports/shophelp-judge.md

# Sweep thresholds on val rows from the live dump; pass --write to freeze YAML.
calibrate:
	python scripts/calibrate_thresholds.py --rows reports/shophelp-live.json --write
