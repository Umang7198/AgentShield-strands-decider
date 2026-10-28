# Setup

AgentShield is a decision-gated support desk (Hearth & Cable / ShopHelp). Maya (the agent) proposes tool calls; a policy engine and the Strands Decider 2B allow, clarify, send to Priya, or block.

Two ways to run it:

| | App (UI + API + SQLite) | Decider 2B |
|---|---|---|
| **Local** | `python app.py` | Started for you on `:8099` (MLX on Apple silicon) |
| **Docker** | `docker compose up --build` | Run on the **host**; the container calls it |

The Decider is Apple MLX on a Mac. Do not put it inside the Linux Docker image. The container talks to the host at `http://host.docker.internal:8099`.

Desk UI: **http://127.0.0.1:8080/**

---

## Prerequisites

- Python **3.12** (not 3.13)
- [uv](https://docs.astral.sh/uv/) or pip
- An OpenAI-compatible host LLM (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_ID`)
- [strands-decider](https://pypi.org/project/strands-decider/) for the real gate (`pip install strands-decider`)
- Docker Desktop only if you use the container path

---

## 1. Environment file

Never commit `.env`.

```bash
cp .env.example .env
```

Fill at least:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
MODEL_ID=...
```

| Variable | Local `python app.py` | Docker app |
|---|---|---|
| `DECIDER_URL` | `http://127.0.0.1:8099` | `http://host.docker.internal:8099` |
| `FAKE_DECIDER` | unset / `0` | `0` for the real host Decider; default in compose is `1` (policy only) |

Inside Docker, `127.0.0.1` is the container, not your Mac. Use `host.docker.internal` there.

---

## 2. Local (no Docker)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -e ".[strands,dev]"
# or: pip install -e ".[strands,dev]"
pip install strands-decider
```

Start the desk. If nothing is listening on `:8099`, this also starts the Decider (`--device mlx`):

```bash
source .venv/bin/activate
python app.py
# same: make app
```

Open **http://127.0.0.1:8080/**.

Useful flags:

```bash
python app.py --fake-decider      # no 2B model; policy still runs
python app.py --port 8080 --no-browser
```

Run the Decider in its own terminal instead:

```bash
make decider
# strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --device mlx --port 8099
```

First load can take a minute. Check:

```bash
python scripts/check_decider.py
python scripts/check_llm.py
```

SQLite file: `./agentshield.db` (git-ignored). Reset in the UI reseeds orders and pending cards; audit rows stay.

On Linux without a GPU, serve the Decider with `--device cpu` and set `DECIDER_DEVICE=cpu`.

---

## 3. Docker (app in a container, Decider on the host)

This is the path for a new machine that should only “up the container” for the desk, while the 2B model stays on the Mac.

**Terminal A — Decider (host, MLX):**

```bash
source .venv/bin/activate
make decider
```

Wait until it prints that it is serving on `http://127.0.0.1:8099`.

**`.env` for Docker** (in addition to the LLM keys):

```bash
FAKE_DECIDER=0
DECIDER_URL=http://host.docker.internal:8099
```

If `.env` still has `DECIDER_URL=http://127.0.0.1:8099`, Compose will pass that into the container and the app will not see the host Decider.

**Terminal B — app:**

```bash
docker compose up --build
# same: make docker-up
```

Logs should include:

```text
Decider already running at http://host.docker.internal:8099
Desk UI + API: http://127.0.0.1:8080/
```

Open **http://127.0.0.1:8080/**. Gate tape `source` should show `decider` (or `policy` when a YAML rule fires), not a fake client.

Stop: Ctrl+C, or `docker compose down` / `make docker-down`. SQLite lives in the `agentshield_data` volume.

Do not run `python app.py` and Docker on **8080** at the same time.

### If the container cannot reach the Decider

The server binds `127.0.0.1` by default. On Docker Desktop for Mac that is usually enough. If health checks from the container fail, restart the Decider on all interfaces (no auth — only on a trusted machine):

```bash
strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 \
  --device mlx --host 0.0.0.0 --port 8099
```

From the host:

```bash
curl -s http://127.0.0.1:8099/health
docker compose exec app python -c "import httpx; print(httpx.get('http://host.docker.internal:8099/health').status_code)"
```

Both should return 200.

### Fake Decider in Docker

Compose defaults to `FAKE_DECIDER=1` if you omit it. Policy still gates; the 2B model is not called. Set `FAKE_DECIDER=0` for the real host Decider.

---

## 4. Optional Postgres

SQLite is the default (local file or `/data/agentshield.db` in Docker). Postgres is extra:

```bash
uv pip install -e ".[postgres]"
docker compose --profile postgres up -d postgres
```

```bash
DATABASE_URL=postgresql+psycopg://agentshield:agentshield@127.0.0.1:5432/agentshield
```

From the app container, use host `postgres` and profile networking, not `127.0.0.1`.

---

## 5. Tests and eval

```bash
source .venv/bin/activate
make test          # pytest (skips live Decider / LLM)
make eval          # oracle eval, no model server
make eval-live     # against the local Decider on :8099
```

---

## 6. Desk recap

Four panes: customer chat (Maya), gate tape, Priya’s approval desk, ops board. Scripted scenes sit in the top bar.

| Chip | Meaning |
|---|---|
| allow | Tool ran |
| clarify | Maya must ask; tool did not run |
| require_approval | Paused for Priya |
| block | Hard stop |

`source` is who decided: `policy` (YAML), `decider` (2B + matrix), or `fallback` (Decider down).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `missing in .env: OPENAI_API_KEY, ...` | Copy `.env.example` and fill the LLM fields |
| `strands-decider is not on PATH` | `pip install strands-decider`, or `python app.py --fake-decider` |
| Docker app ignores the Mac Decider | `FAKE_DECIDER=0` and `DECIDER_URL=http://host.docker.internal:8099` |
| Port 8080 already in use | Stop the other `python app.py` / Compose stack |
| First gate call ~10–15 s | Decider warmup, not a hang |
| Chat refused while Priya has a card | Approve or deny, or Reset |
