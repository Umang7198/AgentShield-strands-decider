"""One command: ShopHelp backend + frontend (and the Decider if it is not already up).

    python app.py
    python app.py --fake-decider
    python app.py --port 8080 --no-browser
    docker compose up --build

Open http://127.0.0.1:8080/ when it prints the desk URL.
Needs OPENAI_API_KEY, OPENAI_BASE_URL and MODEL_ID in .env.
Persists to sqlite:///agentshield.db unless DATABASE_URL is set.
The Docker image binds 0.0.0.0 and uses FAKE_DECIDER=1 (SQLite in /data).
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

DECIDER_MODEL = os.environ.get("DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
DECIDER_DEVICE = os.environ.get("DECIDER_DEVICE", "mlx")

_decider_proc: subprocess.Popen[bytes] | None = None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _on_verdict() -> Callable[..., None]:
    from agentshield.pipeline import Verdict

    def emit(verdict: Verdict) -> None:
        d = verdict.decision
        print(
            f"  [gate] {d.outcome.value:17} {verdict.call.name:<22} "
            f"{d.source.value:8} {verdict.total_ms:6.0f}ms  {d.reason}",
            flush=True,
        )

    return emit


def _model() -> object:
    from strands.models.openai import OpenAIModel

    key = os.environ.get("OPENAI_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    model_id = os.environ.get("MODEL_ID", "")
    missing = [
        n
        for n, v in (("OPENAI_API_KEY", key), ("OPENAI_BASE_URL", base), ("MODEL_ID", model_id))
        if not v
    ]
    if missing:
        raise SystemExit(f"missing in .env: {', '.join(missing)}")
    return OpenAIModel(
        client_args={"api_key": key, "base_url": base, "timeout": 60.0},
        model_id=model_id,
        params={"max_tokens": 1024},
    )


def _checker(fake: bool, decider_url: str) -> Any:
    from agentshield.decider import FakeDeciderClient, HttpDeciderClient
    from agentshield.decider.checker import DeciderChecker

    if fake:
        return DeciderChecker(FakeDeciderClient({"arg:*": 0.95, "consent": 0.95, "claim": 0.95}))
    return DeciderChecker(HttpDeciderClient(decider_url, timeout_s=20.0))


def _decider_url() -> str:
    return os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")


def _decider_port(url: str) -> int:
    parsed = urlparse(url)
    if parsed.port:
        return parsed.port
    return 443 if parsed.scheme == "https" else 80


def decider_ready(url: str, timeout_s: float = 2.0) -> bool:
    try:
        response = httpx.get(f"{url}/health", timeout=timeout_s)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _stop_decider() -> None:
    global _decider_proc
    proc = _decider_proc
    _decider_proc = None
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()


def ensure_decider(url: str) -> None:
    """Use an already-running Decider, or start one in this process tree."""
    global _decider_proc
    if decider_ready(url):
        print(f"Decider already running at {url}", flush=True)
        return
    binary = shutil.which("strands-decider")
    if not binary:
        raise SystemExit(
            "strands-decider is not on PATH. Install it, or run: python app.py --fake-decider"
        )
    port = _decider_port(url)
    print(
        f"Starting Decider on :{port} ({DECIDER_DEVICE}). First load can take a minute…",
        flush=True,
    )
    _decider_proc = subprocess.Popen(
        [
            binary,
            "serve",
            DECIDER_MODEL,
            "--device",
            DECIDER_DEVICE,
            "--port",
            str(port),
        ],
        cwd=ROOT,
        start_new_session=True,
    )
    atexit.register(_stop_decider)
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        if _decider_proc.poll() is not None:
            raise SystemExit(f"Decider exited early with code {_decider_proc.returncode}")
        if decider_ready(url, timeout_s=1.0):
            print(f"Decider ready at {url}", flush=True)
            return
        time.sleep(1.0)
    _stop_decider()
    raise SystemExit(f"Decider did not become ready at {url} within 180s")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ShopHelp: FastAPI backend + desk frontend in one process"
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("APP_HOST", "127.0.0.1"),
        help="bind address (0.0.0.0 inside Docker)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("APP_PORT", "8080")),
        help="desk UI and API port",
    )
    parser.add_argument(
        "--fake-decider",
        action="store_true",
        help="skip the local Decider process (policy still runs)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open the desk in a browser",
    )
    args = parser.parse_args()

    from agentshield.api import create_app
    from agentshield.approvals import ApprovalDesk
    from agentshield.db import Database, default_url, describe_url
    from agentshield.shophelp.runtime import ShopHelpRuntime, banner

    decider_url = _decider_url()
    fake = args.fake_decider or _env_flag("FAKE_DECIDER")
    if not fake:
        ensure_decider(decider_url)

    db_url = default_url(ROOT)
    db = Database.from_url(db_url)
    desk = ApprovalDesk(db.approvals)
    runtime = ShopHelpRuntime(
        _model(),
        _checker(fake, decider_url),
        approvals=desk,
        on_verdict=_on_verdict(),
        persist=db,
    )
    api = create_app(desk, runtime=runtime)
    public = f"http://127.0.0.1:{args.port}/"
    print(banner(), flush=True)
    print(f"Database: {describe_url(db_url)}", flush=True)
    print(f"Decider: {'fake' if fake else decider_url}", flush=True)
    print(f"Desk UI + API: {public}", flush=True)
    print("Ctrl+C to stop.", flush=True)

    if not args.no_browser:

        def _open() -> None:
            time.sleep(0.6)
            webbrowser.open(public)

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    uvicorn.run(api, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
