"""Run the gated ShopHelp support agent.

Needs the host LLM in ``.env``. The Decider defaults to the local server; pass ``--fake-decider``
to skip it (policy still runs).

    python scripts/shophelp_demo.py
    python scripts/shophelp_demo.py "Where is order A1001?"
    python scripts/shophelp_demo.py --fake-decider "Refund 40 dollars on A1001"
    python scripts/shophelp_demo.py --port 8080   # desk UI at http://127.0.0.1:8080/
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Running as a script: make sure the src layout is importable without an install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentshield.decider.checker import DeciderChecker  # noqa: E402

load_dotenv(ROOT / ".env")


def _on_verdict() -> Callable[..., None]:
    from agentshield.pipeline import Verdict

    def emit(verdict: Verdict) -> None:
        d = verdict.decision
        print(
            f"  [gate] {d.outcome.value:17} {verdict.call.name:<22} "
            f"{d.source.value:8} {verdict.total_ms:6.0f}ms  {d.reason}"
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


def _checker(fake: bool) -> DeciderChecker:
    from agentshield.decider import FakeDeciderClient, HttpDeciderClient

    if fake:
        return DeciderChecker(FakeDeciderClient({"arg:*": 0.95, "consent": 0.95, "claim": 0.95}))
    url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")
    return DeciderChecker(HttpDeciderClient(url, timeout_s=20.0))


def _print_state(state: dict[str, object]) -> None:
    if state.get("paused"):
        print(f"  [paused] {state.get('reply')}")
        print("           type 'approve' or 'deny', or open the desk UI if --port is set")
        return
    reply = str(state.get("reply") or "").rstrip()
    if reply:
        print(reply)
    gates: list[dict[str, object]] = []
    transcript = state.get("transcript")
    rows = transcript if isinstance(transcript, list) else []
    for row in reversed(rows):
        if isinstance(row, dict) and row.get("role") == "gate":
            gates.append(row)
            continue
        break
    for row in reversed(gates):
        print(f"  [gate] {row.get('text')}")


def _serve(app_api: Any, port: int) -> None:
    import uvicorn

    uvicorn.run(app_api, host="127.0.0.1", port=port, log_level="warning")


def main() -> int:
    parser = argparse.ArgumentParser(description="ShopHelp support agent, gated by AgentShield")
    parser.add_argument("prompt", nargs="?", help="one user message; omit for a REPL")
    parser.add_argument(
        "--fake-decider",
        action="store_true",
        help="use a scripted Decider (always confident) so you can run without the local server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="serve the Hearth & Cable desk UI on this port (same process as the agent)",
    )
    args = parser.parse_args()

    from agentshield.api import create_app
    from agentshield.approvals import ApprovalDesk
    from agentshield.shophelp.runtime import ShopHelpRuntime, banner

    desk = ApprovalDesk()
    runtime = ShopHelpRuntime(
        _model(), _checker(args.fake_decider), approvals=desk, on_verdict=_on_verdict()
    )
    print(banner())
    if args.port:
        api = create_app(desk, runtime=runtime)
        threading.Thread(target=_serve, args=(api, args.port), daemon=True).start()
        print(f"Desk UI: http://127.0.0.1:{args.port}/")

    if args.prompt:
        print(f"you> {args.prompt}")
        _print_state(runtime.chat(args.prompt))
        return 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line in {"quit", "exit"}:
            return 0
        if line == "reset":
            runtime.reset()
            print("  catalogue reset")
            continue
        if line in {"approve", "deny"}:
            pending = runtime.approvals.pending()
            if not pending:
                print("  nothing pending")
                continue
            state = runtime.resolve(pending[0].id, line, by="demo")
            print(f"  [{line}d] by demo")
            _print_state(state)
            continue
        _print_state(runtime.chat(line))


if __name__ == "__main__":
    raise SystemExit(main())
