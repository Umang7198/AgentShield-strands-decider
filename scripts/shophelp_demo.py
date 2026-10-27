"""Run the gated ShopHelp support agent.

Needs the host LLM in ``.env``. The Decider defaults to the local server; pass ``--fake-decider``
to skip it (policy still runs).

    python scripts/shophelp_demo.py
    python scripts/shophelp_demo.py "Where is order A1001?"
    python scripts/shophelp_demo.py --fake-decider "Refund 40 dollars on A1001"
    python scripts/shophelp_demo.py --port 8080   # review UI at http://127.0.0.1:8080/review
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
        return DeciderChecker(FakeDeciderClient({"arg:*": 0.95, "consent": 0.95}))
    url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")
    return DeciderChecker(HttpDeciderClient(url, timeout_s=20.0))


def _chat(app: object, text: str) -> None:
    from agentshield.shophelp import ShopHelpApp

    assert isinstance(app, ShopHelpApp)
    result = app.agent(text)
    _print_result(app, result)


def _print_result(app: object, result: object) -> None:
    from agentshield.shophelp import ShopHelpApp

    assert isinstance(app, ShopHelpApp)
    stop = getattr(result, "stop_reason", None)
    if stop == "interrupt":
        pending = app.approvals.pending() if app.approvals else []
        if pending:
            rec = pending[0]
            print(f"  [paused] {rec.prompt}")
            print("           type 'approve' or 'deny', or open /review if --port is set")
        else:
            print("  [paused] waiting for human approval")
        return
    print(str(result).rstrip())


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
        help="serve the approval review UI on this port (same process as the agent)",
    )
    args = parser.parse_args()

    from agentshield.api import create_app
    from agentshield.approvals import ApprovalDesk
    from agentshield.shophelp import build_shophelp

    desk = ApprovalDesk()
    app = build_shophelp(
        _model(), _checker(args.fake_decider), on_verdict=_on_verdict(), approvals=desk
    )
    print("ShopHelp ready. Try: 'Where is order A1001?'  (quit to exit)")
    print("Orders: A1001 $89.99 (Sam), A2002 $249.00 (Sam), A3003 $19.50 (Ann).")
    if args.port:
        api = create_app(desk)
        threading.Thread(target=_serve, args=(api, args.port), daemon=True).start()
        print(f"Review UI: http://127.0.0.1:{args.port}/review")

    if args.prompt:
        print(f"you> {args.prompt}")
        _chat(app, args.prompt)
        return 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line in {"quit", "exit"}:
            return 0
        if line in {"approve", "deny"} and app.approvals:
            pending = app.approvals.pending()
            if not pending:
                print("  nothing pending")
                continue
            fn = app.approvals.approve if line == "approve" else app.approvals.deny
            rec, result = fn(pending[0].id, by="demo")
            print(f"  [{rec.status.value}] by demo")
            if result is not None:
                _print_result(app, result)
            continue
        _chat(app, line)


if __name__ == "__main__":
    raise SystemExit(main())
