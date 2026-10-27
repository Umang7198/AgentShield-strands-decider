"""Run the gated ShopHelp support agent.

Needs the host LLM in ``.env``. The Decider defaults to the local server; pass ``--fake-decider``
to skip it (policy still runs).

    python scripts/shophelp_demo.py
    python scripts/shophelp_demo.py "Where is order A1001?"
    python scripts/shophelp_demo.py --fake-decider "Refund 500 dollars for A1001"
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

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


def _chat(agent: object, text: str) -> None:
    result = agent(text)  # type: ignore[operator]
    stop = getattr(result, "stop_reason", None)
    if stop == "interrupt":
        print("  [paused] a high-risk call is waiting for human approval (approval API is next).")
        return
    print(str(result).rstrip())


def main() -> int:
    parser = argparse.ArgumentParser(description="ShopHelp support agent, gated by AgentShield")
    parser.add_argument("prompt", nargs="?", help="one user message; omit for a REPL")
    parser.add_argument(
        "--fake-decider",
        action="store_true",
        help="use a scripted Decider (always confident) so you can run without the local server",
    )
    args = parser.parse_args()

    from agentshield.shophelp import build_shophelp

    app = build_shophelp(_model(), _checker(args.fake_decider), on_verdict=_on_verdict())
    print("ShopHelp ready. Try: 'Where is order A1001?'  (quit to exit)")
    print("Orders: A1001 $89.99 (Sam), A2002 $249.00 (Sam), A3003 $19.50 (Ann).")

    if args.prompt:
        print(f"you> {args.prompt}")
        _chat(app.agent, args.prompt)
        return 0

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line or line in {"quit", "exit"}:
            return 0
        _chat(app.agent, line)


if __name__ == "__main__":
    raise SystemExit(main())
