"""Run the ShopHelp evaluation set and write a markdown report.

python scripts/run_eval.py              # oracle Decider (no model server)
python scripts/run_eval.py --live       # local Decider at DECIDER_URL
python scripts/run_eval.py --modes full,policy,none
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agentshield.eval.dataset import build_dataset  # noqa: E402
from agentshield.eval.metrics import score  # noqa: E402
from agentshield.eval.report import dataset_summary, render, write_report  # noqa: E402
from agentshield.eval.runner import EvalHarness, Mode, OracleHarness  # noqa: E402


def _modes(raw: str) -> tuple[Mode, ...]:
    return tuple(Mode(part.strip()) for part in raw.split(",") if part.strip())


async def _go(args: argparse.Namespace) -> int:
    scenarios = build_dataset()
    modes = _modes(args.modes)
    if args.live:
        from agentshield.decider import DeciderChecker, HttpDeciderClient

        url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")
        harness: EvalHarness = EvalHarness(DeciderChecker(HttpDeciderClient(url, timeout_s=20.0)))
        backend = f"live {url}"
    else:
        harness = OracleHarness()
        backend = "oracle (gold labels as p_yes)"

    results = await harness.run(scenarios, modes)
    metrics = {mode: score(rows) for mode, rows in results.items()}
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    kind = "live" if args.live else "oracle"
    out_dir = ROOT / "eval" / "results"
    md_path = Path(args.out) if args.out else out_dir / f"{stamp}-{kind}.md"
    text = render(
        metrics=metrics,
        n_scenarios=len(scenarios),
        backend=backend,
        extra=dataset_summary(scenarios),
    )
    write_report(md_path, text)
    json_path = md_path.with_suffix(".json")
    json_path.write_text(
        json.dumps({m: metrics[m].model_dump(mode="json") for m in metrics}, indent=2),
        encoding="utf-8",
    )
    print(text)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AgentShield evaluation harness")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the real Decider (needs `make decider`); default is the oracle FakeDecider",
    )
    parser.add_argument(
        "--modes",
        default="full,policy,decider,none",
        help="comma-separated: full,policy,decider,none",
    )
    parser.add_argument(
        "--out", default="", help="markdown report path (default eval/results/<stamp>.md)"
    )
    args = parser.parse_args()
    return asyncio.run(_go(args))


if __name__ == "__main__":
    raise SystemExit(main())
