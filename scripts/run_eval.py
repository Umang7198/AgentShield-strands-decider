"""Run the ShopHelp evaluation set and write a markdown report.

python scripts/run_eval.py                         # oracle Decider (no model server)
python scripts/run_eval.py --live --split test     # local Decider at DECIDER_URL
python scripts/run_eval.py --live --out reports/shophelp-live.md
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agentshield.eval.dataset import build_dataset  # noqa: E402
from agentshield.eval.hardware import eval_metadata  # noqa: E402
from agentshield.eval.metrics import score  # noqa: E402
from agentshield.eval.report import dataset_summary, dump_eval_json, render, write_report  # noqa: E402
from agentshield.eval.runner import EvalHarness, Mode, OracleHarness  # noqa: E402
from agentshield.eval.split import scenarios_in, write_splits  # noqa: E402


def _modes(raw: str) -> tuple[Mode, ...]:
    return tuple(Mode(part.strip()) for part in raw.split(",") if part.strip())


async def _go(args: argparse.Namespace) -> int:
    if args.write_splits:
        write_splits()
    scenarios = list(build_dataset())
    split = args.split
    chosen = list(scenarios_in(scenarios, split))
    if args.limit:
        chosen = chosen[: args.limit]
    modes = _modes(args.modes)
    if args.live:
        from agentshield.decider import DeciderChecker, HttpDeciderClient

        url = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")
        harness: EvalHarness = EvalHarness(DeciderChecker(HttpDeciderClient(url, timeout_s=20.0)))
        backend = f"live {url}"
        meta = eval_metadata(backend=backend)
    else:
        harness = OracleHarness()
        backend = "oracle (gold labels as p_yes)"
        meta = eval_metadata(backend=backend, decider_model="oracle")

    meta = {**meta, "split": split, "n_limit": args.limit}
    print(f"eval {backend} split={split} n={len(chosen)} modes={[m.value for m in modes]}")
    results = await harness.run(chosen, modes)
    metrics = {mode: score(rows) for mode, rows in results.items()}
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    kind = "live" if args.live else "oracle"
    if args.out:
        md_path = Path(args.out)
        if not md_path.is_absolute():
            md_path = ROOT / md_path
    elif args.live:
        md_path = ROOT / "reports" / f"shophelp-{kind}-{split}.md"
    else:
        md_path = ROOT / "eval" / "results" / f"{stamp}-{kind}-{split}.md"
    text = render(
        metrics=metrics,
        n_scenarios=len(chosen),
        backend=backend,
        extra=dataset_summary(chosen),
        meta=meta,
    )
    write_report(md_path, text)
    json_path = md_path.with_suffix(".json")
    dump_eval_json(json_path, metrics=metrics, rows=results, meta=meta, n_scenarios=len(chosen))
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
        "--split",
        choices=("val", "test", "all"),
        default="all",
        help="frozen split from eval/splits.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="optional cap (debug)")
    parser.add_argument(
        "--out",
        default="",
        help="markdown report path (live default: reports/shophelp-live-<split>.md)",
    )
    parser.add_argument(
        "--write-splits",
        action="store_true",
        help="refresh eval/splits.json from the current dataset before running",
    )
    args = parser.parse_args()
    return asyncio.run(_go(args))


if __name__ == "__main__":
    raise SystemExit(main())
