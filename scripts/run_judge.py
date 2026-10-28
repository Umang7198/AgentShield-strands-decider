"""LLM-as-judge baseline on the same ShopHelp questions the Decider answers.

Uses the host OpenAI-compatible LLM from .env. Writes markdown+JSON under reports/.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agentshield.decider.checker import DeciderChecker  # noqa: E402
from agentshield.eval.dataset import build_dataset  # noqa: E402
from agentshield.eval.hardware import eval_metadata  # noqa: E402
from agentshield.eval.judge import LlmJudgeClient, estimate_usd, load_prices  # noqa: E402
from agentshield.eval.metrics import score  # noqa: E402
from agentshield.eval.report import dataset_summary, dump_eval_json, render, write_report  # noqa: E402
from agentshield.eval.runner import EvalHarness, Mode  # noqa: E402
from agentshield.eval.split import scenarios_in  # noqa: E402


def _modes(raw: str) -> tuple[Mode, ...]:
    return tuple(Mode(part.strip()) for part in raw.split(",") if part.strip())


async def _go(args: argparse.Namespace) -> int:
    scenarios = list(scenarios_in(build_dataset(), args.split))
    if args.limit:
        scenarios = scenarios[: args.limit]
    prices = load_prices()
    client = LlmJudgeClient(timeout_s=args.timeout)
    harness = EvalHarness(DeciderChecker(client))
    modes = _modes(args.modes)
    print(f"judge model={client.model_id} split={args.split} n={len(scenarios)}")
    results = await harness.run(scenarios, modes)
    metrics = {mode: score(rows) for mode, rows in results.items()}
    usd = estimate_usd(client.input_tokens, client.output_tokens, prices)
    meta = {
        **eval_metadata(backend=f"llm-judge {client.model_id}", decider_model=client.model_id),
        "split": args.split,
        "judge_model": client.model_id,
        "base_url": os.environ.get("OPENAI_BASE_URL", ""),
        "prices": prices,
        "tokens": {
            "input": client.input_tokens,
            "output": client.output_tokens,
            "calls": client.calls,
        },
        "estimated_usd": round(usd, 6),
    }
    md_path = Path(args.out)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    extra = dataset_summary(scenarios) + (
        "\nPrices are documented list rates in `eval/prices.yaml`, not a billed invoice.\n"
    )
    text = render(
        metrics=metrics,
        n_scenarios=len(scenarios),
        backend=f"llm-judge {client.model_id}",
        extra=extra,
        meta=meta,
    )
    write_report(md_path, text)
    dump_eval_json(
        md_path.with_suffix(".json"),
        metrics=metrics,
        rows=results,
        meta=meta,
        n_scenarios=len(scenarios),
    )
    print(text)
    print(f"wrote {md_path}")
    print(f"wrote {md_path.with_suffix('.json')}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-as-judge baseline for AgentShield")
    parser.add_argument("--split", choices=("val", "test", "all"), default="test")
    parser.add_argument("--modes", default="full,decider")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--out", default="reports/shophelp-judge.md")
    args = parser.parse_args()
    return asyncio.run(_go(args))


if __name__ == "__main__":
    raise SystemExit(main())
