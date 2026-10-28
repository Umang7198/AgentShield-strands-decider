"""Merge live Decider + LLM-judge dumps into one comparison table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentshield.eval.metrics import Metrics  # noqa: E402
from agentshield.eval.report import load_eval_json  # noqa: E402


def _row(name: str, metrics: Metrics, extra: str = "") -> str:
    ece = "n/a" if metrics.ece is None else f"{metrics.ece:.3f}"
    p95 = "n/a" if metrics.latency_p95_ms is None else f"{metrics.latency_p95_ms:.0f}"
    p50 = "n/a" if metrics.latency_p50_ms is None else f"{metrics.latency_p50_ms:.0f}"
    return (
        f"| `{name}` | {metrics.n} | {metrics.accuracy:.1%} | {metrics.false_stop_rate:.1%} | "
        f"{metrics.missed_violation_rate:.1%} | {ece} | {p50} | {p95} | {extra} |"
    )


def _metrics(payload: dict[str, Any], mode: str) -> Metrics:
    return Metrics.model_validate(payload["metrics"][mode])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", default="reports/shophelp-live.json")
    parser.add_argument("--judge", default="reports/shophelp-judge.json")
    parser.add_argument("--out", default="reports/shophelp-baselines.md")
    args = parser.parse_args()

    live_path = ROOT / args.live if not Path(args.live).is_absolute() else Path(args.live)
    judge_path = ROOT / args.judge if not Path(args.judge).is_absolute() else Path(args.judge)
    live = load_eval_json(live_path)
    judge = load_eval_json(judge_path)
    live_meta = live.get("meta") or {}
    judge_meta = judge.get("meta") or {}

    lines = [
        "# ShopHelp gate vs baselines",
        "",
        "Gold labels are for the **full** pipeline. `none` never stops a call. `policy` is YAML "
        "only. `full` is policy then Decider then the matrix. `llm-judge` answers the same "
        "noul questions (`arg:*`, `consent`, `claim`) with the host LLM.",
        "",
        f"- Live dump: `{live_path.relative_to(ROOT)}` (question set "
        f"`{live_meta.get('question_set_version', '?')}`, model `{live_meta.get('decider_model', '?')}`)",
        f"- Judge dump: `{judge_path.relative_to(ROOT)}` (model `{judge_meta.get('judge_model', '?')}`)",
        f"- Judge estimated USD: {judge_meta.get('estimated_usd', 'n/a')} "
        f"(input {((judge_meta.get('tokens') or {}).get('input', 0))}, "
        f"output {((judge_meta.get('tokens') or {}).get('output', 0))})",
        "",
        "| system | n | accuracy | false-stop | missed violation | ECE | p50 ms | p95 ms | notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if "none" in live["metrics"]:
        lines.append(_row("none", _metrics(live, "none"), "unguarded"))
    if "policy" in live["metrics"]:
        lines.append(_row("policy", _metrics(live, "policy"), "YAML only"))
    lines.append(_row("full / decider-2B", _metrics(live, "full"), "AgentShield"))
    lines.append(
        _row(
            "llm-judge",
            _metrics(judge, "full" if "full" in judge["metrics"] else next(iter(judge["metrics"]))),
            f"~${float(judge_meta.get('estimated_usd') or 0):.4f}",
        )
    )
    lines.extend(["", "See `docs/benchmark-results.md` for sample size and limitations.", ""])
    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"live_meta": live_meta, "judge_meta": judge_meta}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(out.read_text(encoding="utf-8"))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
