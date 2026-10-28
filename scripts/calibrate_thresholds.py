"""Sweep matrix thresholds on val; freeze YAML; score frozen values on test.

Needs a live Decider dump (`scripts/run_eval.py --live`) so `p_ok` values are real.
Policy BLOCK / REQUIRE_APPROVAL rows are never overridden.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from agentshield.core.thresholds_io import dump_thresholds, write_thresholds  # noqa: E402
from agentshield.eval.calibrate import rescore, sweep  # noqa: E402
from agentshield.eval.dataset import build_dataset  # noqa: E402
from agentshield.eval.metrics import score  # noqa: E402
from agentshield.eval.report import load_eval_json, rows_from_dump  # noqa: E402
from agentshield.eval.split import scenarios_in  # noqa: E402


def _metrics_row(name: str, metrics: object) -> str:
    m = metrics  # Metrics
    ece = "n/a" if m.ece is None else f"{m.ece:.3f}"  # type: ignore[attr-defined]
    return (
        f"| `{name}` | {m.n} | {m.accuracy:.1%} | {m.false_stop_rate:.1%} | "  # type: ignore[attr-defined]
        f"{m.missed_violation_rate:.1%} | {ece} |"  # type: ignore[attr-defined]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate AgentShield decision-matrix thresholds")
    parser.add_argument(
        "--rows",
        default="reports/shophelp-live-all.json",
        help="JSON dump from run_eval.py --live (must include per-row check_scores)",
    )
    parser.add_argument("--write", action="store_true", help="write policies/thresholds.yaml")
    parser.add_argument("--out", default="reports/shophelp-thresholds.md")
    args = parser.parse_args()

    rows_path = Path(args.rows)
    if not rows_path.is_absolute():
        rows_path = ROOT / rows_path
    payload = load_eval_json(rows_path)
    all_rows = rows_from_dump(payload, "full")
    data = build_dataset()
    by_id = {s.id: s for s in data}
    val_scenarios = scenarios_in(data, "val")
    test_scenarios = scenarios_in(data, "test")
    val_ids = {s.id for s in val_scenarios}
    test_ids = {s.id for s in test_scenarios}
    val_rows = tuple(r for r in all_rows if r.scenario_id in val_ids)
    test_rows = tuple(r for r in all_rows if r.scenario_id in test_ids)
    if not val_rows:
        print("no val rows in dump; run with --split all or --split val", file=sys.stderr)
        return 1

    result = sweep(val_rows, [by_id[r.scenario_id] for r in val_rows])
    winner = result["winner"]
    val_before = score(val_rows)
    val_after = result["winner_metrics"]
    test_before = score(test_rows) if test_rows else None
    test_after = (
        score(rescore(test_rows, [by_id[r.scenario_id] for r in test_rows], winner))
        if test_rows
        else None
    )

    notes = (
        "Chosen on the val split of the live Decider dump; frozen for test. "
        "Selection: lowest missed-violation rate, then lowest false-stop, then highest accuracy. "
        "Policy BLOCK still cannot be overridden by confidence."
    )
    if args.write:
        write_thresholds(winner, source="val-sweep-live", notes=notes)

    md_path = Path(args.out)
    if not md_path.is_absolute():
        md_path = ROOT / md_path
    md_path.parent.mkdir(parents=True, exist_ok=True)
    dumped = dump_thresholds(winner, source="val-sweep-live", notes=notes)
    lines = [
        "# Threshold calibration",
        "",
        f"- Source dump: `{rows_path.relative_to(ROOT)}`",
        f"- Val n={len(val_rows)}, test n={len(test_rows)}, candidates={result['n_candidates']}",
        "- Policy short-circuit rows keep their YAML outcome (fail-closed BLOCK is never lifted).",
        "",
        "## Frozen thresholds",
        "",
        "```yaml",
        yaml.safe_dump(dumped, sort_keys=False).rstrip(),
        "```",
        "",
        "## Val (selection) vs test (frozen)",
        "",
        "| slice | n | accuracy | false-stop | missed violation | ECE |",
        "|---|---:|---:|---:|---:|---:|",
        _metrics_row("val before", val_before),
        _metrics_row("val after", val_after),
    ]
    if test_before is not None and test_after is not None:
        lines.append(_metrics_row("test before", test_before))
        lines.append(_metrics_row("test after", test_after))
    lines.extend(["", f"- Wrote `policies/thresholds.yaml`: **{args.write}**", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    json_path = md_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "n_val": len(val_rows),
                "n_test": len(test_rows),
                "n_candidates": result["n_candidates"],
                "thresholds": dumped,
                "val_before": val_before.model_dump(mode="json"),
                "val_after": val_after.model_dump(mode="json"),
                "test_before": test_before.model_dump(mode="json") if test_before else None,
                "test_after": test_after.model_dump(mode="json") if test_after else None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(md_path.read_text(encoding="utf-8"))
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
