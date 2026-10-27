"""Markdown report for one evaluation run."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from agentshield.eval.metrics import Metrics
from agentshield.eval.scenario import Scenario


def render(
    *,
    metrics: dict[str, Metrics],
    n_scenarios: int,
    backend: str,
    extra: str = "",
) -> str:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# AgentShield evaluation",
        "",
        f"- When: {now}",
        f"- Scenarios: {n_scenarios}",
        f"- Decider backend: {backend}",
        "",
        "Gold labels are for the **full** pipeline (policy, then Decider, then the decision matrix).",
        "Ablations reuse the same labels so the gaps are visible.",
        "",
        extra.rstrip(),
        "",
    ]
    for mode, m in metrics.items():
        lines.extend(_section(mode, m))
    lines.extend(_compare(metrics))
    return "\n".join(line for line in lines if line is not None) + "\n"


def _section(mode: str, m: Metrics) -> list[str]:
    lat = (
        f"p50={m.latency_p50_ms:.1f} p95={m.latency_p95_ms:.1f} p99={m.latency_p99_ms:.1f} "
        f"mean={m.mean_ms:.1f}"
        if m.mean_ms is not None
        else "n/a"
    )
    ece = "n/a" if m.ece is None else f"{m.ece:.3f}"
    rows = [
        f"## Mode `{mode}`",
        "",
        f"- n = {m.n}, exact outcome match **{m.accuracy:.1%}** ({m.correct}/{m.n})",
        f"- False-stop (over-refusal): **{m.false_stop_rate:.1%}**",
        f"- False-block (BLOCK on a gold ALLOW): **{m.false_block_rate:.1%}**",
        f"- Missed violation (ALLOW on a gold stop): **{m.missed_violation_rate:.1%}**",
        f"- Stop precision / recall: {m.stop_precision:.1%} / {m.stop_recall:.1%}",
        f"- ECE (check `p_ok` vs gold): {ece}",
        f"- Latency ms: {lat}",
        "",
        "### Confusion (rows = gold, columns = predicted)",
        "",
        "| gold \\ predicted | allow | clarify | require_approval | block |",
        "|---|---:|---:|---:|---:|",
    ]
    for gold in ("allow", "clarify", "require_approval", "block"):
        c = m.confusion.get(gold)
        if c is None:
            rows.append(f"| {gold} | 0 | 0 | 0 | 0 |")
        else:
            rows.append(f"| {gold} | {c.allow} | {c.clarify} | {c.require_approval} | {c.block} |")
    if m.checks:
        rows.extend(
            [
                "",
                "### Per-check (p_ok >= 0.5 vs gold)",
                "",
                "| kind | n | accuracy | precision | recall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for ch in m.checks:
            rows.append(
                f"| {ch.kind} | {ch.n} | {ch.accuracy:.1%} | {ch.precision:.1%} | {ch.recall:.1%} |"
            )
    rows.extend(["", "### Accuracy by family", ""])
    for family, acc in m.by_family.items():
        rows.append(f"- `{family}`: {acc:.1%}")
    rows.extend(["", "### Accuracy by tag", ""])
    for tag, acc in m.by_tag.items():
        rows.append(f"- `{tag}`: {acc:.1%}")
    rows.append("")
    return rows


def _compare(metrics: dict[str, Metrics]) -> list[str]:
    if len(metrics) < 2:
        return []
    lines = [
        "## Ablation comparison",
        "",
        "| mode | accuracy | false-stop | missed violation | p95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode, m in metrics.items():
        p95 = "n/a" if m.latency_p95_ms is None else f"{m.latency_p95_ms:.0f}"
        lines.append(
            f"| `{mode}` | {m.accuracy:.1%} | {m.false_stop_rate:.1%} | "
            f"{m.missed_violation_rate:.1%} | {p95} |"
        )
    lines.append("")
    return lines


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dataset_summary(scenarios: Sequence[Scenario]) -> str:
    families = Counter(s.family for s in scenarios)
    tags = Counter(tag for s in scenarios for tag in s.tags)
    outcomes = Counter(s.expected.value for s in scenarios)
    lines = ["## Dataset", ""]
    lines.append("Families: " + ", ".join(f"{k}={v}" for k, v in sorted(families.items())))
    lines.append("Gold outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    lines.append("Tags: " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    lines.append("")
    return "\n".join(lines)
