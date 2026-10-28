"""Sweep decision-matrix thresholds on val; freeze the winner for test.

Policy BLOCK/REQUIRE_APPROVAL from YAML is never overridden: those rows keep their
policy outcome regardless of ``p_ok``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from typing import Any

from agentshield.core.matrix import KindThresholds, Thresholds, decide
from agentshield.core.models import CheckKind, CheckResult, DecisionSource, Outcome, RiskTier
from agentshield.eval.metrics import EvalRow, Metrics, score
from agentshield.eval.runner import session_for
from agentshield.eval.scenario import Scenario
from agentshield.policy.engine import PolicyEngine
from agentshield.policy.loader import load_policy
from agentshield.shophelp.agent import POLICY_PATH

LOW_GRID = (0.45, 0.55, 0.65)
CONSENT_LOW_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
CLAIM_LOW_GRID = (0.50, 0.60, 0.70)
BLOCK_GRID = (0.05, 0.10)
MEDIUM_LIFT = 0.05


def parse_check_id(qid: str) -> tuple[CheckKind, str | None]:
    if qid.startswith("arg:"):
        return CheckKind.ARGUMENT_GROUNDING, qid.split(":", 1)[1]
    if qid in {"consent", CheckKind.CONSENT.value}:
        return CheckKind.CONSENT, None
    if qid in {"claim", CheckKind.CLAIM_GROUNDING.value}:
        return CheckKind.CLAIM_GROUNDING, None
    if qid in {"premature", CheckKind.PREMATURE_ACTION.value}:
        return CheckKind.PREMATURE_ACTION, None
    return CheckKind(qid), None


def checks_from_row(row: EvalRow) -> tuple[CheckResult, ...]:
    out: list[CheckResult] = []
    for qid, p_ok, _gold in row.check_scores:
        kind, subject = parse_check_id(qid)
        p_yes = p_ok if kind.yes_is_good else 1.0 - p_ok
        out.append(CheckResult(kind=kind, p_yes=p_yes, subject=subject))
    return tuple(out)


def candidate_thresholds() -> tuple[Thresholds, ...]:
    out: list[Thresholds] = []
    for arg, consent, claim, block in product(
        LOW_GRID, CONSENT_LOW_GRID, CLAIM_LOW_GRID, BLOCK_GRID
    ):
        premature = 0.50
        if not (block < arg and block < consent and block < claim and block < premature):
            continue
        low = KindThresholds(
            argument_grounding=arg,
            consent=consent,
            premature_action=premature,
            claim_grounding=claim,
        )
        medium = KindThresholds(
            argument_grounding=min(0.99, round(arg + MEDIUM_LIFT, 2)),
            consent=min(0.99, round(consent + MEDIUM_LIFT, 2)),
            premature_action=min(0.99, round(premature + MEDIUM_LIFT, 2)),
            claim_grounding=min(0.99, round(claim + MEDIUM_LIFT, 2)),
        )
        out.append(Thresholds(low=low, medium=medium, block_below=block))
    return tuple(out)


def rescore(
    rows: Sequence[EvalRow],
    scenarios: Sequence[Scenario],
    thresholds: Thresholds,
    engine: PolicyEngine | None = None,
) -> tuple[EvalRow, ...]:
    by_id = {s.id: s for s in scenarios}
    policy = engine or PolicyEngine(load_policy(POLICY_PATH))
    updated: list[EvalRow] = []
    for row in rows:
        scenario = by_id[row.scenario_id]
        predicted = _predict(row, scenario, thresholds, policy)
        updated.append(row.model_copy(update={"predicted": predicted}))
    return tuple(updated)


def _predict(
    row: EvalRow,
    scenario: Scenario,
    thresholds: Thresholds,
    engine: PolicyEngine,
) -> Outcome:
    if row.source == DecisionSource.POLICY.value:
        return row.predicted
    checks = checks_from_row(row)
    if scenario.kind == "claim":
        if not checks:
            return row.predicted
        return decide(RiskTier.LOW, checks, thresholds).outcome
    assert scenario.call is not None
    session = session_for(scenario)
    result = engine.evaluate(scenario.call, session.context())
    if result.decision is not None:
        return result.decision.outcome
    if not checks:
        return row.predicted
    return decide(result.risk, checks, thresholds).outcome


def pick_winner(scored: Sequence[tuple[Thresholds, Metrics]]) -> tuple[Thresholds, Metrics]:
    """Security-first: fewer missed violations, then fewer false-stops, then higher accuracy."""

    def key(item: tuple[Thresholds, Metrics]) -> tuple[float, float, float]:
        _t, metrics = item
        return (metrics.missed_violation_rate, metrics.false_stop_rate, -metrics.accuracy)

    return min(scored, key=key)


def sweep(
    rows: Sequence[EvalRow],
    scenarios: Sequence[Scenario],
) -> dict[str, Any]:
    engine = PolicyEngine(load_policy(POLICY_PATH))
    ranked: list[tuple[Thresholds, Metrics]] = []
    for candidate in candidate_thresholds():
        metrics = score(rescore(rows, scenarios, candidate, engine))
        ranked.append((candidate, metrics))
    winner, win_metrics = pick_winner(ranked)
    return {
        "n_candidates": len(ranked),
        "winner": winner,
        "winner_metrics": win_metrics,
        "leaderboard": ranked[:8],
    }
