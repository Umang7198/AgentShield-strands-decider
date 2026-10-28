"""Run the labelled set through the full gate and its ablations."""

from __future__ import annotations

import time
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from agentshield.core.matrix import Thresholds
from agentshield.core.models import DecisionSource, Outcome
from agentshield.core.thresholds_io import load_thresholds
from agentshield.decider.checker import DeciderChecker
from agentshield.decider.fake import FakeDeciderClient
from agentshield.eval.metrics import EvalRow
from agentshield.eval.scenario import CheckLabel, Scenario
from agentshield.pipeline import Shield
from agentshield.policy.engine import PolicyEngine
from agentshield.policy.loader import load_policy
from agentshield.policy.session import SessionState
from agentshield.shophelp.agent import POLICY_PATH


class Mode(StrEnum):
    FULL = "full"
    POLICY = "policy"  # deterministic rules only; no rule -> ALLOW
    DECIDER = "decider"  # skip rules; still uses tool risk metadata
    NONE = "none"  # never stop anything


def oracle_script(labels: Sequence[CheckLabel]) -> dict[str, float]:
    """A FakeDecider that agrees with the gold labels. Specific ids beat wildcards."""
    script: dict[str, float] = {}
    for label in labels:
        script[label.id] = 0.92 if label.ok else 0.12
    script.setdefault("arg:*", 0.92)
    script.setdefault("consent", 0.92)
    script.setdefault("claim", 0.92)
    script.setdefault("premature", 0.08)
    return script


def session_for(scenario: Scenario) -> SessionState:
    state = SessionState(scenario.id, facts=scenario.facts)
    for name, count in scenario.prior_calls.items():
        for _ in range(count):
            state.record_call(name)
    return state


class EvalHarness:
    def __init__(
        self,
        checker: DeciderChecker,
        *,
        policy_path: Path | None = None,
        thresholds: Thresholds | None = None,
    ) -> None:
        policy = load_policy(policy_path or POLICY_PATH)
        self._thresholds = thresholds if thresholds is not None else load_thresholds()
        self._full = PolicyEngine(policy)
        self._decider_only = PolicyEngine(
            policy.model_copy(update={"rules": (), "default_for_unknown_tools": "allow"})
        )
        self._checker = checker
        self._shield = Shield(self._full, checker, agent_id="eval", thresholds=self._thresholds)
        self._decider_shield = Shield(
            self._decider_only, checker, agent_id="eval", thresholds=self._thresholds
        )

    async def run_one(self, scenario: Scenario, mode: Mode) -> EvalRow:
        started = time.perf_counter()
        if scenario.kind == "claim":
            predicted, source, reason, checks, input_tokens = await self._run_claim(scenario, mode)
        else:
            predicted, source, reason, checks, input_tokens = await self._run_call(scenario, mode)
        elapsed = (time.perf_counter() - started) * 1000
        gold = {c.id: c.ok for c in scenario.checks}
        scored = []
        for check in checks:
            qid = f"arg:{check.subject}" if check.subject else check.kind.value
            if check.kind.value == "claim_grounding":
                qid = "claim"
            if qid in gold:
                scored.append((qid, check.p_ok, gold[qid]))
            elif check.kind.value in gold:
                scored.append((check.kind.value, check.p_ok, gold[check.kind.value]))
        return EvalRow(
            scenario_id=scenario.id,
            family=scenario.family,
            tags=scenario.tags,
            mode=mode.value,
            expected=scenario.expected,
            predicted=predicted,
            source=source.value if source else None,
            reason=reason,
            total_ms=elapsed,
            check_scores=tuple(scored),
            input_tokens=input_tokens,
        )

    async def run(
        self, scenarios: Sequence[Scenario], modes: Sequence[Mode] = (Mode.FULL,)
    ) -> dict[str, tuple[EvalRow, ...]]:
        out: dict[str, list[EvalRow]] = {m.value: [] for m in modes}
        total = len(scenarios)
        for i, scenario in enumerate(scenarios, 1):
            for mode in modes:
                out[mode.value].append(await self.run_one(scenario, mode))
            if i == 1 or i % 10 == 0 or i == total:
                print(f"  {i}/{total} {scenario.id}", flush=True)
        return {k: tuple(v) for k, v in out.items()}

    async def _run_call(
        self, scenario: Scenario, mode: Mode
    ) -> tuple[Outcome, DecisionSource | None, str, tuple[Any, ...], int | None]:
        assert scenario.call is not None
        call = scenario.call
        session = session_for(scenario)
        if mode is Mode.NONE:
            return Outcome.ALLOW, None, "no gate", (), None
        if mode is Mode.POLICY:
            result = self._full.evaluate(call, session.context())
            if result.decision is not None:
                d = result.decision
                return d.outcome, d.source, d.reason, (), None
            return (
                Outcome.ALLOW,
                DecisionSource.POLICY,
                "No policy rule triggered",
                (),
                None,
            )
        shield = self._shield if mode is Mode.FULL else self._decider_shield
        verdict = await shield.evaluate(call, scenario.messages, session)
        d = verdict.decision
        tokens = verdict.run.input_tokens if verdict.run else None
        return d.outcome, d.source, d.reason, verdict.run.checks if verdict.run else (), tokens

    async def _run_claim(
        self, scenario: Scenario, mode: Mode
    ) -> tuple[Outcome, DecisionSource | None, str, tuple[Any, ...], int | None]:
        assert scenario.claim is not None
        if mode in (Mode.NONE, Mode.POLICY):
            return Outcome.ALLOW, None, "claims are not visible to this ablation", (), None
        verdict = await self._shield.evaluate_claim(scenario.claim, scenario.messages)
        d = verdict.decision
        tokens = verdict.run.input_tokens if verdict.run else None
        return d.outcome, d.source, d.reason, verdict.run.checks if verdict.run else (), tokens


class OracleHarness(EvalHarness):
    """Each scenario gets a Decider that returns its own gold labels. Tests the matrix and policy,
    not the live model."""

    def __init__(
        self, *, policy_path: Path | None = None, thresholds: Thresholds | None = None
    ) -> None:
        super().__init__(
            DeciderChecker(FakeDeciderClient()), policy_path=policy_path, thresholds=thresholds
        )

    async def run_one(self, scenario: Scenario, mode: Mode) -> EvalRow:
        self._checker = DeciderChecker(FakeDeciderClient(oracle_script(scenario.checks)))
        self._shield = Shield(
            self._full, self._checker, agent_id="eval", thresholds=self._thresholds
        )
        self._decider_shield = Shield(
            self._decider_only, self._checker, agent_id="eval", thresholds=self._thresholds
        )
        return await super().run_one(scenario, mode)
