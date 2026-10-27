"""The decision pipeline: policy first, then Decider evidence, then the decision matrix.

Framework-agnostic: it takes a proposed call and the conversation so far, and returns a
:class:`Verdict`. Adapters (such as the Strands handler) translate the verdict for their
framework.

Order matters:

1. **Policy engine** (deterministic). A rule that triggers decides on its own and the Decider is
   not consulted: forbidden actions never depend on a model, and it saves latency.
2. **Decider checks** (argument grounding, consent) for everything else.
3. **Decision matrix** turns the evidence into ALLOW / CLARIFY / REQUIRE_APPROVAL / BLOCK.

If the Decider is unreachable the result is a fail-closed fallback, never an ALLOW.

Every evaluation is written to the audit sinks (if any) before the verdict is returned. A sink
that fails raises, so a broken log cannot silently let a call through when this runs behind the
Strands handler (``on_error="deny"``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from opentelemetry.trace import Span
from pydantic import BaseModel, ConfigDict

from agentshield.audit.record import Redactor, build_record
from agentshield.audit.store import AuditSink
from agentshield.core.matrix import Thresholds, decide, fallback_decision
from agentshield.core.models import Decision, RiskTier, ToolCall
from agentshield.decider.checker import CheckRun, DeciderChecker
from agentshield.decider.errors import DeciderError
from agentshield.decider.state import Message
from agentshield.policy.engine import PolicyEngine
from agentshield.policy.models import PolicyResult
from agentshield.policy.session import SessionState
from agentshield.telemetry import SPAN_DECIDER, SPAN_EVALUATE, SPAN_POLICY, Telemetry

CLAIM_TOOL = "agent_claim"
"""Synthetic tool name used when the gate judges an assistant claim, not a tool call."""


class Verdict(BaseModel):
    """A decision plus everything that produced it (for the audit log)."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    decision: Decision
    policy: PolicyResult
    run: CheckRun | None = None  # None when the policy decided or the Decider failed
    decider_error: str | None = None
    total_ms: float
    audit_id: str | None = None


class Shield:
    def __init__(
        self,
        engine: PolicyEngine,
        checker: DeciderChecker,
        *,
        thresholds: Thresholds | None = None,
        include_premature: bool = False,
        agent_id: str = "agentshield",
        sinks: Sequence[AuditSink] = (),
        redactor: Redactor | None = None,
        telemetry: Telemetry | None = None,
    ) -> None:
        self._engine = engine
        self._checker = checker
        self._thresholds = thresholds or Thresholds()
        self._include_premature = include_premature
        self._agent_id = agent_id
        self._sinks = tuple(sinks)
        self._redactor = redactor or Redactor()
        self._telemetry = telemetry or Telemetry()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def evaluate(
        self,
        call: ToolCall,
        messages: Sequence[Message],
        session: SessionState | None = None,
    ) -> Verdict:
        started = time.perf_counter()
        session_id = session.session_id if session else "anonymous"
        ctx = session.context() if session else None

        with self._telemetry.tracer.start_as_current_span(SPAN_EVALUATE) as span:
            with self._telemetry.tracer.start_as_current_span(SPAN_POLICY):
                policy_started = time.perf_counter()
                policy = self._engine.evaluate(call, ctx)
                policy_ms = (time.perf_counter() - policy_started) * 1000

            policy_decision = policy.decision
            if policy_decision is not None:
                verdict = self._verdict(call, policy_decision, policy, started)
                return self._emit(verdict, policy, policy_ms, session_id, span)

            try:
                with self._telemetry.tracer.start_as_current_span(SPAN_DECIDER):
                    run = await self._checker.check_call(
                        call, messages, include_premature=self._include_premature
                    )
            except DeciderError as exc:
                decision = fallback_decision(policy.risk, f"{type(exc).__name__}: {exc}")
                verdict = self._verdict(
                    call, decision, policy, started, error=str(exc) or type(exc).__name__
                )
                return self._emit(verdict, policy, policy_ms, session_id, span)

            decision = decide(policy.risk, run.checks, self._thresholds)
            verdict = self._verdict(call, decision, policy, started, run=run)
            return self._emit(verdict, policy, policy_ms, session_id, span)

    async def evaluate_claim(
        self,
        claim: str,
        messages: Sequence[Message],
        session: SessionState | None = None,
    ) -> Verdict:
        """Judge an assistant statement against the conversation (no policy, low risk).

        The claim itself must not be in ``messages`` or it will look self-supporting.
        """
        started = time.perf_counter()
        session_id = session.session_id if session else "anonymous"
        call = ToolCall(name=CLAIM_TOOL, arguments={"claim": claim})
        policy = PolicyResult(tool=CLAIM_TOOL, risk=RiskTier.LOW, known_tool=True)
        with self._telemetry.tracer.start_as_current_span(SPAN_EVALUATE) as span:
            try:
                with self._telemetry.tracer.start_as_current_span(SPAN_DECIDER):
                    run = await self._checker.check_claim(claim, messages)
            except DeciderError as exc:
                decision = fallback_decision(RiskTier.LOW, f"{type(exc).__name__}: {exc}")
                verdict = self._verdict(
                    call, decision, policy, started, error=str(exc) or type(exc).__name__
                )
                return self._emit(verdict, policy, 0.0, session_id, span)
            decision = decide(RiskTier.LOW, run.checks, self._thresholds)
            verdict = self._verdict(call, decision, policy, started, run=run)
            return self._emit(verdict, policy, 0.0, session_id, span)

    def _emit(
        self,
        verdict: Verdict,
        policy: PolicyResult,
        policy_ms: float,
        session_id: str,
        span: Span,
    ) -> Verdict:
        record = build_record(
            call=verdict.call,
            decision=verdict.decision,
            policy=policy,
            run=verdict.run,
            decider_error=verdict.decider_error,
            total_ms=verdict.total_ms,
            policy_ms=policy_ms,
            session_id=session_id,
            agent_id=self._agent_id,
            policy_name=self._engine.policy.name,
            policy_version=self._engine.policy.version,
            redactor=self._redactor,
        )
        for sink in self._sinks:
            sink.record(record)
        self._telemetry.record(record, span)
        return verdict.model_copy(update={"audit_id": record.id})

    @staticmethod
    def _verdict(
        call: ToolCall,
        decision: Decision,
        policy: PolicyResult,
        started: float,
        *,
        run: CheckRun | None = None,
        error: str | None = None,
    ) -> Verdict:
        return Verdict(
            call=call,
            decision=decision,
            policy=policy,
            run=run,
            decider_error=error,
            total_ms=(time.perf_counter() - started) * 1000,
        )
