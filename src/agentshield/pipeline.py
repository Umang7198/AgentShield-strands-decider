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
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from agentshield.core.matrix import Thresholds, decide, fallback_decision
from agentshield.core.models import Decision, ToolCall
from agentshield.decider.checker import CheckRun, DeciderChecker
from agentshield.decider.errors import DeciderError
from agentshield.decider.state import Message
from agentshield.policy.engine import PolicyEngine
from agentshield.policy.models import PolicyResult
from agentshield.policy.session import SessionState


class Verdict(BaseModel):
    """A decision plus everything that produced it (for the audit log)."""

    model_config = ConfigDict(frozen=True)

    call: ToolCall
    decision: Decision
    policy: PolicyResult
    run: CheckRun | None = None  # None when the policy decided or the Decider failed
    decider_error: str | None = None
    total_ms: float


class Shield:
    def __init__(
        self,
        engine: PolicyEngine,
        checker: DeciderChecker,
        *,
        thresholds: Thresholds | None = None,
        include_premature: bool = False,
    ) -> None:
        self._engine = engine
        self._checker = checker
        self._thresholds = thresholds or Thresholds()
        self._include_premature = include_premature

    async def evaluate(
        self,
        call: ToolCall,
        messages: Sequence[Message],
        session: SessionState | None = None,
    ) -> Verdict:
        started = time.perf_counter()
        ctx = session.context() if session else None
        policy = self._engine.evaluate(call, ctx)

        policy_decision = policy.decision
        if policy_decision is not None:
            return self._verdict(call, policy_decision, policy, started)

        try:
            run = await self._checker.check_call(
                call, messages, include_premature=self._include_premature
            )
        except DeciderError as exc:
            decision = fallback_decision(policy.risk, f"{type(exc).__name__}: {exc}")
            return self._verdict(
                call, decision, policy, started, error=str(exc) or type(exc).__name__
            )

        decision = decide(policy.risk, run.checks, self._thresholds)
        return self._verdict(call, decision, policy, started, run=run)

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
