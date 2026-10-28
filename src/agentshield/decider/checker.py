"""Runs the Decider checks for one proposed call and returns typed evidence."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from agentshield.core.models import CheckResult, ToolCall
from agentshield.decider.client import DeciderClient, DeciderResponse, NoulQuestion
from agentshield.decider.errors import DeciderBadResponse
from agentshield.decider.questions import (
    DEFAULT_MAX_ARGUMENT_CHECKS,
    QuestionPlan,
    build_call_questions,
    build_claim_questions,
)
from agentshield.decider.state import BuiltState, Message, StateBudget, build_state


class CheckRun(BaseModel):
    """The checks plus everything the audit log needs to understand how they were produced."""

    model_config = ConfigDict(frozen=True)

    checks: tuple[CheckResult, ...]
    state: BuiltState
    unchecked_args: tuple[str, ...] = ()
    question_set_version: str
    model: str | None = None
    wall_latency_ms: float
    server_latency_ms: float | None = None
    input_tokens: int | None = None


class DeciderChecker:
    """Builds the state and questions, asks the Decider once, and converts the answers.

    Raises a :class:`~agentshield.decider.errors.DeciderError` subclass when no usable answers
    came back; the pipeline turns that into a fail-closed fallback decision.
    """

    def __init__(
        self,
        client: DeciderClient,
        *,
        budget: StateBudget | None = None,
        max_argument_checks: int = DEFAULT_MAX_ARGUMENT_CHECKS,
    ) -> None:
        self._client = client
        self._budget = budget or StateBudget()
        self._max_argument_checks = max_argument_checks

    async def ask(self, state: str, questions: Mapping[str, NoulQuestion]) -> DeciderResponse:
        """Raw noul questions (used by handbook RAG; not a tool-gate CheckRun)."""
        return await self._client.ask(state, questions)

    async def check_call(
        self,
        call: ToolCall,
        messages: Sequence[Message],
        *,
        skip_args: Collection[str] = (),
        include_premature: bool = False,
    ) -> CheckRun:
        state = build_state(messages, call, self._budget)
        plan = build_call_questions(
            call,
            max_argument_checks=self._max_argument_checks,
            skip_args=skip_args,
            include_premature=include_premature,
        )
        return await self._run(state, plan)

    async def check_claim(self, claim: str, messages: Sequence[Message]) -> CheckRun:
        state = build_state(messages, None, self._budget)
        return await self._run(state, build_claim_questions(claim))

    async def _run(self, state: BuiltState, plan: QuestionPlan) -> CheckRun:
        response: DeciderResponse = await self._client.ask(state.text, plan.wire())
        checks = []
        for planned in plan.questions:
            if planned.id not in response.answers:  # defensive: clients validate this already
                raise DeciderBadResponse(f"missing answer for {planned.id!r}")
            checks.append(
                CheckResult(
                    kind=planned.kind,
                    p_yes=response.answers[planned.id],
                    subject=planned.subject,
                )
            )
        return CheckRun(
            checks=tuple(checks),
            state=state,
            unchecked_args=plan.unchecked_args,
            question_set_version=plan.version,
            model=response.model,
            wall_latency_ms=response.wall_latency_ms,
            server_latency_ms=response.server_latency_ms,
            input_tokens=response.input_tokens,
        )
