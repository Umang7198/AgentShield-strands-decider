"""The yes/no questions AgentShield asks the Decider about a proposed call.

Wording matters a lot for a small decision model, so the templates are versioned
(:data:`QUESTION_SET_VERSION`) and the version is recorded with every decision. Each question
carries ``true``/``false`` descriptions that sharpen the boundary. Change a template -> bump the
version -> re-run the evaluation.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from agentshield.core.models import CheckKind, ToolCall
from agentshield.decider.client import NoulQuestion
from agentshield.decider.state import neutralise, render_value

QUESTION_SET_VERSION = "v2"
# v1 -> v2: shorter instructions (about 20% lower latency), argument grounding keeps the richer
# criteria, consent uses short criteria; the premature-action question is no longer asked by
# default. Chosen from a live 15-scenario comparison of four wordings; see core/matrix.py.

CONSENT_ID = "consent"
PREMATURE_ID = "premature"
CLAIM_ID = "claim"
ARG_PREFIX = "arg:"

DEFAULT_MAX_ARGUMENT_CHECKS = 6


@dataclass(frozen=True, slots=True)
class PlannedQuestion:
    id: str
    kind: CheckKind
    subject: str | None
    question: NoulQuestion


@dataclass(frozen=True, slots=True)
class QuestionPlan:
    """The questions for one call, plus anything we did not ask about."""

    version: str
    questions: tuple[PlannedQuestion, ...]
    unchecked_args: tuple[str, ...] = ()

    def wire(self) -> Mapping[str, NoulQuestion]:
        return {q.id: q.question for q in self.questions}


def _argument_question(call: ToolCall, name: str, value_text: str) -> NoulQuestion:
    return NoulQuestion(
        instructions=(
            f"Tool call: {neutralise(call.name)}. Did the user state, or clearly imply, "
            f"the value of '{neutralise(name)}' ({value_text})?"
        ),
        true_means="the user or a tool result provided this value, or clearly implies it",
        false_means="the agent guessed, invented or assumed this value; it is not in the conversation",
    )


def _consent_question(call: ToolCall) -> NoulQuestion:
    return NoulQuestion(
        instructions=(
            f"Tool call: {neutralise(call.name)}. Did the user ask for this action or agree to it?"
        ),
        true_means="the user asked for it or agreed",
        false_means="the user did not ask for it, e.g. they only asked a question",
    )


def _premature_question(call: ToolCall) -> NoulQuestion:
    # Not measured and off by default: in the first live trial it scored 0.09-0.41 for good and
    # bad calls alike, so it did not separate them.
    return NoulQuestion(
        instructions=(
            f"Tool call: {neutralise(call.name)}. Must the agent ask the user something first?"
        ),
        true_means="clarification is needed before this call should run",
        false_means="the call is fully specified and can proceed",
    )


def build_call_questions(
    call: ToolCall,
    *,
    max_argument_checks: int = DEFAULT_MAX_ARGUMENT_CHECKS,
    skip_args: Collection[str] = (),
    include_premature: bool = False,
    max_value_chars: int = 200,
) -> QuestionPlan:
    """Questions for a proposed call: grounding per argument and consent, plus, if asked for,
    whether the agent is acting prematurely (off by default; it did not separate good from bad).

    At most ``max_argument_checks`` arguments are checked, in the order given. The rest are
    reported in ``unchecked_args`` so the caller and the audit log can see the gap.
    """
    if max_argument_checks < 0:
        raise ValueError("max_argument_checks must not be negative")

    candidates = [name for name in call.arguments if name not in skip_args]
    checked = candidates[:max_argument_checks]
    unchecked = tuple(candidates[max_argument_checks:])

    planned: list[PlannedQuestion] = [
        PlannedQuestion(
            id=f"{ARG_PREFIX}{name}",
            kind=CheckKind.ARGUMENT_GROUNDING,
            subject=name,
            question=_argument_question(
                call, name, render_value(call.arguments[name], max_value_chars)
            ),
        )
        for name in checked
    ]
    planned.append(PlannedQuestion(CONSENT_ID, CheckKind.CONSENT, None, _consent_question(call)))
    if include_premature:
        planned.append(
            PlannedQuestion(
                PREMATURE_ID, CheckKind.PREMATURE_ACTION, None, _premature_question(call)
            )
        )
    return QuestionPlan(QUESTION_SET_VERSION, tuple(planned), unchecked)


def build_claim_questions(claim: str, *, max_chars: int = 600) -> QuestionPlan:
    """One question: is something the agent told the user supported by the conversation?"""
    text = neutralise(claim.strip())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    # Not yet measured against the live Decider (the call checks were); treat as provisional.
    question = NoulQuestion(
        instructions=(
            f'The agent told the user: "{text}". '
            "Is this supported by the conversation and the tool results?"
        ),
        true_means="the statement is backed by the conversation or a tool result",
        false_means="the statement is not backed by anything above; it was made up or assumed",
    )
    return QuestionPlan(
        QUESTION_SET_VERSION,
        (PlannedQuestion(CLAIM_ID, CheckKind.CLAIM_GROUNDING, None, question),),
    )
