"""Core data models shared by every AgentShield component.

These are plain, validated, immutable value objects. They contain no I/O and no model calls.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Outcome(StrEnum):
    """What AgentShield decides to do with a proposed tool call."""

    ALLOW = "allow"
    CLARIFY = "clarify"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


class RiskTier(StrEnum):
    """How much damage a wrong call could do. Set per tool, not per call."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DecisionSource(StrEnum):
    """Which part of the pipeline produced the final decision."""

    POLICY = "policy"  # deterministic rule engine
    DECIDER = "decider"  # decision-model checks + decision matrix
    FALLBACK = "fallback"  # decider unavailable or errored


class CheckKind(StrEnum):
    """The questions the Decider is asked about a proposed call."""

    ARGUMENT_GROUNDING = "argument_grounding"  # yes = the argument is supported by the user
    CONSENT = "consent"  # yes = the user asked for or agreed to this action
    PREMATURE_ACTION = "premature_action"  # yes = the agent should clarify first (BAD)
    CLAIM_GROUNDING = "claim_grounding"  # yes = the agent's claim is supported by tool results

    @property
    def yes_is_good(self) -> bool:
        """Whether a "yes" answer means the check is satisfied."""
        return self is not CheckKind.PREMATURE_ACTION


class ToolCall(BaseModel):
    """A tool call proposed by the agent, before it executes."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_use_id: str | None = None


class CheckResult(BaseModel):
    """One Decider answer about a proposed call.

    The Decider answers yes/no questions with ``p_yes``, the probability of "yes". This class
    turns that into two numbers that do not depend on how the question was phrased:

    * ``p_ok``: probability that the check is satisfied (flips for questions where "yes" is bad).
    * ``confidence``: how sure the model is either way, ``max(p_yes, 1 - p_yes)``.
    """

    model_config = ConfigDict(frozen=True)

    kind: CheckKind
    p_yes: float = Field(ge=0.0, le=1.0)
    subject: str | None = None  # e.g. the argument name for ARGUMENT_GROUNDING
    latency_ms: float | None = Field(default=None, ge=0.0)

    # Rounded so values sitting exactly on a threshold (e.g. 1 - 0.08 = 0.9199999999999999)
    # are not pushed to the wrong side by floating-point noise.
    @property
    def p_ok(self) -> float:
        return round(self.p_yes if self.kind.yes_is_good else 1.0 - self.p_yes, 9)

    @property
    def confidence(self) -> float:
        return round(max(self.p_yes, 1.0 - self.p_yes), 9)

    @property
    def label(self) -> str:
        return f"{self.kind.value}[{self.subject}]" if self.subject else self.kind.value


class Decision(BaseModel):
    """The final answer for one proposed tool call."""

    model_config = ConfigDict(frozen=True)

    outcome: Outcome
    risk: RiskTier
    source: DecisionSource
    reason: str
    checks: tuple[CheckResult, ...] = ()
    limiting_check: CheckResult | None = None  # the weakest check, if checks were run
    rule_id: str | None = None  # set when source is POLICY
