"""One labelled ShopHelp case for the evaluation harness."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentshield.core.models import Outcome, ToolCall
from agentshield.decider.state import Message


class CheckLabel(BaseModel):
    """What a perfect Decider should conclude for one question."""

    model_config = ConfigDict(frozen=True)

    id: str  # e.g. arg:order_id, consent, claim
    ok: bool


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    family: str
    tags: tuple[str, ...]
    kind: Literal["call", "claim"] = "call"
    call: ToolCall | None = None
    claim: str | None = None
    messages: tuple[Message, ...]
    facts: dict[str, Any] = Field(default_factory=dict)
    prior_calls: dict[str, int] = Field(default_factory=dict)
    expected: Outcome
    checks: tuple[CheckLabel, ...] = ()
    notes: str = ""

    @property
    def should_stop(self) -> bool:
        return self.expected is not Outcome.ALLOW
