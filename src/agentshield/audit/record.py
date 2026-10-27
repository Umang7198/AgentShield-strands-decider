"""Immutable audit record for one gate evaluation.

The conversation itself is not stored: that is PII-heavy and belongs, if anywhere, on a pending
approval. What we keep is enough to replay the *decision*: the call, the evidence, and why.
"""

from __future__ import annotations

import copy
import uuid
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from agentshield.core.models import (
    CheckKind,
    CheckResult,
    Decision,
    DecisionSource,
    Outcome,
    RiskTier,
    ToolCall,
)
from agentshield.decider.checker import CheckRun
from agentshield.policy.models import PolicyResult

REDACTED = "***"


class AuditCheck(BaseModel):
    """A Decider check, stored as numbers so a later polarity change cannot rewrite history."""

    model_config = ConfigDict(frozen=True)

    kind: CheckKind
    subject: str | None = None
    p_yes: float
    p_ok: float
    confidence: float

    @classmethod
    def from_check(cls, check: CheckResult) -> AuditCheck:
        return cls(
            kind=check.kind,
            subject=check.subject,
            p_yes=check.p_yes,
            p_ok=check.p_ok,
            confidence=check.confidence,
        )


class MatchedRule(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    effect: str
    detail: str = ""


class AuditRecord(BaseModel):
    """One append-only row. Frozen so a stored record cannot be edited in memory either."""

    model_config = ConfigDict(frozen=True)

    id: str
    timestamp: datetime
    session_id: str
    agent_id: str
    tool: str
    arguments: dict[str, Any]
    arguments_redacted: bool
    tool_use_id: str | None = None
    outcome: Outcome
    risk: RiskTier
    source: DecisionSource
    reason: str
    rule_id: str | None = None
    limiting_check: str | None = None
    checks: tuple[AuditCheck, ...] = ()
    matched_rules: tuple[MatchedRule, ...] = ()
    policy_name: str
    policy_version: int
    question_set_version: str | None = None
    decider_model: str | None = None
    decider_error: str | None = None
    lost_evidence: bool = False
    dropped_messages: int = 0
    unchecked_args: tuple[str, ...] = ()
    policy_ms: float
    decider_ms: float | None = None
    total_ms: float


class Redactor:
    """Replace argument values before they are written to the audit log.

    Keys are always kept, so you can still see *which* arguments were passed. Matching is
    case-insensitive. Nested dicts are walked.
    """

    def __init__(self, *, all_values: bool = False, keys: Collection[str] = ()) -> None:
        self.all_values = all_values
        self._keys = {k.casefold() for k in keys}

    def apply(self, arguments: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        copied = copy.deepcopy(dict(arguments))
        if not self.all_values and not self._keys:
            return copied, False
        redacted = _redact(copied, self.all_values, self._keys)
        if not isinstance(redacted, dict):  # defensive: top-level arguments are always a dict
            raise TypeError("redacted arguments must be a dict")
        return redacted, True


def _redact(value: Any, all_values: bool, keys: set[str], key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _redact(v, all_values, keys, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, all_values, keys) for v in value]
    if all_values or (key is not None and key.casefold() in keys):
        return REDACTED
    return value


def build_record(
    *,
    call: ToolCall,
    decision: Decision,
    policy: PolicyResult,
    run: CheckRun | None,
    decider_error: str | None,
    total_ms: float,
    policy_ms: float,
    session_id: str,
    agent_id: str,
    policy_name: str,
    policy_version: int,
    redactor: Redactor | None = None,
) -> AuditRecord:
    arguments, was_redacted = (redactor or Redactor()).apply(call.arguments)
    limiting = decision.limiting_check
    return AuditRecord(
        id=str(uuid.uuid4()),
        timestamp=datetime.now(UTC),
        session_id=session_id,
        agent_id=agent_id,
        tool=call.name,
        arguments=arguments,
        arguments_redacted=was_redacted,
        tool_use_id=call.tool_use_id,
        outcome=decision.outcome,
        risk=decision.risk,
        source=decision.source,
        reason=decision.reason,
        rule_id=decision.rule_id,
        limiting_check=limiting.label if limiting else None,
        checks=tuple(AuditCheck.from_check(c) for c in (run.checks if run else decision.checks)),
        matched_rules=tuple(
            MatchedRule(rule_id=m.rule_id, effect=m.effect.value, detail=m.detail)
            for m in policy.matches
        ),
        policy_name=policy_name,
        policy_version=policy_version,
        question_set_version=run.question_set_version if run else None,
        decider_model=run.model if run else None,
        decider_error=decider_error,
        lost_evidence=run.state.lost_evidence if run else False,
        dropped_messages=run.state.dropped_messages if run else 0,
        unchecked_args=run.unchecked_args if run else (),
        policy_ms=policy_ms,
        decider_ms=run.wall_latency_ms if run else None,
        total_ms=total_ms,
    )
