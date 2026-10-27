from typing import Any

import pytest
from pydantic import ValidationError

from agentshield.audit import REDACTED, AuditRecord, Redactor, build_record
from agentshield.core import (
    CheckKind,
    CheckResult,
    Decision,
    DecisionSource,
    Outcome,
    RiskTier,
    ToolCall,
)
from agentshield.policy.models import PolicyResult


def _record(
    arguments: dict[str, Any] | None = None, redactor: Redactor | None = None
) -> AuditRecord:
    call = ToolCall(name="cancel_order", arguments=arguments or {"order_id": "A1"})
    decision = Decision(
        outcome=Outcome.ALLOW,
        risk=RiskTier.MEDIUM,
        source=DecisionSource.DECIDER,
        reason="ok",
        limiting_check=CheckResult(kind=CheckKind.CONSENT, p_yes=0.9),
    )
    policy = PolicyResult(tool="cancel_order", risk=RiskTier.MEDIUM, known_tool=True)
    return build_record(
        call=call,
        decision=decision,
        policy=policy,
        run=None,
        decider_error=None,
        total_ms=12.5,
        policy_ms=0.4,
        session_id="s1",
        agent_id="shophelp",
        policy_name="shophelp",
        policy_version=1,
        redactor=redactor,
    )


class TestRedactor:
    def test_default_keeps_values(self) -> None:
        rec = _record({"order_id": "A1", "note": "secret"})
        assert rec.arguments == {"order_id": "A1", "note": "secret"}
        assert rec.arguments_redacted is False

    def test_all_values_keeps_keys(self) -> None:
        rec = _record({"order_id": "A1", "amount": 40}, Redactor(all_values=True))
        assert rec.arguments == {"order_id": REDACTED, "amount": REDACTED}
        assert rec.arguments_redacted is True

    def test_named_keys_are_case_insensitive(self) -> None:
        rec = _record({"API_KEY": "k", "order_id": "A1"}, Redactor(keys=["api_key"]))
        assert rec.arguments["API_KEY"] == REDACTED
        assert rec.arguments["order_id"] == "A1"

    def test_nested_dicts_are_walked(self) -> None:
        rec = _record({"profile": {"email": "a@b.c", "id": "C1"}}, Redactor(keys=["email"]))
        assert rec.arguments["profile"]["email"] == REDACTED
        assert rec.arguments["profile"]["id"] == "C1"

    def test_original_call_is_not_mutated(self) -> None:
        args = {"order_id": "A1"}
        rec = _record(args, Redactor(all_values=True))
        assert args["order_id"] == "A1"
        assert rec.arguments["order_id"] == REDACTED


class TestBuildRecord:
    def test_is_frozen(self) -> None:
        rec = _record()
        with pytest.raises(ValidationError):
            rec.tool = "nope"  # type: ignore[misc]

    def test_ids_are_unique(self) -> None:
        assert _record().id != _record().id

    def test_does_not_store_the_conversation(self) -> None:
        assert "conversation" not in _record().model_dump()
