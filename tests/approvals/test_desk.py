from datetime import timedelta

import pytest

from agentshield.approvals import (
    ApprovalDesk,
    ApprovalNotFound,
    ApprovalNotPending,
    ApprovalStatus,
    excerpt,
)
from agentshield.core import Decision, DecisionSource, Outcome, RiskTier, ToolCall
from agentshield.decider.state import Message
from agentshield.pipeline import Verdict
from agentshield.policy.models import PolicyResult
from agentshield.policy.session import SessionState


def _verdict(**args: object) -> Verdict:
    call = ToolCall(
        name="issue_refund",
        arguments=args or {"order_id": "A1", "amount": 40},
        tool_use_id="tu-1",
    )
    return Verdict(
        call=call,
        decision=Decision(
            outcome=Outcome.REQUIRE_APPROVAL,
            risk=RiskTier.HIGH,
            source=DecisionSource.DECIDER,
            reason="high risk",
        ),
        policy=PolicyResult(tool="issue_refund", risk=RiskTier.HIGH, known_tool=True),
        total_ms=1.0,
    )


class TestExcerpt:
    def test_keeps_recent_messages(self) -> None:
        messages = [Message(role="user", content=f"m{i}") for i in range(20)]
        text = excerpt(messages)
        assert "m19" in text
        assert "m0" not in text


class TestDesk:
    def test_submit_and_list(self) -> None:
        desk = ApprovalDesk()
        rec = desk.submit(
            _verdict(), SessionState("s1"), [Message(role="user", content="refund me")]
        )
        assert rec.status is ApprovalStatus.PENDING
        assert rec.session_id == "s1"
        assert "refund me" in rec.excerpt
        assert desk.pending()[0].id == rec.id

    def test_approve_resumes_the_registered_agent(self) -> None:
        seen: list[object] = []

        def agent(prompt: object) -> object:
            seen.append(prompt)
            return prompt

        desk = ApprovalDesk()
        rec = desk.submit(_verdict(), SessionState("s1"))
        desk.bind_interrupt(rec.id, "int-1")
        desk.register_agent("s1", agent)
        updated, result = desk.approve(rec.id, by="alice")
        assert updated.status is ApprovalStatus.APPROVED
        assert updated.resolved_by == "alice"
        assert result == [{"interruptResponse": {"interruptId": "int-1", "response": "yes"}}]
        assert seen == [result]

    def test_cannot_approve_twice(self) -> None:
        desk = ApprovalDesk()
        rec = desk.submit(_verdict(), SessionState("s"))
        desk.record(rec.id, ApprovalStatus.APPROVED, by="a")
        with pytest.raises(ApprovalNotPending):
            desk.approve(rec.id, by="b")

    def test_unknown_id(self) -> None:
        with pytest.raises(ApprovalNotFound):
            ApprovalDesk().get("missing")

    def test_timeout_denies_and_resumes(self) -> None:
        seen: list[object] = []
        desk = ApprovalDesk(timeout=timedelta(0))
        rec = desk.submit(_verdict(), SessionState("s1"))
        desk.bind_interrupt(rec.id, "int-1")
        desk.register_agent("s1", seen.append)
        expired = desk.expire_due()
        assert expired[0].status is ApprovalStatus.EXPIRED
        assert expired[0].resolved_by == "timeout"
        payload = seen[0]
        assert isinstance(payload, list)
        assert payload[0]["interruptResponse"]["response"] == "no"
        assert desk.pending() == []
