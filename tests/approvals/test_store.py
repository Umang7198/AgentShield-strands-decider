from pathlib import Path

from agentshield.approvals import ApprovalDesk, ApprovalStatus, SqlApprovalStore
from agentshield.core import Decision, DecisionSource, Outcome, RiskTier, ToolCall
from agentshield.pipeline import Verdict
from agentshield.policy.models import PolicyResult
from agentshield.policy.session import SessionState


def _verdict() -> Verdict:
    return Verdict(
        call=ToolCall(name="issue_refund", arguments={"amount": 40}, tool_use_id="tu"),
        decision=Decision(
            outcome=Outcome.REQUIRE_APPROVAL,
            risk=RiskTier.HIGH,
            source=DecisionSource.DECIDER,
            reason="high risk",
        ),
        policy=PolicyResult(tool="issue_refund", risk=RiskTier.HIGH, known_tool=True),
        total_ms=1.0,
    )


def test_sqlite_roundtrip_and_update(tmp_path: Path) -> None:
    store = SqlApprovalStore.sqlite(tmp_path / "a.db")
    desk = ApprovalDesk(store)
    rec = desk.submit(_verdict(), SessionState("s"))
    desk.record(rec.id, ApprovalStatus.APPROVED, by="alice")
    reopened = ApprovalDesk(SqlApprovalStore.sqlite(tmp_path / "a.db"))
    got = reopened.get(rec.id)
    assert got.status is ApprovalStatus.APPROVED
    assert got.resolved_by == "alice"
    assert got.arguments == {"amount": 40}


def test_sqlite_clear_drops_cards(tmp_path: Path) -> None:
    store = SqlApprovalStore.sqlite(tmp_path / "a.db")
    desk = ApprovalDesk(store)
    desk.submit(_verdict(), SessionState("s"))
    assert desk.pending()
    store.clear()
    assert desk.pending() == []
