from fastapi.testclient import TestClient

from agentshield.api import create_app
from agentshield.approvals import ApprovalDesk
from agentshield.core import Decision, DecisionSource, Outcome, RiskTier, ToolCall
from agentshield.pipeline import Verdict
from agentshield.policy.models import PolicyResult
from agentshield.policy.session import SessionState


def _submit(desk: ApprovalDesk) -> str:
    verdict = Verdict(
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
    return desk.submit(verdict, SessionState("s")).id


def test_list_approve_deny_and_review_page() -> None:
    desk = ApprovalDesk()
    approval_id = _submit(desk)
    resumed: list[object] = []
    desk.bind_interrupt(approval_id, "int-1")
    desk.register_agent("s", resumed.append)
    client = TestClient(create_app(desk))

    listed = client.get("/approvals").json()
    assert listed[0]["id"] == approval_id
    assert client.get("/review").status_code == 200
    assert client.get("/health").json() == {"status": "ok"}

    denied = client.post(f"/approvals/{approval_id}/deny", json={"approver": "bob", "note": "nope"})
    assert denied.status_code == 200
    assert denied.json()["status"] == "denied"
    payload = resumed[0]
    assert isinstance(payload, list)
    assert payload[0]["interruptResponse"]["response"] == "no"

    replay = client.post(f"/approvals/{approval_id}/deny", json={"approver": "bob"})
    assert replay.status_code == 409


def test_token_is_required_when_configured() -> None:
    desk = ApprovalDesk()
    client = TestClient(create_app(desk, token="secret"))
    assert client.get("/approvals").status_code == 401
    ok = client.get("/approvals", headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert ok.json() == []
