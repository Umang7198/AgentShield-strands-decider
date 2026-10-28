from fastapi.testclient import TestClient

from agentshield.api import create_app
from agentshield.approvals import ApprovalDesk
from agentshield.decider import DeciderChecker, FakeDeciderClient
from agentshield.shophelp.runtime import ShopHelpRuntime
from tests.fakes import ScriptedModel

GOOD = {"arg:*": 0.95, "consent": 0.95, "claim": 0.95}


def client() -> TestClient:
    desk = ApprovalDesk()
    runtime = ShopHelpRuntime(
        ScriptedModel([], final="How can I help?"),
        DeciderChecker(FakeDeciderClient(GOOD)),
        approvals=desk,
    )
    return TestClient(create_app(desk, runtime=runtime))


def test_desk_pages_and_state() -> None:
    c = client()
    home = c.get("/")
    assert home.status_code == 200
    assert b"Hearth" in home.content
    assert b"/js/app.js" in home.content
    assert c.get("/review").status_code == 200
    assert c.get("/css/app.css").status_code == 200
    assert c.get("/js/app.js").status_code == 200
    state = c.get("/demo/state").json()
    assert state["store"] == "Hearth & Cable"
    assert state["supervisor"] == "Priya"
    assert len(state["scenes"]) == 11


def test_chat_while_paused_returns_200() -> None:
    desk = ApprovalDesk()
    runtime = ShopHelpRuntime(
        ScriptedModel(
            [
                ("get_order_status", {"order_id": "A1001"}),
                ("issue_refund", {"order_id": "A1001", "amount": 40}),
            ]
        ),
        DeciderChecker(FakeDeciderClient(GOOD)),
        approvals=desk,
    )
    c = TestClient(create_app(desk, runtime=runtime))
    first = c.post("/demo/chat", json={"text": "Refund 40 dollars on A1001, I confirm."})
    assert first.status_code == 200
    assert first.json()["paused"] is True
    second = c.post("/demo/chat", json={"text": "now cancel it"})
    assert second.status_code == 200
    assert second.json()["paused"] is True
    assert second.json()["board"]["refunds"] == []


def test_claim_scene_and_reset() -> None:
    desk = ApprovalDesk()
    runtime = ShopHelpRuntime(
        ScriptedModel([], final="done"),
        DeciderChecker(FakeDeciderClient({**GOOD, "claim": 0.12})),
        approvals=desk,
    )
    c = TestClient(create_app(desk, runtime=runtime))
    flagged = c.post("/demo/claim", json={"claim": "Your refund has been processed."}).json()
    assert flagged["tape"][-1]["outcome"] == "clarify"
    reset = c.post("/demo/reset").json()
    assert reset["tape"] == []
    assert reset["board"]["refunds"] == []
