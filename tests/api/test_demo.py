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
    assert c.get("/review").status_code == 200
    state = c.get("/demo/state").json()
    assert state["store"] == "Hearth & Cable"
    assert state["supervisor"] == "Priya"
    assert len(state["scenes"]) == 10


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
