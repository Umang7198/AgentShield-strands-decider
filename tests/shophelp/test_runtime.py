from agentshield.approvals import ApprovalDesk
from agentshield.core import Outcome
from agentshield.decider import DeciderChecker, FakeDeciderClient
from agentshield.shophelp.runtime import SCENES, ShopHelpRuntime
from agentshield.shophelp.world import STORE_NAME
from tests.fakes import ScriptedModel

GOOD = {
    "arg:*": 0.95,
    "consent": 0.95,
    "claim": 0.95,
    "relevance": 0.91,
    "sufficiency": 0.88,
}


def make(
    calls: list[tuple[str, dict[str, object]]] | None = None,
    *,
    script: dict[str, float] | None = None,
    final: str = "A1001 has shipped.",
) -> ShopHelpRuntime:
    return ShopHelpRuntime(
        ScriptedModel(calls or [], final=final),
        DeciderChecker(FakeDeciderClient(GOOD if script is None else script)),
        approvals=ApprovalDesk(),
    )


def test_board_is_hearth_and_cable() -> None:
    state = make().state()
    assert state["store"] == STORE_NAME
    orders = {row["order_id"]: row for row in state["board"]["orders"]}
    assert orders["A1001"]["status"] == "shipped"
    assert orders["A1001"]["ticket_id"] == "T-1042"
    assert orders["A3003"]["within_return_window"] is False
    assert len(SCENES) == 11


def test_policy_nudge_is_not_shown_in_the_transcript() -> None:
    runtime = make(final="14-day window from delivery.")
    state = runtime.chat("How long do I have to return a kettle?")
    user = next(row for row in state["transcript"] if row["role"] == "user")
    assert user["text"] == "How long do I have to return a kettle?"
    assert "lookup_handbook" not in user["text"]
    assert "[Gate]" not in user["text"]


def test_reset_restores_the_catalogue() -> None:
    runtime = make([("cancel_order", {"order_id": "A2002"})])
    runtime.chat("Please cancel order A2002.")
    assert runtime.app.store.get_order("A2002")["status"] == "cancelled"
    runtime.reset()
    assert runtime.app.store.get_order("A2002")["status"] == "processing"
    assert runtime.tape == []
    assert runtime.transcript == []


def test_ungrounded_claim_is_flagged_on_the_tape() -> None:
    runtime = make(script={**GOOD, "claim": 0.12})
    state = runtime.check_claim("Your refund has been processed.")
    assert state["tape"][-1]["tool"] == "agent_claim"
    assert state["tape"][-1]["outcome"] == Outcome.CLARIFY.value
    assert state["transcript"][-1]["role"] == "gate"


def test_chat_while_paused_does_not_resume_with_a_string() -> None:
    runtime = make(
        [
            ("get_order_status", {"order_id": "A1001"}),
            ("issue_refund", {"order_id": "A1001", "amount": 40}),
        ]
    )
    paused = runtime.chat("Yes, refund 40 dollars on A1001. I confirm.")
    assert paused["paused"] is True
    assert runtime.approvals.pending()
    again = runtime.chat("and cancel it too")
    assert again["paused"] is True
    assert runtime.app.store.refunds == []
    assert "Approve or deny" in str(again["reply"])


def test_handbook_lookup_appends_rag_evidence_to_the_tape() -> None:
    runtime = make(
        [("lookup_handbook", {"question": "How long is the return window?"})],
        final="You have 14 days from delivery.",
    )
    state = runtime.chat("How long do I have to return a kettle?")
    tools = [row["tool"] for row in state["tape"]]
    assert "lookup_handbook" in tools
    rag = next(row for row in state["tape"] if row["tool"] == "handbook_evidence")
    assert rag["source"] == "rag"
    assert rag["outcome"] == Outcome.ALLOW.value
    kinds = {c["kind"] for c in rag["checks"]}
    assert kinds == {"relevance", "sufficiency"}
    assert rag["retried"] is False


def test_handbook_abstain_shows_clarify_on_the_tape() -> None:
    runtime = make(
        [("lookup_handbook", {"question": "What is GDPR delete?"})],
        script={**GOOD, "relevance": 0.12, "sufficiency": 0.10, "claim": 0.95},
        final="I do not have enough handbook evidence.",
    )
    state = runtime.chat("What is the GDPR delete process in the handbook?")
    rag = next(row for row in state["tape"] if row["tool"] == "handbook_evidence")
    assert rag["outcome"] == Outcome.CLARIFY.value
    assert rag["retried"] is True


def test_supported_claim_is_not_flagged() -> None:
    runtime = make(script={**GOOD, "claim": 0.92})
    state = runtime.check_claim("Your refund of $40 was issued.")
    assert state["tape"][-1]["outcome"] == Outcome.ALLOW.value
    assert state["transcript"][-1]["role"] == "assistant"
