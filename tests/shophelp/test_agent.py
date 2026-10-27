"""ShopHelp through a real Strands Agent. The host LLM is scripted; the Decider is fake."""

from typing import Any

from agentshield.approvals import ApprovalDesk, ApprovalStatus
from agentshield.audit import MemoryAuditLog
from agentshield.core import DecisionSource, Outcome
from agentshield.decider import DeciderChecker, FakeDeciderClient
from agentshield.shophelp import ShopHelpStore, build_shophelp
from tests.fakes import ScriptedModel

GOOD = {"arg:*": 0.95, "consent": 0.95}


def run(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    script: dict[str, float] | None = None,
    prompt: str = "help me",
    desk: ApprovalDesk | None = None,
) -> Any:
    log = MemoryAuditLog()
    app = build_shophelp(
        ScriptedModel(calls),
        DeciderChecker(FakeDeciderClient(GOOD if script is None else script)),
        store=ShopHelpStore.seed(),
        sinks=[log],
        approvals=desk,
    )
    result = app.agent(prompt)
    return app, result, log


def outcomes(log: MemoryAuditLog) -> list[tuple[str, Outcome, str | None]]:
    return [(r.tool, r.outcome, r.rule_id) for r in reversed(log.query())]


class TestLookupsFeedThePolicy:
    def test_order_lookup_writes_order_total(self) -> None:
        app, _, _ = run([("get_order_status", {"order_id": "A1001"})])
        assert app.handler.session.context().facts["order_total"] == 89.99
        assert app.store.get_order("A1001")["status"] == "shipped"

    def test_profile_lookup_writes_customer_email(self) -> None:
        app, _, _ = run([("get_customer_profile", {"customer_id": "C42"})])
        assert app.handler.session.context().facts["customer_email"] == "sam@example.com"

    def test_failed_lookup_does_not_invent_a_fact(self) -> None:
        app, _, _ = run([("get_order_status", {"order_id": "NOPE"})])
        assert "order_total" not in app.handler.session.context().facts


class TestPolicyUsesThoseFacts:
    def test_refund_without_a_lookup_is_blocked(self) -> None:
        app, _, log = run([("issue_refund", {"order_id": "A1001", "amount": 40})])
        assert app.store.refunds == []
        assert outcomes(log) == [("issue_refund", Outcome.BLOCK, "refund-exceeds-order-total")]

    def test_refund_over_the_looked_up_total_is_blocked(self) -> None:
        app, _, log = run(
            [
                ("get_order_status", {"order_id": "A1001"}),
                ("issue_refund", {"order_id": "A1001", "amount": 500}),
            ]
        )
        assert app.store.refunds == []
        assert ("issue_refund", Outcome.BLOCK, "refund-exceeds-order-total") in outcomes(log)

    def test_in_budget_refund_after_lookup_goes_to_a_human(self) -> None:
        # High-risk: the matrix never auto-allows, even with strong evidence.
        app, result, log = run(
            [
                ("get_order_status", {"order_id": "A1001"}),
                ("issue_refund", {"order_id": "A1001", "amount": 40}),
            ]
        )
        assert app.store.refunds == []
        assert result.stop_reason == "interrupt"
        assert ("issue_refund", Outcome.REQUIRE_APPROVAL, None) in outcomes(log)

    def test_email_to_a_stranger_is_blocked_after_profile_lookup(self) -> None:
        app, _, log = run(
            [
                ("get_customer_profile", {"customer_id": "C42"}),
                (
                    "send_customer_email",
                    {"to": "stranger@evil.test", "subject": "hi", "body": "x"},
                ),
            ]
        )
        assert app.store.emails == []
        assert ("send_customer_email", Outcome.BLOCK, "email-only-to-profile-address") in outcomes(
            log
        )

    def test_email_to_the_profile_address_is_allowed(self) -> None:
        app, _, log = run(
            [
                ("get_customer_profile", {"customer_id": "C42"}),
                (
                    "send_customer_email",
                    {"to": "sam@example.com", "subject": "Your receipt", "body": "Thanks"},
                ),
            ]
        )
        assert app.store.emails[0]["to"] == "sam@example.com"
        assert ("send_customer_email", Outcome.ALLOW, None) in outcomes(log)


class TestForbiddenAndConsent:
    def test_delete_never_runs(self) -> None:
        app, _, log = run([("delete_customer_data", {"customer_id": "C42"})])
        assert "C42" in app.store.customers
        assert outcomes(log)[0][1] is Outcome.BLOCK
        assert log.query()[0].source is DecisionSource.POLICY
        assert log.query()[0].decider_model is None

    def test_cancel_without_consent_does_not_run(self) -> None:
        app, _, _ = run(
            [("cancel_order", {"order_id": "A2002"})],
            script={**GOOD, "consent": 0.1},
        )
        assert app.store.get_order("A2002")["status"] == "processing"

    def test_confirmed_cancel_runs_and_counts(self) -> None:
        app, _, _ = run([("cancel_order", {"order_id": "A2002"})])
        assert app.store.get_order("A2002")["status"] == "cancelled"
        assert app.handler.session.context().prior_calls == {"cancel_order": 1}


class TestHumanApproval:
    def test_approving_a_refund_lets_it_run(self) -> None:
        desk = ApprovalDesk()
        app, result, _ = run(
            [
                ("get_order_status", {"order_id": "A1001"}),
                ("issue_refund", {"order_id": "A1001", "amount": 40}),
            ],
            desk=desk,
        )
        assert result.stop_reason == "interrupt"
        pending = desk.pending()
        assert len(pending) == 1
        assert pending[0].tool == "issue_refund"
        assert pending[0].interrupt_id
        rec, _ = desk.approve(pending[0].id, by="alice")
        assert rec.status is ApprovalStatus.APPROVED
        assert rec.resolved_by == "alice"
        assert app.store.refunds == [{"order_id": "A1001", "amount": 40.0}]

    def test_denying_a_refund_leaves_the_store_untouched(self) -> None:
        desk = ApprovalDesk()
        app, _, _ = run(
            [
                ("get_order_status", {"order_id": "A1001"}),
                ("issue_refund", {"order_id": "A1001", "amount": 40}),
            ],
            desk=desk,
        )
        desk.deny(desk.pending()[0].id, by="bob", note="too risky")
        assert app.store.refunds == []
        assert desk.query(status=ApprovalStatus.DENIED)[0].note == "too risky"
