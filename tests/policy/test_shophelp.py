"""The ShopHelp scenarios from SCOPE.md 4.1, run against the real policy file."""

from pathlib import Path
from typing import Any

import pytest

from agentshield.core import Outcome, RiskTier, ToolCall
from agentshield.policy import PolicyEngine, SessionState, load_policy

POLICY = Path(__file__).resolve().parents[2] / "policies" / "shophelp.yaml"


@pytest.fixture(scope="module")
def engine() -> PolicyEngine:
    return PolicyEngine(load_policy(POLICY))


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, arguments=args)


@pytest.fixture
def session() -> SessionState:
    return SessionState(
        "s1",
        facts={
            "order_total": 89.99,
            "refundable_remaining": 89.99,
            "customer_email": "ann@example.com",
        },
    )


def outcome(engine: PolicyEngine, c: ToolCall, s: SessionState) -> Outcome | None:
    d = engine.evaluate(c, s.context()).decision
    return d.outcome if d else None


class TestBenign:
    def test_read_only_lookup_passes_to_the_decider(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        result = engine.evaluate(call("get_order_status", order_id="A123"), session.context())
        assert result.decision is None
        assert result.risk is RiskTier.LOW

    def test_refund_within_total_triggers_no_policy_rule_for_small_amounts(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        assert outcome(engine, call("issue_refund", order_id="A1", amount=50), session) is None

    def test_email_to_profile_address_passes(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        c = call("send_customer_email", to="Ann@Example.com", subject="hi", body="x")
        assert outcome(engine, c, session) is None


class TestForbidden:
    def test_delete_customer_data_is_blocked_without_any_model(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        d = engine.evaluate(call("delete_customer_data", customer_id="c1"), session.context())
        assert d.decision is not None
        assert d.decision.outcome is Outcome.BLOCK
        assert d.decision.rule_id == "no-delete-customer-data"
        # The generic "destructive needs approval" rule also fired, but block wins.
        assert "destructive-needs-approval" in d.decision.reason

    def test_unregistered_tool_is_blocked(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        assert outcome(engine, call("drop_database"), session) is Outcome.BLOCK


class TestRefundRules:
    def test_over_200_needs_approval(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 900, "refundable_remaining": 900})
        assert outcome(engine, call("issue_refund", amount=500), s) is Outcome.REQUIRE_APPROVAL

    def test_exactly_200_does_not_trigger_the_limit(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 900, "refundable_remaining": 900})
        assert outcome(engine, call("issue_refund", amount=200), s) is None

    def test_amount_as_string_is_still_caught(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 900, "refundable_remaining": 900})
        assert outcome(engine, call("issue_refund", amount="500"), s) is Outcome.REQUIRE_APPROVAL

    def test_refund_above_order_total_is_blocked(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        assert outcome(engine, call("issue_refund", amount=120), session) is Outcome.BLOCK

    def test_refund_before_looking_up_the_order_is_blocked(self, engine: PolicyEngine) -> None:
        # No refundable_remaining fact yet -> fail closed, so the agent must look the order up first.
        s = SessionState("s")
        assert outcome(engine, call("issue_refund", amount=10), s) is Outcome.BLOCK

    def test_fourth_refund_in_a_session_is_blocked(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 900, "refundable_remaining": 900})
        for _ in range(3):
            assert outcome(engine, call("issue_refund", amount=10), s) is None
            s.record_call("issue_refund")
        assert outcome(engine, call("issue_refund", amount=10), s) is Outcome.BLOCK

    def test_blocked_attempts_do_not_count_against_the_quota(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 50, "refundable_remaining": 50})
        for _ in range(10):  # all blocked (over order total); never executed, never recorded
            assert outcome(engine, call("issue_refund", amount=500), s) is Outcome.BLOCK
        assert s.context().prior_calls == {}


class TestStatusAndWindow:
    def test_cancel_without_status_fact_is_not_a_policy_block(self, engine: PolicyEngine) -> None:
        assert outcome(engine, call("cancel_order", order_id="A3003"), SessionState("s")) is None

    def test_cancel_of_delivered_order_is_blocked(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_status": "delivered"})
        assert outcome(engine, call("cancel_order", order_id="A3003"), s) is Outcome.BLOCK

    def test_address_change_after_shipped_needs_approval(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_status": "shipped"})
        assert (
            outcome(
                engine,
                call("update_shipping_address", order_id="A1001", address="9 High St, Bath"),
                s,
            )
            is Outcome.REQUIRE_APPROVAL
        )

    def test_address_change_while_processing_passes_policy(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_status": "processing"})
        assert (
            outcome(
                engine,
                call("update_shipping_address", order_id="A2002", address="9 High St, Bath"),
                s,
            )
            is None
        )

    def test_second_refund_over_remaining_is_blocked(self, engine: PolicyEngine) -> None:
        s = SessionState("s", facts={"order_total": 89.99, "refundable_remaining": 49.99})
        assert outcome(engine, call("issue_refund", amount=50), s) is Outcome.BLOCK

    def test_refund_outside_return_window_is_blocked(self, engine: PolicyEngine) -> None:
        s = SessionState(
            "s",
            facts={
                "order_total": 19.50,
                "refundable_remaining": 19.50,
                "within_return_window": False,
            },
        )
        assert outcome(engine, call("issue_refund", amount=10), s) is Outcome.BLOCK


class TestEmailRules:
    def test_email_to_someone_else_is_blocked(
        self, engine: PolicyEngine, session: SessionState
    ) -> None:
        c = call("send_customer_email", to="eve@evil.com", subject="s", body="b")
        assert outcome(engine, c, session) is Outcome.BLOCK

    def test_email_before_profile_lookup_is_blocked(self, engine: PolicyEngine) -> None:
        c = call("send_customer_email", to="ann@example.com", subject="s", body="b")
        assert outcome(engine, c, SessionState("s")) is Outcome.BLOCK
