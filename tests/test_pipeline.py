"""The pipeline: real ShopHelp policy + scripted Decider (no model needed)."""

from pathlib import Path
from typing import Any

import pytest

from agentshield.audit import MemoryAuditLog, Redactor
from agentshield.core import (
    CheckKind,
    DecisionSource,
    KindThresholds,
    Outcome,
    RiskTier,
    Thresholds,
    ToolCall,
)
from agentshield.decider import DeciderChecker, DeciderTimeout, FakeDeciderClient, Message
from agentshield.pipeline import Shield
from agentshield.policy import PolicyEngine, SessionState, load_policy

POLICY = Path(__file__).resolve().parents[1] / "policies" / "shophelp.yaml"

GOOD = {"arg:*": 0.95, "consent": 0.95}
CHAT = [Message(role="user", content="Please cancel order A1001.")]


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(name=name, arguments=args)


def make(
    script: dict[str, float] | None = None, **fake_kwargs: Any
) -> tuple[Shield, FakeDeciderClient]:
    fake = FakeDeciderClient(script if script is not None else GOOD, **fake_kwargs)
    shield = Shield(PolicyEngine(load_policy(POLICY)), DeciderChecker(fake))
    return shield, fake


@pytest.fixture
def session() -> SessionState:
    return SessionState("s1", facts={"order_total": 89.99, "customer_email": "ann@example.com"})


class TestPolicyFirst:
    async def test_forbidden_action_is_blocked_without_asking_the_decider(self) -> None:
        shield, fake = make()
        v = await shield.evaluate(call("delete_customer_data", customer_id="C1"), CHAT)
        assert v.decision.outcome is Outcome.BLOCK
        assert v.decision.source is DecisionSource.POLICY
        assert v.decision.rule_id == "no-delete-customer-data"
        assert v.run is None
        assert fake.calls == []

    async def test_unknown_tool_is_blocked(self) -> None:
        shield, fake = make()
        v = await shield.evaluate(call("launch_missiles"), CHAT)
        assert v.decision.outcome is Outcome.BLOCK
        assert fake.calls == []

    async def test_policy_beats_a_confident_decider(self, session: SessionState) -> None:
        # The Decider would happily approve, but the refund exceeds the order total.
        shield, _ = make()
        v = await shield.evaluate(call("issue_refund", order_id="A1", amount=500), CHAT, session)
        assert v.decision.outcome is Outcome.BLOCK
        assert v.decision.rule_id == "refund-exceeds-order-total"

    async def test_session_quota_is_enforced(self, session: SessionState) -> None:
        for _ in range(3):
            session.record_call("issue_refund")
        shield, _ = make()
        v = await shield.evaluate(call("issue_refund", order_id="A1", amount=10), CHAT, session)
        assert v.decision.rule_id == "max-3-refunds-per-session"

    async def test_missing_fact_fails_closed(self) -> None:
        shield, _ = make()
        v = await shield.evaluate(call("issue_refund", order_id="A1", amount=10), CHAT)
        assert v.decision.outcome is Outcome.BLOCK  # no order_total known


class TestDeciderEvidence:
    async def test_well_supported_low_risk_call_is_allowed(self) -> None:
        shield, fake = make()
        v = await shield.evaluate(call("get_order_status", order_id="A1001"), CHAT)
        assert v.decision.outcome is Outcome.ALLOW
        assert v.decision.source is DecisionSource.DECIDER
        assert v.run is not None
        assert {c.kind for c in v.run.checks} == {CheckKind.ARGUMENT_GROUNDING, CheckKind.CONSENT}
        assert len(fake.calls) == 1

    async def test_invented_argument_needs_clarification(self) -> None:
        shield, _ = make({"arg:order_id": 0.2, **GOOD})
        v = await shield.evaluate(call("cancel_order", order_id="A9999"), CHAT)
        assert v.decision.outcome is Outcome.CLARIFY
        assert v.decision.limiting_check is not None
        assert v.decision.limiting_check.subject == "order_id"

    async def test_missing_consent_stops_a_write(self) -> None:
        shield, _ = make({**GOOD, "consent": 0.1})
        v = await shield.evaluate(call("cancel_order", order_id="A1001"), CHAT)
        assert v.decision.outcome is Outcome.CLARIFY
        assert v.decision.limiting_check is not None
        assert v.decision.limiting_check.kind is CheckKind.CONSENT

    async def test_high_risk_goes_to_a_human_even_when_well_supported(
        self, session: SessionState
    ) -> None:
        shield, _ = make()
        v = await shield.evaluate(call("issue_refund", order_id="A1", amount=40), CHAT, session)
        assert v.decision.outcome is Outcome.REQUIRE_APPROVAL
        assert v.decision.risk is RiskTier.HIGH
        assert v.run is not None  # evidence is still gathered for the approver

    async def test_hopeless_high_risk_call_is_blocked_not_escalated(
        self, session: SessionState
    ) -> None:
        shield, _ = make({"arg:amount": 0.01, **GOOD})
        v = await shield.evaluate(call("issue_refund", order_id="A1", amount=40), CHAT, session)
        assert v.decision.outcome is Outcome.BLOCK

    async def test_conversation_and_call_reach_the_decider(self) -> None:
        shield, fake = make()
        await shield.evaluate(call("cancel_order", order_id="A1001"), CHAT)
        assert "Please cancel order A1001." in fake.calls[0].state
        assert "cancel_order(" in fake.calls[0].state

    async def test_custom_thresholds_are_used(self) -> None:
        strict = KindThresholds(
            argument_grounding=0.99, consent=0.99, premature_action=0.99, claim_grounding=0.99
        )
        engine = PolicyEngine(load_policy(POLICY))
        shield = Shield(
            engine,
            DeciderChecker(FakeDeciderClient(GOOD)),
            thresholds=Thresholds(low=strict, medium=strict),
        )
        v = await shield.evaluate(call("get_order_status", order_id="A1001"), CHAT)
        assert v.decision.outcome is Outcome.CLARIFY

    async def test_premature_check_can_be_switched_on(self) -> None:
        fake = FakeDeciderClient({**GOOD, "premature": 0.9})
        shield = Shield(
            PolicyEngine(load_policy(POLICY)), DeciderChecker(fake), include_premature=True
        )
        v = await shield.evaluate(call("get_order_status", order_id="A1001"), CHAT)
        assert v.decision.outcome is Outcome.CLARIFY
        assert "premature" in fake.calls[0].questions


class TestDeciderFailure:
    @pytest.mark.parametrize(
        ("tool", "args", "expected"),
        [
            ("get_order_status", {"order_id": "A1"}, Outcome.CLARIFY),
            ("cancel_order", {"order_id": "A1"}, Outcome.REQUIRE_APPROVAL),
        ],
    )
    async def test_falls_back_fail_closed(
        self, tool: str, args: dict[str, Any], expected: Outcome
    ) -> None:
        shield, _ = make(error=DeciderTimeout("slow"))
        v = await shield.evaluate(call(tool, **args), CHAT)
        assert v.decision.outcome is expected
        assert v.decision.source is DecisionSource.FALLBACK
        assert v.decider_error == "slow"
        assert v.run is None

    async def test_policy_still_works_when_the_decider_is_down(self) -> None:
        shield, _ = make(error=DeciderTimeout("slow"))
        v = await shield.evaluate(call("delete_customer_data", customer_id="C1"), CHAT)
        assert v.decision.source is DecisionSource.POLICY
        assert v.decider_error is None


class TestVerdict:
    async def test_carries_audit_information(self) -> None:
        shield, _ = make()
        c = call("get_order_status", order_id="A1001")
        v = await shield.evaluate(c, CHAT)
        assert v.call == c
        assert v.policy.tool == "get_order_status"
        assert v.total_ms >= 0

    async def test_works_without_a_session(self) -> None:
        shield, _ = make()
        v = await shield.evaluate(call("get_order_status", order_id="A1001"), CHAT)
        assert v.decision.outcome is Outcome.ALLOW


class TestAudit:
    def _shield(self, log: MemoryAuditLog, **kw: Any) -> Shield:
        return Shield(
            PolicyEngine(load_policy(POLICY)),
            DeciderChecker(FakeDeciderClient(GOOD)),
            sinks=[log],
            agent_id="shophelp",
            **kw,
        )

    async def test_every_decision_is_written(self, session: SessionState) -> None:
        log = MemoryAuditLog()
        v = await self._shield(log).evaluate(
            call("get_order_status", order_id="A1001"), CHAT, session
        )
        assert v.audit_id is not None
        rec = log.get(v.audit_id)
        assert rec is not None
        assert rec.session_id == "s1"
        assert rec.agent_id == "shophelp"
        assert rec.policy_name == "shophelp"
        assert rec.outcome is Outcome.ALLOW
        assert rec.arguments == {"order_id": "A1001"}
        assert rec.question_set_version is not None
        assert rec.decider_ms is not None
        assert rec.checks

    async def test_policy_short_circuit_is_still_audited(self) -> None:
        log = MemoryAuditLog()
        v = await self._shield(log).evaluate(call("delete_customer_data", customer_id="C1"), CHAT)
        rec = log.get(v.audit_id)  # type: ignore[arg-type]
        assert rec is not None
        assert rec.source is DecisionSource.POLICY
        assert rec.rule_id == "no-delete-customer-data"
        assert rec.question_set_version is None
        assert rec.matched_rules[0].rule_id == "no-delete-customer-data"

    async def test_redaction_is_audit_only(self) -> None:
        log = MemoryAuditLog()
        c = call("get_order_status", order_id="A1001")
        v = await self._shield(log, redactor=Redactor(all_values=True)).evaluate(c, CHAT)
        assert v.call.arguments["order_id"] == "A1001"
        rec = log.get(v.audit_id) if v.audit_id else None
        assert rec is not None
        assert rec.arguments["order_id"] == "***"

    async def test_a_broken_sink_surfaces(self) -> None:
        class Boom:
            def record(self, record: object) -> None:
                raise RuntimeError("disk full")

        shield = Shield(
            PolicyEngine(load_policy(POLICY)),
            DeciderChecker(FakeDeciderClient(GOOD)),
            sinks=[Boom()],  # type: ignore[list-item]
        )
        with pytest.raises(RuntimeError, match="disk full"):
            await shield.evaluate(call("get_order_status", order_id="A1"), CHAT)
