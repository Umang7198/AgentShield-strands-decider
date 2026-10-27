import pytest

from agentshield.core import CheckKind, Outcome, RiskTier, ToolCall, decide, fallback_decision
from agentshield.decider import (
    QUESTION_SET_VERSION,
    DeciderChecker,
    DeciderTimeout,
    FakeDeciderClient,
    Message,
    StateBudget,
)

REFUND = ToolCall(name="issue_refund", arguments={"order_id": "A123", "amount": 500})
CHAT = [
    Message(role="user", content="My order A123 arrived damaged."),
    Message(role="assistant", content="Sorry to hear that! How can I help?"),
]

GROUNDED = {"arg:*": 0.97, "consent": 0.97, "premature": 0.02}


class TestRun:
    async def test_returns_one_typed_check_per_question(self) -> None:
        run = await DeciderChecker(FakeDeciderClient(GROUNDED)).check_call(REFUND, CHAT)
        assert [(c.kind, c.subject) for c in run.checks] == [
            (CheckKind.ARGUMENT_GROUNDING, "order_id"),
            (CheckKind.ARGUMENT_GROUNDING, "amount"),
            (CheckKind.CONSENT, None),
        ]
        assert [c.p_yes for c in run.checks] == [0.97, 0.97, 0.97]

    async def test_sends_one_request_containing_the_conversation_and_the_call(self) -> None:
        fake = FakeDeciderClient(GROUNDED)
        await DeciderChecker(fake).check_call(REFUND, CHAT)
        assert len(fake.calls) == 1
        sent = fake.calls[0]
        assert "My order A123 arrived damaged." in sent.state
        assert "issue_refund(" in sent.state
        assert set(sent.questions) == {"arg:order_id", "arg:amount", "consent"}

    async def test_run_carries_audit_information(self) -> None:
        run = await DeciderChecker(FakeDeciderClient(GROUNDED)).check_call(REFUND, CHAT)
        assert run.question_set_version == QUESTION_SET_VERSION
        assert run.model == "fake-decider"
        assert run.wall_latency_ms == 1.0
        assert run.server_latency_ms == 1.0
        assert run.state.total_messages == 2
        assert not run.state.lost_evidence

    async def test_unchecked_arguments_are_reported(self) -> None:
        call = ToolCall(name="t", arguments={f"a{i}": i for i in range(9)})
        checker = DeciderChecker(FakeDeciderClient(), max_argument_checks=2)
        run = await checker.check_call(call, CHAT)
        assert run.unchecked_args == tuple(f"a{i}" for i in range(2, 9))

    async def test_lost_evidence_is_visible_to_the_caller(self) -> None:
        many = [Message(role="user", content="m" * 200) for _ in range(100)]
        checker = DeciderChecker(
            FakeDeciderClient(), budget=StateBudget(max_chars=1000, max_message_chars=300)
        )
        run = await checker.check_call(REFUND, many)
        assert run.state.lost_evidence
        assert run.state.dropped_messages > 0

    async def test_premature_check_is_off_by_default(self) -> None:
        run = await DeciderChecker(FakeDeciderClient()).check_call(REFUND, CHAT)
        assert CheckKind.PREMATURE_ACTION not in {c.kind for c in run.checks}

    async def test_premature_check_can_be_enabled(self) -> None:
        run = await DeciderChecker(FakeDeciderClient(GROUNDED)).check_call(
            REFUND, CHAT, include_premature=True
        )
        assert run.checks[-1].kind is CheckKind.PREMATURE_ACTION
        assert run.checks[-1].p_yes == 0.02

    async def test_claim_check(self) -> None:
        fake = FakeDeciderClient({"claim": 0.12})
        run = await DeciderChecker(fake).check_claim("Your refund was issued.", CHAT)
        assert [(c.kind, c.p_yes) for c in run.checks] == [(CheckKind.CLAIM_GROUNDING, 0.12)]
        assert "Proposed tool call" not in fake.calls[0].state


class TestErrors:
    async def test_decider_errors_propagate(self) -> None:
        checker = DeciderChecker(FakeDeciderClient(error=DeciderTimeout("slow")))
        with pytest.raises(DeciderTimeout):
            await checker.check_call(REFUND, CHAT)

    async def test_failure_becomes_a_fail_closed_decision(self) -> None:
        checker = DeciderChecker(FakeDeciderClient(error=DeciderTimeout("slow")))
        try:
            await checker.check_call(REFUND, CHAT)
        except DeciderTimeout as exc:
            d = fallback_decision(RiskTier.MEDIUM, str(exc))
        assert d.outcome is Outcome.REQUIRE_APPROVAL


class TestWithTheDecisionMatrix:
    """The evidence from the checker feeds the step-1 matrix. No model needed."""

    async def decide(
        self, script: dict[str, float], risk: RiskTier, *, premature: bool = False
    ) -> Outcome:
        run = await DeciderChecker(FakeDeciderClient(script)).check_call(
            REFUND, CHAT, include_premature=premature
        )
        return decide(risk, run.checks).outcome

    async def test_well_supported_low_risk_call_is_allowed(self) -> None:
        assert await self.decide(GROUNDED, RiskTier.LOW) is Outcome.ALLOW

    async def test_one_invented_argument_forces_clarification(self) -> None:
        script = {"arg:amount": 0.30, **GROUNDED}
        assert await self.decide(script, RiskTier.LOW) is Outcome.CLARIFY

    async def test_wildly_unsupported_argument_is_blocked(self) -> None:
        script = {"arg:amount": 0.02, **GROUNDED}
        assert await self.decide(script, RiskTier.LOW) is Outcome.BLOCK

    async def test_missing_consent_stops_a_medium_risk_call(self) -> None:
        # The live Decider gave 0.37 for "user only asked where their order is".
        script = {**GROUNDED, "consent": 0.37}
        assert await self.decide(script, RiskTier.MEDIUM) is Outcome.CLARIFY

    async def test_agent_acting_prematurely_is_not_allowed(self) -> None:
        # premature p_yes=0.9 means "should clarify first": p_ok = 0.1, not enough to allow.
        script = {"premature": 0.9, **{k: v for k, v in GROUNDED.items() if k != "premature"}}
        assert await self.decide(script, RiskTier.LOW, premature=True) is Outcome.CLARIFY

    async def test_high_risk_always_goes_to_a_human_even_when_well_supported(self) -> None:
        assert await self.decide(GROUNDED, RiskTier.HIGH) is Outcome.REQUIRE_APPROVAL


class TestFake:
    async def test_first_matching_pattern_wins(self) -> None:
        fake = FakeDeciderClient({"arg:amount": 0.1, "arg:*": 0.9})
        run = await DeciderChecker(fake).check_call(REFUND, CHAT)
        by_subject = {c.subject: c.p_yes for c in run.checks if c.subject}
        assert by_subject == {"order_id": 0.9, "amount": 0.1}

    async def test_unmatched_questions_get_the_default(self) -> None:
        run = await DeciderChecker(FakeDeciderClient(default=0.42)).check_call(REFUND, CHAT)
        assert {c.p_yes for c in run.checks} == {0.42}
