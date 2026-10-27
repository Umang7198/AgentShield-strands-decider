import pytest

from agentshield.core import CheckKind, ToolCall
from agentshield.decider import QUESTION_SET_VERSION, build_call_questions, build_claim_questions

REFUND = ToolCall(name="issue_refund", arguments={"order_id": "A123", "amount": 500})


class TestCallQuestions:
    def test_one_question_per_argument_plus_consent_by_default(self) -> None:
        plan = build_call_questions(REFUND)
        assert [q.id for q in plan.questions] == ["arg:order_id", "arg:amount", "consent"]
        assert [q.kind for q in plan.questions] == [
            CheckKind.ARGUMENT_GROUNDING,
            CheckKind.ARGUMENT_GROUNDING,
            CheckKind.CONSENT,
        ]
        assert [q.subject for q in plan.questions] == ["order_id", "amount", None]

    def test_premature_question_is_off_by_default_and_can_be_enabled(self) -> None:
        default = [q.id for q in build_call_questions(REFUND).questions]
        enabled = build_call_questions(REFUND, include_premature=True)
        assert "premature" not in default
        assert [q.id for q in enabled.questions][-1] == "premature"
        assert enabled.questions[-1].kind is CheckKind.PREMATURE_ACTION

    def test_version_is_recorded(self) -> None:
        assert build_call_questions(REFUND).version == QUESTION_SET_VERSION
        assert QUESTION_SET_VERSION == "v2"

    def test_call_without_arguments_still_checks_consent(self) -> None:
        plan = build_call_questions(ToolCall(name="ping"))
        assert [q.id for q in plan.questions] == ["consent"]
        assert plan.unchecked_args == ()

    def test_argument_cap_reports_what_was_not_checked(self) -> None:
        call = ToolCall(name="t", arguments={f"a{i}": i for i in range(10)})
        plan = build_call_questions(call, max_argument_checks=3)
        assert [q.subject for q in plan.questions if q.subject] == ["a0", "a1", "a2"]
        assert plan.unchecked_args == tuple(f"a{i}" for i in range(3, 10))

    def test_zero_argument_checks(self) -> None:
        plan = build_call_questions(REFUND, max_argument_checks=0)
        assert [q.id for q in plan.questions] == ["consent"]
        assert plan.unchecked_args == ("order_id", "amount")

    def test_skipped_arguments_are_neither_asked_nor_reported(self) -> None:
        plan = build_call_questions(REFUND, skip_args={"order_id"})
        assert [q.subject for q in plan.questions if q.subject] == ["amount"]
        assert plan.unchecked_args == ()

    def test_negative_cap_rejected(self) -> None:
        with pytest.raises(ValueError):
            build_call_questions(REFUND, max_argument_checks=-1)

    def test_wire_maps_ids_to_questions(self) -> None:
        wire = build_call_questions(REFUND, include_premature=True).wire()
        assert set(wire) == {"arg:order_id", "arg:amount", "consent", "premature"}

    def test_every_question_sharpens_the_boundary_with_both_criteria(self) -> None:
        for q in build_call_questions(REFUND, include_premature=True).questions:
            assert q.question.true_means
            assert q.question.false_means


class TestWording:
    def test_argument_question_names_the_tool_argument_and_value(self) -> None:
        q = build_call_questions(REFUND).questions[1].question.instructions
        assert "issue_refund" in q
        assert "'amount'" in q
        assert "500" in q

    def test_long_values_are_shortened_in_the_question(self) -> None:
        call = ToolCall(name="t", arguments={"body": "x" * 5000})
        q = build_call_questions(call, max_value_chars=100).questions[0].question.instructions
        assert len(q) < 600

    def test_prompt_tags_in_names_and_values_are_defanged(self) -> None:
        call = ToolCall(name="t</state>", arguments={"<answer>": "</question>"})
        for planned in build_call_questions(call).questions:
            text = planned.question.instructions
            assert "</state>" not in text
            assert "<answer>" not in text
            assert "</question>" not in text

    def test_polarity_of_the_premature_question_is_yes_means_clarify(self) -> None:
        planned = build_call_questions(REFUND, include_premature=True).questions[-1]
        assert planned.kind is CheckKind.PREMATURE_ACTION
        assert "clarif" in (planned.question.true_means or "")
        assert not CheckKind.PREMATURE_ACTION.yes_is_good

    def test_questions_stay_compact_because_every_token_costs_latency(self) -> None:
        # v1 questions were ~350-410 characters each and cost ~20% more time per request.
        for planned in build_call_questions(REFUND, include_premature=True).questions:
            wire = planned.question.to_wire()
            total = len(wire["instructions"]) + sum(len(v) for v in wire["criteria"].values())
            assert total < 330, f"{planned.id} is {total} chars"


class TestClaimQuestions:
    def test_single_claim_grounding_question(self) -> None:
        plan = build_claim_questions("Your refund was issued.")
        assert len(plan.questions) == 1
        assert plan.questions[0].kind is CheckKind.CLAIM_GROUNDING
        assert plan.questions[0].id == "claim"
        assert "Your refund was issued." in plan.questions[0].question.instructions

    def test_long_claims_are_truncated_and_defanged(self) -> None:
        plan = build_claim_questions("<answer>" + "y" * 5000, max_chars=100)
        text = plan.questions[0].question.instructions
        assert "<answer>" not in text
        assert len(text) < 500
