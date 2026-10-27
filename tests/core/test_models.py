import pytest
from pydantic import ValidationError

from agentshield.core import CheckKind, CheckResult, ToolCall


def check(kind: CheckKind, p_yes: float, subject: str | None = None) -> CheckResult:
    return CheckResult(kind=kind, p_yes=p_yes, subject=subject)


class TestCheckResult:
    @pytest.mark.parametrize(
        "kind", [CheckKind.ARGUMENT_GROUNDING, CheckKind.CONSENT, CheckKind.CLAIM_GROUNDING]
    )
    def test_yes_is_good_kinds_use_p_yes_directly(self, kind: CheckKind) -> None:
        assert check(kind, 0.9).p_ok == 0.9

    def test_premature_action_flips_polarity(self) -> None:
        # "Should the agent clarify first?" yes = bad, so p_ok is the complement.
        assert check(CheckKind.PREMATURE_ACTION, 0.9).p_ok == pytest.approx(0.1)
        assert check(CheckKind.PREMATURE_ACTION, 0.1).p_ok == pytest.approx(0.9)

    @pytest.mark.parametrize(
        ("p_yes", "expected"), [(0.0, 1.0), (0.5, 0.5), (0.37, 0.63), (1.0, 1.0)]
    )
    def test_confidence_is_certainty_either_way(self, p_yes: float, expected: float) -> None:
        assert check(CheckKind.CONSENT, p_yes).confidence == pytest.approx(expected)

    def test_p_ok_is_not_pushed_across_a_threshold_by_float_noise(self) -> None:
        # 1 - 0.08 == 0.9199999999999999 in raw floating point; must compare as 0.92.
        assert check(CheckKind.PREMATURE_ACTION, 0.08).p_ok == 0.92

    @pytest.mark.parametrize("bad", [-0.01, 1.01])
    def test_p_yes_must_be_a_probability(self, bad: float) -> None:
        with pytest.raises(ValidationError):
            check(CheckKind.CONSENT, bad)

    def test_negative_latency_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CheckResult(kind=CheckKind.CONSENT, p_yes=0.5, latency_ms=-1)

    def test_label_includes_subject_when_present(self) -> None:
        assert (
            check(CheckKind.ARGUMENT_GROUNDING, 0.5, "amount").label == "argument_grounding[amount]"
        )
        assert check(CheckKind.CONSENT, 0.5).label == "consent"

    def test_is_immutable(self) -> None:
        c = check(CheckKind.CONSENT, 0.5)
        with pytest.raises(ValidationError):
            c.p_yes = 0.9  # type: ignore[misc]


class TestToolCall:
    def test_defaults(self) -> None:
        call = ToolCall(name="get_order_status")
        assert call.arguments == {}
        assert call.tool_use_id is None

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall(name="")

    def test_is_immutable(self) -> None:
        call = ToolCall(name="issue_refund", arguments={"amount": 5})
        with pytest.raises(ValidationError):
            call.name = "other"  # type: ignore[misc]
