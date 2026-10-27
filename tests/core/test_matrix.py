import pytest
from pydantic import ValidationError

from agentshield.core import (
    CheckKind,
    CheckResult,
    DecisionSource,
    Outcome,
    RiskTier,
    Thresholds,
    decide,
    fallback_decision,
    weakest,
)


def ok(p: float, kind: CheckKind = CheckKind.CONSENT, subject: str | None = None) -> CheckResult:
    """A check whose p_ok equals ``p`` (all yes-is-good kinds)."""
    return CheckResult(kind=kind, p_yes=p, subject=subject)


class TestDefaultThresholds:
    def test_match_scope_document(self) -> None:
        t = Thresholds()
        assert t.allow_for(RiskTier.LOW) == 0.85
        assert t.allow_for(RiskTier.MEDIUM) == 0.92
        assert t.allow_for(RiskTier.HIGH) is None


class TestAllow:
    def test_low_risk_high_confidence_allows(self) -> None:
        d = decide(RiskTier.LOW, [ok(0.95), ok(0.90)])
        assert d.outcome is Outcome.ALLOW
        assert d.source is DecisionSource.DECIDER

    def test_exactly_at_threshold_allows(self) -> None:
        assert decide(RiskTier.LOW, [ok(0.85)]).outcome is Outcome.ALLOW
        assert decide(RiskTier.MEDIUM, [ok(0.92)]).outcome is Outcome.ALLOW

    def test_medium_needs_more_than_low(self) -> None:
        checks = [ok(0.90)]
        assert decide(RiskTier.LOW, checks).outcome is Outcome.ALLOW
        assert decide(RiskTier.MEDIUM, checks).outcome is Outcome.CLARIFY


class TestClarify:
    def test_below_threshold_clarifies(self) -> None:
        d = decide(RiskTier.LOW, [ok(0.60)])
        assert d.outcome is Outcome.CLARIFY
        assert "0.85" in d.reason

    def test_one_weak_check_is_enough_even_if_others_are_strong(self) -> None:
        d = decide(RiskTier.LOW, [ok(0.99), ok(0.50, subject="amount"), ok(0.97)])
        assert d.outcome is Outcome.CLARIFY

    def test_real_world_observed_consent_value_is_not_allowed(self) -> None:
        # Observed from the live Decider: "user only asked where their order is" gave p_yes=0.37.
        d = decide(RiskTier.MEDIUM, [ok(0.37)])
        assert d.outcome is Outcome.CLARIFY


class TestRequireApproval:
    @pytest.mark.parametrize("p", [0.60, 0.99, 1.0])
    def test_high_risk_never_auto_allows(self, p: float) -> None:
        assert decide(RiskTier.HIGH, [ok(p)]).outcome is Outcome.REQUIRE_APPROVAL


class TestBlock:
    @pytest.mark.parametrize("risk", list(RiskTier))
    def test_clearly_unsupported_blocks_at_every_tier(self, risk: RiskTier) -> None:
        assert decide(risk, [ok(0.02)]).outcome is Outcome.BLOCK

    def test_at_block_threshold_blocks(self) -> None:
        assert decide(RiskTier.LOW, [ok(0.05)]).outcome is Outcome.BLOCK

    def test_just_above_block_threshold_does_not_block(self) -> None:
        assert decide(RiskTier.LOW, [ok(0.06)]).outcome is Outcome.CLARIFY

    def test_block_beats_high_risk_approval(self) -> None:
        # Don't bother a human with a call the evidence says is nonsense.
        assert decide(RiskTier.HIGH, [ok(0.01)]).outcome is Outcome.BLOCK

    def test_premature_action_polarity_drives_block(self) -> None:
        # p_yes=0.97 on "should clarify first?" -> p_ok=0.03 -> block.
        d = decide(RiskTier.LOW, [CheckResult(kind=CheckKind.PREMATURE_ACTION, p_yes=0.97)])
        assert d.outcome is Outcome.BLOCK

    def test_premature_action_polarity_drives_allow(self) -> None:
        d = decide(RiskTier.LOW, [CheckResult(kind=CheckKind.PREMATURE_ACTION, p_yes=0.08)])
        assert d.outcome is Outcome.ALLOW  # p_ok = 0.92


class TestEvidenceReporting:
    def test_limiting_check_is_the_weakest(self) -> None:
        weak = ok(0.40, CheckKind.ARGUMENT_GROUNDING, "amount")
        d = decide(RiskTier.LOW, [ok(0.95), weak, ok(0.90)])
        assert d.limiting_check == weak
        assert "argument_grounding[amount]" in d.reason

    def test_all_checks_are_kept_on_the_decision(self) -> None:
        checks = [ok(0.95), ok(0.90)]
        assert decide(RiskTier.LOW, checks).checks == tuple(checks)

    def test_weakest_tie_goes_to_first(self) -> None:
        a, b = ok(0.5, subject="a"), ok(0.5, subject="b")
        assert weakest([a, b]) is a

    def test_decision_is_deterministic(self) -> None:
        checks = [ok(0.7), ok(0.9)]
        assert decide(RiskTier.LOW, checks) == decide(RiskTier.LOW, checks)


class TestNoEvidence:
    def test_decide_rejects_empty_checks(self) -> None:
        with pytest.raises(ValueError, match="at least one check"):
            decide(RiskTier.LOW, [])


class TestFallback:
    def test_low_risk_asks_user(self) -> None:
        d = fallback_decision(RiskTier.LOW, "timeout")
        assert d.outcome is Outcome.CLARIFY
        assert d.source is DecisionSource.FALLBACK
        assert "timeout" in d.reason

    @pytest.mark.parametrize("risk", [RiskTier.MEDIUM, RiskTier.HIGH])
    def test_riskier_tiers_go_to_a_human(self, risk: RiskTier) -> None:
        assert fallback_decision(risk, "boom").outcome is Outcome.REQUIRE_APPROVAL

    @pytest.mark.parametrize("risk", list(RiskTier))
    def test_never_allows(self, risk: RiskTier) -> None:
        assert fallback_decision(risk, "x").outcome is not Outcome.ALLOW


class TestCustomThresholds:
    def test_stricter_thresholds_change_outcome(self) -> None:
        strict = Thresholds(allow_low=0.99, allow_medium=0.995)
        assert decide(RiskTier.LOW, [ok(0.95)], strict).outcome is Outcome.CLARIFY

    def test_block_below_zero_disables_blocking_of_nonzero_evidence(self) -> None:
        t = Thresholds(block_below=0.0)
        assert decide(RiskTier.LOW, [ok(0.01)], t).outcome is Outcome.CLARIFY

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"block_below": 0.9},  # not below allow_low
            {"allow_low": 0.95, "allow_medium": 0.90},  # riskier tier is looser
            {"allow_low": 1.5},  # out of range
            {"block_below": -0.1},
        ],
    )
    def test_invalid_thresholds_rejected(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            Thresholds(**kwargs)
