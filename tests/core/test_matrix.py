import pytest
from pydantic import ValidationError

from agentshield.core import (
    CheckKind,
    CheckResult,
    DecisionSource,
    KindThresholds,
    Outcome,
    RiskTier,
    Thresholds,
    decide,
    fallback_decision,
    weakest,
)

ARG = CheckKind.ARGUMENT_GROUNDING
CONSENT = CheckKind.CONSENT
PREMATURE = CheckKind.PREMATURE_ACTION
CLAIM = CheckKind.CLAIM_GROUNDING

T = Thresholds()


def chk(p_ok: float, kind: CheckKind = CONSENT, subject: str | None = None) -> CheckResult:
    """A check whose p_ok equals ``p_ok``, whatever the kind's polarity."""
    p_yes = p_ok if kind.yes_is_good else 1.0 - p_ok
    return CheckResult(kind=kind, p_yes=p_yes, subject=subject)


class TestDefaultThresholds:
    def test_medium_is_never_looser_than_low(self) -> None:
        for kind in CheckKind:
            assert T.low.for_kind(kind) <= T.medium.for_kind(kind)

    def test_everything_sits_above_the_block_threshold(self) -> None:
        for kind in CheckKind:
            assert T.block_below < T.low.for_kind(kind)

    def test_consent_needs_a_stricter_bar_than_argument_grounding(self) -> None:
        # Measured live: argument grounding separates cleanly, consent overlaps.
        assert T.medium.consent > T.medium.argument_grounding

    def test_high_risk_has_no_threshold(self) -> None:
        for kind in CheckKind:
            assert T.allow_for(RiskTier.HIGH, kind) is None

    def test_allow_for_looks_up_tier_and_kind(self) -> None:
        assert T.allow_for(RiskTier.LOW, CONSENT) == T.low.consent
        assert T.allow_for(RiskTier.MEDIUM, ARG) == T.medium.argument_grounding


class TestAllow:
    def test_low_risk_with_strong_evidence_allows(self) -> None:
        d = decide(RiskTier.LOW, [chk(0.9, ARG, "id"), chk(0.9, CONSENT)])
        assert d.outcome is Outcome.ALLOW
        assert d.source is DecisionSource.DECIDER

    @pytest.mark.parametrize("kind", list(CheckKind))
    @pytest.mark.parametrize("risk", [RiskTier.LOW, RiskTier.MEDIUM])
    def test_exactly_at_the_threshold_allows(self, kind: CheckKind, risk: RiskTier) -> None:
        need = T.allow_for(risk, kind)
        assert need is not None
        assert decide(risk, [chk(need, kind)]).outcome is Outcome.ALLOW

    @pytest.mark.parametrize("kind", list(CheckKind))
    @pytest.mark.parametrize("risk", [RiskTier.LOW, RiskTier.MEDIUM])
    def test_just_below_the_threshold_clarifies(self, kind: CheckKind, risk: RiskTier) -> None:
        need = T.allow_for(risk, kind)
        assert need is not None
        assert decide(risk, [chk(need - 0.01, kind)]).outcome is Outcome.CLARIFY

    def test_medium_is_stricter_than_low(self) -> None:
        checks = [chk(0.70, CONSENT)]
        assert decide(RiskTier.LOW, checks).outcome is Outcome.ALLOW
        assert decide(RiskTier.MEDIUM, checks).outcome is Outcome.CLARIFY

    def test_each_kind_is_judged_by_its_own_bar(self) -> None:
        # 0.70 is plenty for argument grounding at medium risk, but not for consent.
        assert decide(RiskTier.MEDIUM, [chk(0.70, ARG)]).outcome is Outcome.ALLOW
        assert decide(RiskTier.MEDIUM, [chk(0.70, CONSENT)]).outcome is Outcome.CLARIFY

    def test_allow_reports_the_check_closest_to_its_threshold(self) -> None:
        tight = chk(0.82, CONSENT)  # 0.02 above 0.80
        d = decide(RiskTier.MEDIUM, [chk(0.99, ARG, "id"), tight])
        assert d.outcome is Outcome.ALLOW
        assert d.limiting_check == tight


class TestClarify:
    def test_one_failing_check_is_enough(self) -> None:
        d = decide(RiskTier.LOW, [chk(0.99, ARG, "id"), chk(0.30, ARG, "amount"), chk(0.9)])
        assert d.outcome is Outcome.CLARIFY
        assert "amount" in d.reason

    def test_limiting_check_is_the_failing_one_not_merely_the_lowest(self) -> None:
        # At medium risk: argument grounding 0.66 clears 0.65, consent 0.79 misses 0.80.
        # The lowest p_ok is the argument, but the consent is what actually stops the call.
        arg, consent = chk(0.66, ARG, "id"), chk(0.79, CONSENT)
        assert weakest([arg, consent]) == arg
        d = decide(RiskTier.MEDIUM, [arg, consent])
        assert d.outcome is Outcome.CLARIFY
        assert d.limiting_check == consent
        assert "needed 0.80" in d.reason

    def test_reports_the_check_furthest_below_its_threshold(self) -> None:
        near, far = chk(0.62, ARG, "near"), chk(0.20, ARG, "far")
        d = decide(RiskTier.MEDIUM, [near, far])
        assert d.limiting_check == far


class TestRequireApproval:
    @pytest.mark.parametrize("p", [0.30, 0.99, 1.0])
    def test_high_risk_never_auto_allows(self, p: float) -> None:
        assert decide(RiskTier.HIGH, [chk(p)]).outcome is Outcome.REQUIRE_APPROVAL


class TestBlock:
    @pytest.mark.parametrize("risk", list(RiskTier))
    def test_clearly_unsupported_blocks_at_every_tier(self, risk: RiskTier) -> None:
        assert decide(risk, [chk(0.02)]).outcome is Outcome.BLOCK

    def test_at_block_threshold_blocks(self) -> None:
        assert decide(RiskTier.LOW, [chk(0.05)]).outcome is Outcome.BLOCK

    def test_just_above_block_threshold_does_not_block(self) -> None:
        assert decide(RiskTier.LOW, [chk(0.06)]).outcome is Outcome.CLARIFY

    def test_block_beats_high_risk_approval(self) -> None:
        # Don't bother a human with a call the evidence says is nonsense.
        assert decide(RiskTier.HIGH, [chk(0.01)]).outcome is Outcome.BLOCK

    def test_premature_action_polarity_drives_block(self) -> None:
        # p_yes=0.97 on "must the agent ask first?" -> p_ok=0.03 -> block.
        d = decide(RiskTier.LOW, [CheckResult(kind=PREMATURE, p_yes=0.97)])
        assert d.outcome is Outcome.BLOCK

    def test_premature_action_polarity_drives_allow(self) -> None:
        d = decide(RiskTier.LOW, [CheckResult(kind=PREMATURE, p_yes=0.08)])
        assert d.outcome is Outcome.ALLOW  # p_ok = 0.92

    def test_block_reports_the_weakest_check(self) -> None:
        weak = chk(0.01, ARG, "amount")
        d = decide(RiskTier.LOW, [chk(0.9), weak])
        assert d.limiting_check == weak


class TestObservedOnTheLiveDecider:
    """Regression tests using real numbers from the live 15-scenario measurement. Provisional,
    like the thresholds themselves: they pin today's behaviour, not a promise about accuracy."""

    def test_confirmed_cancel_is_allowed(self) -> None:
        checks = [chk(0.82, ARG, "order_id"), chk(0.96, CONSENT)]
        assert decide(RiskTier.MEDIUM, checks).outcome is Outcome.ALLOW

    def test_cancel_after_a_status_question_needs_clarification(self) -> None:
        checks = [chk(0.80, ARG, "order_id"), chk(0.08, CONSENT)]
        assert decide(RiskTier.MEDIUM, checks).outcome is Outcome.CLARIFY

    def test_cancel_after_a_can_i_question_needs_clarification(self) -> None:
        # "Is it possible to cancel...?" scored 0.74 for consent; the bar of 0.80 stops it.
        checks = [chk(0.75, ARG, "order_id"), chk(0.74, CONSENT)]
        assert decide(RiskTier.MEDIUM, checks).outcome is Outcome.CLARIFY

    @pytest.mark.parametrize("p", [0.12, 0.31, 0.35])  # highest "bad" argument scores seen
    def test_invented_or_guessed_arguments_never_pass(self, p: float) -> None:
        for risk in (RiskTier.LOW, RiskTier.MEDIUM):
            checks = [chk(p, ARG, "amount"), chk(0.9)]
            assert decide(risk, checks).outcome is not Outcome.ALLOW

    @pytest.mark.parametrize("p", [0.78, 0.82, 0.87, 0.89])  # lowest "good" argument scores seen
    def test_genuinely_stated_arguments_pass_at_low_and_medium(self, p: float) -> None:
        for risk in (RiskTier.LOW, RiskTier.MEDIUM):
            assert decide(risk, [chk(p, ARG, "order_id"), chk(0.9)]).outcome is Outcome.ALLOW

    def test_good_read_only_lookup_is_allowed(self) -> None:
        checks = [chk(0.78, ARG, "order_id"), chk(0.83, CONSENT)]
        assert decide(RiskTier.LOW, checks).outcome is Outcome.ALLOW


class TestEvidenceReporting:
    def test_all_checks_are_kept_on_the_decision(self) -> None:
        checks = [chk(0.9), chk(0.9, ARG, "id")]
        assert decide(RiskTier.LOW, checks).checks == tuple(checks)

    def test_weakest_tie_goes_to_first(self) -> None:
        a, b = chk(0.5, ARG, "a"), chk(0.5, ARG, "b")
        assert weakest([a, b]) is a

    def test_decision_is_deterministic(self) -> None:
        checks = [chk(0.7), chk(0.9, ARG, "id")]
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


def kind_thresholds(**overrides: float) -> KindThresholds:
    base = {
        "argument_grounding": 0.5,
        "consent": 0.5,
        "premature_action": 0.5,
        "claim_grounding": 0.5,
    }
    return KindThresholds(**{**base, **overrides})


class TestCustomThresholds:
    def test_stricter_thresholds_change_outcome(self) -> None:
        strict = Thresholds(
            low=kind_thresholds(consent=0.99), medium=kind_thresholds(consent=0.995)
        )
        assert decide(RiskTier.LOW, [chk(0.95, CONSENT)], strict).outcome is Outcome.CLARIFY

    def test_thresholds_can_differ_per_kind(self) -> None:
        t = Thresholds(
            low=kind_thresholds(argument_grounding=0.2, consent=0.9),
            medium=kind_thresholds(argument_grounding=0.2, consent=0.9),
        )
        assert decide(RiskTier.LOW, [chk(0.3, ARG, "x")], t).outcome is Outcome.ALLOW
        assert decide(RiskTier.LOW, [chk(0.3, CONSENT)], t).outcome is Outcome.CLARIFY

    def test_block_below_zero_disables_blocking_of_nonzero_evidence(self) -> None:
        t = Thresholds(block_below=0.0)
        assert decide(RiskTier.LOW, [chk(0.01)], t).outcome is Outcome.CLARIFY

    @pytest.mark.parametrize(
        "make",
        [
            # block_below not below a low threshold
            lambda: Thresholds(low=kind_thresholds(consent=0.04), block_below=0.05),
            # a riskier tier that is looser than the safer one
            lambda: Thresholds(
                low=kind_thresholds(consent=0.9), medium=kind_thresholds(consent=0.8)
            ),
            # values outside [0, 1]
            lambda: kind_thresholds(consent=1.5),
            lambda: kind_thresholds(consent=-0.1),
            lambda: Thresholds(block_below=-0.1),
            lambda: Thresholds(block_below=1.1),
        ],
    )
    def test_invalid_thresholds_rejected(self, make: object) -> None:
        with pytest.raises(ValidationError):
            make()  # type: ignore[operator]
