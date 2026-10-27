"""The decision matrix: risk tier x Decider evidence -> outcome.

Pure function, no I/O. The deterministic policy engine runs *before* this and can short-circuit
the whole pipeline; this module only handles what the policy engine lets through.

Rules, evaluated in order:

1. Any check with ``p_ok <= block_below``  -> BLOCK (evidence says the call is clearly unsupported).
2. Risk is HIGH                            -> REQUIRE_APPROVAL (a human always signs off).
3. Every check meets *its own* threshold for this risk tier -> ALLOW.
4. Otherwise                               -> CLARIFY.

Thresholds are per risk tier **and per check kind**, because the checks are not equally reliable:
measured on the live Decider, argument grounding separates good from bad calls cleanly, while
consent overlaps much more, so consent needs a stricter bar. The defaults below are provisional
starting points taken from a 15-scenario measurement; they are meant to be re-calibrated on the
evaluation set, which is why they live in :class:`Thresholds` and are configurable.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentshield.core.models import (
    CheckKind,
    CheckResult,
    Decision,
    DecisionSource,
    Outcome,
    RiskTier,
)


class KindThresholds(BaseModel):
    """Minimum ``p_ok`` for each kind of check to allow a call at one risk tier."""

    model_config = ConfigDict(frozen=True)

    argument_grounding: float = Field(ge=0.0, le=1.0)
    consent: float = Field(ge=0.0, le=1.0)
    premature_action: float = Field(ge=0.0, le=1.0)
    claim_grounding: float = Field(ge=0.0, le=1.0)

    def for_kind(self, kind: CheckKind) -> float:
        return float(getattr(self, kind.value))


def _low_defaults() -> KindThresholds:
    return KindThresholds(
        argument_grounding=0.55, consent=0.60, premature_action=0.50, claim_grounding=0.60
    )


def _medium_defaults() -> KindThresholds:
    return KindThresholds(
        argument_grounding=0.65, consent=0.80, premature_action=0.60, claim_grounding=0.75
    )


class Thresholds(BaseModel):
    """Per-tier, per-kind thresholds. Validated so impossible orderings are rejected.

    There is no HIGH tier entry: high-risk calls never auto-allow.
    """

    model_config = ConfigDict(frozen=True)

    low: KindThresholds = Field(default_factory=_low_defaults)
    medium: KindThresholds = Field(default_factory=_medium_defaults)
    block_below: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Thresholds:
        for kind in CheckKind:
            lo, med = self.low.for_kind(kind), self.medium.for_kind(kind)
            if not self.block_below < lo:
                raise ValueError(f"block_below must be below the low threshold for {kind.value}")
            if lo > med:
                raise ValueError(
                    f"low threshold for {kind.value} must not exceed the medium one "
                    "(riskier tiers need more evidence)"
                )
        return self

    def allow_for(self, risk: RiskTier, kind: CheckKind) -> float | None:
        """Minimum ``p_ok`` to allow this kind of check at this tier, or None if the tier never
        auto-allows."""
        if risk is RiskTier.LOW:
            return self.low.for_kind(kind)
        if risk is RiskTier.MEDIUM:
            return self.medium.for_kind(kind)
        return None


def weakest(checks: Sequence[CheckResult]) -> CheckResult:
    """The check least likely to be satisfied. Ties go to the earliest."""
    return min(checks, key=lambda c: c.p_ok)


def decide(
    risk: RiskTier,
    checks: Sequence[CheckResult],
    thresholds: Thresholds | None = None,
) -> Decision:
    """Turn Decider evidence into a decision.

    ``checks`` must not be empty: a call with no evidence is a caller error. If the Decider is
    unavailable, use :func:`fallback_decision` instead.

    ``Decision.limiting_check`` is the check that decided the outcome: the weakest one for BLOCK
    and REQUIRE_APPROVAL, the one furthest below its threshold for CLARIFY, and the one closest to
    its threshold for ALLOW.
    """
    if not checks:
        raise ValueError("decide() needs at least one check; use fallback_decision() if none ran")

    t = thresholds or Thresholds()

    def make(outcome: Outcome, reason: str, limiting: CheckResult) -> Decision:
        return Decision(
            outcome=outcome,
            risk=risk,
            source=DecisionSource.DECIDER,
            reason=reason,
            checks=tuple(checks),
            limiting_check=limiting,
        )

    lowest = weakest(checks)
    lowest_detail = f"{lowest.label} p_ok={lowest.p_ok:.2f}"

    if lowest.p_ok <= t.block_below:
        return make(
            Outcome.BLOCK,
            f"Evidence says the call is unsupported ({lowest_detail} <= {t.block_below:.2f})",
            lowest,
        )

    if risk is RiskTier.HIGH:
        return make(
            Outcome.REQUIRE_APPROVAL,
            f"High-risk action always needs human approval (weakest: {lowest_detail})",
            lowest,
        )

    def need(check: CheckResult) -> float:
        threshold = t.allow_for(risk, check.kind)
        assert threshold is not None  # HIGH was handled above
        return threshold

    def margin(check: CheckResult) -> float:
        return check.p_ok - need(check)

    failing = [c for c in checks if c.p_ok < need(c)]
    if failing:
        worst = min(failing, key=margin)
        return make(
            Outcome.CLARIFY,
            f"Not enough evidence for {risk.value} risk: {worst.label} "
            f"p_ok={worst.p_ok:.2f} < needed {need(worst):.2f}",
            worst,
        )

    tightest = min(checks, key=margin)
    return make(
        Outcome.ALLOW,
        f"All checks satisfied for {risk.value} risk "
        f"(tightest: {tightest.label} p_ok={tightest.p_ok:.2f} >= {need(tightest):.2f})",
        tightest,
    )


def fallback_decision(risk: RiskTier, cause: str) -> Decision:
    """Decision to use when the Decider could not be consulted (timeout, error, bad response).

    Fails closed: LOW risk asks the user, anything riskier goes to a human.
    """
    outcome = Outcome.CLARIFY if risk is RiskTier.LOW else Outcome.REQUIRE_APPROVAL
    return Decision(
        outcome=outcome,
        risk=risk,
        source=DecisionSource.FALLBACK,
        reason=f"Decider unavailable ({cause}); failing closed for {risk.value} risk",
    )
