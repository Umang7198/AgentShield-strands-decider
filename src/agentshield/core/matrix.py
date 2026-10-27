"""The decision matrix: risk tier x Decider evidence -> outcome.

Pure function, no I/O. The deterministic policy engine runs *before* this and can short-circuit
the whole pipeline; this module only handles what the policy engine lets through.

Rules, evaluated in order:

1. Any check with ``p_ok <= block_below``  -> BLOCK (evidence says the call is clearly unsupported).
2. Risk is HIGH                            -> REQUIRE_APPROVAL (a human always signs off).
3. Weakest ``p_ok`` >= the tier's ALLOW threshold -> ALLOW.
4. Otherwise                               -> CLARIFY.

The default numbers are illustrative starting points (SCOPE.md 5.5). They are meant to be
re-calibrated on the evaluation set, so they live in :class:`Thresholds` and are configurable.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agentshield.core.models import (
    CheckResult,
    Decision,
    DecisionSource,
    Outcome,
    RiskTier,
)


class Thresholds(BaseModel):
    """Per-tier confidence thresholds. Validated so impossible orderings are rejected."""

    model_config = ConfigDict(frozen=True)

    allow_low: float = Field(default=0.85, ge=0.0, le=1.0)
    allow_medium: float = Field(default=0.92, ge=0.0, le=1.0)
    block_below: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_ordering(self) -> Thresholds:
        if not self.block_below < self.allow_low:
            raise ValueError("block_below must be strictly less than allow_low")
        if self.allow_low > self.allow_medium:
            raise ValueError("allow_low must not exceed allow_medium (riskier tiers need more)")
        return self

    def allow_for(self, risk: RiskTier) -> float | None:
        """Minimum ``p_ok`` to ALLOW at this tier, or None if the tier never auto-allows."""
        if risk is RiskTier.LOW:
            return self.allow_low
        if risk is RiskTier.MEDIUM:
            return self.allow_medium
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
    """
    if not checks:
        raise ValueError("decide() needs at least one check; use fallback_decision() if none ran")

    t = thresholds or Thresholds()
    low = weakest(checks)
    detail = f"{low.label} p_ok={low.p_ok:.2f}"

    def make(outcome: Outcome, reason: str) -> Decision:
        return Decision(
            outcome=outcome,
            risk=risk,
            source=DecisionSource.DECIDER,
            reason=reason,
            checks=tuple(checks),
            limiting_check=low,
        )

    if low.p_ok <= t.block_below:
        return make(
            Outcome.BLOCK,
            f"Evidence says the call is unsupported ({detail} <= {t.block_below:.2f})",
        )

    allow_at = t.allow_for(risk)
    if allow_at is None:
        return make(
            Outcome.REQUIRE_APPROVAL,
            f"High-risk action always needs human approval (weakest: {detail})",
        )

    if low.p_ok >= allow_at:
        return make(
            Outcome.ALLOW,
            f"All checks satisfied for {risk.value} risk (weakest: {detail})",
        )

    return make(
        Outcome.CLARIFY,
        f"Not enough evidence for {risk.value} risk: {detail} < {allow_at:.2f}",
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
