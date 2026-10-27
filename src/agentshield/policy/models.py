"""Schema for policy files, plus the context and result types used by the engine.

A policy file declares the tools an agent may use (with a risk tier and tags) and a list of
rules. A rule *triggers* when its ``when`` clause matches a proposed call, and then applies its
``effect`` (``block`` or ``require_approval``). Rules describe what is **not** allowed, so a
call that matches no rule simply continues to the Decider checks.

Everything is validated when the file is loaded, so a typo fails at startup, not mid-incident.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agentshield.core.models import Decision, DecisionSource, Outcome, RiskTier

type Scalar = str | int | float | bool

CTX_PREFIX = "$ctx."
"""A string operand starting with this refers to a fact in :class:`PolicyContext`."""

UNKNOWN_TOOL_RULE_ID = "unknown-tool"


class _Strict(BaseModel):
    """Base for schema models: unknown keys are errors, so typos cannot silently disable a rule."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class Effect(StrEnum):
    BLOCK = "block"
    REQUIRE_APPROVAL = "require_approval"


class ToolSpec(_Strict):
    """Risk metadata for one tool."""

    risk: RiskTier
    tags: tuple[str, ...] = ()
    description: str = ""


class ArgCondition(_Strict):
    """A test on one argument. Exactly one operator must be set.

    Comparison operators (``eq`` ``ne`` ``gt`` ``gte`` ``lt`` ``lte``) accept a literal or a
    ``$ctx.<fact>`` reference. Strings compare case-insensitively and ignoring surrounding
    whitespace; numeric strings such as ``"250"`` compare as numbers.
    """

    eq: Scalar | None = None
    ne: Scalar | None = None
    gt: Scalar | None = None
    gte: Scalar | None = None
    lt: Scalar | None = None
    lte: Scalar | None = None
    in_: tuple[Scalar, ...] | None = Field(default=None, alias="in")
    not_in: tuple[Scalar, ...] | None = None
    matches: str | None = None
    not_matches: str | None = None
    email_domain_in: tuple[str, ...] | None = None
    email_domain_not_in: tuple[str, ...] | None = None
    exists: bool | None = None

    @model_validator(mode="after")
    def _exactly_one_operator(self) -> ArgCondition:
        names = [n for n in _OPERATORS if getattr(self, n) is not None]
        if len(names) != 1:
            found = ", ".join(n.rstrip("_") for n in names) or "none"
            raise ValueError(f"exactly one operator is required per condition (found: {found})")
        for pattern in (self.matches, self.not_matches):
            if pattern is not None:
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        for name in ("in_", "not_in", "email_domain_in", "email_domain_not_in"):
            value = getattr(self, name)
            if value is not None and len(value) == 0:
                raise ValueError(f"{name.rstrip('_')} must not be an empty list")
        return self

    def operator(self) -> tuple[str, Any]:
        """The single operator that is set, as ``(name, operand)``."""
        for name in _OPERATORS:
            value = getattr(self, name)
            if value is not None:
                return name.rstrip("_"), value
        raise AssertionError("unreachable: validated to have exactly one operator")


_OPERATORS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "in_",
    "not_in",
    "matches",
    "not_matches",
    "email_domain_in",
    "email_domain_not_in",
    "exists",
)


class CountCondition(_Strict):
    """A test on how many times the same tool already ran in this session."""

    eq: int | None = None
    gt: int | None = None
    gte: int | None = None
    lt: int | None = None
    lte: int | None = None

    @model_validator(mode="after")
    def _at_least_one(self) -> CountCondition:
        if all(getattr(self, n) is None for n in ("eq", "gt", "gte", "lt", "lte")):
            raise ValueError("prior_calls needs at least one of: eq, gt, gte, lt, lte")
        return self

    def holds(self, count: int) -> bool:
        checks = (
            (self.eq, lambda n, v: n == v),
            (self.gt, lambda n, v: n > v),
            (self.gte, lambda n, v: n >= v),
            (self.lt, lambda n, v: n < v),
            (self.lte, lambda n, v: n <= v),
        )
        return all(limit is None or test(count, limit) for limit, test in checks)


class When(_Strict):
    """Which calls a rule applies to. All given parts must hold (AND).

    ``tool`` is a tool name, a list of names, or ``"*"`` for any tool. ``tags`` matches a tool
    that has at least one of the listed tags. ``facts`` tests session facts the same way ``args``
    tests arguments, except a *missing* fact does not trigger (the agent may not have looked the
    order up yet). ``$ctx`` references on arguments still fail-closed when missing. Use several
    rules for OR logic.
    """

    tool: tuple[str, ...] | None = None
    tags: tuple[str, ...] = ()
    args: dict[str, ArgCondition] = Field(default_factory=dict)
    facts: dict[str, ArgCondition] = Field(default_factory=dict)
    prior_calls: CountCondition | None = None

    @field_validator("tool", mode="before")
    @classmethod
    def _single_name_to_tuple(cls, value: Any) -> Any:
        return (value,) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _needs_a_selector(self) -> When:
        if not self.tool and not self.tags:
            raise ValueError('"when" needs a tool selector: set tool (or "*") and/or tags')
        return self


class Rule(_Strict):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    description: str = ""
    effect: Effect
    when: When


class Policy(_Strict):
    version: Literal[1]
    name: str = Field(min_length=1)
    default_for_unknown_tools: Literal["block", "require_approval", "allow"] = "block"
    tools: dict[str, ToolSpec] = Field(default_factory=dict)
    rules: tuple[Rule, ...] = ()

    @model_validator(mode="after")
    def _cross_references(self) -> Policy:
        seen: set[str] = set()
        known_tags = {tag for spec in self.tools.values() for tag in spec.tags}
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id {rule.id!r}")
            seen.add(rule.id)
            if rule.id == UNKNOWN_TOOL_RULE_ID:
                raise ValueError(f"rule id {UNKNOWN_TOOL_RULE_ID!r} is reserved")
            for name in rule.when.tool or ():
                if name != "*" and name not in self.tools:
                    raise ValueError(f"rule {rule.id!r} refers to unknown tool {name!r}")
            for tag in rule.when.tags:
                if tag not in known_tags:
                    raise ValueError(f"rule {rule.id!r} refers to tag {tag!r} that no tool has")
        return self


# --- Runtime types -------------------------------------------------------------------------


class PolicyContext(BaseModel):
    """Facts the engine may consult, supplied by the caller.

    ``facts`` holds values learned earlier in the conversation (for example ``order_total`` after
    an order lookup). ``prior_calls`` counts how many times each tool already ran this session.
    The engine never mutates this; see :class:`~agentshield.policy.session.SessionState`.
    """

    model_config = ConfigDict(frozen=True)

    facts: dict[str, Any] = Field(default_factory=dict)
    prior_calls: dict[str, int] = Field(default_factory=dict)


class RuleMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    rule_id: str
    effect: Effect
    description: str
    detail: str = ""


class PolicyResult(BaseModel):
    """Everything the engine found out about one proposed call."""

    model_config = ConfigDict(frozen=True)

    tool: str
    risk: RiskTier
    known_tool: bool
    matches: tuple[RuleMatch, ...] = ()

    @property
    def decision(self) -> Decision | None:
        """The policy's verdict, or None if no rule triggered (continue to the Decider)."""
        if not self.matches:
            return None
        # A block always wins over a required approval; otherwise the earliest rule wins.
        chosen = next(
            (m for m in self.matches if m.effect is Effect.BLOCK),
            self.matches[0],
        )
        outcome = Outcome.BLOCK if chosen.effect is Effect.BLOCK else Outcome.REQUIRE_APPROVAL
        reason = f"Policy rule '{chosen.rule_id}': {chosen.description or 'rule triggered'}"
        if chosen.detail:
            reason += f" [{chosen.detail}]"
        others = [m.rule_id for m in self.matches if m is not chosen]
        if others:
            reason += f" (also triggered: {', '.join(others)})"
        return Decision(
            outcome=outcome,
            risk=self.risk,
            source=DecisionSource.POLICY,
            reason=reason,
            rule_id=chosen.rule_id,
        )
