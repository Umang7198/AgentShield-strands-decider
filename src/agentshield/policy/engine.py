"""The deterministic policy engine.

Pure and stateless: the same call, policy and context always give the same result, and no model
is ever consulted. State such as "how many refunds already happened" is passed in through
:class:`~agentshield.policy.models.PolicyContext`.

Fail-closed conventions (a rule *triggers* when it cannot be sure the call is fine):

* A ``$ctx.<fact>`` reference to a fact that is not available triggers the rule.
* A numeric comparison where either side is not a finite number triggers the rule.
* ``not_in`` / ``email_domain_not_in`` (allow-lists) trigger on values that cannot be checked.
* An argument that is **absent** never triggers a comparison; use ``exists`` to require one.
* An unknown tool follows ``default_for_unknown_tools`` (``block`` by default).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from agentshield.core.models import RiskTier, ToolCall
from agentshield.policy.models import (
    CTX_PREFIX,
    UNKNOWN_TOOL_RULE_ID,
    ArgCondition,
    Effect,
    Policy,
    PolicyContext,
    PolicyResult,
    Rule,
    RuleMatch,
    ToolSpec,
    When,
)

_NUMERIC_OPS = {
    "gt": lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
}


class PolicyEngine:
    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    @property
    def policy(self) -> Policy:
        return self._policy

    def risk_for(self, tool_name: str) -> RiskTier:
        """Risk tier of a tool. Unknown tools are treated as HIGH."""
        spec = self._policy.tools.get(tool_name)
        return spec.risk if spec else RiskTier.HIGH

    def evaluate(self, call: ToolCall, ctx: PolicyContext | None = None) -> PolicyResult:
        ctx = ctx or PolicyContext()
        spec = self._policy.tools.get(call.name)
        matches: list[RuleMatch] = []

        if spec is None and self._policy.default_for_unknown_tools != "allow":
            matches.append(
                RuleMatch(
                    rule_id=UNKNOWN_TOOL_RULE_ID,
                    effect=Effect(self._policy.default_for_unknown_tools),
                    description=f"Tool '{call.name}' is not registered in the policy",
                )
            )

        for rule in self._policy.rules:
            if not _selects(rule.when, call.name, spec):
                continue
            detail = _conditions_detail(rule.when, call, ctx)
            if detail is not None:
                matches.append(_match(rule, detail))

        return PolicyResult(
            tool=call.name,
            risk=spec.risk if spec else RiskTier.HIGH,
            known_tool=spec is not None,
            matches=tuple(matches),
        )


def _match(rule: Rule, detail: str) -> RuleMatch:
    return RuleMatch(
        rule_id=rule.id,
        effect=rule.effect,
        description=rule.description,
        detail=detail,
    )


def _selects(when: When, tool_name: str, spec: ToolSpec | None) -> bool:
    if when.tool and "*" not in when.tool and tool_name not in when.tool:
        return False
    if when.tags and not (spec and any(tag in spec.tags for tag in when.tags)):
        return False
    return True


def _conditions_detail(when: When, call: ToolCall, ctx: PolicyContext) -> str | None:
    """Return a description of why the rule triggered, or None if it did not.

    All conditions must hold for the rule to trigger.
    """
    parts: list[str] = []
    for arg_name, cond in when.args.items():
        detail = _eval_arg(cond, arg_name, call.arguments, ctx)
        if detail is None:
            return None
        parts.append(detail)
    for fact_name, cond in when.facts.items():
        detail = _eval_fact(cond, fact_name, ctx)
        if detail is None:
            return None
        parts.append(detail)
    if when.prior_calls is not None:
        count = ctx.prior_calls.get(call.name, 0)
        if not when.prior_calls.holds(count):
            return None
        parts.append(f"{count} prior call(s) this session")
    return "; ".join(parts)


def _eval_fact(cond: ArgCondition, name: str, ctx: PolicyContext) -> str | None:
    """Test a session fact. Absent facts do **not** fail-closed (unlike ``$ctx`` on arguments)."""
    if name not in ctx.facts:
        return None
    return _eval_arg(cond, name, {name: ctx.facts[name]}, ctx)


def _eval_arg(
    cond: ArgCondition, arg: str, arguments: Mapping[str, Any], ctx: PolicyContext
) -> str | None:
    op, operand = cond.operator()
    present = arg in arguments

    if op == "exists":
        return f"{arg} {'present' if present else 'absent'}" if present == operand else None
    if not present:
        return None
    value = arguments[arg]

    if op in ("in", "not_in"):
        member = any(_equal(value, item) for item in operand)
        if (op == "in") == member:
            return f"{arg}={_short(value)} {op.replace('_', ' ')} {_short(list(operand))}"
        return None

    if op in ("matches", "not_matches"):
        found = re.search(operand, str(value)) is not None
        if (op == "matches") == found:
            return f"{arg}={_short(value)} {op.replace('_', ' ')} {operand!r}"
        return None

    if op in ("email_domain_in", "email_domain_not_in"):
        domain = _email_domain(value)
        allowed = {d.strip().casefold() for d in operand}
        if op == "email_domain_in":
            hit = domain in allowed
        else:  # allow-list: anything we cannot verify counts as a violation
            hit = domain is None or domain not in allowed
        if hit:
            return f"{arg}={_short(value)} domain {'not in' if 'not' in op else 'in'} allow-list"
        return None

    # Scalar comparison, possibly against a context fact.
    resolved, missing = _resolve(operand, ctx)
    if missing is not None:
        return f"{arg} compared to unavailable fact '{missing}' (fail-closed)"

    if op in ("eq", "ne"):
        same = _equal(value, resolved)
        return f"{arg}={_short(value)} {op} {_short(resolved)}" if same == (op == "eq") else None

    left, right = _to_decimal(value), _to_decimal(resolved)
    if left is None or right is None:
        return f"{arg}={_short(value)} is not comparable as a number (fail-closed)"
    if _NUMERIC_OPS[op](left, right):
        return f"{arg}={_short(value)} {op} {_short(resolved)}"
    return None


def _resolve(operand: Any, ctx: PolicyContext) -> tuple[Any, str | None]:
    """Resolve ``$ctx.<fact>`` references. Returns ``(value, missing_fact_name)``."""
    if isinstance(operand, str) and operand.startswith(CTX_PREFIX):
        key = operand[len(CTX_PREFIX) :]
        if key not in ctx.facts:
            return None, key
        return ctx.facts[key], None
    return operand, None


def _to_decimal(value: Any) -> Decimal | None:
    """Finite number from an int, float or numeric string; None for anything else (incl. bool)."""
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def _norm(value: Any) -> Any:
    return value.strip().casefold() if isinstance(value, str) else value


def _equal(a: Any, b: Any) -> bool:
    da, db = _to_decimal(a), _to_decimal(b)
    if da is not None and db is not None:
        return da == db
    return bool(_norm(a) == _norm(b))


def _email_domain(value: Any) -> str | None:
    if not isinstance(value, str) or "@" not in value:
        return None
    domain = value.rsplit("@", 1)[1].strip().casefold()
    return domain or None


def _short(value: Any, limit: int = 60) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
