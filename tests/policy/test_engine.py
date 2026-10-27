from typing import Any

import pytest

from agentshield.core import DecisionSource, Outcome, RiskTier, ToolCall
from agentshield.policy import Effect, Policy, PolicyContext, PolicyEngine, SessionState


def make(rules: list[dict[str, Any]], default: str = "block") -> PolicyEngine:
    policy = Policy.model_validate(
        {
            "version": 1,
            "name": "test",
            "default_for_unknown_tools": default,
            "tools": {
                "look": {"risk": "low", "tags": ["read_only"]},
                "mail": {"risk": "medium", "tags": ["outbound"]},
                "pay": {"risk": "high", "tags": ["financial", "irreversible"]},
                "wipe": {"risk": "high", "tags": ["destructive"]},
            },
            "rules": rules,
        }
    )
    return PolicyEngine(policy)


def rule(
    rid: str = "r",
    effect: str = "block",
    tool: str | list[str] | None = "pay",
    **when: Any,
) -> dict[str, Any]:
    w: dict[str, Any] = dict(when)
    if tool is not None:
        w["tool"] = tool
    return {"id": rid, "effect": effect, "description": f"desc {rid}", "when": w}


def call(name: str = "pay", **args: Any) -> ToolCall:
    return ToolCall(name=name, arguments=args)


def triggered(engine: PolicyEngine, c: ToolCall, ctx: PolicyContext | None = None) -> bool:
    return bool(engine.evaluate(c, ctx).matches)


class TestSelection:
    def test_no_rules_no_match_and_no_decision(self) -> None:
        result = make([]).evaluate(call("look"))
        assert result.matches == ()
        assert result.decision is None
        assert result.known_tool

    def test_tool_name_selects(self) -> None:
        e = make([rule(tool="pay")])
        assert triggered(e, call("pay"))
        assert not triggered(e, call("look"))

    def test_tool_list(self) -> None:
        e = make([rule(tool=["pay", "mail"])])
        assert triggered(e, call("pay"))
        assert triggered(e, call("mail"))
        assert not triggered(e, call("look"))

    def test_wildcard(self) -> None:
        e = make([rule(tool="*")])
        assert all(triggered(e, call(n)) for n in ("look", "mail", "pay", "wipe"))

    def test_tags_select_any_match(self) -> None:
        e = make([rule(tool=None, tags=["destructive", "outbound"])])
        assert triggered(e, call("wipe"))
        assert triggered(e, call("mail"))
        assert not triggered(e, call("pay"))

    def test_tool_and_tags_must_both_hold(self) -> None:
        e = make([rule(tool="pay", tags=["destructive"])])
        assert not triggered(e, call("pay"))


class TestNumericComparisons:
    @pytest.mark.parametrize(
        ("op", "limit", "amount", "expected"),
        [
            ("gt", 200, 201, True),
            ("gt", 200, 200, False),
            ("gte", 200, 200, True),
            ("lt", 10, 9, True),
            ("lt", 10, 10, False),
            ("lte", 10, 10, True),
        ],
    )
    def test_operators(self, op: str, limit: int, amount: int, expected: bool) -> None:
        e = make([rule(args={"amount": {op: limit}})])
        assert triggered(e, call(amount=amount)) is expected

    def test_numeric_strings_are_treated_as_numbers(self) -> None:
        # LLMs often pass "250" instead of 250; that must not slip past a limit.
        e = make([rule(args={"amount": {"gt": 200}})])
        assert triggered(e, call(amount="250"))
        assert triggered(e, call(amount=" 250.50 "))
        assert not triggered(e, call(amount="150"))

    def test_decimal_precision(self) -> None:
        e = make([rule(args={"amount": {"gt": 0.3}})])
        assert not triggered(e, call(amount=0.3))
        assert not triggered(e, call(amount="0.30"))
        assert triggered(e, call(amount="0.30001"))

    @pytest.mark.parametrize("bad", ["abc", "NaN", "Infinity", "-inf", None, True, [1], {"a": 1}])
    def test_non_numeric_value_triggers_fail_closed(self, bad: Any) -> None:
        e = make([rule(args={"amount": {"gt": 200}})])
        result = e.evaluate(call(amount=bad))
        assert result.matches, f"{bad!r} should trigger"
        assert "fail-closed" in result.matches[0].detail

    def test_absent_argument_does_not_trigger_comparison(self) -> None:
        e = make([rule(args={"amount": {"gt": 200}})])
        assert not triggered(e, call())


class TestEqualityAndMembership:
    def test_eq_and_ne(self) -> None:
        eq = make([rule(args={"env": {"eq": "production"}})])
        ne = make([rule(args={"env": {"ne": "staging"}})])
        assert triggered(eq, call(env="production"))
        assert not triggered(eq, call(env="staging"))
        assert triggered(ne, call(env="production"))
        assert not triggered(ne, call(env="staging"))

    @pytest.mark.parametrize("value", ["PRODUCTION", "Production", "  production  "])
    def test_string_comparison_ignores_case_and_whitespace(self, value: str) -> None:
        # Otherwise "PRODUCTION" would bypass a rule written for "production".
        e = make([rule(args={"env": {"eq": "production"}})])
        assert triggered(e, call(env=value))

    def test_numeric_equality_across_types(self) -> None:
        e = make([rule(args={"n": {"eq": 5}})])
        assert triggered(e, call(n="5"))
        assert triggered(e, call(n=5.0))

    def test_in_and_not_in(self) -> None:
        deny = make([rule(args={"table": {"in": ["users", "payments"]}})])
        allow = make([rule(args={"table": {"not_in": ["notes"]}})])
        assert triggered(deny, call(table="Users"))
        assert not triggered(deny, call(table="notes"))
        assert triggered(allow, call(table="users"))
        assert not triggered(allow, call(table="NOTES"))

    def test_exists(self) -> None:
        must_have = make([rule(args={"reason": {"exists": False}})])
        forbidden = make([rule(args={"force": {"exists": True}})])
        assert triggered(must_have, call())
        assert not triggered(must_have, call(reason="x"))
        assert triggered(forbidden, call(force=False))
        assert not triggered(forbidden, call())


class TestPatterns:
    def test_matches_and_not_matches(self) -> None:
        deny = make([rule(args={"path": {"matches": r"^/etc/|\.\./"}})])
        allow = make([rule(args={"path": {"not_matches": r"^/tmp/"}})])
        assert triggered(deny, call(path="/etc/passwd"))
        assert triggered(deny, call(path="/tmp/../etc/passwd"))
        assert not triggered(deny, call(path="/tmp/ok.txt"))
        assert triggered(allow, call(path="/home/x"))
        assert not triggered(allow, call(path="/tmp/x"))

    def test_matches_non_string_values_via_str(self) -> None:
        e = make([rule(args={"id": {"matches": r"^4\d+$"}})])
        assert triggered(e, call(id=4417))

    def test_email_domain_allow_list_blocks_other_domains(self) -> None:
        e = make([rule(tool="mail", args={"to": {"email_domain_not_in": ["shop.com"]}})])
        assert not triggered(e, call("mail", to="bob@shop.com"))
        assert not triggered(e, call("mail", to="Bob@SHOP.COM"))
        assert triggered(e, call("mail", to="bob@evil.com"))
        assert triggered(e, call("mail", to="bob@shop.com.evil.com"))

    @pytest.mark.parametrize("bad", ["not-an-email", "bob@", 42, None])
    def test_email_allow_list_fails_closed_on_unverifiable_values(self, bad: Any) -> None:
        e = make([rule(tool="mail", args={"to": {"email_domain_not_in": ["shop.com"]}})])
        assert triggered(e, call("mail", to=bad))

    def test_email_domain_deny_list(self) -> None:
        e = make([rule(tool="mail", args={"to": {"email_domain_in": ["competitor.com"]}})])
        assert triggered(e, call("mail", to="x@Competitor.com"))
        assert not triggered(e, call("mail", to="x@shop.com"))
        assert not triggered(e, call("mail", to="not-an-email"))


class TestContextFacts:
    def test_reference_is_resolved(self) -> None:
        e = make([rule(args={"amount": {"gt": "$ctx.order_total"}})])
        ctx = PolicyContext(facts={"order_total": 89.99})
        assert triggered(e, call(amount=100), ctx)
        assert not triggered(e, call(amount=89.99), ctx)

    def test_missing_fact_triggers_fail_closed(self) -> None:
        e = make([rule(args={"amount": {"gt": "$ctx.order_total"}})])
        result = e.evaluate(call(amount=1), PolicyContext())
        assert result.matches
        assert "order_total" in result.matches[0].detail

    def test_missing_fact_with_no_context_at_all(self) -> None:
        e = make([rule(args={"to": {"ne": "$ctx.customer_email"}})])
        assert triggered(e, call(to="a@b.com"))

    def test_string_fact_comparison_is_case_insensitive(self) -> None:
        e = make([rule(args={"to": {"ne": "$ctx.customer_email"}})])
        ctx = PolicyContext(facts={"customer_email": "Ann@Shop.com"})
        assert not triggered(e, call(to="ann@shop.com"), ctx)
        assert triggered(e, call(to="eve@shop.com"), ctx)


class TestSessionFacts:
    def test_in_on_a_known_fact(self) -> None:
        e = make([rule(tool="pay", facts={"order_status": {"in": ["delivered", "cancelled"]}})])
        assert triggered(e, call(), PolicyContext(facts={"order_status": "delivered"}))
        assert not triggered(e, call(), PolicyContext(facts={"order_status": "processing"}))

    def test_missing_fact_does_not_trigger(self) -> None:
        e = make([rule(tool="pay", facts={"order_status": {"eq": "delivered"}})])
        assert not triggered(e, call(), PolicyContext())

    def test_boolean_fact(self) -> None:
        e = make([rule(tool="pay", facts={"within_return_window": {"eq": False}})])
        assert triggered(e, call(), PolicyContext(facts={"within_return_window": False}))
        assert not triggered(e, call(), PolicyContext(facts={"within_return_window": True}))


class TestAllConditionsMustHold:
    def test_args_are_anded(self) -> None:
        e = make([rule(args={"amount": {"gt": 100}, "currency": {"eq": "usd"}})])
        assert triggered(e, call(amount=150, currency="USD"))
        assert not triggered(e, call(amount=150, currency="eur"))
        assert not triggered(e, call(amount=50, currency="usd"))

    def test_prior_calls(self) -> None:
        e = make([rule(prior_calls={"gte": 3})])
        assert not triggered(e, call(), PolicyContext(prior_calls={"pay": 2}))
        assert triggered(e, call(), PolicyContext(prior_calls={"pay": 3}))

    def test_prior_calls_only_count_the_same_tool(self) -> None:
        e = make([rule(prior_calls={"gte": 1})])
        assert not triggered(e, call("pay"), PolicyContext(prior_calls={"mail": 9}))

    def test_prior_calls_and_args_combined(self) -> None:
        e = make([rule(prior_calls={"gte": 1}, args={"amount": {"gt": 10}})])
        ctx = PolicyContext(prior_calls={"pay": 1})
        assert triggered(e, call(amount=11), ctx)
        assert not triggered(e, call(amount=5), ctx)

    def test_rule_with_no_conditions_triggers_on_selection_alone(self) -> None:
        assert triggered(make([rule()]), call())


class TestUnknownTools:
    @pytest.mark.parametrize(
        ("mode", "effect"),
        [("block", Effect.BLOCK), ("require_approval", Effect.REQUIRE_APPROVAL)],
    )
    def test_follow_the_configured_default(self, mode: str, effect: Effect) -> None:
        result = make([], default=mode).evaluate(call("mystery"))
        assert not result.known_tool
        assert result.matches[0].effect is effect
        assert result.matches[0].rule_id == "unknown-tool"

    def test_allow_mode_lets_them_through_but_as_high_risk(self) -> None:
        result = make([], default="allow").evaluate(call("mystery"))
        assert result.matches == ()
        assert result.risk is RiskTier.HIGH

    def test_unknown_tool_is_blocked_by_default_decision(self) -> None:
        decision = make([]).evaluate(call("mystery")).decision
        assert decision is not None
        assert decision.outcome is Outcome.BLOCK
        assert decision.risk is RiskTier.HIGH

    def test_wildcard_rules_still_apply_to_unknown_tools(self) -> None:
        e = make([rule(tool="*", effect="require_approval")], default="allow")
        assert triggered(e, call("mystery"))


class TestDecisionMapping:
    def test_block(self) -> None:
        d = make([rule(effect="block")]).evaluate(call()).decision
        assert d is not None
        assert d.outcome is Outcome.BLOCK
        assert d.source is DecisionSource.POLICY
        assert d.rule_id == "r"
        assert d.risk is RiskTier.HIGH
        assert "desc r" in d.reason

    def test_require_approval(self) -> None:
        d = make([rule(effect="require_approval")]).evaluate(call()).decision
        assert d is not None
        assert d.outcome is Outcome.REQUIRE_APPROVAL

    def test_block_beats_approval_regardless_of_order(self) -> None:
        approve_first = make(
            [rule("approve", "require_approval"), rule("deny", "block")],
        ).evaluate(call())
        d = approve_first.decision
        assert d is not None
        assert d.outcome is Outcome.BLOCK
        assert d.rule_id == "deny"
        assert "approve" in d.reason  # also-triggered rules are reported

    def test_earliest_rule_wins_among_equals(self) -> None:
        d = make([rule("first"), rule("second")]).evaluate(call()).decision
        assert d is not None
        assert d.rule_id == "first"
        assert "second" in d.reason

    def test_detail_is_included_and_truncated(self) -> None:
        e = make([rule(args={"amount": {"gt": 200}})])
        d = e.evaluate(call(amount=500)).decision
        assert d is not None
        assert "amount=500" in d.reason

        long = make([rule(args={"note": {"matches": "x"}})]).evaluate(call(note="x" * 500))
        assert long.decision is not None
        assert len(long.decision.reason) < 250

    def test_risk_comes_from_the_tool_registry(self) -> None:
        e = make([rule("a", tool="look"), rule("b", tool="mail")])
        assert e.evaluate(call("look")).risk is RiskTier.LOW
        assert e.evaluate(call("mail")).risk is RiskTier.MEDIUM
        assert e.risk_for("pay") is RiskTier.HIGH
        assert e.risk_for("mystery") is RiskTier.HIGH


class TestDeterminism:
    def test_same_input_same_result(self) -> None:
        e = make([rule(args={"amount": {"gt": 200}}), rule("b", "require_approval")])
        ctx = PolicyContext(prior_calls={"pay": 1})
        assert e.evaluate(call(amount=500), ctx) == e.evaluate(call(amount=500), ctx)

    def test_evaluation_does_not_mutate_context(self) -> None:
        ctx = PolicyContext(facts={"a": 1}, prior_calls={"pay": 1})
        before = ctx.model_copy(deep=True)
        make([rule(prior_calls={"gte": 1})]).evaluate(call(), ctx)
        assert ctx == before


class TestSessionState:
    def test_records_executed_calls_into_the_context(self) -> None:
        s = SessionState("s1", facts={"customer_email": "a@b.com"})
        s.record_call("pay")
        s.record_call("pay")
        s.set_fact("order_total", 10)
        ctx = s.context()
        assert ctx.prior_calls == {"pay": 2}
        assert ctx.facts == {"customer_email": "a@b.com", "order_total": 10}

    def test_context_is_a_snapshot(self) -> None:
        s = SessionState("s1")
        ctx = s.context()
        s.record_call("pay")
        assert ctx.prior_calls == {}
        assert s.context().prior_calls == {"pay": 1}

    def test_drives_a_quota_rule_end_to_end(self) -> None:
        e = make([rule(prior_calls={"gte": 2})])
        s = SessionState("s1")
        assert not triggered(e, call(), s.context())
        s.record_call("pay")
        s.record_call("pay")
        assert triggered(e, call(), s.context())
