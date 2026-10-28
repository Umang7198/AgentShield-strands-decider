from pathlib import Path

import pytest

from agentshield.core import RiskTier
from agentshield.policy import PolicyError, load_policy, parse_policy

POLICY_DIR = Path(__file__).resolve().parents[2] / "policies"

HEADER = """
version: 1
name: t
tools:
  pay: {risk: high, tags: [financial]}
  look: {risk: low}
"""


def wrap(rules: str, header: str = HEADER) -> str:
    return header + "rules:\n" + rules


class TestShopHelpFile:
    def test_loads(self) -> None:
        policy = load_policy(POLICY_DIR / "shophelp.yaml")
        assert policy.name == "shophelp"
        assert policy.default_for_unknown_tools == "block"
        assert policy.tools["issue_refund"].risk is RiskTier.HIGH
        assert policy.tools["get_order_status"].risk is RiskTier.LOW
        assert policy.tools["lookup_handbook"].risk is RiskTier.LOW
        assert len(policy.rules) == 10

    def test_rule_ids_are_unique(self) -> None:
        policy = load_policy(POLICY_DIR / "shophelp.yaml")
        ids = [r.id for r in policy.rules]
        assert len(ids) == len(set(ids))


class TestValid:
    def test_single_tool_name_becomes_a_tuple(self) -> None:
        p = parse_policy(wrap("  - {id: r, effect: block, when: {tool: pay}}\n"))
        assert p.rules[0].when.tool == ("pay",)

    def test_tool_list_and_wildcard(self) -> None:
        text = wrap(
            "  - {id: a, effect: block, when: {tool: [pay, look]}}\n"
            '  - {id: b, effect: block, when: {tool: "*"}}\n'
        )
        assert len(parse_policy(text).rules) == 2

    def test_in_keyword_is_accepted(self) -> None:
        text = wrap("  - {id: r, effect: block, when: {tool: pay, args: {x: {in: [1, 2]}}}}\n")
        assert parse_policy(text).rules[0].when.args["x"].operator() == ("in", (1, 2))


class TestInvalid:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("{{{ not yaml", "not valid YAML"),
            ("- just\n- a list\n", "top level must be a mapping"),
            ("name: t\n", "version"),
            (HEADER.replace("version: 1", "version: 2") + "rules: []\n", "version"),
            (wrap("  - {id: r, effect: explode, when: {tool: pay}}\n"), "effect"),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay}, severity: high}\n"),
                "severity",
            ),
            (
                wrap(
                    "  - {id: r, effect: block, when: {tool: pay}}\n"
                    "  - {id: r, effect: block, when: {tool: look}}\n"
                ),
                "duplicate rule id",
            ),
            (wrap("  - {id: Bad_ID, effect: block, when: {tool: pay}}\n"), "id"),
            (wrap("  - {id: unknown-tool, effect: block, when: {tool: pay}}\n"), "reserved"),
            (
                wrap("  - {id: r, effect: block, when: {tool: payy}}\n"),
                "unknown tool 'payy'",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tags: [nope]}}\n"),
                "tag 'nope'",
            ),
            (wrap("  - {id: r, effect: block, when: {}}\n"), "needs a tool selector"),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, args: {a: {}}}}\n"),
                "exactly one operator",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, args: {a: {gt: 1, lt: 5}}}}\n"),
                "exactly one operator",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, args: {a: {gtt: 1}}}}\n"),
                "gtt",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, args: {a: {matches: '('}}}}\n"),
                "invalid regex",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, args: {a: {in: []}}}}\n"),
                "empty list",
            ),
            (
                wrap("  - {id: r, effect: block, when: {tool: pay, prior_calls: {}}}\n"),
                "at least one",
            ),
            (HEADER.replace("risk: high", "risk: extreme") + "rules: []\n", "risk"),
        ],
    )
    def test_rejected_with_a_useful_message(self, text: str, expected: str) -> None:
        with pytest.raises(PolicyError, match=expected):
            parse_policy(text)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PolicyError, match="cannot read"):
            load_policy(tmp_path / "nope.yaml")

    def test_error_names_the_source(self) -> None:
        with pytest.raises(PolicyError, match=r"my-policy\.yaml"):
            parse_policy("name: t\n", source="my-policy.yaml")
