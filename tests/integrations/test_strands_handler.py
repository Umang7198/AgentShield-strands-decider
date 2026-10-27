from collections.abc import AsyncGenerator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from strands import Agent, tool
from strands.interventions import Confirm, Deny, Guide, Proceed
from strands.models.model import Model

from agentshield.core import (
    CheckKind,
    CheckResult,
    Decision,
    DecisionSource,
    Outcome,
    RiskTier,
    ToolCall,
)
from agentshield.decider import DeciderChecker, FakeDeciderClient
from agentshield.integrations.strands_handler import (
    AgentShieldHandler,
    approval_prompt,
    clarify_feedback,
    messages_from_strands,
    result_payload,
    to_action,
)
from agentshield.pipeline import Shield, Verdict
from agentshield.policy import PolicyEngine, SessionState, load_policy
from agentshield.policy.facts import FactMap

POLICY = Path(__file__).resolve().parents[2] / "policies" / "shophelp.yaml"
GOOD = {"arg:*": 0.95, "consent": 0.95}


# --------------------------------------------------------------------------- conversion


class TestMessagesFromStrands:
    def test_text_keeps_its_role(self) -> None:
        out = messages_from_strands(
            [
                {"role": "user", "content": [{"text": "hi"}]},
                {"role": "assistant", "content": [{"text": "hello"}]},
            ]
        )
        assert [(m.role, m.content) for m in out] == [("user", "hi"), ("assistant", "hello")]

    def test_tool_results_become_tool_messages_named_after_their_tool(self) -> None:
        out = messages_from_strands(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "t1", "name": "get_order_status", "input": {}}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "t1",
                                "status": "success",
                                "content": [{"json": {"total": 89.99}}, {"text": "ok"}],
                            }
                        }
                    ],
                },
            ]
        )
        tool_msg = out[-1]
        assert tool_msg.role == "tool"
        assert tool_msg.name == "get_order_status"
        assert "89.99" in tool_msg.content
        assert "ok" in tool_msg.content

    def test_the_call_being_judged_is_excluded_but_earlier_calls_are_kept(self) -> None:
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "old", "name": "a", "input": {"x": 1}}},
                    {"toolUse": {"toolUseId": "now", "name": "b", "input": {"y": 2}}},
                ],
            }
        ]
        out = messages_from_strands(msgs, exclude_tool_use_id="now")
        assert len(out) == 1
        assert "Called a" in out[0].content

    def test_error_results_are_marked(self) -> None:
        out = messages_from_strands(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": "t",
                                "status": "error",
                                "content": [{"text": "boom"}],
                            }
                        }
                    ],
                }
            ]
        )
        assert out[0].content.startswith("(error)")

    def test_unknown_blocks_and_blank_text_are_ignored(self) -> None:
        out = messages_from_strands(
            [
                {
                    "role": "assistant",
                    "content": [{"reasoningContent": {}}, {"text": "  "}, {"image": {}}],
                }
            ]
        )
        assert out == []

    def test_result_without_matching_call_still_converts(self) -> None:
        out = messages_from_strands(
            [
                {
                    "role": "user",
                    "content": [
                        {"toolResult": {"toolUseId": "?", "status": "success", "content": []}}
                    ],
                }
            ]
        )
        assert out[0].name is None


# --------------------------------------------------------------------------- mapping


def decision(outcome: Outcome, limiting: CheckResult | None = None) -> Decision:
    return Decision(
        outcome=outcome,
        risk=RiskTier.MEDIUM,
        source=DecisionSource.DECIDER,
        reason="because",
        limiting_check=limiting,
    )


CALL = ToolCall(name="cancel_order", arguments={"order_id": "A1"})


class TestToAction:
    def test_allow_proceeds(self) -> None:
        assert isinstance(to_action(decision(Outcome.ALLOW), CALL), Proceed)

    def test_block_denies_with_the_reason(self) -> None:
        action = to_action(decision(Outcome.BLOCK), CALL)
        assert isinstance(action, Deny)
        assert "because" in action.reason

    def test_clarify_guides(self) -> None:
        assert isinstance(to_action(decision(Outcome.CLARIFY), CALL), Guide)

    def test_require_approval_confirms_and_shows_the_call(self) -> None:
        action = to_action(decision(Outcome.REQUIRE_APPROVAL), CALL)
        assert isinstance(action, Confirm)
        assert "cancel_order" in action.prompt
        assert "A1" in action.prompt
        assert action.response is None  # pauses for a real human, never self-approves


class TestFeedback:
    def test_ungrounded_argument_names_the_argument(self) -> None:
        check = CheckResult(kind=CheckKind.ARGUMENT_GROUNDING, p_yes=0.1, subject="order_id")
        text = clarify_feedback(decision(Outcome.CLARIFY, check), CALL)
        assert "'order_id'" in text
        assert "not run" in text

    def test_missing_consent_asks_whether_to_proceed(self) -> None:
        check = CheckResult(kind=CheckKind.CONSENT, p_yes=0.1)
        text = clarify_feedback(decision(Outcome.CLARIFY, check), CALL)
        assert "go ahead" in text

    def test_other_cases_fall_back_to_the_reason(self) -> None:
        assert "because" in clarify_feedback(decision(Outcome.CLARIFY), CALL)

    def test_approval_prompt_is_readable(self) -> None:
        assert "cancel_order(order_id='A1')" in approval_prompt(
            decision(Outcome.REQUIRE_APPROVAL), CALL
        )


# --------------------------------------------------------------------------- handler


def make_handler(
    script: dict[str, float] | None = None, **kw: Any
) -> tuple[AgentShieldHandler, list[Verdict]]:
    seen: list[Verdict] = []
    shield = Shield(
        PolicyEngine(load_policy(POLICY)),
        DeciderChecker(FakeDeciderClient(GOOD if script is None else script, **kw)),
    )
    session = SessionState(
        "s",
        facts={
            "order_total": 89.99,
            "refundable_remaining": 89.99,
            "customer_email": "ann@example.com",
        },
    )
    return AgentShieldHandler(shield, session, on_verdict=seen.append), seen


def fake_event(name: str, tool_input: Any, user_text: str = "Cancel order A1001.") -> Any:
    return SimpleNamespace(
        tool_use={"toolUseId": "tu1", "name": name, "input": tool_input},
        agent=SimpleNamespace(
            messages=[
                {"role": "user", "content": [{"text": user_text}]},
                {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "tu1", "name": name, "input": tool_input}}
                    ],
                },
            ]
        ),
    )


class TestHandler:
    def test_is_configured_fail_closed(self) -> None:
        handler, _ = make_handler()
        assert handler.on_error == "deny"
        assert handler.name == "agentshield"

    async def test_allow(self) -> None:
        handler, seen = make_handler()
        action = await handler.before_tool_call(
            fake_event("get_order_status", {"order_id": "A1001"})
        )
        assert isinstance(action, Proceed)
        assert len(seen) == 1

    async def test_block(self) -> None:
        handler, _ = make_handler()
        action = await handler.before_tool_call(
            fake_event("delete_customer_data", {"customer_id": "C"})
        )
        assert isinstance(action, Deny)

    async def test_clarify(self) -> None:
        handler, _ = make_handler({**GOOD, "consent": 0.1})
        action = await handler.before_tool_call(fake_event("cancel_order", {"order_id": "A1001"}))
        assert isinstance(action, Guide)

    async def test_approval(self) -> None:
        handler, _ = make_handler()
        action = await handler.before_tool_call(
            fake_event("issue_refund", {"order_id": "A1001", "amount": 40})
        )
        assert isinstance(action, Confirm)

    async def test_the_pending_call_is_not_part_of_the_evidence_history(self) -> None:
        handler, seen = make_handler()
        await handler.before_tool_call(fake_event("get_order_status", {"order_id": "A1001"}))
        state = seen[0].run.state.text  # type: ignore[union-attr]
        assert state.count("get_order_status(") == 1  # once, as the proposed call
        assert "Called get_order_status" not in state

    async def test_non_object_input_is_denied(self) -> None:
        handler, seen = make_handler()
        action = await handler.before_tool_call(fake_event("get_order_status", "A1001"))
        assert isinstance(action, Deny)
        assert seen == []

    async def test_executed_calls_count_toward_quotas_blocked_ones_do_not(self) -> None:
        handler, _ = make_handler()
        executed: Any = SimpleNamespace(tool_use={"name": "issue_refund"}, cancel_message=None)
        cancelled: Any = SimpleNamespace(tool_use={"name": "issue_refund"}, cancel_message="DENIED")
        handler.after_tool_call(executed)
        handler.after_tool_call(cancelled)
        assert handler.session.context().prior_calls == {"issue_refund": 1}

    async def test_successful_lookup_writes_facts_cancelled_ones_do_not(self) -> None:
        handler, _ = make_handler()
        handler.session = SessionState("s")
        handler._facts = FactMap({"get_order_status": {"total": "order_total"}})
        ok: Any = SimpleNamespace(
            tool_use={"name": "get_order_status"},
            cancel_message=None,
            result={
                "status": "success",
                "content": [{"text": '{"total": 89.99, "order_id": "A1001"}'}],
            },
        )
        denied: Any = SimpleNamespace(
            tool_use={"name": "get_order_status"},
            cancel_message="DENIED",
            result={
                "status": "success",
                "content": [{"text": '{"total": 1}'}],
            },
        )
        handler.after_tool_call(ok)
        handler.after_tool_call(denied)
        assert handler.session.context().facts == {"order_total": 89.99}


class TestResultPayload:
    def test_json_block(self) -> None:
        assert result_payload({"status": "success", "content": [{"json": {"total": 1}}]}) == {
            "total": 1
        }

    def test_json_in_text(self) -> None:
        assert result_payload({"status": "success", "content": [{"text": '{"total": 1}'}]}) == {
            "total": 1
        }

    def test_errors_and_prose_are_ignored(self) -> None:
        assert result_payload({"status": "error", "content": [{"json": {"total": 1}}]}) is None
        assert result_payload({"status": "success", "content": [{"text": "ok"}]}) is None
        assert result_payload(None) is None


# --------------------------------------------------------------------------- real Agent


class ScriptedModel(Model):
    """Replays scripted turns: first a tool call, then a final text answer."""

    def __init__(self, tool_name: str, tool_input: dict[str, Any]) -> None:
        self.tool_name, self.tool_input = tool_name, tool_input
        self.turns = 0

    def update_config(self, **model_config: Any) -> None: ...

    def get_config(self) -> Any:
        return {}

    def structured_output(self, *a: Any, **k: Any) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def stream(self, messages: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        import json

        self.turns += 1
        yield {"messageStart": {"role": "assistant"}}
        if self.turns == 1:
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"name": self.tool_name, "toolUseId": "tu-1"}}
                }
            }
            yield {
                "contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(self.tool_input)}}}
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": "done"}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


def run_agent(
    tool_name: str, tool_input: dict[str, Any], script: dict[str, float] | None = None
) -> tuple[Any, list[str], AgentShieldHandler, Agent]:
    executed: list[str] = []

    @tool
    def cancel_order(order_id: str) -> str:
        """Cancel an order."""
        executed.append(order_id)
        return "cancelled"

    @tool
    def delete_customer_data(customer_id: str) -> str:
        """Delete data."""
        executed.append(customer_id)
        return "deleted"

    @tool
    def issue_refund(order_id: str, amount: float) -> str:
        """Refund."""
        executed.append(order_id)
        return "refunded"

    handler, _ = make_handler(script)
    agent = Agent(
        model=ScriptedModel(tool_name, tool_input),
        tools=[cancel_order, delete_customer_data, issue_refund],
        interventions=[handler],
        callback_handler=None,
    )
    result = agent("Please cancel order A1001.")
    return result, executed, handler, agent


class TestWithARealAgent:
    def test_allowed_call_runs_and_is_counted(self) -> None:
        _, executed, handler, _ = run_agent("cancel_order", {"order_id": "A1001"})
        assert executed == ["A1001"]
        assert handler.session.context().prior_calls == {"cancel_order": 1}

    def test_blocked_call_never_runs_and_the_model_sees_the_denial(self) -> None:
        _, executed, handler, agent = run_agent("delete_customer_data", {"customer_id": "C1"})
        assert executed == []
        assert handler.session.context().prior_calls == {}
        assert "DENIED" in tool_result_text(agent)

    def test_unclear_call_never_runs_and_the_model_is_told_what_to_ask(self) -> None:
        _, executed, _, agent = run_agent(
            "cancel_order", {"order_id": "A1001"}, {**GOOD, "consent": 0.1}
        )
        assert executed == []
        text = tool_result_text(agent)
        assert "GUIDANCE" in text
        assert "go ahead" in text

    def test_approval_pauses_the_agent_before_the_tool_runs(self) -> None:
        result, executed, _, _ = run_agent("issue_refund", {"order_id": "A1001", "amount": 40})
        assert executed == []
        assert result.stop_reason == "interrupt"
        assert result.interrupts


def tool_result_text(agent: Agent) -> str:
    parts: list[str] = []
    for message in agent.messages:
        for block in message["content"]:
            if "toolResult" in block:
                parts.extend(c.get("text", "") for c in block["toolResult"]["content"])
    return "\n".join(parts)
