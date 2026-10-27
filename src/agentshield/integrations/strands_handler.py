"""Strands Agents adapter: gates every tool call through the :class:`~agentshield.pipeline.Shield`.

Usage::

    handler = AgentShieldHandler(shield, SessionState("session-1"))
    agent = Agent(model=..., tools=[...], interventions=[handler])

Outcome mapping:

====================  ===========================================================
ALLOW                 ``Proceed``: the tool runs.
CLARIFY               ``Guide``: the tool is cancelled and the model is told what to ask.
REQUIRE_APPROVAL      Pauses for a human. With an ``ApprovalDesk``, the review UI/API
                      resume the agent; without one, Strands ``Confirm`` is used.
BLOCK                 ``Deny``: the tool is cancelled with the reason.
====================  ===========================================================

Fail-closed: if anything in the handler raises, ``on_error`` is ``"deny"``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
from strands.interrupt import InterruptException
from strands.interventions import Confirm, Deny, Guide, InterventionHandler, OnError, Proceed

from agentshield.approvals.desk import ApprovalDesk, interrupt_was_yes
from agentshield.approvals.models import Approval, ApprovalStatus
from agentshield.core.models import CheckKind, Decision, Outcome, ToolCall
from agentshield.decider.state import Message
from agentshield.pipeline import Shield, Verdict
from agentshield.policy.facts import FactMap
from agentshield.policy.session import SessionState

VerdictListener = Callable[[Verdict], None]


class AgentShieldHandler(InterventionHandler):
    def __init__(
        self,
        shield: Shield,
        session: SessionState | None = None,
        *,
        on_verdict: VerdictListener | None = None,
        facts: FactMap | None = None,
        approvals: ApprovalDesk | None = None,
    ) -> None:
        self._shield = shield
        self.session = session or SessionState("default")
        self._on_verdict = on_verdict
        self._facts = facts
        self._approvals = approvals

    @property
    def name(self) -> str:
        return "agentshield"

    @property
    def on_error(self) -> OnError:
        return "deny"  # a broken guard must never silently let calls through

    async def before_tool_call(
        self, event: BeforeToolCallEvent, **kwargs: Any
    ) -> Proceed | Deny | Guide | Confirm:
        tool_use = event.tool_use
        arguments = tool_use.get("input")
        if not isinstance(arguments, dict):
            return Deny(reason="Tool input must be a JSON object; the call was not understood.")

        call = ToolCall(
            name=tool_use["name"], arguments=arguments, tool_use_id=tool_use.get("toolUseId")
        )
        if self._approvals is not None:
            existing = self._approvals.by_tool_use(call.tool_use_id)
            if existing is not None:
                return self._resume_approval(event, existing)
        messages = messages_from_strands(event.agent.messages, exclude_tool_use_id=call.tool_use_id)
        verdict = await self._shield.evaluate(call, messages, self.session)
        if self._on_verdict is not None:
            self._on_verdict(verdict)
        if verdict.decision.outcome is Outcome.REQUIRE_APPROVAL and self._approvals is not None:
            return self._request_approval(event, verdict, messages)
        return to_action(verdict.decision, call)

    def _request_approval(
        self, event: BeforeToolCallEvent, verdict: Verdict, messages: Sequence[Message]
    ) -> Proceed | Deny:
        desk = self._approvals
        assert desk is not None
        rec = desk.submit(verdict, self.session, messages, agent_id=self._shield.agent_id)
        return self._interrupt_for(event, rec)

    def _resume_approval(self, event: BeforeToolCallEvent, rec: Approval) -> Proceed | Deny:
        return self._interrupt_for(event, rec)

    def _interrupt_for(self, event: BeforeToolCallEvent, rec: Approval) -> Proceed | Deny:
        desk = self._approvals
        assert desk is not None
        try:
            response = event.interrupt(self.name, reason=rec.prompt)
        except InterruptException as exc:
            desk.bind_interrupt(rec.id, exc.interrupt.id)
            raise
        approved = interrupt_was_yes(response)
        if rec.status is ApprovalStatus.PENDING:
            rec = desk.record(
                rec.id,
                ApprovalStatus.APPROVED if approved else ApprovalStatus.DENIED,
                by="interrupt",
            )
        if approved and rec.status is ApprovalStatus.APPROVED:
            return Proceed(reason=f"Approved by {rec.resolved_by}")
        who = rec.resolved_by or "human"
        return Deny(reason=f"Denied by {who}")

    def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Proceed:
        # Quotas ("max 3 refunds") count only calls that really ran, not blocked ones.
        if event.cancel_message is not None:
            return Proceed()
        name = event.tool_use["name"]
        self.session.record_call(name)
        if self._facts is not None:
            payload = result_payload(getattr(event, "result", None))
            if payload is not None:
                self._facts.apply(name, payload, self.session)
        return Proceed()


def to_action(decision: Decision, call: ToolCall) -> Proceed | Deny | Guide | Confirm:
    match decision.outcome:
        case Outcome.ALLOW:
            return Proceed(reason=decision.reason)
        case Outcome.BLOCK:
            return Deny(reason=decision.reason)
        case Outcome.CLARIFY:
            return Guide(feedback=clarify_feedback(decision, call), reason=decision.reason)
        case Outcome.REQUIRE_APPROVAL:
            return Confirm(prompt=approval_prompt(decision, call), reason=decision.reason)
    raise AssertionError(f"unhandled outcome {decision.outcome!r}")  # pragma: no cover


def clarify_feedback(decision: Decision, call: ToolCall) -> str:
    """Tell the model what to do instead of calling the tool: ask the user."""
    head = f"Do not call '{call.name}' yet; it was not run."
    check = decision.limiting_check
    if check is not None and check.kind is CheckKind.ARGUMENT_GROUNDING and check.subject:
        return f"{head} Ask the user to confirm the value of '{check.subject}' before retrying."
    if check is not None and check.kind is CheckKind.CONSENT:
        return f"{head} Ask the user whether they want you to go ahead with this action."
    return f"{head} Ask the user for the missing information or confirmation. ({decision.reason})"


def approval_prompt(decision: Decision, call: ToolCall) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in call.arguments.items())
    return f"Approve {call.name}({args})? {decision.reason}"


def messages_from_strands(
    messages: Sequence[Any], *, exclude_tool_use_id: str | None = None
) -> list[Message]:
    """Convert Strands messages into the Decider's evidence format.

    * text blocks keep their role;
    * tool results (which Strands stores in ``user`` messages) become ``tool`` messages named
      after the tool that produced them;
    * the tool call being judged is left out, because it is passed to the Decider separately;
    * reasoning, images and other block kinds are ignored.
    """
    names: dict[str, str] = {}
    for message in messages:
        for block in message.get("content", []):
            use = block.get("toolUse")
            if use:
                names[use["toolUseId"]] = use["name"]

    out: list[Message] = []
    for message in messages:
        role = message.get("role")
        for block in message.get("content", []):
            if "text" in block and role in ("user", "assistant"):
                if block["text"].strip():
                    out.append(Message(role=role, content=block["text"]))
            elif "toolUse" in block:
                use = block["toolUse"]
                if use["toolUseId"] != exclude_tool_use_id:
                    out.append(
                        Message(
                            role="assistant",
                            content=f"Called {use['name']} with {_dump(use.get('input'))}",
                        )
                    )
            elif "toolResult" in block:
                result = block["toolResult"]
                out.append(
                    Message(
                        role="tool",
                        name=names.get(result.get("toolUseId", "")),
                        content=_result_text(result),
                    )
                )
    return out


def _result_text(result: dict[str, Any]) -> str:
    parts = []
    for item in result.get("content", []):
        if "text" in item:
            parts.append(item["text"])
        elif "json" in item:
            parts.append(_dump(item["json"]))
    text = "\n".join(parts)
    return f"(error) {text}" if result.get("status") == "error" else text


def result_payload(result: Any) -> dict[str, Any] | None:
    """JSON object from a Strands tool result, or None if the call failed or was not an object."""
    if not isinstance(result, dict) or result.get("status") == "error":
        return None
    for item in result.get("content", []):
        if not isinstance(item, dict):
            continue
        if "json" in item and isinstance(item["json"], dict):
            return item["json"]
        if "text" in item:
            try:
                parsed: Any = json.loads(item["text"])
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _dump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(value)
