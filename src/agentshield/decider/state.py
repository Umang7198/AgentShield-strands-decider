"""Turn a conversation plus a proposed tool call into the text the Decider reads.

Two things matter here, both learned from the Decider's server code:

1. **Over-long states are cut from the end, silently.** The proposed call is the most important
   part, so we never let the server truncate: we budget the text ourselves, drop the *oldest*
   messages first, and always keep the call and the latest user message.
2. **The prompt uses XML-like tags** (``<state>``, ``<question>``, ``<options>``, ``<answer>``).
   Conversation text is untrusted (a customer, a web page, a tool result), so those tags are
   neutralised, otherwise text could forge the prompt's structure.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agentshield.core.models import ToolCall

Role = Literal["user", "assistant", "tool"]

_ELLIPSIS = " …[truncated]… "
_PROMPT_TAGS = re.compile(r"<(\s*/?\s*)(state|question|options|answer)\b", re.IGNORECASE)
_MIN_KEPT_CHARS = 80


class Message(BaseModel):
    """One conversation turn. ``name`` is the tool name when ``role == "tool"``."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str
    name: str | None = None


class StateBudget(BaseModel):
    """Size limits, in characters. The Decider's window is 4096 tokens, roughly 3-4 chars each;
    the defaults leave generous room for the questions themselves."""

    model_config = ConfigDict(frozen=True)

    max_chars: int = Field(default=6000, ge=500)
    max_message_chars: int = Field(default=1200, ge=_MIN_KEPT_CHARS)
    max_value_chars: int = Field(default=300, ge=20)


class BuiltState(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    total_messages: int
    included_messages: int
    shortened_messages: int

    @property
    def dropped_messages(self) -> int:
        return self.total_messages - self.included_messages

    @property
    def lost_evidence(self) -> bool:
        """True if any message was dropped or cut short, so a 'not grounded' answer may just
        mean the supporting text did not fit."""
        return self.dropped_messages > 0 or self.shortened_messages > 0


def neutralise(text: str) -> str:
    """Defang the Decider's own prompt tags in untrusted text by swapping their ``<`` for U+2039."""
    return _PROMPT_TAGS.sub(lambda m: f"\u2039{m.group(1)}{m.group(2)}", text)


def shorten(text: str, limit: int) -> str:
    """Cut to ``limit`` characters, keeping the start and the end (the middle is least useful)."""
    if len(text) <= limit:
        return text
    room = max(limit - len(_ELLIPSIS), 2)
    head = room * 2 // 3
    tail = room - head
    return text[:head] + _ELLIPSIS + (text[-tail:] if tail else "")


def render_value(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = repr(value)
    return shorten(neutralise(text), limit)


def render_call(call: ToolCall, budget: StateBudget) -> str:
    args = ", ".join(
        f"{neutralise(name)}={render_value(value, budget.max_value_chars)}"
        for name, value in call.arguments.items()
    )
    block = f"Proposed tool call:\n{neutralise(call.name)}({args})"
    return shorten(block, budget.max_chars // 2)


def _render_message(message: Message) -> str:
    label = f"tool result: {neutralise(message.name)}" if message.name else "tool result"
    prefix = {"user": "[user]", "assistant": "[assistant]", "tool": f"[{label}]"}[message.role]
    return f"{prefix} {neutralise(message.content).strip()}"


def build_state(
    messages: Sequence[Message],
    call: ToolCall | None = None,
    budget: StateBudget | None = None,
) -> BuiltState:
    """Render ``messages`` (oldest first) and the proposed ``call`` within ``budget``.

    The call is always included in full (values shortened). Messages are added newest-first until
    the budget is used up; older ones are dropped and a marker says how many.
    """
    b = budget or StateBudget()
    header = "Conversation (oldest first):"
    call_block = render_call(call, b) if call else ""
    omitted_marker = "[{n} earlier message(s) omitted]"
    # Space left for message text; reserve room for the header, call block and omitted marker.
    available = b.max_chars - len(header) - len(call_block) - len(omitted_marker) - 16

    kept: list[str] = []
    shortened = 0
    used = 0
    last_user = max((i for i, m in enumerate(messages) if m.role == "user"), default=-1)

    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        rendered = _render_message(message)
        cut = shorten(rendered, b.max_message_chars)
        was_shortened = cut != rendered
        cost = len(cut) + 1
        if used + cost > available:
            if index == last_user and available - used >= _MIN_KEPT_CHARS:
                # The latest user message is the main evidence of intent and consent: keep it,
                # squeezed, rather than lose it.
                cut = shorten(rendered, available - used - 1)
                was_shortened = True
                cost = len(cut) + 1
            else:
                break
        kept.append(cut)
        shortened += was_shortened
        used += cost

    kept.reverse()
    dropped = len(messages) - len(kept)

    lines = [header]
    if dropped:
        lines.append(omitted_marker.format(n=dropped))
    lines.extend(kept if kept else ["(no messages)"])
    if call_block:
        lines.extend(["", call_block])

    return BuiltState(
        text="\n".join(lines),
        total_messages=len(messages),
        included_messages=len(kept),
        shortened_messages=shortened,
    )
