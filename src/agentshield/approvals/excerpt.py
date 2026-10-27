"""A short, reviewable slice of the conversation. Not the full transcript."""

from __future__ import annotations

from collections.abc import Sequence

from agentshield.decider.state import Message

_MAX_MESSAGES = 8
_MAX_CHARS = 2000
_PER_MESSAGE = 300


def excerpt(messages: Sequence[Message], *, max_chars: int = _MAX_CHARS) -> str:
    kept = list(messages[-_MAX_MESSAGES:])
    lines = [f"[{m.role}] {m.content[:_PER_MESSAGE]}" for m in kept]
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
