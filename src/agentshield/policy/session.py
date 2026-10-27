"""Per-session state that feeds the policy engine's context."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from agentshield.policy.models import PolicyContext


class SessionState:
    """Mutable bookkeeping for one agent session.

    The policy engine itself is stateless. The caller owns this object and:

    * calls :meth:`set_fact` when it learns something (e.g. ``order_total`` from a lookup), and
    * calls :meth:`record_call` after a tool call has **actually executed** (not when it was
      merely proposed, so blocked calls do not count against quotas).
    """

    def __init__(self, session_id: str, facts: Mapping[str, Any] | None = None) -> None:
        self.session_id = session_id
        self._facts: dict[str, Any] = dict(facts or {})
        self._calls: Counter[str] = Counter()

    def set_fact(self, key: str, value: Any) -> None:
        self._facts[key] = value

    def record_call(self, tool_name: str) -> None:
        self._calls[tool_name] += 1

    def context(self) -> PolicyContext:
        """An immutable snapshot for one policy evaluation."""
        return PolicyContext(facts=dict(self._facts), prior_calls=dict(self._calls))
