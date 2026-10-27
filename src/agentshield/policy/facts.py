"""Copy fields out of a successful tool result into :class:`~agentshield.policy.session.SessionState`.

The policy engine is stateless; it only sees facts the session already knows. Lookups such as
``get_order_status`` have to write ``order_total`` here, otherwise a later refund is fail-closed
(the ``$ctx.order_total`` fact is missing).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agentshield.policy.session import SessionState


class FactMap:
    """Declarative mapping: tool name -> {result field: session fact name}."""

    def __init__(self, mapping: Mapping[str, Mapping[str, str]]) -> None:
        self._mapping = {tool: dict(fields) for tool, fields in mapping.items()}

    def apply(self, tool_name: str, payload: Mapping[str, Any], session: SessionState) -> None:
        fields = self._mapping.get(tool_name)
        if not fields:
            return
        for source, fact in fields.items():
            if source in payload and payload[source] is not None:
                session.set_fact(fact, payload[source])
