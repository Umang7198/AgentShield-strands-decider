"""Test doubles shared across suites. Not part of the package."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any

from strands.models.model import Model


class ScriptedModel(Model):
    """Each agent turn consumes the next planned tool call; then it answers in text.

    Used so ShopHelp and handler tests do not need a live host LLM.
    """

    def __init__(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]],
        *,
        final: str = "done",
    ) -> None:
        self._calls = list(calls)
        self._index = 0
        self.final = final

    def update_config(self, **model_config: Any) -> None: ...

    def get_config(self) -> Any:
        return {}

    def structured_output(self, *a: Any, **k: Any) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def stream(self, messages: Any, *args: Any, **kwargs: Any) -> AsyncGenerator[Any, None]:
        yield {"messageStart": {"role": "assistant"}}
        if self._index < len(self._calls):
            name, payload = self._calls[self._index]
            self._index += 1
            yield {
                "contentBlockStart": {
                    "start": {"toolUse": {"name": name, "toolUseId": f"tu-{self._index}"}}
                }
            }
            yield {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(payload)}}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
            return
        yield {"contentBlockDelta": {"delta": {"text": self.final}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
