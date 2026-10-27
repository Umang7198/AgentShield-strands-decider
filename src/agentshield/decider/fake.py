"""A scripted, deterministic stand-in for the Decider, for tests and offline demos."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass

from agentshield.decider.client import DeciderResponse, NoulQuestion


@dataclass(frozen=True, slots=True)
class RecordedCall:
    state: str
    questions: Mapping[str, NoulQuestion]


class FakeDeciderClient:
    """Answers each question from ``script`` (question id pattern -> p_yes).

    Patterns use shell wildcards and the first match wins, so ``{"arg:amount": 0.1, "arg:*": 0.9}``
    makes the amount look invented and everything else well supported. Unmatched questions get
    ``default``. Set ``error`` to make every call raise, to test fallback behaviour.
    """

    def __init__(
        self,
        script: Mapping[str, float] | None = None,
        *,
        default: float = 0.5,
        error: Exception | None = None,
        latency_ms: float = 1.0,
    ) -> None:
        self._script = dict(script or {})
        self._default = default
        self.error = error
        self._latency_ms = latency_ms
        self.calls: list[RecordedCall] = []

    async def ask(self, state: str, questions: Mapping[str, NoulQuestion]) -> DeciderResponse:
        self.calls.append(RecordedCall(state=state, questions=dict(questions)))
        if self.error is not None:
            raise self.error
        answers = {qid: self._answer(qid) for qid in questions}
        return DeciderResponse(
            answers=answers,
            wall_latency_ms=self._latency_ms,
            server_latency_ms=self._latency_ms,
            model="fake-decider",
        )

    def _answer(self, qid: str) -> float:
        for pattern, p in self._script.items():
            if fnmatch.fnmatchcase(qid, pattern):
                return p
        return self._default
