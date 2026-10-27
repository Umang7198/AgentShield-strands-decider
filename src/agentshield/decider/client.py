"""Async client for the Strands Decider ``/v1/systemone`` endpoint (yes/no questions only)."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol, Self

import httpx

from agentshield.decider.errors import (
    DeciderBadResponse,
    DeciderRejected,
    DeciderTimeout,
    DeciderUnavailable,
)

ENDPOINT = "/v1/systemone"


@dataclass(frozen=True, slots=True)
class NoulQuestion:
    """A yes/no question. ``true_means`` / ``false_means`` sharpen the decision boundary."""

    instructions: str
    true_means: str | None = None
    false_means: str | None = None

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        criteria = {
            key: text
            for key, text in (("true", self.true_means), ("false", self.false_means))
            if text
        }
        if criteria:
            wire["criteria"] = criteria
        return wire


@dataclass(frozen=True, slots=True)
class DeciderResponse:
    """Probabilities of "yes" keyed by question id, plus timing."""

    answers: Mapping[str, float]
    wall_latency_ms: float
    server_latency_ms: float | None = None
    model: str | None = None
    input_tokens: int | None = None


class DeciderClient(Protocol):
    """Anything that can answer a batch of yes/no questions about one state."""

    async def ask(self, state: str, questions: Mapping[str, NoulQuestion]) -> DeciderResponse: ...


class HttpDeciderClient:
    """Talks to a running ``strands-decider serve`` (or any Jev-compatible server).

    All questions go in one request so the server reads the state once. Only connection failures
    are retried (the server may be restarting); a timeout is *not* retried, because the model may
    still be working and a retry would just double the load and the wait.
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8099",
        *,
        timeout_s: float = 5.0,
        max_retries: int = 1,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        self._max_retries = max_retries
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        # A new client per request so we never reuse sockets bound to a closed asyncio loop.
        # Strands, FastAPI and the demo each call asyncio.run() on their own loop.
        return httpx.AsyncClient(
            base_url=self._base_url, timeout=self._timeout_s, transport=self._transport
        )

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def ask(self, state: str, questions: Mapping[str, NoulQuestion]) -> DeciderResponse:
        if not questions:
            raise ValueError("ask() needs at least one question")
        payload = {
            "state": state,
            "questions": {qid: q.to_wire() for qid, q in questions.items()},
        }
        started = time.perf_counter()
        response = await self._post(payload)
        wall_ms = (time.perf_counter() - started) * 1000
        return _parse(response, questions, wall_ms)

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for _ in range(self._max_retries + 1):
            try:
                async with self._client() as http:
                    response = await http.post(ENDPOINT, json=payload)
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_error = exc
                continue
            except httpx.TimeoutException as exc:
                raise DeciderTimeout(f"Decider timed out: {type(exc).__name__}") from exc
            except httpx.HTTPError as exc:
                raise DeciderUnavailable(f"Decider request failed: {exc!r}") from exc
            return _check_status(response)
        raise DeciderUnavailable(f"cannot connect to the Decider: {last_error!r}")


def _check_status(response: httpx.Response) -> httpx.Response:
    if response.status_code >= 500:
        raise DeciderUnavailable(f"Decider server error ({response.status_code})")
    if response.status_code >= 400:
        raise DeciderRejected(response.status_code, _detail(response))
    return response


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail", response.text)
    except (ValueError, AttributeError):
        detail = response.text
    return str(detail)[:300]


def _parse(
    response: httpx.Response, questions: Mapping[str, NoulQuestion], wall_ms: float
) -> DeciderResponse:
    try:
        body = response.json()
        raw_answers = body["answers"]
    except (ValueError, KeyError, TypeError) as exc:
        raise DeciderBadResponse("response is not the expected JSON shape") from exc
    if not isinstance(raw_answers, dict):
        raise DeciderBadResponse("'answers' is not an object")

    answers: dict[str, float] = {}
    for qid in questions:
        answer = raw_answers.get(qid)
        if not isinstance(answer, dict) or answer.get("type") != "noul":
            raise DeciderBadResponse(f"no yes/no answer for question {qid!r}")
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, int | float) or not math.isfinite(p):
            raise DeciderBadResponse(f"answer for {qid!r} is not a number")
        if not 0.0 <= p <= 1.0:
            raise DeciderBadResponse(f"answer for {qid!r} is not a probability: {p}")
        answers[qid] = float(p)

    server_ms = body.get("latency_ms")
    usage = body.get("usage")
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    return DeciderResponse(
        answers=answers,
        wall_latency_ms=wall_ms,
        server_latency_ms=float(server_ms) if isinstance(server_ms, int | float) else None,
        model=body.get("model") if isinstance(body.get("model"), str) else None,
        input_tokens=tokens if isinstance(tokens, int) else None,
    )
