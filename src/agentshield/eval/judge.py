"""LLM-as-judge baseline: same noul questions as the Decider, answered by the host LLM."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import yaml

from agentshield.decider.client import DeciderResponse, NoulQuestion
from agentshield.decider.errors import DeciderBadResponse, DeciderUnavailable

PRICES_PATH = Path(__file__).resolve().parents[3] / "eval" / "prices.yaml"


def load_prices(path: Path | None = None) -> dict[str, Any]:
    data = yaml.safe_load((path or PRICES_PATH).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("prices file is not a mapping")
    return {
        "input_per_1m": _price(os.environ.get("LLM_JUDGE_INPUT_PER_1M"), data.get("input_per_1m")),
        "output_per_1m": _price(
            os.environ.get("LLM_JUDGE_OUTPUT_PER_1M"), data.get("output_per_1m")
        ),
        "model_note": str(data.get("model_note", "")),
    }


def _price(override: str | None, fallback: object) -> float:
    if override is not None:
        return float(override)
    if isinstance(fallback, (int, float)) and not isinstance(fallback, bool):
        return float(fallback)
    if isinstance(fallback, str):
        return float(fallback)
    raise ValueError(f"price is not a number: {fallback!r}")


def estimate_usd(input_tokens: int, output_tokens: int, prices: Mapping[str, Any]) -> float:
    return input_tokens / 1_000_000 * float(
        prices["input_per_1m"]
    ) + output_tokens / 1_000_000 * float(prices["output_per_1m"])


class LlmJudgeClient:
    """OpenAI-compatible chat completions: one request per scenario, all noul questions.

    Expects a JSON object mapping question id → ``p_yes`` in [0, 1].
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float = 60.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self._base = (
            base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL", "")
        ).rstrip("/")
        self._model = model if model is not None else os.environ.get("MODEL_ID", "")
        self._timeout_s = timeout_s
        self._transport = transport
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    @property
    def model_id(self) -> str:
        return self._model

    def reset_usage(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    async def ask(self, state: str, questions: Mapping[str, NoulQuestion]) -> DeciderResponse:
        if not questions:
            raise ValueError("ask() needs at least one question")
        if not self._api_key or not self._base or not self._model:
            raise DeciderUnavailable(
                "LLM judge missing OPENAI_API_KEY, OPENAI_BASE_URL or MODEL_ID"
            )
        started = time.perf_counter()
        answers = await self._batch(state, questions)
        wall_ms = (time.perf_counter() - started) * 1000
        return DeciderResponse(
            answers=answers,
            wall_latency_ms=wall_ms,
            model=self._model,
            input_tokens=self.input_tokens or None,
        )

    async def _batch(self, state: str, questions: Mapping[str, NoulQuestion]) -> dict[str, float]:
        items = {
            qid: {
                "question": q.instructions,
                "yes_means": q.true_means or "yes",
                "no_means": q.false_means or "no",
            }
            for qid, q in questions.items()
        }
        payload = {
            "model": self._model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a yes/no judge for a customer-support agent. "
                        "Reply with JSON only: an object mapping each question id to a "
                        'float p_yes in [0, 1]. Example: {"consent": 0.9, "arg:order_id": 0.1}.'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation / state:\n{state}\n\n"
                        f"Questions:\n{json.dumps(items, indent=2)}"
                    ),
                },
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout_s, transport=self._transport
            ) as http:
                response = await http.post(
                    f"{self._base}/chat/completions", json=payload, headers=headers
                )
        except httpx.HTTPError as exc:
            raise DeciderUnavailable(f"LLM judge request failed: {exc!r}") from exc
        if response.status_code >= 400:
            raise DeciderUnavailable(
                f"LLM judge HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DeciderBadResponse("LLM judge response is not chat-completions JSON") from exc
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        self.input_tokens += int(usage.get("prompt_tokens") or 0)
        self.output_tokens += int(usage.get("completion_tokens") or 0)
        self.calls += 1
        parsed = _parse_answers(str(content), tuple(questions))
        return parsed


def _parse_answers(content: str, qids: tuple[str, ...]) -> dict[str, float]:
    data = _extract_json(content)
    if len(qids) == 1 and isinstance(data, dict) and "p_yes" in data and qids[0] not in data:
        return {qids[0]: _clip(float(data["p_yes"]))}
    if not isinstance(data, dict):
        raise DeciderBadResponse(f"LLM judge did not return a JSON object: {content[:160]!r}")
    out: dict[str, float] = {}
    for qid in qids:
        if qid not in data:
            raise DeciderBadResponse(f"LLM judge missing {qid!r} in {content[:160]!r}")
        raw = data[qid]
        if isinstance(raw, dict) and "p_yes" in raw:
            raw = raw["p_yes"]
        out[qid] = _clip(float(raw))
    return out


def _extract_json(content: str) -> Any:
    text = content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _parse_p_yes(content: str) -> float:
    text = content.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "p_yes" in data:
            return _clip(float(data["p_yes"]))
    except (json.JSONDecodeError, TypeError, ValueError):
        match = re.search(r"\{[^{}]*p_yes[^{}]*\}", text)
        if match:
            try:
                data = json.loads(match.group(0))
                return _clip(float(data["p_yes"]))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                pass
    lowered = text.lower()
    if re.search(r"\bno\b", lowered) and not re.search(r"\byes\b", lowered):
        return 0.12
    if re.search(r"\byes\b", lowered):
        return 0.88
    raise DeciderBadResponse(f"LLM judge did not return p_yes: {text[:160]!r}")


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
