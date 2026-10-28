import json

import httpx
import pytest

from agentshield.decider.client import NoulQuestion
from agentshield.eval.judge import LlmJudgeClient, _parse_p_yes, estimate_usd, load_prices


def test_parse_p_yes_json_and_yes_no() -> None:
    assert _parse_p_yes('{"p_yes": 0.81}') == pytest.approx(0.81)
    assert _parse_p_yes("yes") == pytest.approx(0.88)
    assert _parse_p_yes("No.") == pytest.approx(0.12)
    assert _parse_p_yes('prefix {"p_yes": 0.4} suffix') == pytest.approx(0.4)


def test_load_prices_and_estimate() -> None:
    prices = load_prices()
    assert prices["input_per_1m"] > 0
    usd = estimate_usd(1_000_000, 1_000_000, prices)
    assert usd == pytest.approx(prices["input_per_1m"] + prices["output_per_1m"])


async def test_llm_judge_client_parses_chat_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        assert body["model"] == "mock-judge"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"consent": 0.73, "arg:order_id": 0.91}'}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4},
            },
        )

    client = LlmJudgeClient(
        api_key="sk-test",
        base_url="http://judge.test/v1",
        model="mock-judge",
        transport=httpx.MockTransport(handler),
    )
    response = await client.ask(
        "user: refund A1001",
        {
            "consent": NoulQuestion(instructions="Did the user agree?"),
            "arg:order_id": NoulQuestion(instructions="Did the user state A1001?"),
        },
    )
    assert response.answers["consent"] == pytest.approx(0.73)
    assert response.answers["arg:order_id"] == pytest.approx(0.91)
    assert client.input_tokens == 12
    assert client.output_tokens == 4
    assert client.calls == 1
