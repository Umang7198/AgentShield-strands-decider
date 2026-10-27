import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from agentshield.decider import (
    DeciderBadResponse,
    DeciderRejected,
    DeciderTimeout,
    DeciderUnavailable,
    HttpDeciderClient,
    NoulQuestion,
)

Q = {"consent": NoulQuestion("Did the user agree?", "they agreed", "they did not")}

GOOD: dict[str, Any] = {
    "model": "strands-decider-2B-qwen3.5-v1-2610",
    "answers": {"consent": {"type": "noul", "noul": 0.8581}},
    "usage": {"input_tokens": 86, "output_tokens": 1},
    "latency_ms": 489.01,
}


def client_for(
    handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any
) -> HttpDeciderClient:
    return HttpDeciderClient(transport=httpx.MockTransport(handler), **kwargs)


def ok(body: dict[str, Any] = GOOD) -> Callable[[httpx.Request], httpx.Response]:
    return lambda request: httpx.Response(200, json=body)


class TestRequestShape:
    async def test_posts_state_and_questions_to_systemone(self) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=GOOD)

        async with client_for(handler) as client:
            await client.ask("the state", Q)

        assert seen[0].method == "POST"
        assert seen[0].url.path == "/v1/systemone"
        body = json.loads(seen[0].content)
        assert body["state"] == "the state"
        assert body["questions"]["consent"] == {
            "type": "noul",
            "instructions": "Did the user agree?",
            "criteria": {"true": "they agreed", "false": "they did not"},
        }

    async def test_criteria_are_omitted_when_not_given(self) -> None:
        assert "criteria" not in NoulQuestion("q").to_wire()

    async def test_all_questions_go_in_one_request(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={"answers": {k: {"type": "noul", "noul": 0.5} for k in ("a", "b", "c")}},
            )

        async with client_for(handler) as client:
            await client.ask("s", {k: NoulQuestion(k) for k in ("a", "b", "c")})
        assert calls == 1

    async def test_empty_question_set_is_rejected_locally(self) -> None:
        async with client_for(ok()) as client:
            with pytest.raises(ValueError, match="at least one"):
                await client.ask("s", {})


class TestParsing:
    async def test_returns_probabilities_and_timing(self) -> None:
        async with client_for(ok()) as client:
            r = await client.ask("s", Q)
        assert r.answers == {"consent": 0.8581}
        assert r.server_latency_ms == 489.01
        assert r.wall_latency_ms >= 0
        assert r.model == "strands-decider-2B-qwen3.5-v1-2610"
        assert r.input_tokens == 86

    async def test_missing_optional_fields_are_fine(self) -> None:
        body = {"answers": {"consent": {"type": "noul", "noul": 0.1}}}
        async with client_for(ok(body)) as client:
            r = await client.ask("s", Q)
        assert r.server_latency_ms is None
        assert r.model is None
        assert r.input_tokens is None

    async def test_extra_answers_are_ignored(self) -> None:
        body = {
            "answers": {
                "consent": {"type": "noul", "noul": 0.1},
                "bonus": {"type": "noul", "noul": 0.9},
            }
        }
        async with client_for(ok(body)) as client:
            r = await client.ask("s", Q)
        assert set(r.answers) == {"consent"}

    @pytest.mark.parametrize(
        "body",
        [
            {},
            {"answers": []},
            {"answers": {}},
            {"answers": {"consent": "yes"}},
            {"answers": {"consent": {"type": "choice", "choice": "a"}}},
            {"answers": {"consent": {"type": "noul"}}},
            {"answers": {"consent": {"type": "noul", "noul": "0.9"}}},
            {"answers": {"consent": {"type": "noul", "noul": True}}},
            {"answers": {"consent": {"type": "noul", "noul": 1.5}}},
            {"answers": {"consent": {"type": "noul", "noul": -0.1}}},
            {"answers": {"consent": {"type": "noul", "noul": None}}},
        ],
    )
    async def test_unusable_bodies_raise_bad_response(self, body: dict[str, Any]) -> None:
        async with client_for(ok(body)) as client:
            with pytest.raises(DeciderBadResponse):
                await client.ask("s", Q)

    async def test_nan_is_rejected(self) -> None:
        raw = b'{"answers": {"consent": {"type": "noul", "noul": NaN}}}'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=raw, headers={"content-type": "application/json"})

        async with client_for(handler) as client:
            with pytest.raises(DeciderBadResponse, match=r"not a probability|not a number"):
                await client.ask("s", Q)

    async def test_non_json_body(self) -> None:
        async with client_for(lambda r: httpx.Response(200, text="<html>oops</html>")) as client:
            with pytest.raises(DeciderBadResponse):
                await client.ask("s", Q)

    async def test_one_bad_answer_fails_the_whole_batch(self) -> None:
        body = {"answers": {"a": {"type": "noul", "noul": 0.9}, "b": {"type": "noul", "noul": 7}}}
        async with client_for(ok(body)) as client:
            with pytest.raises(DeciderBadResponse, match="'b'"):
                await client.ask("s", {"a": NoulQuestion("a"), "b": NoulQuestion("b")})


class TestHttpErrors:
    async def test_4xx_is_rejected_with_the_servers_detail(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={"detail": "prompt exceeds the context window"})

        async with client_for(handler) as client:
            with pytest.raises(DeciderRejected, match="context window") as info:
                await client.ask("s", Q)
        assert info.value.status_code == 422

    async def test_5xx_is_unavailable(self) -> None:
        async with client_for(lambda r: httpx.Response(503)) as client:
            with pytest.raises(DeciderUnavailable, match="503"):
                await client.ask("s", Q)

    async def test_4xx_with_non_json_body(self) -> None:
        async with client_for(lambda r: httpx.Response(400, text="bad")) as client:
            with pytest.raises(DeciderRejected, match="bad"):
                await client.ask("s", Q)


class TestTimeoutsAndRetries:
    async def test_read_timeout_is_a_timeout_and_is_not_retried(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("slow", request=request)

        async with client_for(handler, max_retries=3) as client:
            with pytest.raises(DeciderTimeout):
                await client.ask("s", Q)
        assert calls == 1

    async def test_connect_error_is_retried_then_succeeds(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json=GOOD)

        async with client_for(handler, max_retries=1) as client:
            r = await client.ask("s", Q)
        assert calls == 2
        assert r.answers["consent"] == 0.8581

    async def test_connect_error_gives_up_after_max_retries(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        async with client_for(handler, max_retries=2) as client:
            with pytest.raises(DeciderUnavailable, match="cannot connect"):
                await client.ask("s", Q)
        assert calls == 3

    async def test_zero_retries(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ConnectError("refused", request=request)

        async with client_for(handler, max_retries=0) as client:
            with pytest.raises(DeciderUnavailable):
                await client.ask("s", Q)
        assert calls == 1

    async def test_other_transport_errors_are_unavailable(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.RemoteProtocolError("garbled", request=request)

        async with client_for(handler) as client:
            with pytest.raises(DeciderUnavailable):
                await client.ask("s", Q)

    @pytest.mark.parametrize("kwargs", [{"timeout_s": 0}, {"timeout_s": -1}, {"max_retries": -1}])
    def test_invalid_settings_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            HttpDeciderClient(**kwargs)
