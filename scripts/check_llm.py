"""Connectivity test for the host-agent LLM (OpenAI-compatible endpoint).

Run:  python scripts/check_llm.py
Reads OPENAI_API_KEY, OPENAI_BASE_URL and MODEL_ID from the environment or a git-ignored .env.
The API key is never printed.

Checks, in order (stops at the first failure):
  1. plain chat completion
  2. OpenAI-style tool calling (the model returns a tool call)
  3. a Strands agent calls a dummy tool end to end
"""

from __future__ import annotations

import os
import sys
from typing import Any

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
MODEL_ID = os.environ.get("MODEL_ID", "")


def short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def check_env() -> bool:
    missing = [
        n
        for n, v in (
            ("OPENAI_API_KEY", API_KEY),
            ("OPENAI_BASE_URL", BASE_URL),
            ("MODEL_ID", MODEL_ID),
        )
        if not v
    ]
    if missing:
        print(f"[FAIL] missing in .env / environment: {', '.join(missing)}")
        return False
    print(f"[env] base_url={BASE_URL} model={MODEL_ID} api_key=<set, {len(API_KEY)} chars>")
    return True


def check_plain() -> bool:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=20,
        )
    except Exception as exc:
        print(f"[FAIL 1/3 plain completion] {short(exc)}")
        return False
    print(f"[PASS 1/3 plain completion] reply={resp.choices[0].message.content!r}")
    return True


def check_tool_call() -> bool:
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60)
    tools: Any = [
        {
            "type": "function",
            "function": {
                "name": "get_order_status",
                "description": "Look up the status of an order by id.",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            },
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": "What is the status of order A123?"}],
            tools=tools,
            max_tokens=200,
        )
    except Exception as exc:
        print(f"[FAIL 2/3 tool calling] {short(exc)}")
        return False
    calls = resp.choices[0].message.tool_calls or []
    if not calls:
        print(
            "[FAIL 2/3 tool calling] model answered without a tool call "
            f"(finish_reason={resp.choices[0].finish_reason}). The endpoint may drop `tools`."
        )
        return False
    fn: Any = getattr(calls[0], "function", None)
    print(f"[PASS 2/3 tool calling] {getattr(fn, 'name', '?')}({getattr(fn, 'arguments', '')})")
    return True


def check_strands() -> bool:
    try:
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel
    except ImportError as exc:
        print(f"[FAIL 3/3 strands agent] import error: {exc}. Run: make install")
        return False

    called: list[str] = []

    @tool
    def get_order_status(order_id: str) -> str:
        """Look up the status of an order by id."""
        called.append(order_id)
        return f"Order {order_id} is shipped."

    model = OpenAIModel(
        client_args={"api_key": API_KEY, "base_url": BASE_URL},
        model_id=MODEL_ID,
        params={"max_tokens": 300},
    )
    agent = Agent(model=model, tools=[get_order_status], callback_handler=None)
    try:
        result = agent("What is the status of order A123? Use the tool.")
    except Exception as exc:
        print(f"[FAIL 3/3 strands agent] {short(exc)}")
        return False
    if not called:
        print("[FAIL 3/3 strands agent] agent finished but never called the tool")
        return False
    print(f"[PASS 3/3 strands agent] tool called with {called}; final={str(result)[:120]!r}")
    return True


def main() -> int:
    if not check_env():
        return 1
    for step in (check_plain, check_tool_call, check_strands):
        if not step():
            return 1
    print("\nAll checks passed: the endpoint works for plain chat and tool calling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
