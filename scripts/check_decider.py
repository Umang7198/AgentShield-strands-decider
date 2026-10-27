"""Smoke test for the local Strands Decider server.

Run:  python scripts/check_decider.py
Needs the server running (`make decider`). Reads DECIDER_URL from the environment or .env.
"""

from __future__ import annotations

import os
import statistics
import sys
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

URL = os.environ.get("DECIDER_URL", "http://127.0.0.1:8099").rstrip("/")

STATE = "User: where is my order?\nAgent: calling cancel_order(order_id=4417)"
QUESTIONS = {
    "consent": {
        "type": "noul",
        "instructions": "Did the user ask for or agree to cancelling the order?",
    },
    "arg_grounded": {
        "type": "noul",
        "instructions": "Is the order id 4417 stated by the user?",
    },
}


def main() -> int:
    try:
        health = httpx.get(f"{URL}/health", timeout=5)
        print(f"[health] {health.status_code}")
    except httpx.HTTPError as exc:
        print(f"[FAIL] cannot reach {URL}: {exc}\nStart it with: make decider")
        return 1

    latencies: list[float] = []
    answers: dict = {}
    for i in range(6):
        start = time.perf_counter()
        resp = httpx.post(
            f"{URL}/v1/systemone",
            json={"state": STATE, "questions": QUESTIONS},
            timeout=30,
        )
        wall_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
        body = resp.json()
        answers = body["answers"]
        latencies.append(wall_ms)
        print(f"[call {i + 1}] wall={wall_ms:7.1f} ms  server={body.get('latency_ms')} ms")

    print("\nanswers (noul = probability of YES):")
    for name, ans in answers.items():
        print(f"  {name}: {ans.get('noul')}")

    warm = latencies[1:]
    print(
        f"\ncold={latencies[0]:.0f} ms  warm median={statistics.median(warm):.0f} ms  "
        f"warm max={max(warm):.0f} ms"
    )
    print("[PASS] decider server OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
