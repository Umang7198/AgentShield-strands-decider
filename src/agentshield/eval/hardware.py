"""Machine + model metadata for a committed evaluation report."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any

from agentshield.decider.questions import QUESTION_SET_VERSION


def hardware_summary() -> dict[str, Any]:
    info: dict[str, Any] = {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if platform.system() == "Darwin":
        info["mac_chip"] = _sysctl("machdep.cpu.brand_string")
        info["mac_mem_gb"] = _bytes_to_gb(_sysctl("hw.memsize"))
    return info


def eval_metadata(*, backend: str, decider_model: str | None = None) -> dict[str, Any]:
    return {
        "backend": backend,
        "question_set_version": QUESTION_SET_VERSION,
        "decider_model": decider_model
        or os.environ.get("DECIDER_MODEL", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"),
        "hardware": hardware_summary(),
    }


def _sysctl(key: str) -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", key], check=False, capture_output=True, text=True, timeout=2
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = result.stdout.strip()
    return text or None


def _bytes_to_gb(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return round(int(raw) / (1024**3), 1)
    except ValueError:
        return None
