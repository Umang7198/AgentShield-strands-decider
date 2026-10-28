"""Load frozen decision-matrix thresholds from YAML (chosen on val, applied on test)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from agentshield.core.matrix import KindThresholds, Thresholds

THRESHOLDS_PATH = Path(__file__).resolve().parents[3] / "policies" / "thresholds.yaml"


def load_thresholds(path: Path | None = None) -> Thresholds:
    target = path or THRESHOLDS_PATH
    if not target.is_file():
        return Thresholds()
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"thresholds file {target} is not a mapping")
    return thresholds_from_dict(data)


def thresholds_from_dict(data: dict[str, Any]) -> Thresholds:
    low = data.get("low") or {}
    medium = data.get("medium") or {}
    return Thresholds(
        low=KindThresholds(**low)
        if low
        else KindThresholds(
            argument_grounding=0.55, consent=0.60, premature_action=0.50, claim_grounding=0.60
        ),
        medium=KindThresholds(**medium)
        if medium
        else KindThresholds(
            argument_grounding=0.65, consent=0.80, premature_action=0.60, claim_grounding=0.75
        ),
        block_below=float(data.get("block_below", 0.05)),
    )


def dump_thresholds(thresholds: Thresholds, *, source: str, notes: str = "") -> dict[str, Any]:
    return {
        "version": 1,
        "source": source,
        "notes": notes,
        "block_below": thresholds.block_below,
        "low": thresholds.low.model_dump(mode="json"),
        "medium": thresholds.medium.model_dump(mode="json"),
    }


def write_thresholds(
    thresholds: Thresholds,
    path: Path | None = None,
    *,
    source: str,
    notes: str = "",
) -> None:
    target = path or THRESHOLDS_PATH
    payload = dump_thresholds(thresholds, source=source, notes=notes)
    text = yaml.safe_dump(payload, sort_keys=False)
    target.write_text(text, encoding="utf-8")
