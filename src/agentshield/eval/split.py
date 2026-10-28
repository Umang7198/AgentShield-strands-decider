"""Val/test split that keeps paraphrase templates on the same side.

Gold labels stay on the full pipeline. Groups are ``family`` + expected outcome, and every
id that differs only by a trailing integer (``lookup/stated/A1001/0`` vs ``/1``) is one
leak-group so near-duplicates cannot sit in both val and test.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from agentshield.eval.dataset import build_dataset
from agentshield.eval.scenario import Scenario

SplitName = Literal["val", "test"]

SPLITS_PATH = Path(__file__).resolve().parents[3] / "eval" / "splits.json"
SEED = 42
VAL_FRAC = 0.4


def leak_group(scenario_id: str) -> str:
    """Strip a trailing paraphrase index (``.../0``) so those rows travel together."""
    parts = scenario_id.split("/")
    if parts and parts[-1].isdigit():
        return "/".join(parts[:-1])
    return scenario_id


def assign_splits(
    scenarios: Sequence[Scenario],
    *,
    seed: int = SEED,
    val_frac: float = VAL_FRAC,
) -> dict[str, SplitName]:
    groups: dict[str, list[Scenario]] = defaultdict(list)
    for scenario in scenarios:
        groups[leak_group(scenario.id)].append(scenario)

    strata: dict[tuple[str, str], list[str]] = defaultdict(list)
    for gid, members in groups.items():
        strata[(members[0].family, members[0].expected.value)].append(gid)

    mapping: dict[str, SplitName] = {}
    rng = random.Random(seed)
    for _key, gids in sorted(strata.items()):
        ordered = list(gids)
        rng.shuffle(ordered)
        if len(ordered) == 1:
            split: SplitName = "val" if rng.random() < val_frac else "test"
            for scenario in groups[ordered[0]]:
                mapping[scenario.id] = split
            continue
        n_val = min(len(ordered) - 1, max(1, round(len(ordered) * val_frac)))
        for i, gid in enumerate(ordered):
            split = "val" if i < n_val else "test"
            for scenario in groups[gid]:
                mapping[scenario.id] = split
    missing = [s.id for s in scenarios if s.id not in mapping]
    if missing:
        raise RuntimeError(f"split missed {len(missing)} scenarios, e.g. {missing[0]}")
    return mapping


def write_splits(path: Path | None = None) -> dict[str, SplitName]:
    scenarios = build_dataset()
    mapping = assign_splits(scenarios)
    target = path or SPLITS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": SEED,
        "val_frac": VAL_FRAC,
        "n": len(mapping),
        "assignment": mapping,
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return mapping


def load_splits(path: Path | None = None) -> dict[str, SplitName]:
    target = path or SPLITS_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    assignment = raw["assignment"]
    return {key: ("val" if value == "val" else "test") for key, value in assignment.items()}


def scenarios_in(
    scenarios: Sequence[Scenario],
    split: SplitName | Literal["all"],
    *,
    mapping: dict[str, SplitName] | None = None,
) -> tuple[Scenario, ...]:
    if split == "all":
        return tuple(scenarios)
    assigned = mapping or load_splits()
    return tuple(s for s in scenarios if assigned[s.id] == split)
