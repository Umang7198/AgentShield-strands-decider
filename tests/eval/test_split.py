from collections import Counter

from agentshield.eval.dataset import build_dataset
from agentshield.eval.split import (
    assign_splits,
    leak_group,
    load_splits,
    scenarios_in,
    write_splits,
)


def test_paraphrase_rows_share_a_leak_group() -> None:
    assert leak_group("lookup/stated/A1001/0") == leak_group("lookup/stated/A1001/2")
    assert leak_group("lookup/stated/A1001/0") != leak_group("lookup/stated/A2002/0")
    assert leak_group("profile/invented/C42") == "profile/invented/C42"


def test_frozen_split_covers_every_scenario_without_leakage() -> None:
    data = build_dataset()
    mapping = write_splits()  # refresh committed file if the dataset grew
    frozen = load_splits()
    assert set(frozen) == {s.id for s in data}
    assert frozen == mapping
    groups: dict[str, set[str]] = {}
    for sid, split in frozen.items():
        groups.setdefault(leak_group(sid), set()).add(split)
    assert all(len(sides) == 1 for sides in groups.values())
    counts = Counter(frozen.values())
    assert counts["val"] > 40
    assert counts["test"] > 60
    val = scenarios_in(data, "val")
    test = scenarios_in(data, "test")
    assert len(val) + len(test) == len(data)
    assert {s.id for s in val}.isdisjoint({s.id for s in test})
    val_families = {s.family for s in val}
    test_families = {s.family for s in test}
    # Stratification should put the large families on both sides.
    assert "lookup_stated" in val_families and "lookup_stated" in test_families


def test_assign_splits_is_deterministic() -> None:
    data = build_dataset()
    a = assign_splits(data, seed=42)
    b = assign_splits(data, seed=42)
    assert a == b
