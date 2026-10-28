from agentshield.core.matrix import Thresholds
from agentshield.core.models import Outcome
from agentshield.core.thresholds_io import load_thresholds, thresholds_from_dict
from agentshield.eval.calibrate import candidate_thresholds, pick_winner, rescore
from agentshield.eval.dataset import build_dataset
from agentshield.eval.metrics import Metrics, score
from agentshield.eval.runner import Mode, OracleHarness


def test_frozen_file_loads() -> None:
    loaded = load_thresholds()
    assert 0.0 <= loaded.block_below < loaded.low.consent
    assert loaded.low.argument_grounding <= loaded.medium.argument_grounding


def test_thresholds_from_dict_roundtrip() -> None:
    t = Thresholds()
    again = thresholds_from_dict(
        {
            "block_below": t.block_below,
            "low": t.low.model_dump(),
            "medium": t.medium.model_dump(),
        }
    )
    assert again == t


async def test_rescore_does_not_override_policy_blocks() -> None:
    data = [s for s in build_dataset() if s.expected is Outcome.BLOCK][:8]
    rows = (await OracleHarness().run(data, (Mode.FULL,)))["full"]
    policy_rows = [r for r in rows if r.source == "policy"]
    assert policy_rows
    rescored = rescore(policy_rows, data, Thresholds())
    assert [r.predicted for r in rescored] == [r.predicted for r in policy_rows]
    assert all(r.predicted is Outcome.BLOCK for r in rescored)


def test_pick_winner_prefers_fewer_missed_violations() -> None:
    a = Thresholds()
    b = Thresholds()
    worse = Metrics(
        n=10,
        correct=8,
        accuracy=0.8,
        false_stop_rate=0.0,
        false_block_rate=0.0,
        missed_violation_rate=0.4,
        stop_precision=1.0,
        stop_recall=0.6,
        ece=None,
        latency_p50_ms=1,
        latency_p95_ms=1,
        latency_p99_ms=1,
        mean_ms=1,
        by_family={},
        by_tag={},
        checks=(),
        confusion={},
    )
    better = worse.model_copy(update={"missed_violation_rate": 0.1, "accuracy": 0.7})
    winner, metrics = pick_winner(((a, worse), (b, better)))
    assert winner is b
    assert metrics.missed_violation_rate == 0.1


def test_candidate_grid_is_valid() -> None:
    grid = candidate_thresholds()
    assert len(grid) > 10
    for t in grid:
        assert t.block_below < t.low.consent
        assert t.low.consent <= t.medium.consent


async def test_oracle_rescore_keeps_perfect_accuracy() -> None:
    data = build_dataset()[:20]
    rows = (await OracleHarness().run(data, (Mode.FULL,)))["full"]
    assert score(rescore(rows, data, Thresholds())).accuracy == 1.0
