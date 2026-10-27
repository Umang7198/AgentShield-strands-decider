from agentshield.eval.dataset import build_dataset
from agentshield.eval.metrics import score
from agentshield.eval.report import render
from agentshield.eval.runner import Mode, OracleHarness


async def test_oracle_full_pipeline_matches_gold_labels() -> None:
    data = build_dataset()
    rows = (await OracleHarness().run(data, (Mode.FULL,)))["full"]
    m = score(rows)
    assert m.n == len(data)
    assert m.accuracy == 1.0
    assert m.false_stop_rate == 0.0
    assert m.missed_violation_rate == 0.0
    assert m.false_block_rate == 0.0


async def test_policy_only_misses_unconsented_calls() -> None:
    data = [s for s in build_dataset() if "unconsented" in s.tags]
    rows = (await OracleHarness().run(data, (Mode.POLICY,)))["policy"]
    m = score(rows)
    assert m.missed_violation_rate == 1.0
    assert m.accuracy == 0.0


async def test_no_gate_never_stops_anything() -> None:
    data = build_dataset()[:40]
    rows = (await OracleHarness().run(data, (Mode.NONE,)))["none"]
    m = score(rows)
    assert all(r.predicted.value == "allow" for r in rows)
    assert m.false_stop_rate == 0.0
    assert m.missed_violation_rate > 0


async def test_report_mentions_modes() -> None:
    data = build_dataset()[:5]
    results = await OracleHarness().run(data, (Mode.FULL, Mode.NONE))
    text = render(
        metrics={k: score(v) for k, v in results.items()},
        n_scenarios=5,
        backend="oracle",
    )
    assert "Mode `full`" in text
    assert "Ablation comparison" in text
