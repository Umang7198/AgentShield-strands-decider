from agentshield.core import Outcome
from agentshield.eval.metrics import EvalRow, expected_calibration_error, percentile, score


def _row(expected: Outcome, predicted: Outcome, p_ok: float = 0.9, gold_ok: bool = True) -> EvalRow:
    return EvalRow(
        scenario_id="x",
        family="f",
        tags=("benign",),
        mode="full",
        expected=expected,
        predicted=predicted,
        source=None,
        reason="",
        total_ms=10.0,
        check_scores=(("consent", p_ok, gold_ok),),
    )


def test_perfect_run() -> None:
    rows = [_row(Outcome.ALLOW, Outcome.ALLOW) for _ in range(4)] + [
        _row(Outcome.BLOCK, Outcome.BLOCK, p_ok=0.1, gold_ok=False)
    ]
    m = score(rows)
    assert m.accuracy == 1.0
    assert m.false_stop_rate == 0.0
    assert m.missed_violation_rate == 0.0


def test_missed_violation_and_false_stop() -> None:
    rows = [
        _row(Outcome.ALLOW, Outcome.BLOCK),
        _row(Outcome.CLARIFY, Outcome.ALLOW),
        _row(Outcome.ALLOW, Outcome.ALLOW),
    ]
    m = score(rows)
    assert m.false_stop_rate == 0.5
    assert m.false_block_rate == 0.5
    assert m.missed_violation_rate == 1.0


def test_percentile_and_ece() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    ece = expected_calibration_error([(0.9, True), (0.1, False), (0.9, True)])
    assert ece is not None
    assert ece < 0.2
    assert expected_calibration_error([]) is None
