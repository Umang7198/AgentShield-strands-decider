from pathlib import Path

import pytest

from agentshield.audit import FanoutAuditLog, MemoryAuditLog, SqlAuditLog
from agentshield.core import Outcome
from tests.audit.test_record import _record


class _Exploding:
    def record(self, record: object) -> None:
        raise RuntimeError("disk full")


def _roundtrip(store: MemoryAuditLog | SqlAuditLog) -> None:
    a = _record({"order_id": "A1"})
    b = _record({"order_id": "B2"})
    # timestamps can match; query is newest-first by timestamp then insert order for memory
    store.record(a)
    store.record(b)
    assert store.get(a.id) == a
    assert store.get("missing") is None
    by_session = store.query(session_id="s1")
    assert {r.id for r in by_session} == {a.id, b.id}
    assert store.query(tool="nope") == []
    assert store.query(outcome=Outcome.ALLOW, limit=1)


class TestMemory:
    def test_roundtrip_and_filters(self) -> None:
        log = MemoryAuditLog()
        _roundtrip(log)
        assert len(log) == 2

    def test_query_is_newest_first(self) -> None:
        log = MemoryAuditLog()
        first, second = _record(), _record()
        log.record(first)
        log.record(second)
        assert [r.id for r in log.query()] == [second.id, first.id]


class TestSqlite:
    def test_in_memory_roundtrip(self) -> None:
        _roundtrip(SqlAuditLog.memory())

    def test_file_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.db"
        store = SqlAuditLog.sqlite(path)
        rec = _record({"amount": 40})
        store.record(rec)
        reopened = SqlAuditLog.sqlite(path)
        got = reopened.get(rec.id)
        assert got is not None
        assert got.arguments == {"amount": 40}
        assert got.outcome is Outcome.ALLOW

    def test_has_no_update_or_delete(self) -> None:
        store = SqlAuditLog.memory()
        assert not hasattr(store, "update")
        assert not hasattr(store, "delete")
        rec = _record()
        store.record(rec)
        rec.model_copy(update={"reason": "tampered"})
        assert store.get(rec.id) is not None
        assert store.get(rec.id).reason == "ok"  # type: ignore[union-attr]


class TestFanout:
    def test_writes_to_every_sink(self) -> None:
        a, b = MemoryAuditLog(), MemoryAuditLog()
        rec = _record()
        FanoutAuditLog([a, b]).record(rec)
        assert a.get(rec.id) == rec
        assert b.get(rec.id) == rec

    def test_attempts_the_rest_then_raises(self) -> None:
        ok = MemoryAuditLog()
        with pytest.raises(RuntimeError, match="disk full"):
            FanoutAuditLog([_Exploding(), ok]).record(_record())
        assert len(ok) == 1
