"""Append-only audit storage.

The public API can insert and query. It has no update or delete, so a caller cannot rewrite
history even if they have a handle to the store. SQLite is the default; any SQLAlchemy URL
works, including Postgres once a driver is installed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import DateTime, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON

from agentshield.audit.record import AuditRecord
from agentshield.core.models import Outcome


@runtime_checkable
class AuditSink(Protocol):
    def record(self, record: AuditRecord) -> None: ...


class MemoryAuditLog:
    """Process-local log, useful in tests and as a fallback when no database is configured."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._rows: list[AuditRecord] = []

    def record(self, record: AuditRecord) -> None:
        with self._lock:
            self._rows.append(record)

    def get(self, record_id: str) -> AuditRecord | None:
        with self._lock:
            return next((r for r in self._rows if r.id == record_id), None)

    def query(
        self,
        *,
        session_id: str | None = None,
        outcome: Outcome | None = None,
        tool: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        with self._lock:
            rows = list(self._rows)
        rows.reverse()  # newest first
        out = []
        for row in rows:
            if session_id is not None and row.session_id != session_id:
                continue
            if outcome is not None and row.outcome != outcome:
                continue
            if tool is not None and row.tool != tool:
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._rows)


class _Base(DeclarativeBase):
    pass


class _AuditRow(_Base):
    __tablename__ = "audit_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    outcome: Mapped[str] = mapped_column(String(32), index=True)
    tool: Mapped[str] = mapped_column(String(128), index=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSON)


class SqlAuditLog:
    """Append-only SQL store. Schema is created on init if it is missing."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _Base.metadata.create_all(engine)

    @classmethod
    def from_url(cls, url: str) -> SqlAuditLog:
        return cls(_engine_from_url(url))

    @classmethod
    def sqlite(cls, path: str | Path) -> SqlAuditLog:
        return cls.from_url(f"sqlite:///{Path(path)}")

    @classmethod
    def memory(cls) -> SqlAuditLog:
        return cls.from_url("sqlite://")

    def record(self, record: AuditRecord) -> None:
        row = _AuditRow(
            id=record.id,
            timestamp=record.timestamp,
            session_id=record.session_id,
            outcome=record.outcome.value,
            tool=record.tool,
            body=record.model_dump(mode="json"),
        )
        with Session(self._engine) as session:
            session.add(row)
            session.commit()

    def get(self, record_id: str) -> AuditRecord | None:
        with Session(self._engine) as session:
            row = session.get(_AuditRow, record_id)
            return AuditRecord.model_validate(row.body) if row else None

    def query(
        self,
        *,
        session_id: str | None = None,
        outcome: Outcome | None = None,
        tool: str | None = None,
        limit: int = 100,
    ) -> list[AuditRecord]:
        stmt = select(_AuditRow).order_by(_AuditRow.timestamp.desc()).limit(limit)
        if session_id is not None:
            stmt = stmt.where(_AuditRow.session_id == session_id)
        if outcome is not None:
            stmt = stmt.where(_AuditRow.outcome == outcome.value)
        if tool is not None:
            stmt = stmt.where(_AuditRow.tool == tool)
        with Session(self._engine) as session:
            return [AuditRecord.model_validate(row.body) for row in session.scalars(stmt)]


class FanoutAuditLog:
    """Write to every sink. Attempts all of them even if one fails, then raises the first error."""

    def __init__(self, sinks: Sequence[AuditSink]) -> None:
        self._sinks = tuple(sinks)

    def record(self, record: AuditRecord) -> None:
        errors: list[Exception] = []
        for sink in self._sinks:
            try:
                sink.record(record)
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def _engine_from_url(url: str) -> Engine:
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)
