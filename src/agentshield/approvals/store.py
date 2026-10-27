"""Approval storage. Unlike the audit log, rows are updated when a human decides."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, runtime_checkable

from sqlalchemy import DateTime, String, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool
from sqlalchemy.types import JSON

from agentshield.approvals.errors import ApprovalNotFound
from agentshield.approvals.models import Approval, ApprovalStatus


@runtime_checkable
class ApprovalStore(Protocol):
    def put(self, approval: Approval) -> None: ...
    def get(self, approval_id: str) -> Approval: ...
    def query(
        self,
        *,
        status: ApprovalStatus | None = None,
        session_id: str | None = None,
        tool_use_id: str | None = None,
        limit: int = 100,
    ) -> list[Approval]: ...


class MemoryApprovalStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._rows: dict[str, Approval] = {}

    def put(self, approval: Approval) -> None:
        with self._lock:
            self._rows[approval.id] = approval

    def get(self, approval_id: str) -> Approval:
        with self._lock:
            rec = self._rows.get(approval_id)
        if rec is None:
            raise ApprovalNotFound(approval_id)
        return rec

    def query(
        self,
        *,
        status: ApprovalStatus | None = None,
        session_id: str | None = None,
        tool_use_id: str | None = None,
        limit: int = 100,
    ) -> list[Approval]:
        with self._lock:
            rows = list(self._rows.values())
        rows.sort(key=lambda r: r.created_at, reverse=True)
        out: list[Approval] = []
        for row in rows:
            if status is not None and row.status != status:
                continue
            if session_id is not None and row.session_id != session_id:
                continue
            if tool_use_id is not None and row.tool_use_id != tool_use_id:
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out


class _Base(DeclarativeBase):
    pass


class _ApprovalRow(_Base):
    __tablename__ = "approvals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    tool_use_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    body: Mapped[dict[str, Any]] = mapped_column(JSON)


class SqlApprovalStore:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _Base.metadata.create_all(engine)

    @classmethod
    def from_url(cls, url: str) -> SqlApprovalStore:
        return cls(_engine_from_url(url))

    @classmethod
    def sqlite(cls, path: str | Path) -> SqlApprovalStore:
        return cls.from_url(f"sqlite:///{Path(path)}")

    @classmethod
    def memory(cls) -> SqlApprovalStore:
        return cls.from_url("sqlite://")

    def put(self, approval: Approval) -> None:
        row = _ApprovalRow(
            id=approval.id,
            created_at=approval.created_at,
            status=approval.status.value,
            session_id=approval.session_id,
            tool_use_id=approval.tool_use_id,
            body=approval.model_dump(mode="json"),
        )
        with Session(self._engine) as session:
            session.merge(row)
            session.commit()

    def get(self, approval_id: str) -> Approval:
        with Session(self._engine) as session:
            row = session.get(_ApprovalRow, approval_id)
            if row is None:
                raise ApprovalNotFound(approval_id)
            return Approval.model_validate(row.body)

    def query(
        self,
        *,
        status: ApprovalStatus | None = None,
        session_id: str | None = None,
        tool_use_id: str | None = None,
        limit: int = 100,
    ) -> list[Approval]:
        stmt = select(_ApprovalRow).order_by(_ApprovalRow.created_at.desc()).limit(limit)
        if status is not None:
            stmt = stmt.where(_ApprovalRow.status == status.value)
        if session_id is not None:
            stmt = stmt.where(_ApprovalRow.session_id == session_id)
        if tool_use_id is not None:
            stmt = stmt.where(_ApprovalRow.tool_use_id == tool_use_id)
        with Session(self._engine) as session:
            return [Approval.model_validate(row.body) for row in session.scalars(stmt)]


def _engine_from_url(url: str) -> Engine:
    kwargs: dict[str, Any] = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(url, **kwargs)
