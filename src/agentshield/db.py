"""Shared SQLAlchemy engine and the live-app database bundle.

``DATABASE_URL`` selects the backend. SQLite is the default for ``python app.py``.
Postgres is optional (``docker compose --profile postgres``). The default Docker image uses SQLite.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from sqlalchemy import String, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
from sqlalchemy.pool import StaticPool

if TYPE_CHECKING:
    from agentshield.audit.record import AuditRecord


def normalize_url(url: str) -> str:
    """Accept ``postgres://`` and plain ``postgresql://`` as psycopg 3 URLs."""
    cleaned = url.strip()
    if cleaned.startswith("postgres://"):
        cleaned = "postgresql://" + cleaned[len("postgres://") :]
    if cleaned.startswith("postgresql://") and "+" not in cleaned.split("://", 1)[0]:
        cleaned = "postgresql+psycopg://" + cleaned[len("postgresql://") :]
    return cleaned


def engine_from_url(url: str) -> Engine:
    normalized = normalize_url(url)
    kwargs: dict[str, Any] = {}
    if normalized.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if normalized in ("sqlite://", "sqlite:///:memory:"):
            kwargs["poolclass"] = StaticPool
    return create_engine(normalized, **kwargs)


def default_url(root: Path | None = None) -> str:
    """``DATABASE_URL`` if set, otherwise a SQLite file next to the project."""
    env = os.environ.get("DATABASE_URL", "").strip()
    if env:
        return env
    base = root if root is not None else Path.cwd()
    return f"sqlite:///{(base / 'agentshield.db').resolve()}"


def describe_url(url: str) -> str:
    """Log-safe summary: no passwords."""
    normalized = normalize_url(url)
    parsed = urlparse(normalized)
    if parsed.scheme.startswith("sqlite"):
        return normalized
    host = parsed.hostname or "localhost"
    db = parsed.path.lstrip("/") or "?"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{host}{port}/{db}"


class _MetaBase(DeclarativeBase):
    pass


class _MetaRow(_MetaBase):
    __tablename__ = "demo_meta"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(256))


class Database:
    """One engine: ShopHelp catalogue, append-only audit log, approval cards."""

    def __init__(self, url: str) -> None:
        from agentshield.approvals.store import SqlApprovalStore
        from agentshield.audit.store import SqlAuditLog
        from agentshield.shophelp.sql_store import SqlShopHelpStore

        self.url = normalize_url(url)
        self.engine = engine_from_url(self.url)
        _MetaBase.metadata.create_all(self.engine)
        self.audit = SqlAuditLog(self.engine)
        self.approvals = SqlApprovalStore(self.engine)
        self.shophelp = SqlShopHelpStore(self.engine)
        self.shophelp.ensure_seeded()

    @classmethod
    def from_url(cls, url: str) -> Database:
        return cls(url)

    @classmethod
    def from_env(cls, root: Path | None = None) -> Database:
        return cls(default_url(root))

    def get_meta(self, key: str) -> str | None:
        with Session(self.engine) as session:
            row = session.get(_MetaRow, key)
            return None if row is None else row.value

    def set_meta(self, key: str, value: str) -> None:
        with Session(self.engine) as session:
            session.merge(_MetaRow(key=key, value=value))
            session.commit()

    def tape_after(self) -> datetime | None:
        raw = self.get_meta("tape_after")
        if not raw:
            return None
        return datetime.fromisoformat(raw)

    def mark_tape_reset(self) -> None:
        """Desk Reset: keep audit history, hide it from the current tape."""
        self.set_meta("tape_after", datetime.now(UTC).isoformat())


def query_tape(
    db: Database,
    session_id: str,
    *,
    limit: int = 100,
) -> list[AuditRecord]:
    """Newest-first audit rows that belong on the current desk tape."""
    after = db.tape_after()
    rows = db.audit.query(session_id=session_id, limit=limit)
    if after is not None:
        rows = [row for row in rows if row.timestamp > after]
    return rows
