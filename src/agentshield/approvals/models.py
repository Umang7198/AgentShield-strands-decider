"""Pending and resolved human approvals."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentshield.audit.record import AuditCheck
from agentshield.core.models import DecisionSource, RiskTier


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class Approval(BaseModel):
    """One human review of a proposed tool call.

    Mutable in the store (status changes) but replaced as a whole object, never edited in place.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    created_at: datetime
    expires_at: datetime
    status: ApprovalStatus = ApprovalStatus.PENDING
    session_id: str
    agent_id: str
    tool: str
    arguments: dict[str, Any]
    tool_use_id: str | None = None
    risk: RiskTier
    source: DecisionSource
    reason: str
    rule_id: str | None = None
    checks: tuple[AuditCheck, ...] = ()
    excerpt: str = ""
    audit_id: str | None = None
    interrupt_id: str | None = None
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    note: str = ""

    @property
    def prompt(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"[{self.id[:8]}] Approve {self.tool}({args})? {self.reason}"


class ResolveRequest(BaseModel):
    approver: str = Field(min_length=1)
    note: str = ""
