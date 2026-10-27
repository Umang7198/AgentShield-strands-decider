"""The human-approval desk: create, list, expire, approve, deny, and resume the agent."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from agentshield.approvals.errors import ApprovalNotPending
from agentshield.approvals.excerpt import excerpt
from agentshield.approvals.models import Approval, ApprovalStatus
from agentshield.approvals.store import ApprovalStore, MemoryApprovalStore
from agentshield.audit.record import AuditCheck
from agentshield.decider.state import Message
from agentshield.pipeline import Verdict
from agentshield.policy.session import SessionState
from agentshield.telemetry import Telemetry

YES = "yes"
NO = "no"

ResumeAgent = Callable[[Any], Any]


class ApprovalDesk:
    """In-process workflow. The FastAPI layer is a thin wrapper around this.

    Register the live Strands agent with :meth:`register_agent` so approve/deny can resume it.
    If no agent is registered, the decision is still recorded and the caller must resume.
    """

    def __init__(
        self,
        store: ApprovalStore | None = None,
        *,
        timeout: timedelta = timedelta(minutes=15),
        telemetry: Telemetry | None = None,
    ) -> None:
        self._store = store or MemoryApprovalStore()
        self._timeout = timeout
        self._telemetry = telemetry
        self._agents: dict[str, ResumeAgent] = {}

    def register_agent(self, session_id: str, agent: ResumeAgent) -> None:
        self._agents[session_id] = agent

    def submit(
        self,
        verdict: Verdict,
        session: SessionState,
        messages: Sequence[Message] = (),
        *,
        agent_id: str = "agentshield",
    ) -> Approval:
        now = datetime.now(UTC)
        checks: tuple[AuditCheck, ...] = ()
        if verdict.run is not None:
            checks = tuple(AuditCheck.from_check(c) for c in verdict.run.checks)
        elif verdict.decision.checks:
            checks = tuple(AuditCheck.from_check(c) for c in verdict.decision.checks)
        rec = Approval(
            id=str(uuid.uuid4()),
            created_at=now,
            expires_at=now + self._timeout,
            session_id=session.session_id,
            agent_id=agent_id,
            tool=verdict.call.name,
            arguments=dict(verdict.call.arguments),
            tool_use_id=verdict.call.tool_use_id,
            risk=verdict.decision.risk,
            source=verdict.decision.source,
            reason=verdict.decision.reason,
            rule_id=verdict.decision.rule_id,
            checks=checks,
            excerpt=excerpt(messages),
            audit_id=verdict.audit_id,
        )
        self._store.put(rec)
        return rec

    def bind_interrupt(self, approval_id: str, interrupt_id: str) -> Approval:
        rec = self._store.get(approval_id)
        if rec.interrupt_id == interrupt_id:
            return rec
        updated = rec.model_copy(update={"interrupt_id": interrupt_id})
        self._store.put(updated)
        return updated

    def get(self, approval_id: str) -> Approval:
        self.expire_due()
        return self._store.get(approval_id)

    def by_tool_use(self, tool_use_id: str | None) -> Approval | None:
        if not tool_use_id:
            return None
        rows = self._store.query(tool_use_id=tool_use_id, limit=1)
        return rows[0] if rows else None

    def pending(self, *, session_id: str | None = None) -> list[Approval]:
        return self.query(status=ApprovalStatus.PENDING, session_id=session_id)

    def query(
        self,
        *,
        status: ApprovalStatus | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[Approval]:
        self.expire_due()
        return self._store.query(status=status, session_id=session_id, limit=limit)

    def record(
        self,
        approval_id: str,
        status: ApprovalStatus,
        *,
        by: str,
        note: str = "",
    ) -> Approval:
        """Mark resolved without resuming the agent (the handler is already on the resume path)."""
        rec = self._store.get(approval_id)
        if rec.status is not ApprovalStatus.PENDING:
            raise ApprovalNotPending(approval_id, rec.status.value)
        wait_ms = (datetime.now(UTC) - rec.created_at).total_seconds() * 1000
        updated = rec.model_copy(
            update={
                "status": status,
                "resolved_at": datetime.now(UTC),
                "resolved_by": by,
                "note": note,
            }
        )
        self._store.put(updated)
        if self._telemetry is not None:
            self._telemetry.record_approval(status.value, wait_ms)
        return updated

    def approve(self, approval_id: str, *, by: str, note: str = "") -> tuple[Approval, Any | None]:
        rec = self.record(approval_id, ApprovalStatus.APPROVED, by=by, note=note)
        return rec, self._resume(rec, YES)

    def deny(self, approval_id: str, *, by: str, note: str = "") -> tuple[Approval, Any | None]:
        rec = self.record(approval_id, ApprovalStatus.DENIED, by=by, note=note)
        return rec, self._resume(rec, NO)

    def expire_due(self, *, now: datetime | None = None) -> list[Approval]:
        now = now or datetime.now(UTC)
        expired: list[Approval] = []
        for rec in self._store.query(status=ApprovalStatus.PENDING, limit=1000):
            if rec.expires_at <= now:
                updated = self.record(
                    rec.id, ApprovalStatus.EXPIRED, by="timeout", note="approval timed out"
                )
                self._resume(updated, NO)
                expired.append(updated)
        return expired

    def _resume(self, rec: Approval, response: str) -> Any | None:
        agent = self._agents.get(rec.session_id)
        if agent is None or rec.interrupt_id is None:
            return None
        return agent(
            [{"interruptResponse": {"interruptId": rec.interrupt_id, "response": response}}]
        )


def interrupt_was_yes(response: Any) -> bool:
    """Same rule Strands uses for Confirm: True, 'y' or 'yes'."""
    if response is True:
        return True
    if isinstance(response, str):
        return response.lower().strip() in {"y", "yes"}
    return False
