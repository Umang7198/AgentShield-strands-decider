"""HTTP API and a minimal review page for pending approvals."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse

from agentshield.approvals.desk import ApprovalDesk
from agentshield.approvals.errors import ApprovalError, ApprovalNotFound, ApprovalNotPending
from agentshield.approvals.models import Approval, ApprovalStatus, ResolveRequest

_REVIEW_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"/><title>AgentShield approvals</title>
<style>
 body{font:14px/1.4 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#111}
 .card{border:1px solid #ddd;border-radius:8px;padding:1rem 1.2rem;margin:1rem 0}
 .meta{color:#555;font-size:12px}
 pre{white-space:pre-wrap;background:#f6f6f6;padding:.6rem;border-radius:4px}
 button{margin-right:.4rem;padding:.3rem .7rem}
</style></head><body>
<h1>Pending approvals</h1>
<p class="meta">Approve or deny high-risk tool calls. Denied and expired calls never run.</p>
<div id="list">Loading…</div>
<script>
const token = localStorage.getItem("approval_token") || "";
const approver = localStorage.getItem("approver") || prompt("Your name (recorded on the decision)") || "reviewer";
localStorage.setItem("approver", approver);
async function load(){
  const r = await fetch("/approvals?status=pending", {headers: headers()});
  const items = await r.json();
  const el = document.getElementById("list");
  if(!items.length){ el.innerHTML = "<p>None pending.</p>"; return; }
  el.innerHTML = items.map(card).join("");
}
function headers(){
  const h = {"Content-Type":"application/json", "X-Approver-Id": approver};
  if(token) h["Authorization"] = "Bearer "+token;
  return h;
}
function card(a){
  const args = JSON.stringify(a.arguments, null, 2);
  return `<div class="card" id="${a.id}">
    <strong>${a.tool}</strong> · ${a.risk} · ${a.source}
    <div class="meta">${a.id} · session ${a.session_id} · expires ${a.expires_at}</div>
    <p>${a.reason}</p>
    <pre>${args}</pre>
    <pre>${a.excerpt || "(no excerpt)"}</pre>
    <button onclick="decide('${a.id}','approve')">Approve</button>
    <button onclick="decide('${a.id}','deny')">Deny</button>
  </div>`;
}
async function decide(id, action){
  const r = await fetch("/approvals/"+id+"/"+action, {
    method:"POST", headers: headers(),
    body: JSON.stringify({approver, note: ""})
  });
  if(!r.ok){ alert(await r.text()); return; }
  load();
}
load();
</script></body></html>
"""


def create_app(
    desk: ApprovalDesk,
    *,
    token: str | None = None,
    runtime: Any | None = None,
) -> FastAPI:
    """``token`` is the shared secret. If omitted, ``APPROVAL_TOKEN`` is read; if still empty,
    the API is open (dev) but every decision still records an approver id.

    Pass a :class:`~agentshield.shophelp.runtime.ShopHelpRuntime` to serve the Hearth & Cable
    desk at ``/`` and ``/review``.
    """
    expected = token if token is not None else os.environ.get("APPROVAL_TOKEN", "")
    app = FastAPI(title="AgentShield", version="0.1.0")

    def auth(authorization: str | None = Header(default=None)) -> None:
        if not expected:
            return
        if authorization != f"Bearer {expected}":
            raise HTTPException(status_code=401, detail="invalid approval token")

    def parse_status(value: str | None) -> ApprovalStatus | None:
        if value is None:
            return None
        try:
            return ApprovalStatus(value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"unknown status {value!r}") from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/approvals", response_model=list[Approval], dependencies=[Depends(auth)])
    def list_approvals(
        status: str | None = Query(default="pending"),
        session_id: str | None = None,
    ) -> list[Approval]:
        return desk.query(status=parse_status(status), session_id=session_id)

    @app.get("/approvals/{approval_id}", response_model=Approval, dependencies=[Depends(auth)])
    def get_approval(approval_id: str) -> Approval:
        try:
            return desk.get(approval_id)
        except ApprovalNotFound as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/approvals/{approval_id}/approve", dependencies=[Depends(auth)])
    def approve(approval_id: str, body: ResolveRequest) -> dict[str, Any]:
        return _resolve(desk.approve, approval_id, body)

    @app.post("/approvals/{approval_id}/deny", dependencies=[Depends(auth)])
    def deny(approval_id: str, body: ResolveRequest) -> dict[str, Any]:
        return _resolve(desk.deny, approval_id, body)

    if runtime is not None:
        from agentshield.api.demo import mount_demo

        mount_demo(app, runtime)
    else:

        @app.get("/review", response_class=HTMLResponse)
        def review() -> str:
            return _REVIEW_HTML

    return app


def _resolve(fn: Any, approval_id: str, body: ResolveRequest) -> dict[str, Any]:
    try:
        rec, _result = fn(approval_id, by=body.approver, note=body.note)
    except ApprovalNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalNotPending as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ApprovalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    dumped: dict[str, Any] = rec.model_dump(mode="json")
    return dumped
