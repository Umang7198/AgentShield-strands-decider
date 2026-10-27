"""HTTP routes for the Hearth & Cable desk UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agentshield.approvals.errors import ApprovalError, ApprovalNotFound, ApprovalNotPending
from agentshield.shophelp.runtime import ShopHelpRuntime

FRONTEND = Path(__file__).resolve().parents[3] / "frontend"


class ChatIn(BaseModel):
    text: str = Field(min_length=1)


class ClaimIn(BaseModel):
    claim: str = Field(min_length=1)


class ResolveIn(BaseModel):
    approver: str = Field(default="Priya", min_length=1)
    note: str = ""


def mount_demo(app: FastAPI, runtime: ShopHelpRuntime) -> None:
    app.mount("/css", StaticFiles(directory=FRONTEND / "css"), name="desk-css")
    app.mount("/js", StaticFiles(directory=FRONTEND / "js"), name="desk-js")

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")

    @app.get("/review")
    def review() -> FileResponse:
        return FileResponse(FRONTEND / "index.html")

    @app.get("/demo/state")
    def state() -> dict[str, Any]:
        return runtime.state()

    @app.post("/demo/chat")
    def chat(body: ChatIn) -> dict[str, Any]:
        return runtime.chat(body.text)

    @app.post("/demo/claim")
    def claim(body: ClaimIn) -> dict[str, Any]:
        return runtime.check_claim(body.claim)

    @app.post("/demo/reset")
    def reset() -> dict[str, Any]:
        return runtime.reset()

    @app.post("/demo/approve/{approval_id}")
    def approve(approval_id: str, body: ResolveIn) -> dict[str, Any]:
        return _resolve(runtime, approval_id, "approve", body)

    @app.post("/demo/deny/{approval_id}")
    def deny(approval_id: str, body: ResolveIn) -> dict[str, Any]:
        return _resolve(runtime, approval_id, "deny", body)


def _resolve(
    runtime: ShopHelpRuntime, approval_id: str, action: str, body: ResolveIn
) -> dict[str, Any]:
    try:
        return runtime.resolve(approval_id, action, by=body.approver, note=body.note)
    except ApprovalNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ApprovalNotPending as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ApprovalError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
