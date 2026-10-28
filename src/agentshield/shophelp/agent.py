"""Wire ShopHelp tools, the ShopHelp policy and the Strands handler into one agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent

from agentshield.approvals.desk import ApprovalDesk
from agentshield.audit.store import AuditSink
from agentshield.core.thresholds_io import load_thresholds
from agentshield.decider.checker import DeciderChecker
from agentshield.integrations.strands_handler import AgentShieldHandler
from agentshield.pipeline import Shield, Verdict
from agentshield.policy.engine import PolicyEngine
from agentshield.policy.facts import FactMap
from agentshield.policy.loader import load_policy
from agentshield.policy.session import SessionState
from agentshield.shophelp.handbook_rag import build_handbook_rag
from agentshield.shophelp.store import ShopHelpCatalogue, ShopHelpStore
from agentshield.shophelp.tools import bind_tools

POLICY_PATH = Path(__file__).resolve().parents[3] / "policies" / "shophelp.yaml"

SHOPHELP_FACTS = FactMap(
    {
        "get_order_status": {
            "total": "order_total",
            "refundable_remaining": "refundable_remaining",
            "status": "order_status",
            "within_return_window": "within_return_window",
        },
        "get_customer_profile": {"email": "customer_email"},
        "cancel_order": {"status": "order_status"},
        "update_shipping_address": {"status": "order_status"},
        "issue_refund": {"refundable_remaining": "refundable_remaining"},
    }
)

SYSTEM_PROMPT = """\
You are ShopHelp, Maya's customer-support copilot at Hearth & Cable, a small UK housewares \
and electronics shop.

Rules:
- Use tools to look things up. Never invent an order id, customer id, email, amount or address.
- Look up the order before cancelling it, changing its address, or issuing a refund.
- Look up the customer profile before sending email, and email only that address.
- Ask the user before any write (cancel, refund, address change, email).
- If a tool is denied or you are told to clarify, follow that instruction. Do not retry the \
same call with guessed values.
- Never claim a refund, cancel, email or address change happened unless a tool result shows it.
- For return-window, shipping, or refund-policy questions, call lookup_handbook first and \
answer only from those passages. If it abstains, say you do not have enough handbook evidence. \
Example: "How long do I have to return a kettle?" -> lookup_handbook, then quote the 14-day window. \
Do not invent handbook rules.
"""


@dataclass
class ShopHelpApp:
    agent: Agent
    handler: AgentShieldHandler
    store: ShopHelpCatalogue
    shield: Shield
    approvals: ApprovalDesk | None = None


def build_shophelp(
    model: Any,
    checker: DeciderChecker,
    *,
    session: SessionState | None = None,
    store: ShopHelpCatalogue | None = None,
    sinks: Sequence[AuditSink] = (),
    on_verdict: Callable[[Verdict], None] | None = None,
    approvals: ApprovalDesk | None = None,
) -> ShopHelpApp:
    """A gated ShopHelp agent. ``model`` is any Strands model (live or scripted)."""
    store = store or ShopHelpStore.seed()
    session = session or SessionState("shophelp")
    shield = Shield(
        PolicyEngine(load_policy(POLICY_PATH)),
        checker,
        agent_id="shophelp",
        sinks=sinks,
        thresholds=load_thresholds(),
    )
    handler = AgentShieldHandler(
        shield,
        session,
        on_verdict=on_verdict,
        facts=SHOPHELP_FACTS,
        approvals=approvals,
    )
    agent = Agent(
        model=model,
        tools=bind_tools(store, rag=build_handbook_rag(checker)),
        interventions=[handler],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    if approvals is not None:
        approvals.register_agent(session.session_id, agent)
    return ShopHelpApp(
        agent=agent, handler=handler, store=store, shield=shield, approvals=approvals
    )
