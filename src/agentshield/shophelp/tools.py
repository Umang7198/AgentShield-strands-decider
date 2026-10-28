"""Strands tool wrappers around :class:`~agentshield.shophelp.store.ShopHelpStore`."""

from __future__ import annotations

from typing import Any

from strands import tool

from agentshield.rag import PolicyRag, RagAction
from agentshield.shophelp.store import ShopHelpCatalogue


def bind_tools(store: ShopHelpCatalogue, rag: PolicyRag | None = None) -> list[Any]:
    """Build ShopHelp tools closed over ``store`` (and optional handbook RAG).

    Returning a dict (not a sentence) so the fact binder can read ``total`` and ``email``
    without scraping prose.
    """

    @tool
    def get_order_status(order_id: str) -> dict[str, Any]:
        """Look up an order by id. Returns status, totals, remaining refundable, address."""
        return store.get_order(order_id)

    @tool
    def get_customer_profile(customer_id: str) -> dict[str, Any]:
        """Look up a customer by id. Returns name, email and address."""
        return store.get_customer(customer_id)

    @tool
    def send_customer_email(to: str, subject: str, body: str) -> dict[str, Any]:
        """Send an email to a customer. Only the address on their profile is allowed."""
        return store.send_email(to, subject, body)

    @tool
    def update_shipping_address(order_id: str, address: str) -> dict[str, Any]:
        """Change the shipping address of an order that has not yet been delivered."""
        return store.update_address(order_id, address)

    @tool
    def cancel_order(order_id: str) -> dict[str, Any]:
        """Cancel an order that has not been delivered yet."""
        return store.cancel(order_id)

    @tool
    def issue_refund(order_id: str, amount: float) -> dict[str, Any]:
        """Issue a refund for an order. Amount is in US dollars."""
        return store.refund(order_id, amount)

    @tool
    def delete_customer_data(customer_id: str) -> dict[str, Any]:
        """Permanently delete a customer's personal data. Not permitted for this agent."""
        return store.delete_customer(customer_id)

    tools = [
        get_order_status,
        get_customer_profile,
        send_customer_email,
        update_shipping_address,
        cancel_order,
        issue_refund,
        delete_customer_data,
    ]
    if rag is None:
        return tools

    handbook = rag

    @tool
    async def lookup_handbook(question: str) -> dict[str, Any]:
        """Look up Hearth & Cable return, shipping, and refund policy. Use before answering policy questions."""
        result = await handbook.answer(question)
        if result.action is RagAction.ABSTAIN:
            return {
                "ok": False,
                "decision": "abstain",
                "retried": result.retried,
                "reason": result.reason,
                "p_relevance": result.p_relevance,
                "p_sufficient": result.p_sufficient,
                "instruction": (
                    "You do not have enough handbook evidence. Say so. Do not invent policy."
                ),
            }
        return {
            "ok": True,
            "decision": "generate",
            "retried": result.retried,
            "reason": result.reason,
            "p_relevance": result.p_relevance,
            "p_sufficient": result.p_sufficient,
            "passages": [
                {"id": p.id, "title": p.title, "text": p.text, "source": p.source}
                for p in result.passages
            ],
            "instruction": "Answer using only these passages. If something is missing, say so.",
        }

    return [*tools, lookup_handbook]
