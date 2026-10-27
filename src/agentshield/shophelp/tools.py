"""Strands tool wrappers around :class:`~agentshield.shophelp.store.ShopHelpStore`."""

from __future__ import annotations

from typing import Any

from strands import tool

from agentshield.shophelp.store import ShopHelpStore


def bind_tools(store: ShopHelpStore) -> list[Any]:
    """Build the seven ShopHelp tools closed over ``store``.

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

    return [
        get_order_status,
        get_customer_profile,
        send_customer_email,
        update_shipping_address,
        cancel_order,
        issue_refund,
        delete_customer_data,
    ]
