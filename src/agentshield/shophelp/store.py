"""In-memory Hearth & Cable catalogue. No network: enough for the gate and the desk UI."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from agentshield.shophelp.world import (
    CUSTOMERS,
    ORDERS,
    TICKETS,
    CustomerSeed,
    OrderSeed,
    TicketSeed,
    company,
    return_until,
    within_return_window,
)


class ShopHelpError(LookupError):
    """Unknown id or an illegal state change (already cancelled, and so on)."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Order:
    order_id: str
    customer_id: str
    total: float
    status: str
    items: str
    sku: str
    address: str
    notes: str = ""
    tracking: str = ""
    placed_at: str = ""
    shipped_at: str = ""
    delivered_at: str = ""
    return_until: str = ""
    within_return_window: bool = True
    refunded_so_far: float = 0.0
    payment_last4: str = "4242"
    ticket_id: str = ""

    @classmethod
    def from_seed(cls, seed: OrderSeed) -> Order:
        until = return_until(seed)
        return cls(
            order_id=seed.order_id,
            customer_id=seed.customer_id,
            total=seed.total,
            status=seed.status,
            items=seed.items,
            sku=seed.sku,
            address=seed.address,
            notes=seed.notes,
            tracking=seed.tracking,
            placed_at=seed.placed_at,
            shipped_at=seed.shipped_at,
            delivered_at=seed.delivered_at,
            return_until=until.isoformat(),
            within_return_window=within_return_window(seed),
            payment_last4=seed.payment_last4,
            ticket_id=seed.ticket_id,
        )

    @property
    def refundable_remaining(self) -> float:
        return round(self.total - self.refunded_so_far, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "total": self.total,
            "status": self.status,
            "items": self.items,
            "sku": self.sku,
            "address": self.address,
            "notes": self.notes,
            "tracking": self.tracking,
            "placed_at": self.placed_at,
            "shipped_at": self.shipped_at,
            "delivered_at": self.delivered_at,
            "return_until": self.return_until,
            "within_return_window": self.within_return_window,
            "refunded_so_far": self.refunded_so_far,
            "refundable_remaining": self.refundable_remaining,
            "payment_last4": self.payment_last4,
            "ticket_id": self.ticket_id,
        }


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    address: str
    marketing_consent: bool = False

    @classmethod
    def from_seed(cls, seed: CustomerSeed) -> Customer:
        return cls(
            customer_id=seed.customer_id,
            name=seed.name,
            email=seed.email,
            address=seed.address,
            marketing_consent=seed.marketing_consent,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "address": self.address,
            "marketing_consent": self.marketing_consent,
        }


@dataclass
class Ticket:
    ticket_id: str
    order_id: str
    customer_id: str
    subject: str
    status: str = "open"

    @classmethod
    def from_seed(cls, seed: TicketSeed) -> Ticket:
        return cls(
            ticket_id=seed.ticket_id,
            order_id=seed.order_id,
            customer_id=seed.customer_id,
            subject=seed.subject,
            status=seed.status,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "subject": self.subject,
            "status": self.status,
        }


_CLOSED = frozenset({"cancelled", "delivered"})


@dataclass
class ShopHelpStore:
    orders: dict[str, Order]
    customers: dict[str, Customer]
    tickets: dict[str, Ticket] = field(default_factory=dict)
    emails: list[dict[str, str]] = field(default_factory=list)
    refunds: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def seed(cls) -> ShopHelpStore:
        """Catalogue used by the demo, the desk UI and the tests."""
        return cls(
            orders={oid: Order.from_seed(seed) for oid, seed in ORDERS.items()},
            customers={cid: Customer.from_seed(seed) for cid, seed in CUSTOMERS.items()},
            tickets={t.ticket_id: Ticket.from_seed(t) for t in TICKETS},
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            **company(),
            "orders": [o.as_dict() for o in self.orders.values()],
            "customers": [c.as_dict() for c in self.customers.values()],
            "tickets": [t.as_dict() for t in self.tickets.values()],
            "refunds": list(self.refunds),
            "emails": list(self.emails),
        }

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._require_order(order_id).as_dict()

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        customer = self.customers.get(customer_id.strip())
        if customer is None:
            raise ShopHelpError(f"Unknown customer '{customer_id}'")
        return customer.as_dict()

    def send_email(self, to: str, subject: str, body: str) -> dict[str, str]:
        sent = {"to": to, "subject": subject, "body": body, "at": _now()}
        self.emails.append(sent)
        return {"status": "sent", **sent}

    def update_address(self, order_id: str, address: str) -> dict[str, Any]:
        order = self._require_order(order_id)
        if order.status in _CLOSED:
            raise ShopHelpError(f"Cannot change address of a {order.status} order")
        self.orders[order.order_id] = replace(order, address=address)
        return self.orders[order.order_id].as_dict()

    def cancel(self, order_id: str) -> dict[str, Any]:
        order = self._require_order(order_id)
        if order.status == "cancelled":
            raise ShopHelpError(f"Order '{order_id}' is already cancelled")
        if order.status == "delivered":
            raise ShopHelpError(f"Order '{order_id}' has already been delivered")
        self.orders[order.order_id] = replace(order, status="cancelled")
        return self.orders[order.order_id].as_dict()

    def refund(self, order_id: str, amount: float) -> dict[str, Any]:
        order = self._require_order(order_id)
        value = round(float(amount), 2)
        if value <= 0:
            raise ShopHelpError("Refund amount must be positive")
        if value > order.refundable_remaining:
            raise ShopHelpError(
                f"Refund {value} exceeds remaining {order.refundable_remaining} on '{order_id}'"
            )
        updated = replace(order, refunded_so_far=round(order.refunded_so_far + value, 2))
        self.orders[order.order_id] = updated
        record = {
            "order_id": updated.order_id,
            "amount": value,
            "refunded_so_far": updated.refunded_so_far,
            "refundable_remaining": updated.refundable_remaining,
            "at": _now(),
        }
        self.refunds.append(record)
        return {"status": "refunded", **record}

    def delete_customer(self, customer_id: str) -> dict[str, str]:
        self.get_customer(customer_id)  # raises if unknown
        del self.customers[customer_id]
        return {"status": "deleted", "customer_id": customer_id}

    def _require_order(self, order_id: str) -> Order:
        order = self.orders.get(order_id.strip())
        if order is None:
            raise ShopHelpError(f"Unknown order '{order_id}'")
        return order
