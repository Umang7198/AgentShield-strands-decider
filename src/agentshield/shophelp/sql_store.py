"""SQL-backed Hearth & Cable catalogue. Same operations as :class:`ShopHelpStore`."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Float, Integer, String, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from agentshield.db import engine_from_url
from agentshield.shophelp.store import (
    _CLOSED,
    Customer,
    Order,
    ShopHelpError,
    Ticket,
    _now,
)
from agentshield.shophelp.world import (
    CUSTOMERS,
    DEMO_TODAY,
    ORDERS,
    TICKETS,
    company,
    return_until,
    within_return_window,
)


class _Base(DeclarativeBase):
    pass


class _CustomerRow(_Base):
    __tablename__ = "shophelp_customers"

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    email: Mapped[str] = mapped_column(String(256))
    address: Mapped[str] = mapped_column(String(256))
    marketing_consent: Mapped[bool] = mapped_column(Boolean, default=False)


class _OrderRow(_Base):
    __tablename__ = "shophelp_orders"

    order_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    total: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), index=True)
    items: Mapped[str] = mapped_column(String(256))
    sku: Mapped[str] = mapped_column(String(64))
    address: Mapped[str] = mapped_column(String(256))
    notes: Mapped[str] = mapped_column(String(512), default="")
    tracking: Mapped[str] = mapped_column(String(64), default="")
    placed_at: Mapped[str] = mapped_column(String(32), default="")
    shipped_at: Mapped[str] = mapped_column(String(32), default="")
    delivered_at: Mapped[str] = mapped_column(String(32), default="")
    return_until: Mapped[str] = mapped_column(String(32), default="")
    within_return_window: Mapped[bool] = mapped_column(Boolean, default=True)
    refunded_so_far: Mapped[float] = mapped_column(Float, default=0.0)
    payment_last4: Mapped[str] = mapped_column(String(8), default="4242")
    ticket_id: Mapped[str] = mapped_column(String(32), default="")


class _TicketRow(_Base):
    __tablename__ = "shophelp_tickets"

    ticket_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    customer_id: Mapped[str] = mapped_column(String(32), index=True)
    subject: Mapped[str] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="open")


class _RefundRow(_Base):
    __tablename__ = "shophelp_refunds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(32), index=True)
    amount: Mapped[float] = mapped_column(Float)
    refunded_so_far: Mapped[float] = mapped_column(Float)
    refundable_remaining: Mapped[float] = mapped_column(Float)
    at: Mapped[str] = mapped_column(String(40))


class _EmailRow(_Base):
    __tablename__ = "shophelp_emails"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    to: Mapped[str] = mapped_column(String(256))
    subject: Mapped[str] = mapped_column(String(256))
    body: Mapped[str] = mapped_column(String(2048))
    at: Mapped[str] = mapped_column(String(40))


class SqlShopHelpStore:
    """Persisted catalogue. ``reseed()`` restores world.py; audit rows are not touched."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _Base.metadata.create_all(engine)

    @classmethod
    def from_url(cls, url: str) -> SqlShopHelpStore:
        return cls(engine_from_url(url))

    @classmethod
    def sqlite(cls, path: str | Path) -> SqlShopHelpStore:
        return cls.from_url(f"sqlite:///{Path(path)}")

    def ensure_seeded(self) -> None:
        with Session(self._engine) as session:
            count = session.scalar(select(_CustomerRow.customer_id).limit(1))
        if count is None:
            self.reseed()

    def reseed(self) -> None:
        with Session(self._engine) as session:
            session.execute(delete(_EmailRow))
            session.execute(delete(_RefundRow))
            session.execute(delete(_TicketRow))
            session.execute(delete(_OrderRow))
            session.execute(delete(_CustomerRow))
            for customer in CUSTOMERS.values():
                session.add(
                    _CustomerRow(
                        customer_id=customer.customer_id,
                        name=customer.name,
                        email=customer.email,
                        address=customer.address,
                        marketing_consent=customer.marketing_consent,
                    )
                )
            for order in ORDERS.values():
                until = return_until(order)
                session.add(
                    _OrderRow(
                        order_id=order.order_id,
                        customer_id=order.customer_id,
                        total=order.total,
                        status=order.status,
                        items=order.items,
                        sku=order.sku,
                        address=order.address,
                        notes=order.notes,
                        tracking=order.tracking,
                        placed_at=order.placed_at,
                        shipped_at=order.shipped_at,
                        delivered_at=order.delivered_at,
                        return_until=until.isoformat(),
                        within_return_window=within_return_window(order),
                        refunded_so_far=0.0,
                        payment_last4=order.payment_last4,
                        ticket_id=order.ticket_id,
                    )
                )
            for ticket in TICKETS:
                session.add(
                    _TicketRow(
                        ticket_id=ticket.ticket_id,
                        order_id=ticket.order_id,
                        customer_id=ticket.customer_id,
                        subject=ticket.subject,
                        status=ticket.status,
                    )
                )
            session.commit()

    @property
    def refunds(self) -> list[dict[str, Any]]:
        with Session(self._engine) as session:
            rows = session.scalars(select(_RefundRow).order_by(_RefundRow.id)).all()
            return [_refund_dict(row) for row in rows]

    @property
    def emails(self) -> list[dict[str, str]]:
        with Session(self._engine) as session:
            rows = session.scalars(select(_EmailRow).order_by(_EmailRow.id)).all()
            return [
                {"to": row.to, "subject": row.subject, "body": row.body, "at": row.at}
                for row in rows
            ]

    def snapshot(self) -> dict[str, Any]:
        with Session(self._engine) as session:
            orders = [
                _order_from_row(row).as_dict()
                for row in session.scalars(select(_OrderRow).order_by(_OrderRow.order_id))
            ]
            customers = [
                _customer_from_row(row).as_dict()
                for row in session.scalars(select(_CustomerRow).order_by(_CustomerRow.customer_id))
            ]
            tickets = [
                _ticket_from_row(row).as_dict()
                for row in session.scalars(select(_TicketRow).order_by(_TicketRow.ticket_id))
            ]
            refunds = [
                _refund_dict(row)
                for row in session.scalars(select(_RefundRow).order_by(_RefundRow.id))
            ]
            emails = [
                {"to": row.to, "subject": row.subject, "body": row.body, "at": row.at}
                for row in session.scalars(select(_EmailRow).order_by(_EmailRow.id))
            ]
        return {
            **company(),
            "orders": orders,
            "customers": customers,
            "tickets": tickets,
            "refunds": refunds,
            "emails": emails,
        }

    def get_order(self, order_id: str) -> dict[str, Any]:
        return self._require_order(order_id).as_dict()

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        with Session(self._engine) as session:
            row = session.get(_CustomerRow, customer_id.strip())
            if row is None:
                raise ShopHelpError(f"Unknown customer '{customer_id}'")
            return _customer_from_row(row).as_dict()

    def send_email(self, to: str, subject: str, body: str) -> dict[str, str]:
        sent = {"to": to, "subject": subject, "body": body, "at": _now()}
        with Session(self._engine) as session:
            session.add(_EmailRow(**sent))
            session.commit()
        return {"status": "sent", **sent}

    def update_address(self, order_id: str, address: str) -> dict[str, Any]:
        with Session(self._engine) as session:
            row = self._order_row(session, order_id)
            if row.status in _CLOSED:
                raise ShopHelpError(f"Cannot change address of a {row.status} order")
            row.address = address
            session.commit()
            session.refresh(row)
            return _order_from_row(row).as_dict()

    def cancel(self, order_id: str) -> dict[str, Any]:
        with Session(self._engine) as session:
            row = self._order_row(session, order_id)
            if row.status == "cancelled":
                raise ShopHelpError(f"Order '{order_id}' is already cancelled")
            if row.status == "delivered":
                raise ShopHelpError(f"Order '{order_id}' has already been delivered")
            row.status = "cancelled"
            session.commit()
            session.refresh(row)
            return _order_from_row(row).as_dict()

    def refund(self, order_id: str, amount: float) -> dict[str, Any]:
        value = round(float(amount), 2)
        if value <= 0:
            raise ShopHelpError("Refund amount must be positive")
        with Session(self._engine) as session:
            row = self._order_row(session, order_id)
            remaining = round(row.total - row.refunded_so_far, 2)
            if value > remaining:
                raise ShopHelpError(f"Refund {value} exceeds remaining {remaining} on '{order_id}'")
            row.refunded_so_far = round(row.refunded_so_far + value, 2)
            leftover = round(row.total - row.refunded_so_far, 2)
            at = _now()
            order_id = row.order_id
            refunded_so_far = row.refunded_so_far
            session.add(
                _RefundRow(
                    order_id=order_id,
                    amount=value,
                    refunded_so_far=refunded_so_far,
                    refundable_remaining=leftover,
                    at=at,
                )
            )
            session.commit()
            return {
                "status": "refunded",
                "order_id": order_id,
                "amount": value,
                "refunded_so_far": refunded_so_far,
                "refundable_remaining": leftover,
                "at": at,
            }

    def delete_customer(self, customer_id: str) -> dict[str, str]:
        self.get_customer(customer_id)
        with Session(self._engine) as session:
            row = session.get(_CustomerRow, customer_id.strip())
            if row is not None:
                session.delete(row)
                session.commit()
        return {"status": "deleted", "customer_id": customer_id}

    def _require_order(self, order_id: str) -> Order:
        with Session(self._engine) as session:
            return _order_from_row(self._order_row(session, order_id))

    @staticmethod
    def _order_row(session: Session, order_id: str) -> _OrderRow:
        row = session.get(_OrderRow, order_id.strip())
        if row is None:
            raise ShopHelpError(f"Unknown order '{order_id}'")
        return row


def _refund_dict(row: _RefundRow) -> dict[str, Any]:
    return {
        "order_id": row.order_id,
        "amount": row.amount,
        "refunded_so_far": row.refunded_so_far,
        "refundable_remaining": row.refundable_remaining,
        "at": row.at,
    }


def _order_from_row(row: _OrderRow) -> Order:
    until = row.return_until
    window = row.within_return_window
    if until:
        try:
            window = date.fromisoformat(until) >= DEMO_TODAY
        except ValueError:
            window = row.within_return_window
    return Order(
        order_id=row.order_id,
        customer_id=row.customer_id,
        total=row.total,
        status=row.status,
        items=row.items,
        sku=row.sku,
        address=row.address,
        notes=row.notes,
        tracking=row.tracking,
        placed_at=row.placed_at,
        shipped_at=row.shipped_at,
        delivered_at=row.delivered_at,
        return_until=row.return_until,
        within_return_window=window,
        refunded_so_far=row.refunded_so_far,
        payment_last4=row.payment_last4,
        ticket_id=row.ticket_id,
    )


def _customer_from_row(row: _CustomerRow) -> Customer:
    return Customer(
        customer_id=row.customer_id,
        name=row.name,
        email=row.email,
        address=row.address,
        marketing_consent=row.marketing_consent,
    )


def _ticket_from_row(row: _TicketRow) -> Ticket:
    return Ticket(
        ticket_id=row.ticket_id,
        order_id=row.order_id,
        customer_id=row.customer_id,
        subject=row.subject,
        status=row.status,
    )
