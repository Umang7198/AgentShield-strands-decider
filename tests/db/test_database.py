from pathlib import Path

from agentshield.approvals import ApprovalDesk
from agentshield.core import Outcome
from agentshield.db import Database, describe_url, normalize_url
from agentshield.decider import DeciderChecker, FakeDeciderClient
from agentshield.shophelp.runtime import ShopHelpRuntime
from agentshield.shophelp.world import SESSION_ID
from tests.audit.test_record import _record
from tests.fakes import ScriptedModel

GOOD = {"arg:*": 0.95, "consent": 0.95, "claim": 0.12}


def _runtime(db: Database) -> ShopHelpRuntime:
    return ShopHelpRuntime(
        ScriptedModel([], final="How can I help?"),
        DeciderChecker(FakeDeciderClient(GOOD)),
        approvals=ApprovalDesk(db.approvals),
        persist=db,
    )


def test_normalize_and_describe_hide_password() -> None:
    url = "postgresql://agentshield:secret@127.0.0.1:5432/agentshield"
    assert normalize_url(url).startswith("postgresql+psycopg://")
    summary = describe_url(url)
    assert "secret" not in summary
    assert "agentshield" in summary


def test_seeds_dummy_orders(tmp_path: Path) -> None:
    db = Database.from_url(f"sqlite:///{tmp_path / 'shop.db'}")
    order = db.shophelp.get_order("A1001")
    assert order["status"] == "shipped"
    assert order["customer_id"] == "C42"
    assert db.shophelp.get_customer("C99")["email"] == "ann@example.com"


def test_reset_reseeds_but_keeps_audit(tmp_path: Path) -> None:
    db = Database.from_url(f"sqlite:///{tmp_path / 'shop.db'}")
    runtime = _runtime(db)
    db.shophelp.refund("A1001", 40)
    runtime.check_claim("Your refund has been processed.")
    assert runtime.tape[-1]["outcome"] == Outcome.CLARIFY.value
    assert db.shophelp.get_order("A1001")["refunded_so_far"] == 40.0
    before = len(db.audit.query(session_id=SESSION_ID))

    runtime.reset()
    assert db.shophelp.get_order("A1001")["refunded_so_far"] == 0.0
    assert runtime.tape == []
    assert len(db.audit.query(session_id=SESSION_ID)) == before


def test_new_runtime_hydrates_tape_and_catalogue(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'shop.db'}"
    first = Database.from_url(url)
    runtime = _runtime(first)
    first.shophelp.refund("A1001", 40)
    runtime.check_claim("Your refund has been processed.")

    second = Database.from_url(url)
    revived = _runtime(second)
    assert revived.app.store.get_order("A1001")["refunded_so_far"] == 40.0
    assert revived.tape[-1]["tool"] == "agent_claim"
    assert revived.tape[-1]["outcome"] == Outcome.CLARIFY.value


def test_reset_hides_old_tape_after_restart(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'shop.db'}"
    db = Database.from_url(url)
    runtime = _runtime(db)
    runtime.check_claim("Your refund has been processed.")
    runtime.reset()

    revived_db = Database.from_url(url)
    revived = _runtime(revived_db)
    assert revived.tape == []
    assert len(revived_db.audit.query(session_id=SESSION_ID)) >= 1


def test_audit_sql_roundtrip_on_the_bundle(tmp_path: Path) -> None:
    db = Database.from_url(f"sqlite:///{tmp_path / 'shop.db'}")
    rec = _record({"order_id": "A1001"})
    db.audit.record(rec)
    assert db.audit.get(rec.id) == rec
