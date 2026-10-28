import os

import pytest

from agentshield.db import Database

pytestmark = pytest.mark.postgres


@pytest.mark.skipif(
    os.environ.get("AGENTSHIELD_TEST_POSTGRES") != "1",
    reason="set AGENTSHIELD_TEST_POSTGRES=1 with DATABASE_URL pointing at Postgres",
)
def test_postgres_seeds_and_writes_audit() -> None:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url.startswith(("postgres", "postgresql")):
        pytest.skip("DATABASE_URL is not Postgres")
    db = Database.from_url(url)
    assert db.shophelp.get_order("A1001")["items"] == "wireless headphones"
    db.shophelp.refund("A1001", 10)
    assert db.shophelp.get_order("A1001")["refunded_so_far"] == 10.0
    db.shophelp.reseed()
    assert db.shophelp.get_order("A1001")["refunded_so_far"] == 0.0
