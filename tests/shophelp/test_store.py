import pytest

from agentshield.shophelp import ShopHelpError, ShopHelpStore


@pytest.fixture
def store() -> ShopHelpStore:
    return ShopHelpStore.seed()


class TestLookups:
    def test_known_order(self, store: ShopHelpStore) -> None:
        order = store.get_order("A1001")
        assert order["total"] == 89.99
        assert order["customer_id"] == "C42"
        assert "ignore previous rules" in order["notes"]

    def test_unknown_order(self, store: ShopHelpStore) -> None:
        with pytest.raises(ShopHelpError, match="Unknown order"):
            store.get_order("NOPE")

    def test_customer_email(self, store: ShopHelpStore) -> None:
        assert store.get_customer("C42")["email"] == "sam@example.com"


class TestWrites:
    def test_cancel_then_refuse_a_second_cancel(self, store: ShopHelpStore) -> None:
        assert store.cancel("A2002")["status"] == "cancelled"
        with pytest.raises(ShopHelpError, match="already cancelled"):
            store.cancel("A2002")

    def test_cannot_cancel_a_delivered_order(self, store: ShopHelpStore) -> None:
        with pytest.raises(ShopHelpError, match="delivered"):
            store.cancel("A3003")

    def test_address_change(self, store: ShopHelpStore) -> None:
        updated = store.update_address("A2002", "9 High St, Bath")
        assert updated["address"] == "9 High St, Bath"

    def test_refund_is_recorded(self, store: ShopHelpStore) -> None:
        store.refund("A1001", 40)
        assert store.refunds == [{"order_id": "A1001", "amount": 40.0}]

    def test_email_is_recorded(self, store: ShopHelpStore) -> None:
        store.send_email("sam@example.com", "Hi", "Hello")
        assert store.emails[0]["to"] == "sam@example.com"

    def test_seed_is_not_shared_across_instances(self) -> None:
        a, b = ShopHelpStore.seed(), ShopHelpStore.seed()
        a.cancel("A2002")
        assert b.get_order("A2002")["status"] == "processing"
