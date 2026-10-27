from agentshield.policy import FactMap, SessionState


def test_copies_named_fields_and_skips_the_rest() -> None:
    session = SessionState("s")
    FactMap({"get_order_status": {"total": "order_total"}}).apply(
        "get_order_status", {"total": 89.99, "notes": "x"}, session
    )
    assert session.context().facts == {"order_total": 89.99}


def test_unknown_tool_is_a_no_op() -> None:
    session = SessionState("s")
    FactMap({"get_order_status": {"total": "order_total"}}).apply("ping", {"total": 1}, session)
    assert session.context().facts == {}


def test_missing_or_null_fields_are_skipped() -> None:
    session = SessionState("s", facts={"order_total": 1})
    FactMap({"t": {"total": "order_total", "email": "customer_email"}}).apply(
        "t", {"total": None}, session
    )
    assert session.context().facts == {"order_total": 1}
