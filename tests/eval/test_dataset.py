from collections import Counter

from agentshield.core import Outcome
from agentshield.eval.dataset import build_dataset


def test_dataset_is_large_enough_and_labelled() -> None:
    data = build_dataset()
    assert len(data) >= 150
    assert len({s.id for s in data}) == len(data)
    families = {s.family for s in data}
    for needed in (
        "lookup_stated",
        "lookup_invented",
        "cancel_confirmed",
        "cancel_unconsented",
        "email_wrong_to",
        "refund_over_total",
        "delete_customer",
        "injection_user",
        "claim_grounding",
    ):
        assert needed in families
    tags = {tag for s in data for tag in s.tags}
    for needed in ("benign", "hallucinated", "unconsented", "policy", "injection", "claim"):
        assert needed in tags
    outcomes = Counter(s.expected for s in data)
    assert outcomes[Outcome.ALLOW] > 0
    assert outcomes[Outcome.BLOCK] > 0
    assert outcomes[Outcome.CLARIFY] > 0
    assert outcomes[Outcome.REQUIRE_APPROVAL] > 0
    assert any(s.kind == "claim" for s in data)
