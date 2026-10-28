import os

from agentshield.shophelp.handbook_rag import (
    looks_like_policy_question,
    prompt_for_agent,
    select_rag_backend,
)
from agentshield.shophelp.runtime import SCENES


def test_scene_11_is_a_policy_question() -> None:
    prompt = next(s["prompt"] for s in SCENES if s["id"] == "s11")
    assert looks_like_policy_question(prompt)
    assert "lookup_handbook" in prompt_for_agent(prompt)


def test_writes_are_not_treated_as_policy_questions() -> None:
    assert not looks_like_policy_question("Yes, refund 40 dollars on A1001. I confirm.")
    assert not looks_like_policy_question(
        "Change the shipping address of A1001 to 9 High St, Bath. I confirm."
    )
    assert not looks_like_policy_question("Where is order A1001?")
    assert "lookup_handbook" not in prompt_for_agent("Where is order A1001?")


def test_pytest_forces_hash_backend() -> None:
    embedder, chroma, label = select_rag_backend()
    assert os.environ.get("AGENTSHIELD_RAG_HASH") == "1"
    assert type(embedder).__name__ == "HashEmbedder"
    assert chroma is False
    assert label == "hash/memory"
