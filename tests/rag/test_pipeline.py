from agentshield.decider.checker import DeciderChecker
from agentshield.decider.client import DeciderResponse, NoulQuestion
from agentshield.decider.errors import DeciderUnavailable
from agentshield.decider.fake import FakeDeciderClient
from agentshield.rag import HashEmbedder, PolicyRag, RagAction, build_index, load_handbook
from agentshield.rag.handbook import Passage
from agentshield.rag.index import InMemoryIndex
from agentshield.shophelp.store import ShopHelpStore
from agentshield.shophelp.tools import bind_tools


def test_handbook_splits_returns_and_shipping() -> None:
    passages = load_handbook()
    sources = {p.source for p in passages}
    assert "returns.md" in sources
    assert "shipping.md" in sources
    assert any("14-day" in p.text or "14 days" in p.text for p in passages)


def test_hash_index_retrieves_return_window() -> None:
    index = build_index(embedder=HashEmbedder())
    ranked = index.search("How long is the 14-day return window?", k=3)
    assert ranked
    assert any("return" in p.title.lower() or "return" in p.text.lower() for p, _ in ranked)


class _FlipClient:
    def __init__(self) -> None:
        self.n = 0

    async def ask(self, state: str, questions: dict[str, NoulQuestion]) -> DeciderResponse:
        self.n += 1
        p = 0.91 if self.n > 1 else 0.12
        return DeciderResponse(answers={qid: p for qid in questions}, wall_latency_ms=1.0)


async def test_rag_generates_when_decider_says_enough() -> None:
    seen: list[object] = []
    rag = PolicyRag(
        build_index(),
        DeciderChecker(FakeDeciderClient({"*": 0.92})),
        on_answer=seen.append,
    )
    answer = await rag.answer("What is the return window at Hearth & Cable?")
    assert answer.action is RagAction.GENERATE
    assert answer.retried is False
    assert answer.passages
    assert answer.p_relevance is not None and answer.p_relevance >= 0.55
    assert answer.total_ms >= 0
    assert seen == [answer]


async def test_rag_retries_once_then_generates() -> None:
    rag = PolicyRag(build_index(), DeciderChecker(_FlipClient()))  # type: ignore[arg-type]
    answer = await rag.answer("Can I change a shipped address?")
    assert answer.retried is True
    assert answer.action is RagAction.GENERATE


async def test_rag_abstains_when_decider_is_down() -> None:
    rag = PolicyRag(
        build_index(),
        DeciderChecker(FakeDeciderClient(error=DeciderUnavailable("down"))),
    )
    answer = await rag.answer("What is the return window?")
    assert answer.action is RagAction.ABSTAIN
    assert "unavailable" in answer.reason.lower()


async def test_rag_abstains_when_context_is_unrelated() -> None:
    passage = Passage(
        id="x", title="Unrelated", text="We sell only red balloons in space.", source="x.md"
    )
    index = InMemoryIndex([passage], HashEmbedder())
    rag = PolicyRag(index, DeciderChecker(FakeDeciderClient({"*": 0.1})))
    answer = await rag.answer("What is the GDPR delete process?")
    assert answer.action is RagAction.ABSTAIN
    assert answer.retried is True


def test_lookup_handbook_is_bound_when_rag_present() -> None:
    rag = PolicyRag(build_index(), DeciderChecker(FakeDeciderClient({"*": 0.9})))
    tools = bind_tools(ShopHelpStore.seed(), rag=rag)
    names = [getattr(t, "tool_name", getattr(t, "__name__", "")) for t in tools]
    joined = " ".join(str(n) for n in names)
    assert "lookup_handbook" in joined or any("handbook" in str(t).lower() for t in tools)
