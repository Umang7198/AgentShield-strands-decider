"""Vector index: in-memory cosine (tests) and optional Chroma (live demo)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

from agentshield.rag.embedder import Embedder, HashEmbedder
from agentshield.rag.handbook import Passage, load_handbook


class VectorIndex(Protocol):
    def search(self, query: str, k: int = 4) -> tuple[tuple[Passage, float], ...]: ...


class InMemoryIndex:
    def __init__(self, passages: Sequence[Passage], embedder: Embedder) -> None:
        self._passages = tuple(passages)
        self._embedder = embedder
        self._vectors = embedder.embed([p.text for p in self._passages])

    def search(self, query: str, k: int = 4) -> tuple[tuple[Passage, float], ...]:
        q = self._embedder.embed([query])[0]
        scored = [
            (passage, _cosine(q, vec))
            for passage, vec in zip(self._passages, self._vectors, strict=True)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return tuple(scored[: max(1, k)])


class ChromaIndex:
    """Ephemeral Chroma collection. Requires the optional ``rag`` extra."""

    def __init__(
        self,
        passages: Sequence[Passage],
        embedder: Embedder,
        *,
        collection: str = "shophelp_handbook",
    ) -> None:
        import chromadb

        self._passages = {p.id: p for p in passages}
        self._embedder = embedder
        client = chromadb.EphemeralClient()
        coll = client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        vectors = embedder.embed([p.text for p in passages])
        coll.add(
            ids=[p.id for p in passages],
            embeddings=vectors,
            documents=[p.text for p in passages],
            metadatas=[{"title": p.title, "source": p.source} for p in passages],
        )
        self._coll = coll

    def search(self, query: str, k: int = 4) -> tuple[tuple[Passage, float], ...]:
        q = self._embedder.embed([query])
        result = self._coll.query(query_embeddings=q, n_results=max(1, k))
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        out: list[tuple[Passage, float]] = []
        for pid, dist in zip(ids, distances, strict=False):
            passage = self._passages.get(pid)
            if passage is None:
                continue
            score = 1.0 - float(dist)
            out.append((passage, score))
        return tuple(out)


def build_index(
    *,
    embedder: Embedder | None = None,
    use_chroma: bool = False,
) -> VectorIndex:
    passages = load_handbook()
    encoder = embedder or HashEmbedder()
    if use_chroma:
        return ChromaIndex(passages, encoder)
    return InMemoryIndex(passages, encoder)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
