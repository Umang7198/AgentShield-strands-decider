"""Embedding backends. Tests use HashEmbedder so CI never downloads MiniLM."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-tokens vectors. No network, no native wheels."""

    def __init__(self, dim: int = 64) -> None:
        if dim < 8:
            raise ValueError("dim must be at least 8")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_vector(text, self.dim) for text in texts]


class SentenceTransformerEmbedder:
    """MiniLM via sentence-transformers. Imported lazily so unit tests skip the download."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return [row.tolist() for row in vectors]


def _hash_vector(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "little") % dim
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    return _l2(vec)


def _tokens(text: str) -> list[str]:
    return [
        part for part in "".join(ch.lower() if ch.isalnum() else " " for ch in text).split() if part
    ]


def _l2(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]
