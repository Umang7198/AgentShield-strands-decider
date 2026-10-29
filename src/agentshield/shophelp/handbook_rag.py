"""Wire ShopHelp handbook RAG.

Live default: MiniLM + Chroma when ``pip install -e '.[rag]'`` has been run.
Tests and CI force the hash embedder via ``AGENTSHIELD_RAG_HASH=1`` (see tests/conftest.py)
so pytest never downloads a model.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from collections.abc import Callable

from agentshield.decider.checker import DeciderChecker
from agentshield.rag import (
    HashEmbedder,
    PolicyRag,
    RagAnswer,
    SentenceTransformerEmbedder,
    build_index,
)
from agentshield.rag.embedder import Embedder

log = logging.getLogger(__name__)

# Customer-facing transcript is unchanged; this is appended only to the model prompt.
POLICY_NUDGE = (
    "\n\n[Gate] This is a return/shipping policy question. Call lookup_handbook first and "
    "answer only from those passages. Do not issue_refund, cancel_order, or "
    "update_shipping_address unless the customer asked you to perform that write."
)

_WRITE = re.compile(
    r"(?:refund\s+\$?\d+|issue_refund|cancel (?:it|order)|cancel_order|"
    r"change the shipping address of|update_shipping|email \S+@|delete all data)",
    re.I,
)


def _flag(name: str) -> bool | None:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return None


def _importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def force_hash() -> bool:
    """True in pytest (conftest) or when AGENTSHIELD_RAG_HASH=1."""
    if _flag("AGENTSHIELD_RAG_HASH") is True:
        return True
    return bool(os.environ.get("PYTEST_CURRENT_TEST"))


def looks_like_policy_question(text: str) -> bool:
    """Conservative: handbook Qs only. Writes with an order id / amount stay on the tool path."""
    cleaned = text.strip()
    if not cleaned or _WRITE.search(cleaned):
        return False
    lower = cleaned.lower()
    if "return" in lower and (
        "window" in lower or "policy" in lower or "how long" in lower or "14" in lower
    ):
        return True
    if "ship" in lower and (
        "policy" in lower or "working day" in lower or "how long" in lower or "3-5" in lower
    ):
        return True
    if "change a shipped" in lower or "redirect" in lower:
        return True
    if "handbook" in lower or "refund policy" in lower:
        return True
    return False


def prompt_for_agent(text: str) -> str:
    if looks_like_policy_question(text):
        return text + POLICY_NUDGE
    return text


def select_rag_backend() -> tuple[Embedder, bool, str]:
    """Pick embedder + whether to use Chroma. Hash/memory is the safe fallback."""
    embedder, embed_name = _embedder()
    chroma = _want_chroma()
    return embedder, chroma, f"{embed_name}/{'chroma' if chroma else 'memory'}"


def _embedder() -> tuple[Embedder, str]:
    if force_hash() or _flag("AGENTSHIELD_RAG_MINILM") is False:
        return HashEmbedder(), "hash"
    if not _importable("sentence_transformers"):
        return HashEmbedder(), "hash"
    try:
        return SentenceTransformerEmbedder(), "minilm"
    except Exception as exc:
        log.warning("MiniLM unavailable (%s); using hash embedder", exc)
        return HashEmbedder(), "hash"


def _want_chroma() -> bool:
    if force_hash() or _flag("AGENTSHIELD_RAG_CHROMA") is False:
        return False
    return _importable("chromadb")


def build_handbook_rag(
    checker: DeciderChecker,
    *,
    on_answer: Callable[[RagAnswer], None] | None = None,
) -> PolicyRag:
    embedder, use_chroma, label = select_rag_backend()
    try:
        index = build_index(embedder=embedder, use_chroma=use_chroma)
    except Exception as exc:
        log.warning("Chroma unavailable (%s); using in-memory index", exc)
        index = build_index(embedder=embedder, use_chroma=False)
        label = f"{label.split('/')[0]}/memory"
    log.info("handbook RAG backend: %s", label)
    if not force_hash():
        print(f"  [rag] handbook backend: {label}", flush=True)
    return PolicyRag(index, checker, on_answer=on_answer)
