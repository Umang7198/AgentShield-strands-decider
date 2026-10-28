"""Decider questions for retrieved handbook context (separate from tool-gate v2)."""

from __future__ import annotations

from collections.abc import Sequence

from agentshield.decider.client import NoulQuestion
from agentshield.rag.handbook import Passage

RAG_QUESTION_SET = "rag-v1"


def relevance_question(query: str, passages: Sequence[Passage]) -> NoulQuestion:
    joined = _clip("\n\n".join(f"[{p.title}] {p.text}" for p in passages), 1800)
    return NoulQuestion(
        instructions=(
            f"Customer question: {query}. Do these handbook passages talk about that question?\n"
            f"{joined}"
        ),
        true_means="at least one passage is on-topic for the question",
        false_means="the passages are about something else",
    )


def sufficiency_question(query: str, passages: Sequence[Passage]) -> NoulQuestion:
    joined = _clip("\n\n".join(f"[{p.title}] {p.text}" for p in passages), 1800)
    return NoulQuestion(
        instructions=(
            f"Customer question: {query}. Is this handbook context enough to answer without guessing?\n"
            f"{joined}"
        ),
        true_means="the answer is in the passages; no invented policy is needed",
        false_means="a careful agent would still have to invent or assume missing facts",
    )


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
