"""Retrieve handbook passages, ask the Decider, generate or retry once or abstain."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from agentshield.decider.checker import DeciderChecker
from agentshield.decider.client import NoulQuestion
from agentshield.decider.errors import DeciderError
from agentshield.rag.handbook import Passage
from agentshield.rag.index import VectorIndex
from agentshield.rag.questions import RAG_QUESTION_SET, relevance_question, sufficiency_question

RELEVANCE_ID = "relevance"
SUFFICIENCY_ID = "sufficiency"


class RagAction(StrEnum):
    GENERATE = "generate"
    ABSTAIN = "abstain"


class RagAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: RagAction
    question: str
    passages: tuple[Passage, ...]
    scores: tuple[float, ...] = ()
    p_relevance: float | None = None
    p_sufficient: float | None = None
    retried: bool = False
    reason: str
    context: str = ""
    question_set: str = RAG_QUESTION_SET


class PolicyRag:
    def __init__(
        self,
        index: VectorIndex,
        checker: DeciderChecker,
        *,
        k: int = 4,
        min_relevance: float = 0.55,
        min_sufficient: float = 0.55,
    ) -> None:
        self._index = index
        self._checker = checker
        self._k = k
        self._min_relevance = min_relevance
        self._min_sufficient = min_sufficient

    async def answer(self, question: str) -> RagAnswer:
        first = self._index.search(question, k=self._k)
        judged = await self._judge(question, first)
        if judged.action is RagAction.GENERATE:
            return judged
        retry_query = f"{question} Hearth & Cable return shipping refund policy"
        second = self._index.search(retry_query, k=self._k * 2)
        retried = await self._judge(question, second)
        return retried.model_copy(
            update={
                "retried": True,
                "reason": retried.reason
                if retried.action is RagAction.GENERATE
                else f"retry still insufficient ({retried.reason})",
            }
        )

    async def _judge(self, question: str, ranked: Sequence[tuple[Passage, float]]) -> RagAnswer:
        passages = tuple(p for p, _ in ranked)
        scores = tuple(s for _, s in ranked)
        context = _context(passages)
        if not passages:
            return RagAnswer(
                action=RagAction.ABSTAIN,
                question=question,
                passages=(),
                reason="no handbook passages retrieved",
            )
        try:
            p_rel, p_suf = await self._ask(question, passages)
        except DeciderError as exc:
            return RagAnswer(
                action=RagAction.ABSTAIN,
                question=question,
                passages=passages,
                scores=scores,
                reason=f"Decider unavailable ({exc}); abstaining",
                context=context,
            )
        ok = p_rel >= self._min_relevance and p_suf >= self._min_sufficient
        return RagAnswer(
            action=RagAction.GENERATE if ok else RagAction.ABSTAIN,
            question=question,
            passages=passages,
            scores=scores,
            p_relevance=p_rel,
            p_sufficient=p_suf,
            reason=(
                "handbook is relevant and sufficient"
                if ok
                else (
                    f"relevance {p_rel:.2f} / sufficiency {p_suf:.2f} "
                    f"(need {self._min_relevance:.2f} / {self._min_sufficient:.2f})"
                )
            ),
            context=context,
        )

    async def _ask(self, question: str, passages: Sequence[Passage]) -> tuple[float, float]:
        questions: dict[str, NoulQuestion] = {
            RELEVANCE_ID: relevance_question(question, passages),
            SUFFICIENCY_ID: sufficiency_question(question, passages),
        }
        response = await self._checker.ask(_state(question, passages), questions)
        try:
            return float(response.answers[RELEVANCE_ID]), float(response.answers[SUFFICIENCY_ID])
        except KeyError as exc:
            raise DeciderError(f"missing RAG answer {exc}") from exc


def _state(question: str, passages: Sequence[Passage]) -> str:
    body = "\n\n".join(f"[{p.id} {p.title}]\n{p.text}" for p in passages)
    return f"Customer: {question}\n\nHandbook:\n{body}"


def _context(passages: Sequence[Passage]) -> str:
    return "\n\n".join(f"### {p.title}\n{p.text}" for p in passages)
