"""EvidenceReranker — LLM-assisted evidence selection after hybrid retrieval (J8.5).

Pipeline:
    HybridRetriever.retrieve(query, top_k=40)   → candidates (RetrievedEvidence list)
        ↓
    EvidenceReranker.rerank(query, candidates, top_k=10)
        ↓
    RerankResult  (RankedEvidence list, ordered by LLM relevance)

The LLM receives candidate statements and returns ordered evidence_ids with
relevance scores and rationales.  Hallucinated IDs are discarded; provenance
(retrieval scores, source) is preserved on every RankedEvidence item.

Implementations
---------------
PassthroughReranker  — identity; returns candidates in retrieval order (baseline)
LLMReranker          — Claude Haiku via tool_use structured output
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .retriever import EvidenceType, RetrievedEvidence

LOGGER = logging.getLogger(__name__)

RERANKER_PASSTHROUGH = "passthrough"
RERANKER_LLM_PREFIX = "llm"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class RankedEvidence:
    """One evidence item as ranked by an EvidenceReranker.

    Wraps the original RetrievedEvidence to preserve retrieval provenance
    (lexical_score, semantic_score, source) alongside the new LLM ranking.
    """

    candidate: RetrievedEvidence
    rank: int
    relevance_score: float = field(default=0.0)
    rationale: str = field(default="")

    @property
    def evidence(self):  # noqa: ANN201
        return self.candidate.evidence

    @property
    def statement(self) -> str:
        return self.candidate.statement

    @property
    def evidence_type(self) -> EvidenceType:
        return self.candidate.evidence_type

    @property
    def retrieval_score(self) -> float:
        return self.candidate.score


@dataclass
class RerankResult:
    """Output of EvidenceReranker.rerank()."""

    query: str
    items: list[RankedEvidence]
    candidates_evaluated: int
    reranker: str
    latency_ms: float
    # PH1 — LLM output normalization diagnostics (None for non-LLM rerankers).
    normalization: dict | None = None

    def print_summary(self, *, show_rationale: bool = False) -> None:
        print(
            f"\nQuery:   {self.query!r}\n"
            f"Reranker: {self.reranker}  |  "
            f"Candidates: {self.candidates_evaluated}  |  "
            f"Selected: {len(self.items)}  |  "
            f"Latency: {self.latency_ms:.0f}ms\n"
        )
        if not self.items:
            print("  (no results)\n")
            return

        col_w = 86
        print(f"  {'#':>3}  {'Rel':>5}  {'Ret':>5}  {'Type':<15}  Statement")
        print("  " + "─" * (col_w + 30))
        for item in self.items:
            stmt = item.statement
            if len(stmt) > col_w:
                stmt = stmt[: col_w - 1] + "…"
            print(
                f"  {item.rank:>3}  {item.relevance_score:>5.2f}  "
                f"{item.retrieval_score:>5.3f}  {item.evidence_type:<15}  {stmt}"
            )
            if show_rationale and item.rationale:
                print(f"         └─ {item.rationale}")
        print()


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class EvidenceReranker(ABC):
    """Abstract evidence reranker.

    A reranker takes a list of candidates (already retrieved and roughly
    ordered by hybrid score) and returns a smaller, better-ordered list
    by applying a richer relevance model.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int = 10,
    ) -> RerankResult:
        """Rerank candidates and return top_k items.

        Parameters
        ----------
        query:
            The original retrieval query string.
        candidates:
            Pre-retrieved evidence items (typically 20–60).
        top_k:
            Maximum number of items to return.
        """


# ---------------------------------------------------------------------------
# PassthroughReranker — identity / baseline
# ---------------------------------------------------------------------------


class PassthroughReranker(EvidenceReranker):
    """Returns candidates in their original retrieval order.

    Use as a baseline for A/B comparison against LLMReranker.
    """

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int = 10,
    ) -> RerankResult:
        t0 = time.monotonic()
        items = [
            RankedEvidence(
                candidate=c,
                rank=i + 1,
                relevance_score=round(c.score, 4),
                rationale="",
            )
            for i, c in enumerate(candidates[:top_k])
        ]
        return RerankResult(
            query=query,
            items=items,
            candidates_evaluated=len(candidates),
            reranker=RERANKER_PASSTHROUGH,
            latency_ms=(time.monotonic() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# LLMReranker — Claude Haiku via tool_use
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an evidence selection assistant for nuclear energy investment research.

Your task: given a user query and a set of candidate evidence items retrieved from a knowledge base, \
select and rank the items that BEST answer the query.

Selection criteria (in priority order):
1. Direct relevance — the statement directly addresses the query topic
2. Specificity — specific facts, named claims, data, or quantified statements over vague generalisations
3. Intent alignment — for risk queries prefer barrier/challenge/uncertainty/regulatory evidence; \
for application queries prefer use-case evidence; for technical queries prefer specification evidence
4. Non-redundancy — within the selected set, prefer items covering distinct aspects

Rules:
- Use only evidence_ids that appear verbatim in the candidates list
- Never fabricate, truncate, or alter an evidence_id
- Return a rationale of ≤ 15 words explaining relevance to the specific query\
"""

_USER_TMPL = """\
QUERY: {query}

Select the {top_k} most relevant items from the {n} candidates below, ordered best-first.

CANDIDATES:
{block}

Return rankings using the return_rankings tool.\
"""


class LLMReranker(EvidenceReranker):
    """Claude-backed evidence reranker using tool_use structured output.

    The model receives candidate statements and returns an ordered list of
    evidence_ids with relevance scores (0–1) and short rationales.

    Parameters
    ----------
    client:
        anthropic.Anthropic client. Created automatically if not provided.
    model:
        Model for reranking. Haiku is recommended (fast, cheap, sufficient).
    """

    def __init__(
        self,
        client: object | None = None,
        model: str = "claude-haiku-4-5-20251001",
    ) -> None:
        if client is None:
            import anthropic  # type: ignore[import]
            client = anthropic.Anthropic()
        self._client = client
        self._model = model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int = 10,
    ) -> RerankResult:
        t0 = time.monotonic()

        if not candidates:
            return RerankResult(
                query=query,
                items=[],
                candidates_evaluated=0,
                reranker=f"{RERANKER_LLM_PREFIX}-{self._model}",
                latency_ms=0.0,
            )

        by_id = {c.evidence.evidence_id: c for c in candidates}
        effective_k = min(top_k, len(candidates))

        user_msg = _USER_TMPL.format(
            query=query,
            top_k=effective_k,
            n=len(candidates),
            block=self._format_candidates(candidates),
        )

        raw = self._call_llm(user_msg, effective_k)

        # PH1 — normalize raw LLM output at the boundary BEFORE typed access.
        # The model intermittently emits `rankings` as bare strings or malformed
        # items; normalization coerces bare id-strings to objects and drops the
        # rest so the `.get()` below can never raise. Malformed output degrades to
        # the existing retrieval-order fallback (valid == 0).
        from research_agent.llm_normalize import normalize_llm_items
        normalized, norm_diag = normalize_llm_items(
            raw,
            required_fields=("evidence_id",),
            coerce_str_key="evidence_id",
            component="reranker",
        )

        # Validate: drop hallucinated IDs
        valid = [r for r in normalized if r.get("evidence_id") in by_id]
        n_dropped = len(normalized) - len(valid)
        if n_dropped:
            LOGGER.warning(
                "reranker: dropped %d/%d hallucinated evidence_ids (normalized=%d  valid=%d) — "
                "retrieval-order fallback will apply if valid=0",
                n_dropped, len(normalized), len(normalized), len(valid),
            )
        # Zero valid items → EvidenceAgent's retrieval-order fallback engages.
        norm_diag["fallback_used"] = len(valid) == 0

        # Deduplicate preserving first-occurrence order
        seen: set[str] = set()
        deduped = []
        for r in valid:
            eid = r["evidence_id"]
            if eid not in seen:
                seen.add(eid)
                deduped.append(r)

        def _safe_score(v: Any) -> float:
            # PH1 — a non-numeric relevance_score must not crash the boundary.
            try:
                return max(0.0, min(1.0, float(v)))
            except (TypeError, ValueError):
                return 0.0

        items = [
            RankedEvidence(
                candidate=by_id[r["evidence_id"]],
                rank=i + 1,
                relevance_score=_safe_score(r.get("relevance_score", 0.0)),
                rationale=str(r.get("rationale", "")),
            )
            for i, r in enumerate(deduped[:top_k])
        ]

        latency_ms = (time.monotonic() - t0) * 1000
        LOGGER.info(
            "reranker: query=%r  candidates=%d  selected=%d  latency=%.0fms",
            query, len(candidates), len(items), latency_ms,
        )

        return RerankResult(
            query=query,
            items=items,
            candidates_evaluated=len(candidates),
            reranker=f"{RERANKER_LLM_PREFIX}-{self._model}",
            latency_ms=latency_ms,
            normalization=norm_diag,
        )

    @staticmethod
    def _format_candidates(candidates: list[RetrievedEvidence]) -> str:
        parts = []
        for i, c in enumerate(candidates, start=1):
            src_line = ""
            if c.source:
                src_line = f"\n    source: {c.source.title!r}"
            parts.append(
                f"[{i}] evidence_id: {c.evidence.evidence_id!r}\n"
                f"    type: {c.evidence_type}\n"
                f"    retrieval_score: {c.score:.3f}{src_line}\n"
                f"    statement: {c.statement}"
            )
        return "\n\n".join(parts)

    def _call_llm(self, user_msg: str, top_k: int) -> list[dict]:
        tool = {
            "name": "return_rankings",
            "description": "Return the ranked evidence selection",
            "input_schema": {
                "type": "object",
                "properties": {
                    "rankings": {
                        "type": "array",
                        "description": f"Up to {top_k} selected evidence items, best first",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_id": {
                                    "type": "string",
                                    "description": "Exact evidence_id from the candidates list",
                                },
                                "relevance_score": {
                                    "type": "number",
                                    "description": "Relevance to the query: 0.0 (irrelevant) → 1.0 (perfect)",
                                },
                                "rationale": {
                                    "type": "string",
                                    "description": "≤15-word explanation of why this evidence answers the query",
                                },
                            },
                            "required": ["evidence_id", "relevance_score", "rationale"],
                        },
                    }
                },
                "required": ["rankings"],
            },
        }

        try:
            response = self._client.messages.create(  # type: ignore[attr-defined]
                model=self._model,
                max_tokens=2048,
                system=_SYSTEM,
                tools=[tool],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": user_msg}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "return_rankings":
                    rankings = block.input.get("rankings", [])
                    LOGGER.debug(
                        "reranker: LLM returned %d raw rankings (top_k=%d  stop_reason=%s)",
                        len(rankings), top_k, getattr(response, "stop_reason", "?"),
                    )
                    return rankings
            LOGGER.warning(
                "reranker: model did not call return_rankings (stop_reason=%s) — returning empty. "
                "All %d candidates will be dropped and retrieval-order fallback will apply.",
                getattr(response, "stop_reason", "?"), top_k,
            )
            return []
        except Exception as exc:
            LOGGER.warning(
                "reranker: LLM call failed (%s) — returning empty. "
                "All candidates will be dropped and retrieval-order fallback will apply.",
                exc,
            )
            return []
