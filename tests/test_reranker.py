"""Tests for knowledge/reranker.py — EvidenceReranker (J8.5)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from knowledge.models import Evidence, KnowledgeMetadata
from knowledge.reranker import (
    EvidenceReranker,
    LLMReranker,
    PassthroughReranker,
    RankedEvidence,
    RerankResult,
    RERANKER_PASSTHROUGH,
    RERANKER_LLM_PREFIX,
)
from knowledge.retriever import RetrievedEvidence


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    statement: str,
    score: float = 0.5,
    etype: str = "STRATEGIC",
) -> RetrievedEvidence:
    ev = Evidence(
        statement=statement,
        evidence_type=etype,
        supporting_source_ids=["src-001"],
        extraction_run_id="run-001",
    )
    meta = KnowledgeMetadata(
        evidence_id=ev.evidence_id,
        retrieval_enabled=True,
        overall_score=3.0,
        retrieval_priority=3,
        review_status="AUTO_REVIEWED",
    )
    return RetrievedEvidence(
        evidence=ev,
        metadata=meta,
        score=score,
        rank=1,
    )


def _candidates(n: int = 5) -> list[RetrievedEvidence]:
    statements = [
        "SMR licensing barriers remain a major challenge for near-term deployment.",
        "HALEU fuel availability is critical for advanced SMR deployment.",
        "The BWRX-300 uses natural circulation cooling, eliminating active pumps.",
        "SMRs are intended to replace diesel generators for remote communities.",
        "Construction costs for FOAK SMR units are highly uncertain.",
        "NRC certification of new reactor designs can take 5–10 years.",
        "SMRs have a lower capital outlay per unit than large reactors.",
        "Supply chain constraints limit near-term SMR deployment scale.",
    ]
    return [
        _make_candidate(statements[i % len(statements)], score=round(0.9 - i * 0.05, 2))
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Mock LLM client
# ---------------------------------------------------------------------------


@dataclass
class _ToolUseBlock:
    type: str = "tool_use"
    name: str = "return_rankings"
    input: dict = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.input is None:
            self.input = {}


@dataclass
class _MockResponse:
    content: list[Any]


class MockAnthropicClient:
    """Deterministic mock that reverses candidate order and assigns fixed scores."""

    def __init__(self, evidence_ids: list[str], inject_hallucinated: bool = False):
        self._ids = evidence_ids
        self._inject_hallucinated = inject_hallucinated

    @property
    def messages(self) -> "MockAnthropicClient":
        return self

    def create(self, **kwargs: Any) -> _MockResponse:
        rankings = [
            {
                "evidence_id": eid,
                "relevance_score": round(1.0 - i * 0.1, 2),
                "rationale": f"Relevant item {i + 1}",
            }
            for i, eid in enumerate(reversed(self._ids))
        ]
        if self._inject_hallucinated:
            rankings.insert(1, {
                "evidence_id": "hallucinated-id-does-not-exist",
                "relevance_score": 0.99,
                "rationale": "This ID was fabricated",
            })
        return _MockResponse(
            content=[_ToolUseBlock(input={"rankings": rankings})]
        )


class MockReranker(EvidenceReranker):
    """Deterministic reranker — reverses candidate order. No LLM call."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedEvidence],
        *,
        top_k: int = 10,
    ) -> RerankResult:
        import time
        t0 = time.monotonic()
        top = list(reversed(candidates))[:top_k]
        items = [
            RankedEvidence(
                candidate=c,
                rank=i + 1,
                relevance_score=round(1.0 - i * 0.1, 2),
                rationale=f"Mock rationale {i + 1}",
            )
            for i, c in enumerate(top)
        ]
        return RerankResult(
            query=query,
            items=items,
            candidates_evaluated=len(candidates),
            reranker="mock",
            latency_ms=(time.monotonic() - t0) * 1000,
        )


# ---------------------------------------------------------------------------
# RankedEvidence
# ---------------------------------------------------------------------------


def test_ranked_evidence_properties():
    c = _make_candidate("SMR licensing barriers.", score=0.7, etype="STRATEGIC")
    ranked = RankedEvidence(candidate=c, rank=1, relevance_score=0.9, rationale="Directly relevant.")
    assert ranked.statement == "SMR licensing barriers."
    assert ranked.evidence_type == "STRATEGIC"
    assert ranked.retrieval_score == pytest.approx(0.7)
    assert ranked.evidence is c.evidence


# ---------------------------------------------------------------------------
# RerankResult
# ---------------------------------------------------------------------------


def test_rerank_result_print_no_rationale(capsys):
    candidates = _candidates(3)
    reranker = MockReranker()
    result = reranker.rerank("deployment risks", candidates, top_k=3)
    result.print_summary()
    out = capsys.readouterr().out
    assert "Query:" in out
    assert "Reranker:" in out
    assert "mock" in out


def test_rerank_result_print_with_rationale(capsys):
    candidates = _candidates(3)
    reranker = MockReranker()
    result = reranker.rerank("deployment risks", candidates, top_k=3)
    result.print_summary(show_rationale=True)
    out = capsys.readouterr().out
    assert "Mock rationale" in out
    assert "└─" in out


def test_rerank_result_print_empty(capsys):
    result = RerankResult(query="q", items=[], candidates_evaluated=0, reranker="mock", latency_ms=0.0)
    result.print_summary()
    out = capsys.readouterr().out
    assert "no results" in out


def test_rerank_result_latency_recorded():
    candidates = _candidates(5)
    reranker = MockReranker()
    result = reranker.rerank("query", candidates)
    assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# PassthroughReranker
# ---------------------------------------------------------------------------


def test_passthrough_preserves_order():
    candidates = _candidates(5)
    reranker = PassthroughReranker()
    result = reranker.rerank("deployment risks", candidates, top_k=5)
    assert [item.candidate for item in result.items] == candidates[:5]


def test_passthrough_top_k_truncates():
    candidates = _candidates(8)
    reranker = PassthroughReranker()
    result = reranker.rerank("query", candidates, top_k=3)
    assert len(result.items) == 3


def test_passthrough_rank_sequential():
    candidates = _candidates(4)
    reranker = PassthroughReranker()
    result = reranker.rerank("query", candidates)
    assert [item.rank for item in result.items] == list(range(1, len(result.items) + 1))


def test_passthrough_reranker_name():
    result = PassthroughReranker().rerank("q", _candidates(2))
    assert result.reranker == RERANKER_PASSTHROUGH


def test_passthrough_relevance_score_matches_retrieval():
    candidates = _candidates(3)
    result = PassthroughReranker().rerank("q", candidates, top_k=3)
    for item in result.items:
        assert item.relevance_score == pytest.approx(item.retrieval_score)


def test_passthrough_candidates_evaluated():
    candidates = _candidates(7)
    result = PassthroughReranker().rerank("q", candidates, top_k=3)
    assert result.candidates_evaluated == 7


def test_passthrough_empty_candidates():
    result = PassthroughReranker().rerank("q", [], top_k=5)
    assert result.items == []
    assert result.candidates_evaluated == 0


# ---------------------------------------------------------------------------
# MockReranker (custom deterministic reranker contract)
# ---------------------------------------------------------------------------


def test_mock_reranker_reverses_order():
    candidates = _candidates(4)
    result = MockReranker().rerank("q", candidates, top_k=4)
    assert result.items[0].candidate is candidates[-1]
    assert result.items[-1].candidate is candidates[0]


def test_mock_reranker_includes_rationale():
    result = MockReranker().rerank("q", _candidates(3), top_k=3)
    for item in result.items:
        assert item.rationale != ""


# ---------------------------------------------------------------------------
# LLMReranker with mocked client
# ---------------------------------------------------------------------------


def test_llm_reranker_uses_correct_order():
    candidates = _candidates(4)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    reranker = LLMReranker(client=client)
    result = reranker.rerank("deployment risks SMRs", candidates, top_k=4)

    # Mock reverses the IDs; check that we got them back in reverse order
    assert len(result.items) == 4
    returned_ids = [item.evidence.evidence_id for item in result.items]
    assert returned_ids == list(reversed(ids))


def test_llm_reranker_top_k_truncates():
    candidates = _candidates(6)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    assert len(result.items) == 3


def test_llm_reranker_drops_hallucinated_ids():
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids, inject_hallucinated=True)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=4)
    returned_ids = {item.evidence.evidence_id for item in result.items}
    assert "hallucinated-id-does-not-exist" not in returned_ids


def test_llm_reranker_preserves_provenance():
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    for item in result.items:
        assert item.candidate in candidates
        assert item.evidence is item.candidate.evidence


def test_llm_reranker_relevance_score_clamped():
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    for item in result.items:
        assert 0.0 <= item.relevance_score <= 1.0


def test_llm_reranker_name():
    candidates = _candidates(2)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client, model="claude-haiku-4-5-20251001").rerank("q", candidates)
    assert result.reranker == f"{RERANKER_LLM_PREFIX}-claude-haiku-4-5-20251001"


def test_llm_reranker_candidates_evaluated():
    candidates = _candidates(5)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    assert result.candidates_evaluated == 5


def test_llm_reranker_empty_candidates():
    client = MockAnthropicClient([])
    result = LLMReranker(client=client).rerank("q", [], top_k=5)
    assert result.items == []
    assert result.candidates_evaluated == 0


def test_llm_reranker_rank_sequential():
    candidates = _candidates(4)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockAnthropicClient(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=4)
    assert [item.rank for item in result.items] == list(range(1, len(result.items) + 1))


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class MockClientWithDuplicates:
    """Returns the same evidence_id twice in the rankings."""

    def __init__(self, evidence_ids: list[str]):
        self._ids = evidence_ids

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        first = self._ids[0]
        rankings = [
            {"evidence_id": first, "relevance_score": 0.9, "rationale": "First"},
            {"evidence_id": first, "relevance_score": 0.8, "rationale": "Duplicate"},
            {"evidence_id": self._ids[1], "relevance_score": 0.7, "rationale": "Second"},
        ]
        return _MockResponse(content=[_ToolUseBlock(input={"rankings": rankings})])


def test_llm_reranker_deduplicates():
    candidates = _candidates(4)
    ids = [c.evidence.evidence_id for c in candidates]
    client = MockClientWithDuplicates(ids)

    result = LLMReranker(client=client).rerank("q", candidates, top_k=5)
    returned_ids = [item.evidence.evidence_id for item in result.items]
    assert len(returned_ids) == len(set(returned_ids))


# ---------------------------------------------------------------------------
# LLM error handling
# ---------------------------------------------------------------------------


class MockClientThatFails:
    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        raise RuntimeError("Simulated API failure")


def test_llm_reranker_handles_error_gracefully():
    candidates = _candidates(3)
    result = LLMReranker(client=MockClientThatFails()).rerank("q", candidates, top_k=3)
    assert result.items == []
    assert result.candidates_evaluated == 3


# ---------------------------------------------------------------------------
# PH1 — LLM output boundary normalization (reranker must never crash on
# malformed tool payloads; must degrade to retrieval-order fallback).
# ---------------------------------------------------------------------------

class _ShapeClient:
    """Mock Anthropic client returning an arbitrary `rankings` payload shape."""

    def __init__(self, rankings):
        self._rankings = rankings

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        return _MockResponse(content=[_ToolUseBlock(input={"rankings": self._rankings})])


def test_ph1_bare_string_rankings_do_not_crash():
    """The exact J10 defect: rankings is a list of bare id strings."""
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    client = _ShapeClient(list(reversed(ids)))  # list[str], not list[dict]

    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    # Coerced to objects → valid selection, no exception.
    assert len(result.items) == 3
    assert result.normalization["items_valid"] == 3
    assert result.normalization["fallback_used"] is False


def test_ph1_mixed_shapes_drop_invalid_keep_valid():
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    client = _ShapeClient([
        ids[0],                                   # bare string → coerced
        {"evidence_id": ids[1], "relevance_score": 0.8, "rationale": "ok"},  # valid
        {"relevance_score": 0.5},                 # missing evidence_id → dropped
        99,                                       # non-dict/str → dropped
    ])
    result = LLMReranker(client=client).rerank("q", candidates, top_k=5)
    returned = {i.evidence.evidence_id for i in result.items}
    assert returned == {ids[0], ids[1]}
    assert result.normalization["items_received"] == 4
    assert result.normalization["items_dropped"] == 2


def test_ph1_all_malformed_triggers_fallback_flag():
    candidates = _candidates(3)
    client = _ShapeClient([{"no_id": 1}, 42, None])  # nothing usable
    result = LLMReranker(client=client).rerank("q", candidates, top_k=3)
    assert result.items == []                       # empty → EvidenceAgent falls back
    assert result.normalization["fallback_used"] is True


def test_ph1_non_numeric_score_does_not_crash():
    candidates = _candidates(2)
    ids = [c.evidence.evidence_id for c in candidates]
    client = _ShapeClient([
        {"evidence_id": ids[0], "relevance_score": "high", "rationale": "x"},  # bad score
    ])
    result = LLMReranker(client=client).rerank("q", candidates, top_k=2)
    assert len(result.items) == 1
    assert result.items[0].relevance_score == 0.0   # safe-coerced


def test_ph1_normalization_diagnostics_present_on_valid_run():
    candidates = _candidates(3)
    ids = [c.evidence.evidence_id for c in candidates]
    result = LLMReranker(client=MockAnthropicClient(ids)).rerank("q", candidates, top_k=3)
    norm = result.normalization
    assert norm["component"] == "reranker"
    assert norm["items_received"] == 3
    assert norm["items_valid"] == 3
    assert norm["items_dropped"] == 0
