"""Tests for J8.4 — hybrid retrieval (semantic + lexical).

Uses a mock EmbeddingProvider so tests run without downloading model weights.
The mock assigns deterministic vectors: items about "deployment" or "risk"
get vectors that are semantically close to a "deployment risk" query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from knowledge.embeddings import EmbeddingProvider, embed_evidence_batch, get_provider
from knowledge.models import Evidence, KnowledgeMetadata
from knowledge.retriever import (
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
    EvidenceRetriever,
    RetrievalResult,
    RetrievedEvidence,
)
from knowledge.store import KnowledgeStore


# ---------------------------------------------------------------------------
# Mock embedding provider
# ---------------------------------------------------------------------------


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / (n + 1e-9) for x in v]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings for testing.

    3-dim vector space:
      dim 0 = "deployment / SMR" signal
      dim 1 = "risk / barrier" signal
      dim 2 = "technical / engineering" signal

    Text is mapped to a unit vector based on keyword detection.
    """

    model_name = "mock-3d-v1"
    dimension = 3

    _PROFILES: list[tuple[str, list[float]]] = [
        ("deployment risk smr barrier licens", [0.8, 0.8, 0.1]),
        ("smr deployment", [0.9, 0.1, 0.1]),
        ("technical engineering reactor cooling", [0.1, 0.1, 0.9]),
        ("haleu fuel availability", [0.5, 0.7, 0.2]),
    ]

    def _vec(self, text: str) -> list[float]:
        s = text.lower()
        risk = sum(1 for w in ("risk", "barrier", "licens", "uncertain", "challenge", "cost") if w in s)
        depl = sum(1 for w in ("deployment", "smr", "haleu", "fuel", "schedule") if w in s)
        tech = sum(1 for w in ("technical", "engineering", "reactor", "cooling", "circulation") if w in s)
        return _unit([float(depl) + 0.1, float(risk) + 0.1, float(tech) + 0.1])

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ev(statement: str, etype: str = "STRATEGIC") -> Evidence:
    return Evidence(
        statement=statement,
        evidence_type=etype,
        supporting_source_ids=["src-001"],
        extraction_run_id="run-001",
    )


def _make_meta(evidence_id: str) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        evidence_id=evidence_id,
        retrieval_enabled=True,
        overall_score=3.0,
        retrieval_priority=3,
        review_status="AUTO_REVIEWED",
    )


@pytest.fixture()
def store_with_embeddings(tmp_path: Path) -> KnowledgeStore:
    ks = KnowledgeStore(root=tmp_path / "ks")
    provider = MockEmbeddingProvider()

    items = [
        _make_ev("SMR deployment faces regulatory licensing barriers and NRC certification."),
        _make_ev("HALEU fuel availability is a critical deployment risk for advanced SMRs."),
        _make_ev("The BWRX-300 uses natural circulation cooling, eliminating active pumps.", "TECHNICAL"),
        _make_ev("SMR licensing can take 5 to 10 years from application to first power."),
        _make_ev("SMRs are intended to replace diesel generators for off-grid mining.", "STRATEGIC"),
    ]
    metas = [_make_meta(ev.evidence_id) for ev in items]

    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch(metas, "smr")

    # Pre-embed all items
    embed_evidence_batch(items, ks, provider, force=True)
    return ks


@pytest.fixture()
def retriever(store_with_embeddings: KnowledgeStore) -> EvidenceRetriever:
    return EvidenceRetriever(store_with_embeddings, provider=MockEmbeddingProvider())


# ---------------------------------------------------------------------------
# EmbeddingProvider contract
# ---------------------------------------------------------------------------


def test_mock_provider_returns_one_vector_per_text():
    p = MockEmbeddingProvider()
    results = p.embed(["text one", "text two", "text three"])
    assert len(results) == 3
    assert all(len(v) == 3 for v in results)


def test_mock_provider_embed_one():
    p = MockEmbeddingProvider()
    v = p.embed_one("SMR deployment risk")
    assert len(v) == 3


def test_mock_provider_vectors_are_unit_length():
    p = MockEmbeddingProvider()
    v = p.embed_one("deployment risk barriers")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# embed_evidence_batch
# ---------------------------------------------------------------------------


def test_embed_evidence_batch_persists_vectors(tmp_path: Path):
    ks = KnowledgeStore(root=tmp_path / "ks")
    items = [_make_ev("SMR deployment risk"), _make_ev("cooling system")]
    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch([_make_meta(ev.evidence_id) for ev in items], "smr")

    p = MockEmbeddingProvider()
    embedded, skipped = embed_evidence_batch(items, ks, p)
    assert embedded == 2
    assert skipped == 0

    for ev in items:
        assert ks.read_embedding(ev.evidence_id) is not None
        assert len(ks.read_embedding(ev.evidence_id)) == 3


def test_embed_evidence_batch_skips_existing(tmp_path: Path):
    ks = KnowledgeStore(root=tmp_path / "ks")
    items = [_make_ev("SMR risk")]
    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch([_make_meta(ev.evidence_id) for ev in items], "smr")

    p = MockEmbeddingProvider()
    embed_evidence_batch(items, ks, p)
    embedded2, skipped2 = embed_evidence_batch(items, ks, p)  # second run
    assert embedded2 == 0
    assert skipped2 == 1


def test_embed_evidence_batch_force_overrides_existing(tmp_path: Path):
    ks = KnowledgeStore(root=tmp_path / "ks")
    items = [_make_ev("SMR risk")]
    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch([_make_meta(ev.evidence_id) for ev in items], "smr")

    p = MockEmbeddingProvider()
    embed_evidence_batch(items, ks, p)
    embedded2, skipped2 = embed_evidence_batch(items, ks, p, force=True)
    assert embedded2 == 1
    assert skipped2 == 0


# ---------------------------------------------------------------------------
# Semantic mode
# ---------------------------------------------------------------------------


def test_semantic_mode_returns_result(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks for SMRs", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")
    assert isinstance(result, RetrievalResult)
    assert result.mode == RETRIEVAL_MODE_SEMANTIC
    assert result.semantic_model == "mock-3d-v1"


def test_semantic_mode_finds_relevant_items(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks for SMRs", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")
    assert len(result.items) > 0
    assert all(item.semantic_score > 0 for item in result.items)


def test_semantic_mode_scores_descending(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risk SMR", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)


def test_semantic_requires_provider(store_with_embeddings: KnowledgeStore):
    retriever_no_provider = EvidenceRetriever(store_with_embeddings, provider=None)
    with pytest.raises(ValueError, match="EmbeddingProvider"):
        retriever_no_provider.retrieve("deployment risks", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")


def test_semantic_lexical_score_is_zero(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")
    for item in result.items:
        assert item.lexical_score == 0.0


# ---------------------------------------------------------------------------
# Hybrid mode
# ---------------------------------------------------------------------------


def test_hybrid_mode_returns_result(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks for SMRs", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    assert result.mode == RETRIEVAL_MODE_HYBRID
    assert result.semantic_model == "mock-3d-v1"


def test_hybrid_mode_items_have_both_scores(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risk SMR barrier", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    # Items that appear in both paths should have both scores > 0
    both = [item for item in result.items if item.lexical_score > 0 and item.semantic_score > 0]
    assert len(both) > 0


def test_hybrid_mode_scores_descending(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks SMRs", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_observability_fields(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks SMR barrier", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    # duplicates_removed = items that appear in BOTH lexical and semantic sets
    assert result.duplicates_removed >= 0
    assert result.lexical_candidates >= 0
    assert result.semantic_candidates >= 0
    # merged_candidates ≤ lexical + semantic (no double-counting)
    assert result.merged_candidates <= result.lexical_candidates + result.semantic_candidates


def test_hybrid_no_duplicates_in_results(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR deployment risks barriers", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    ids = [item.evidence.evidence_id for item in result.items]
    assert len(ids) == len(set(ids))


def test_hybrid_top_k_respected(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risk", mode=RETRIEVAL_MODE_HYBRID, domain="smr", top_k=2)
    assert len(result.items) <= 2


def test_hybrid_requires_provider(store_with_embeddings: KnowledgeStore):
    retriever_no_provider = EvidenceRetriever(store_with_embeddings, provider=None)
    with pytest.raises(ValueError, match="EmbeddingProvider"):
        retriever_no_provider.retrieve("deployment risk", mode=RETRIEVAL_MODE_HYBRID, domain="smr")


# ---------------------------------------------------------------------------
# Lexical mode is unchanged (backward compat)
# ---------------------------------------------------------------------------


def test_lexical_mode_unchanged(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks SMRs", mode=RETRIEVAL_MODE_LEXICAL, domain="smr")
    assert result.mode == RETRIEVAL_MODE_LEXICAL
    assert result.semantic_model is None
    for item in result.items:
        assert item.semantic_score == 0.0


def test_invalid_mode_raises(retriever: EvidenceRetriever):
    with pytest.raises(ValueError):
        retriever.retrieve("deployment", mode="fuzzy", domain="smr")


# ---------------------------------------------------------------------------
# Semantic can surface items lexical misses
# ---------------------------------------------------------------------------


def test_semantic_finds_items_lexical_misses(store_with_embeddings: KnowledgeStore):
    """Semantic should find 'cooling circulation' for a 'technical engineering' query
    even when none of those exact words overlap heavily with the query terms."""
    provider = MockEmbeddingProvider()
    retriever = EvidenceRetriever(store_with_embeddings, provider=provider)

    # Semantic query for technical content
    sem_result = retriever.retrieve("reactor technical", mode=RETRIEVAL_MODE_SEMANTIC, domain="smr")
    sem_ids = {item.evidence.evidence_id for item in sem_result.items}

    # The cooling statement should score high semantically
    cooling_items = [
        item for item in sem_result.items
        if "cooling" in item.statement.lower()
    ]
    assert len(cooling_items) > 0


# ---------------------------------------------------------------------------
# RetrievalResult print_summary covers hybrid columns
# ---------------------------------------------------------------------------


def test_hybrid_print_summary_shows_component_scores(retriever: EvidenceRetriever, capsys):
    result = retriever.retrieve("deployment risk SMR", mode=RETRIEVAL_MODE_HYBRID, domain="smr")
    result.print_summary()
    captured = capsys.readouterr()
    assert "Lex" in captured.out
    assert "Sem" in captured.out
    assert "Merged" in captured.out


def test_lexical_print_summary_unchanged(retriever: EvidenceRetriever, capsys):
    result = retriever.retrieve("deployment risks SMRs", mode=RETRIEVAL_MODE_LEXICAL, domain="smr")
    result.print_summary()
    captured = capsys.readouterr()
    assert "Lex" not in captured.out
    assert "Sem" not in captured.out
    assert "Candidates" in captured.out
