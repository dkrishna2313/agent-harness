"""Tests for knowledge/retriever.py — EvidenceRetriever (lexical, J8.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from knowledge.models import Evidence, KnowledgeMetadata
from knowledge.retriever import (
    EvidenceRetriever,
    RetrievalResult,
    RetrievedEvidence,
    _INTENT_RISK,
    _compute_vocab_boost,
    detect_intent,
    tokenize_query,
)
from knowledge.store import KnowledgeStore


# ---------------------------------------------------------------------------
# Query tokenisation
# ---------------------------------------------------------------------------


def test_tokenize_basic():
    tokens = tokenize_query("deployment risks for SMRs")
    assert "deployment" in tokens
    assert "risks" in tokens
    assert "smrs" in tokens
    assert "for" not in tokens  # stopword


def test_tokenize_removes_stopwords():
    tokens = tokenize_query("what are the major risks")
    assert "what" not in tokens
    assert "are" not in tokens
    assert "the" not in tokens
    assert "major" in tokens
    assert "risks" in tokens


def test_tokenize_lowercases():
    tokens = tokenize_query("HALEU Fuel Availability")
    assert "haleu" in tokens
    assert "fuel" in tokens
    assert "availability" in tokens


def test_tokenize_empty():
    assert tokenize_query("") == []
    assert tokenize_query("the an a") == []  # all stopwords


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_evidence(statement: str, etype: str = "STRATEGIC", **kwargs) -> Evidence:
    return Evidence(
        statement=statement,
        evidence_type=etype,
        supporting_source_ids=["src-001"],
        extraction_run_id="run-001",
        **kwargs,
    )


def _make_meta(evidence_id: str, *, retrieval_enabled: bool = True,
               overall_score: float = 3.0, retrieval_priority: int = 3) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        evidence_id=evidence_id,
        retrieval_enabled=retrieval_enabled,
        overall_score=overall_score,
        retrieval_priority=retrieval_priority,
        review_status="AUTO_REVIEWED",
    )


@pytest.fixture()
def store(tmp_path: Path) -> KnowledgeStore:
    ks = KnowledgeStore(root=tmp_path / "ks")

    items = [
        _make_evidence("HALEU fuel availability is a critical deployment risk for advanced SMRs.",
                       "STRATEGIC"),
        _make_evidence("SMR licensing can take 5 to 10 years from application to first power.",
                       "STRATEGIC"),
        _make_evidence("The BWRX-300 uses natural circulation cooling, eliminating active pumps.",
                       "TECHNICAL"),
        _make_evidence("BWRX-300 is estimated to require 50% less capital cost per MW than large reactors.",
                       "TECHNICAL"),
        _make_evidence("Document number 005N9751 revision H.",
                       "ADMINISTRATIVE"),
        _make_evidence("Supply chain constraints remain a significant barrier to SMR deployment at scale.",
                       "STRATEGIC"),
    ]
    metas = [_make_meta(ev.evidence_id, retrieval_enabled=(ev.evidence_type != "ADMINISTRATIVE"))
             for ev in items]
    # Give the HALEU item a higher quality score so it ranks first
    metas[0] = _make_meta(items[0].evidence_id, overall_score=4.5, retrieval_priority=5)

    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch(metas, "smr")
    return ks


@pytest.fixture()
def retriever(store: KnowledgeStore) -> EvidenceRetriever:
    return EvidenceRetriever(store)


# ---------------------------------------------------------------------------
# Basic retrieval
# ---------------------------------------------------------------------------


def test_retrieve_returns_result(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks for SMRs", domain="smr")
    assert isinstance(result, RetrievalResult)
    assert result.query == "deployment risks for SMRs"
    assert result.retrieval_method == "lexical-v1"


def test_retrieve_finds_relevant_items(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks for SMRs", domain="smr")
    statements = [item.statement for item in result.items]
    assert any("deployment" in s.lower() for s in statements)


def test_retrieve_top_k_respected(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR deployment risks", domain="smr", top_k=2)
    assert len(result.items) <= 2


def test_retrieve_ranks_from_one(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks SMRs", domain="smr")
    assert result.items[0].rank == 1
    for i, item in enumerate(result.items, start=1):
        assert item.rank == i


def test_retrieve_scores_descending(retriever: EvidenceRetriever):
    result = retriever.retrieve("deployment risks SMRs", domain="smr")
    scores = [item.score for item in result.items]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_latency_recorded(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMRs", domain="smr")
    assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_retrieval_enabled_only_excludes_admin(retriever: EvidenceRetriever):
    result = retriever.retrieve("document number revision", domain="smr",
                                retrieval_enabled_only=True)
    statements = [item.statement for item in result.items]
    assert not any("Document number" in s for s in statements)


def test_retrieval_enabled_false_includes_all(retriever: EvidenceRetriever):
    result = retriever.retrieve("document number revision", domain="smr",
                                retrieval_enabled_only=False)
    statements = [item.statement for item in result.items]
    assert any("Document number" in s for s in statements)


def test_evidence_type_filter_strategic_only(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR", domain="smr",
                                evidence_types=["STRATEGIC"])
    for item in result.items:
        assert item.evidence_type == "STRATEGIC"


def test_evidence_type_filter_technical_only(retriever: EvidenceRetriever):
    result = retriever.retrieve("BWRX", domain="smr",
                                evidence_types=["TECHNICAL"])
    for item in result.items:
        assert item.evidence_type == "TECHNICAL"


def test_domain_filter(retriever: EvidenceRetriever):
    # Domain "ai_data_centers" has no evidence in fixture → empty result
    result = retriever.retrieve("deployment risks", domain="ai_data_centers")
    assert result.items == []
    assert result.total_candidates == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_higher_quality_score_breaks_ties(retriever: EvidenceRetriever):
    # HALEU item has overall_score=4.5 + priority=5 vs others at 3.0/3
    result = retriever.retrieve("deployment risk SMRs", domain="smr")
    assert "HALEU" in result.items[0].statement


def test_no_match_returns_empty(retriever: EvidenceRetriever):
    result = retriever.retrieve("quantum entanglement photonics", domain="smr")
    assert result.items == []
    assert result.matched_candidates == 0


def test_single_term_partial_match(retriever: EvidenceRetriever):
    result = retriever.retrieve("licensing", domain="smr")
    assert len(result.items) >= 1
    assert any("licens" in item.statement.lower() for item in result.items)


# ---------------------------------------------------------------------------
# Candidates and domains_searched
# ---------------------------------------------------------------------------


def test_total_candidates_counted(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR", domain="smr")
    # 6 evidence items in fixture, 5 with retrieval_enabled=True
    assert result.total_candidates == 6


def test_domains_searched_single(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR", domain="smr")
    assert result.domains_searched == ["smr"]


def test_domains_searched_all_when_none(retriever: EvidenceRetriever):
    result = retriever.retrieve("SMR")
    assert "smr" in result.domains_searched


# ---------------------------------------------------------------------------
# Empty query
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty(retriever: EvidenceRetriever):
    result = retriever.retrieve("the a an", domain="smr")  # all stopwords
    assert result.items == []
    assert result.domains_searched == []


# ---------------------------------------------------------------------------
# RetrievalResult.print_summary
# ---------------------------------------------------------------------------


def test_print_summary_runs_without_error(retriever: EvidenceRetriever, capsys):
    result = retriever.retrieve("deployment SMR risks", domain="smr", top_k=3)
    result.print_summary()
    captured = capsys.readouterr()
    assert "Query:" in captured.out
    assert "lexical-v1" in captured.out


def test_print_summary_empty(retriever: EvidenceRetriever, capsys):
    result = retriever.retrieve("quantum photonics", domain="smr")
    result.print_summary()
    captured = capsys.readouterr()
    assert "no results" in captured.out


# ---------------------------------------------------------------------------
# store additions: find_source, available_domains
# ---------------------------------------------------------------------------


def test_available_domains(store: KnowledgeStore):
    domains = store.available_domains()
    assert "smr" in domains


def test_find_source_returns_none_for_unknown(store: KnowledgeStore):
    assert store.find_source("nonexistent-id") is None


# ---------------------------------------------------------------------------
# J8.3a — intent detection
# ---------------------------------------------------------------------------


def test_detect_intent_risk_from_risks():
    assert detect_intent(tokenize_query("deployment risks for SMRs")) == _INTENT_RISK


def test_detect_intent_risk_from_barriers():
    assert detect_intent(tokenize_query("barriers to SMR deployment")) == _INTENT_RISK


def test_detect_intent_risk_from_licensing():
    assert detect_intent(tokenize_query("SMR licensing requirements")) == _INTENT_RISK


def test_detect_intent_risk_from_uncertainty():
    assert detect_intent(tokenize_query("SMR construction cost uncertainty")) == _INTENT_RISK


def test_detect_intent_risk_from_haleu():
    assert detect_intent(tokenize_query("HALEU availability")) == _INTENT_RISK


def test_detect_intent_none_for_neutral_query():
    assert detect_intent(tokenize_query("remote community diesel replacement")) is None


def test_detect_intent_none_for_empty():
    assert detect_intent([]) is None


# ---------------------------------------------------------------------------
# J8.3a — vocabulary boost
# ---------------------------------------------------------------------------


def test_vocab_boost_zero_for_no_intent():
    assert _compute_vocab_boost("SMR licensing barriers remain high.", None) == 0.0


def test_vocab_boost_zero_for_no_match():
    boost = _compute_vocab_boost("The reactor uses natural circulation cooling.", _INTENT_RISK)
    assert boost == 0.0


def test_vocab_boost_positive_for_risk_statement():
    boost = _compute_vocab_boost(
        "Key licensing barriers and regulatory uncertainty constrain near-term SMR deployment.", _INTENT_RISK
    )
    assert boost > 0.0


def test_vocab_boost_capped_at_max():
    # Statement with many risk-vocab hits should not exceed _INTENT_BOOST_MAX
    very_risky = (
        "Licensing barriers, regulatory uncertainty, supply chain constraints, "
        "HALEU fuel availability shortage, capital cost challenge, financing risks, "
        "construction schedule delays, and NRC certification requirements."
    )
    from knowledge.retriever import _INTENT_BOOST_MAX
    boost = _compute_vocab_boost(very_risky, _INTENT_RISK)
    assert 0.0 < boost <= _INTENT_BOOST_MAX


def test_vocab_boost_monotone_with_matches():
    one_term = _compute_vocab_boost("Licensing is a key concern.", _INTENT_RISK)
    two_terms = _compute_vocab_boost("Licensing barriers are a key concern.", _INTENT_RISK)
    assert two_terms >= one_term


# ---------------------------------------------------------------------------
# J8.3a — intent elevates risk-oriented results
# ---------------------------------------------------------------------------


@pytest.fixture()
def mixed_store(tmp_path: Path) -> KnowledgeStore:
    """Store with one clearly risk-vocabulary item and one high-coverage neutral item."""
    ks = KnowledgeStore(root=tmp_path / "mixed_ks")
    items = [
        _make_evidence(
            "SMR deployment faces regulatory licensing barriers and NRC certification uncertainty.",
            "STRATEGIC"
        ),
        _make_evidence(
            "SMRs offer potential for incremental deployment and modular scaling.",
            "STRATEGIC"
        ),
    ]
    metas = [_make_meta(ev.evidence_id) for ev in items]
    ks.write_evidence_batch(items, "smr")
    ks.write_metadata_batch(metas, "smr")
    return ks


def test_risk_intent_elevates_risk_statement(mixed_store: KnowledgeStore):
    retriever = EvidenceRetriever(mixed_store)
    result = retriever.retrieve("deployment risks for SMRs", domain="smr")
    # The risk-vocabulary statement should rank first
    assert result.items[0].rank == 1
    assert "licens" in result.items[0].statement.lower() or "barrier" in result.items[0].statement.lower()


def test_no_intent_uses_coverage_only(mixed_store: KnowledgeStore):
    retriever = EvidenceRetriever(mixed_store)
    # Neutral query — no risk intent — pure coverage decides
    result = retriever.retrieve("modular scaling SMRs", domain="smr")
    assert any("modular" in item.statement.lower() for item in result.items[:2])
