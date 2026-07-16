"""PH5.5a — Evidence Schema Foundation tests.

Verifies:
  1. v1 payloads (no v2 fields in JSON) still load and validate.
  2. v2 payloads with all new fields validate.
  3. All new v2 fields default safely (None or []).
  4. content_fingerprint is the full SHA-256 (64 hex chars).
  5. statement_fingerprint backward-compat: still 16 hex chars.
  6. content_fingerprint == statement_fingerprint[:16] is NOT guaranteed
     (statement_fingerprint is truncated; content_fingerprint is full).
  7. RetrievalProvenance can be constructed and serialized independently.
  8. Round-trip serialization of Evidence v2 is stable.
  9. Evidence remains frozen (no in-place mutation).
 10. v1 round-trip: old payloads serialize with new null fields, reload cleanly.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from knowledge.models import (
    Evidence,
    RetrievalProvenance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_V1_PAYLOAD = {
    "evidence_id": "ev-v1-001",
    "statement": "The BWRX-300 has a thermal output of 300 MWt.",
    "evidence_type": "TECHNICAL",
    "supporting_source_ids": ["src-abc123"],
    "profile_ids": ["smr"],
    "extraction_run_id": "run-001",
    "entity": "GE Vernova",
    "entity_type": "COMPANY",
    "scope": "UK",
    "category": "reactor design",
    "supersedes": [],
    "superseded_by": None,
    "contradiction_ids": [],
}

_STATEMENT = "The BWRX-300 has a thermal output of 300 MWt."


def _make_evidence(**kwargs) -> Evidence:
    defaults = dict(
        statement=_STATEMENT,
        supporting_source_ids=["src-abc"],
        extraction_run_id="run-001",
        category="reactor design",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


def _make_provenance(**kwargs) -> RetrievalProvenance:
    defaults = dict(
        evidence_id="ev-001",
        retrieval_query="What is the thermal output of BWRX-300?",
        retrieval_mode="hybrid",
        retrieval_rank=1,
        hybrid_score=0.842,
        lexical_score=0.731,
        semantic_score=0.903,
    )
    defaults.update(kwargs)
    return RetrievalProvenance(**defaults)


# ---------------------------------------------------------------------------
# 1. v1 payload backward compatibility
# ---------------------------------------------------------------------------


def test_v1_payload_loads_without_v2_fields():
    """A v1 JSONL payload (no v2 keys) deserializes cleanly."""
    ev = Evidence.model_validate(_V1_PAYLOAD)
    assert ev.evidence_id == "ev-v1-001"
    assert ev.statement == _V1_PAYLOAD["statement"]
    assert ev.category == "reactor design"


def test_v1_payload_loads_from_json_string():
    ev = Evidence.model_validate_json(json.dumps(_V1_PAYLOAD))
    assert ev.evidence_id == "ev-v1-001"


def test_v1_new_fields_default_safely():
    ev = Evidence.model_validate(_V1_PAYLOAD)
    # Passage provenance
    assert ev.excerpt is None
    assert ev.page_number is None
    assert ev.section_heading is None
    assert ev.chunk_id is None
    assert ev.char_offset_start is None
    assert ev.char_offset_end is None
    # Content
    assert ev.topics == []
    assert ev.temporal_reference is None
    assert ev.is_quantitative is False
    # Quality
    assert ev.evidence_confidence is None
    # Grounding
    assert ev.subquestion_assignments == []
    assert ev.investigation_area_assignments == []
    assert ev.grounding_strength is None
    assert ev.coverage_contribution is None
    # Relationships
    assert ev.corroborates == []
    assert ev.informed_hypotheses == []
    assert ev.informed_recommendations == []
    # Identity
    assert ev.schema_version == "1.0"
    assert ev.corpus_version is None


def test_v1_roundtrip_stable():
    """Load v1 payload, serialize, reload — existing field values are preserved."""
    ev = Evidence.model_validate(_V1_PAYLOAD)
    dumped = ev.model_dump()
    ev2 = Evidence.model_validate(dumped)
    assert ev2.evidence_id == ev.evidence_id
    assert ev2.statement == ev.statement
    assert ev2.entity == ev.entity
    assert ev2.category == ev.category
    assert ev2.supporting_source_ids == ev.supporting_source_ids


# ---------------------------------------------------------------------------
# 2. v2 payload validation
# ---------------------------------------------------------------------------


def test_v2_payload_with_all_new_fields():
    ev = _make_evidence(
        schema_version="2.0",
        corpus_version="2026-Q2.1",
        excerpt="The BWRX-300 produces 300 MWt of thermal output.",
        page_number=47,
        section_heading="3.2 Thermal Performance",
        chunk_id="chunk:p47-para2",
        char_offset_start=14200,
        char_offset_end=14600,
        topics=["SMR", "thermal output", "BWRX-300"],
        temporal_reference="2024",
        is_quantitative=True,
        evidence_confidence="HIGH",
        subquestion_assignments=["What is the thermal output?"],
        investigation_area_assignments=["SMR Technology"],
        grounding_strength="STRONG",
        coverage_contribution="STRONG",
        corroborates=["ev-corroborates-001"],
        informed_hypotheses=["hyp-001"],
        informed_recommendations=["rec-001"],
    )
    assert ev.schema_version == "2.0"
    assert ev.corpus_version == "2026-Q2.1"
    assert ev.excerpt == "The BWRX-300 produces 300 MWt of thermal output."
    assert ev.page_number == 47
    assert ev.section_heading == "3.2 Thermal Performance"
    assert ev.chunk_id == "chunk:p47-para2"
    assert ev.char_offset_start == 14200
    assert ev.char_offset_end == 14600
    assert ev.topics == ["SMR", "thermal output", "BWRX-300"]
    assert ev.temporal_reference == "2024"
    assert ev.is_quantitative is True
    assert ev.evidence_confidence == "HIGH"
    assert ev.subquestion_assignments == ["What is the thermal output?"]
    assert ev.investigation_area_assignments == ["SMR Technology"]
    assert ev.grounding_strength == "STRONG"
    assert ev.coverage_contribution == "STRONG"
    assert ev.corroborates == ["ev-corroborates-001"]
    assert ev.informed_hypotheses == ["hyp-001"]
    assert ev.informed_recommendations == ["rec-001"]


def test_v2_roundtrip():
    ev = _make_evidence(
        schema_version="2.0",
        excerpt="Test excerpt.",
        page_number=12,
        topics=["SMR"],
        evidence_confidence="MEDIUM",
        grounding_strength="MODERATE",
    )
    dumped = ev.model_dump()
    ev2 = Evidence.model_validate(dumped)
    assert ev2.schema_version == "2.0"
    assert ev2.excerpt == "Test excerpt."
    assert ev2.page_number == 12
    assert ev2.topics == ["SMR"]
    assert ev2.evidence_confidence == "MEDIUM"
    assert ev2.grounding_strength == "MODERATE"
    assert ev2.evidence_id == ev.evidence_id


# ---------------------------------------------------------------------------
# 3. Literal validation
# ---------------------------------------------------------------------------


def test_evidence_confidence_valid_literals():
    for val in ("HIGH", "MEDIUM", "LOW"):
        ev = _make_evidence(evidence_confidence=val)
        assert ev.evidence_confidence == val


def test_evidence_confidence_invalid_literal_rejected():
    with pytest.raises(ValidationError):
        _make_evidence(evidence_confidence="UNKNOWN")


def test_grounding_strength_valid_literals():
    for val in ("STRONG", "MODERATE", "WEAK"):
        ev = _make_evidence(grounding_strength=val)
        assert ev.grounding_strength == val


def test_coverage_contribution_valid_literals():
    for val in ("STRONG", "MODERATE", "WEAK"):
        ev = _make_evidence(coverage_contribution=val)
        assert ev.coverage_contribution == val


# ---------------------------------------------------------------------------
# 4. content_fingerprint — full SHA-256
# ---------------------------------------------------------------------------


def test_content_fingerprint_is_64_hex_chars():
    ev = _make_evidence()
    assert len(ev.content_fingerprint) == 64
    assert all(c in "0123456789abcdef" for c in ev.content_fingerprint)


def test_content_fingerprint_is_deterministic():
    ev1 = _make_evidence(statement="The reactor outputs 300 MWe.")
    ev2 = _make_evidence(statement="The reactor outputs 300 MWe.")
    assert ev1.content_fingerprint == ev2.content_fingerprint


def test_content_fingerprint_case_insensitive():
    ev1 = _make_evidence(statement="BWRX-300 has 300 MWt output.")
    ev2 = _make_evidence(statement="bwrx-300 has 300 mwt output.")
    assert ev1.content_fingerprint == ev2.content_fingerprint


def test_content_fingerprint_differs_for_different_statements():
    ev1 = _make_evidence(statement="Claim A is true.")
    ev2 = _make_evidence(statement="Claim B is different.")
    assert ev1.content_fingerprint != ev2.content_fingerprint


# ---------------------------------------------------------------------------
# 5. statement_fingerprint backward compat
# ---------------------------------------------------------------------------


def test_statement_fingerprint_is_16_hex_chars():
    """Existing 16-char dedup key is unchanged."""
    ev = _make_evidence()
    assert len(ev.statement_fingerprint) == 16
    assert all(c in "0123456789abcdef" for c in ev.statement_fingerprint)


def test_statement_fingerprint_is_prefix_of_content_fingerprint():
    """statement_fingerprint is the first 16 chars of content_fingerprint."""
    ev = _make_evidence()
    assert ev.content_fingerprint.startswith(ev.statement_fingerprint)


def test_statement_fingerprint_unchanged_for_v1_payload():
    ev = Evidence.model_validate(_V1_PAYLOAD)
    # Verify the fingerprint is still computed correctly from the statement
    import hashlib
    normalised = " ".join(ev.statement.lower().split())
    expected = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]
    assert ev.statement_fingerprint == expected


# ---------------------------------------------------------------------------
# 6. Evidence immutability preserved
# ---------------------------------------------------------------------------


def test_evidence_remains_frozen():
    ev = _make_evidence()
    with pytest.raises((ValidationError, TypeError)):
        ev.statement = "mutated"


def test_evidence_v2_field_immutable():
    ev = _make_evidence(excerpt="original excerpt")
    with pytest.raises((ValidationError, TypeError)):
        ev.excerpt = "changed"


# ---------------------------------------------------------------------------
# 7. RetrievalProvenance construction and serialization
# ---------------------------------------------------------------------------


def test_retrieval_provenance_constructs():
    rp = _make_provenance()
    assert rp.evidence_id == "ev-001"
    assert rp.retrieval_mode == "hybrid"
    assert rp.retrieval_rank == 1
    assert rp.hybrid_score == 0.842
    assert rp.lexical_score == 0.731
    assert rp.semantic_score == 0.903


def test_retrieval_provenance_defaults():
    rp = _make_provenance()
    assert rp.retrieval_model_version is None
    assert rp.metadata_factor is None
    assert rp.reranker == "passthrough"
    assert rp.rerank_score is None
    assert rp.rerank_rationale is None


def test_retrieval_provenance_with_llm_reranker():
    rp = _make_provenance(
        reranker="llm",
        rerank_score=0.92,
        rerank_rationale="Strong semantic match for thermal output query.",
        retrieval_model_version="all-MiniLM-L6-v2:1.2",
        metadata_factor=1.35,
    )
    assert rp.reranker == "llm"
    assert rp.rerank_score == 0.92
    assert rp.rerank_rationale == "Strong semantic match for thermal output query."
    assert rp.retrieval_model_version == "all-MiniLM-L6-v2:1.2"
    assert rp.metadata_factor == 1.35


def test_retrieval_provenance_mode_literals():
    for mode in ("lexical", "semantic", "hybrid"):
        rp = _make_provenance(retrieval_mode=mode)
        assert rp.retrieval_mode == mode


def test_retrieval_provenance_invalid_mode_rejected():
    with pytest.raises(ValidationError):
        _make_provenance(retrieval_mode="bm25")


def test_retrieval_provenance_reranker_literals():
    for reranker in ("passthrough", "llm", "none"):
        rp = _make_provenance(reranker=reranker)
        assert rp.reranker == reranker


def test_retrieval_provenance_invalid_reranker_rejected():
    with pytest.raises(ValidationError):
        _make_provenance(reranker="unknown")


def test_retrieval_provenance_serializes_to_dict():
    rp = _make_provenance()
    d = rp.model_dump()
    assert d["evidence_id"] == "ev-001"
    assert d["retrieval_mode"] == "hybrid"
    assert d["hybrid_score"] == 0.842
    assert d["reranker"] == "passthrough"
    assert d["rerank_score"] is None


def test_retrieval_provenance_roundtrip():
    rp = _make_provenance(
        reranker="llm",
        rerank_score=0.91,
        retrieval_model_version="all-MiniLM-L6-v2:1.2",
    )
    d = rp.model_dump()
    rp2 = RetrievalProvenance.model_validate(d)
    assert rp2.evidence_id == rp.evidence_id
    assert rp2.reranker == "llm"
    assert rp2.rerank_score == 0.91
    assert rp2.retrieval_model_version == "all-MiniLM-L6-v2:1.2"


def test_retrieval_provenance_json_roundtrip():
    rp = _make_provenance()
    j = rp.model_dump_json()
    rp2 = RetrievalProvenance.model_validate_json(j)
    assert rp2.evidence_id == rp.evidence_id
    assert rp2.hybrid_score == rp.hybrid_score


# ---------------------------------------------------------------------------
# 8. Serialization stability — model_dump includes v2 fields
# ---------------------------------------------------------------------------


def test_model_dump_includes_all_v2_fields():
    ev = _make_evidence()
    d = ev.model_dump()
    v2_fields = [
        "schema_version", "corpus_version",
        "excerpt", "page_number", "section_heading", "chunk_id",
        "char_offset_start", "char_offset_end",
        "topics", "temporal_reference", "is_quantitative",
        "evidence_confidence",
        "subquestion_assignments", "investigation_area_assignments",
        "grounding_strength", "coverage_contribution",
        "corroborates", "informed_hypotheses", "informed_recommendations",
        "content_fingerprint",
    ]
    for field in v2_fields:
        assert field in d, f"model_dump missing v2 field: {field}"


def test_model_dump_still_includes_v1_fields():
    ev = _make_evidence()
    d = ev.model_dump()
    v1_fields = [
        "evidence_id", "statement", "evidence_type",
        "supporting_source_ids", "profile_ids", "extraction_run_id",
        "entity", "entity_type", "scope", "category",
        "supersedes", "superseded_by", "contradiction_ids",
        "statement_fingerprint",
    ]
    for field in v1_fields:
        assert field in d, f"model_dump missing v1 field: {field}"


def test_json_serialization_does_not_raise():
    ev = _make_evidence(
        excerpt="Test excerpt.",
        page_number=5,
        topics=["test"],
        evidence_confidence="LOW",
    )
    j = ev.model_dump_json()
    assert "excerpt" in j
    assert "content_fingerprint" in j
    assert "statement_fingerprint" in j
