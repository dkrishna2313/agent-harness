"""Tests for research_agent.extraction_diagnostics (JH1b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.extraction_diagnostics import (
    ALL_FAILURE_STAGES,
    ATTRIBUTION_FAILURE,
    BUDGET_EXCLUSION,
    CROSS_CHUNK,
    DUPLICATE_SUPPRESSION,
    EMPTY_EXTRACTION,
    NO_LLM_OUTPUT,
    PARSER_FAILURE,
    POST_PROCESSING_REJECTION,
    QUALITY_THRESHOLD_REJECTION,
    SCHEMA_VALIDATION_FAILURE,
    SPEC_FAILURE_STAGES,
    UNKNOWN,
    _chunk_has_topic_match,
    _is_garbled,
    _normalise,
    _sentences_with_topic_match,
    _signal_strength,
    analyze_document_failures,
    build_failure_diagnostics,
    build_failure_summary,
    classify_chunk_failure,
    compute_top_missed_chunks,
)
from research_agent.schemas import (
    Chunk,
    ChunkDiagnostic,
    EvidenceItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    text: str,
    doc: str = "doc.pdf",
    chunk_number: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_name=doc,
        chunk_number=chunk_number,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _diag(
    chunk_id: str,
    *,
    doc: str = "doc.pdf",
    chunk_type: str = "evidence_dense",
    sent: bool = True,
    ev_count: int = 0,
    rel: float = 0.6,
    signals: dict | None = None,
) -> ChunkDiagnostic:
    return ChunkDiagnostic(
        chunk_id=chunk_id,
        document_name=doc,
        chunk_size=500,
        relevance_score=rel,
        evidence_candidate_count=2,
        sent_to_claude=sent,
        evidence_items_created=ev_count,
        extraction_decision="rejected" if ev_count == 0 else "accepted",
        rejection_reason="no evidence extracted" if ev_count == 0 else None,
        chunk_type=chunk_type,
        extraction_priority="high",
        candidate_signals=signals or {
            "numeric_claim_count": 5,
            "named_entity_count": 10,
            "unit_count": 2,
            "policy_or_standard_terms": 3,
            "date_count": 4,
            "comparative_claim_count": 1,
        },
    )


def _item(
    doc: str = "doc.pdf",
    snippet: str = "The SMR reactor will cost $3 billion by 2030.",
    chunk_id: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        claim="Cost claim.",
        source_document=doc,
        source_chunk_id=chunk_id,
        evidence_snippet=snippet,
        category="economics",
        relevance="Directly relevant.",
        confidence="high",
    )


_TOPIC_TERMS: dict[str, set[str]] = {
    "economics": {"cost", "billion", "capital", "levelized"},
    "construction": {"construction", "build", "fabrication"},
    "licensing": {"licensing", "regulatory", "nrc", "approval"},
    "grid integration": {"grid", "interconnection", "transmission"},
}


# ---------------------------------------------------------------------------
# _normalise
# ---------------------------------------------------------------------------


def test_normalise_collapses_whitespace():
    assert _normalise("hello   world\n\tfoo") == "hello world foo"


def test_normalise_strips():
    assert _normalise("  abc  ") == "abc"


def test_normalise_empty():
    assert _normalise("") == ""


# ---------------------------------------------------------------------------
# _is_garbled
# ---------------------------------------------------------------------------


def test_is_garbled_short_text():
    assert _is_garbled("x") is True


def test_is_garbled_binary_like():
    text = "\x00\x01\x02\x03\x04\x05" * 20
    assert _is_garbled(text) is True


def test_is_garbled_normal_text():
    text = "The SMR reactor is expected to cost $3 billion and generate 300 MW." * 5
    assert _is_garbled(text) is False


def test_is_garbled_empty():
    assert _is_garbled("") is True


# ---------------------------------------------------------------------------
# _chunk_has_topic_match
# ---------------------------------------------------------------------------


def test_chunk_has_topic_match_true():
    text = "The total cost is $3 billion for construction."
    assert _chunk_has_topic_match(text, _TOPIC_TERMS) is True


def test_chunk_has_topic_match_false():
    text = "Marketing materials show brand positioning for Q4."
    assert _chunk_has_topic_match(text, _TOPIC_TERMS) is False


def test_chunk_has_topic_match_case_insensitive():
    text = "NRC licensing approval was granted in 2024."
    assert _chunk_has_topic_match(text, _TOPIC_TERMS) is True


# ---------------------------------------------------------------------------
# _sentences_with_topic_match
# ---------------------------------------------------------------------------


def test_sentences_with_topic_match_returns_matching_sentences():
    text = "The reactor costs $3 billion. Weather is nice today. NRC approval expected by 2025."
    results = _sentences_with_topic_match(text, _TOPIC_TERMS)
    assert any("billion" in s or "cost" in s for s in results)


def test_sentences_with_topic_match_empty_for_no_match():
    text = "Nothing relevant here about the topic whatsoever."
    assert _sentences_with_topic_match(text, _TOPIC_TERMS) == []


# ---------------------------------------------------------------------------
# _signal_strength
# ---------------------------------------------------------------------------


def test_signal_strength_from_diag_object():
    d = _diag("c1", signals={
        "numeric_claim_count": 10,
        "named_entity_count": 20,
        "unit_count": 5,
        "policy_or_standard_terms": 4,
        "date_count": 3,
        "comparative_claim_count": 2,
    })
    score = _signal_strength(d)
    assert 0 <= score <= 100


def test_signal_strength_from_dict():
    d = {
        "candidate_signals": {
            "numeric_claim_count": 5,
            "named_entity_count": 10,
            "unit_count": 2,
        }
    }
    score = _signal_strength(d)
    assert score > 0


def test_signal_strength_zero_signals():
    d = {"candidate_signals": {}}
    assert _signal_strength(d) == 0


def test_signal_strength_capped_at_100():
    d = {"candidate_signals": {
        "numeric_claim_count": 200,
        "named_entity_count": 200,
        "unit_count": 200,
    }}
    assert _signal_strength(d) == 100


# ---------------------------------------------------------------------------
# classify_chunk_failure
# ---------------------------------------------------------------------------


def test_classify_budget_exclusion():
    chunk = _chunk("c1", "Some SMR cost $3B content.")
    diag = _diag("c1", sent=False)
    stage, reason = classify_chunk_failure(chunk, diag, [], _TOPIC_TERMS)
    assert stage == BUDGET_EXCLUSION
    assert "budget" in reason.lower()


def test_classify_parser_failure():
    chunk = _chunk("c1", "\x00\x01\x02" * 10)
    diag = _diag("c1", sent=True)
    stage, reason = classify_chunk_failure(chunk, diag, [], _TOPIC_TERMS)
    assert stage == PARSER_FAILURE


def test_classify_attribution_failure_when_normalised_match():
    # Use a long snippet so it passes the garbled/short check
    base = "The SMR reactor will cost $3 billion and generate 300 MW of clean power."
    raw_snippet = base.replace(" will ", "  will  ")   # extra whitespace
    chunk_text = base * 3                               # long chunk, contains compacted version
    chunk = _chunk("c1", chunk_text)
    diag = _diag("c1", sent=True)
    ev = _item(doc="doc.pdf", snippet=raw_snippet, chunk_id="")
    stage, reason = classify_chunk_failure(chunk, diag, [ev], _TOPIC_TERMS)
    assert stage == ATTRIBUTION_FAILURE
    assert "normalised" in reason or "whitespace" in reason.lower()


def test_classify_cross_chunk_when_no_match_in_chunk():
    # Long chunk with no topic keyword overlap with the snippet
    chunk_text = ("Unrelated content about general marketing strategy and brand positioning. " * 5)
    chunk = _chunk("c1", chunk_text)
    diag = _diag("c1", sent=True)
    # Evidence from same doc but a completely different section — won't match this chunk
    ev = _item(doc="doc.pdf", snippet="The SMR reactor will cost $3 billion by 2030 and save emissions.", chunk_id="")
    stage, reason = classify_chunk_failure(chunk, diag, [ev], _TOPIC_TERMS)
    assert stage == CROSS_CHUNK


def test_classify_empty_extraction_no_topic_terms():
    chunk = _chunk("c1", "The weather forecast shows sunny skies for the weekend." * 5)
    diag = _diag("c1", sent=True)
    stage, reason = classify_chunk_failure(chunk, diag, [], _TOPIC_TERMS)
    assert stage == EMPTY_EXTRACTION
    assert "topic keyword" in reason.lower()


def test_classify_duplicate_suppression_when_terms_present_but_no_evidence():
    # Topic terms ARE present but no evidence created — classified as dedup/max-cap
    chunk = _chunk("c1", "The construction of the NRC-approved reactor will begin. " * 5)
    diag = _diag("c1", sent=True)
    stage, reason = classify_chunk_failure(chunk, diag, [], _TOPIC_TERMS)
    assert stage == DUPLICATE_SUPPRESSION
    assert "dedup" in reason.lower() or "duplicate" in reason.lower() or "max-items" in reason.lower()


# ---------------------------------------------------------------------------
# build_failure_diagnostics
# ---------------------------------------------------------------------------


def test_build_failure_diagnostics_returns_list():
    chunk = _chunk("c1", "SMR cost $3 billion in 2030.")
    diag = _diag("c1", ev_count=0, chunk_type="evidence_dense")
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    assert isinstance(result, list)


def test_build_failure_diagnostics_skips_chunks_with_evidence():
    chunk = _chunk("c1", "SMR cost $3 billion.")
    diag = _diag("c1", ev_count=3)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    assert result == []


def test_build_failure_diagnostics_required_fields():
    chunk = _chunk("c1", "SMR cost $3 billion in 2030.")
    diag = _diag("c1", ev_count=0)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    assert len(result) == 1
    d = result[0]
    for field in [
        "chunk_id", "document_name", "relevance_score", "candidate_signals",
        "llm_invoked", "llm_response_received",
        "raw_extraction_count", "parsed_extraction_count",
        "validated_extraction_count", "accepted_extraction_count",
        "failure_stage", "failure_reason",
        "raw_llm_extraction_response", "parser_output", "validation_results",
    ]:
        assert field in d, f"missing field: {field}"


def test_build_failure_diagnostics_mock_llm_response_text():
    chunk = _chunk("c1", "SMR cost $3 billion.")
    diag = _diag("c1", ev_count=0)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS, is_mock=True)
    assert "mock" in result[0]["raw_llm_extraction_response"].lower()


def test_build_failure_diagnostics_parser_output_is_list():
    chunk = _chunk("c1", "SMR cost $3 billion. NRC licensing in 2024.")
    diag = _diag("c1", ev_count=0)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    assert isinstance(result[0]["parser_output"], list)


def test_build_failure_diagnostics_validation_results_keys():
    chunk = _chunk("c1", "SMR cost $3 billion.")
    diag = _diag("c1", ev_count=0)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    vr = result[0]["validation_results"]
    assert "passed" in vr
    assert "failed" in vr
    assert "reasons" in vr


def test_build_failure_diagnostics_skips_non_dense_not_sent():
    chunk = _chunk("c1", "Table of contents and disclaimers.", "doc.pdf")
    diag = _diag("c1", ev_count=0, chunk_type="boilerplate", sent=False)
    result = build_failure_diagnostics([diag], [chunk], [], _TOPIC_TERMS)
    # boilerplate + not_sent: not a zero-evidence evidence_dense sent chunk
    assert result == []


# ---------------------------------------------------------------------------
# build_failure_summary
# ---------------------------------------------------------------------------


def test_build_failure_summary_all_stages_present():
    summary = build_failure_summary([])
    for stage in ALL_FAILURE_STAGES:
        assert stage in summary


def test_build_failure_summary_counts():
    diags = [
        {"failure_stage": ATTRIBUTION_FAILURE},
        {"failure_stage": ATTRIBUTION_FAILURE},
        {"failure_stage": EMPTY_EXTRACTION},
        {"failure_stage": CROSS_CHUNK},
    ]
    summary = build_failure_summary(diags)
    assert summary[ATTRIBUTION_FAILURE] == 2
    assert summary[EMPTY_EXTRACTION] == 1
    assert summary[CROSS_CHUNK] == 1


def test_build_failure_summary_zeros_for_absent_stages():
    summary = build_failure_summary([{"failure_stage": EMPTY_EXTRACTION}])
    assert summary[ATTRIBUTION_FAILURE] == 0
    assert summary[NO_LLM_OUTPUT] == 0


# ---------------------------------------------------------------------------
# compute_top_missed_chunks
# ---------------------------------------------------------------------------


def test_compute_top_missed_chunks_sorted_by_signal():
    diags = [
        {"chunk_id": "c1", "document_name": "d.pdf", "signal_strength": 30, "failure_stage": EMPTY_EXTRACTION, "failure_reason": "", "relevance_score": 0.5, "candidate_signals": {}},
        {"chunk_id": "c2", "document_name": "d.pdf", "signal_strength": 90, "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": "", "relevance_score": 0.8, "candidate_signals": {}},
        {"chunk_id": "c3", "document_name": "d.pdf", "signal_strength": 60, "failure_stage": CROSS_CHUNK, "failure_reason": "", "relevance_score": 0.6, "candidate_signals": {}},
    ]
    top = compute_top_missed_chunks(diags, n=2)
    assert len(top) == 2
    assert top[0]["chunk_id"] == "c2"
    assert top[1]["chunk_id"] == "c3"


def test_compute_top_missed_chunks_required_fields():
    diags = [
        {"chunk_id": "c1", "document_name": "d.pdf", "signal_strength": 50,
         "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": "x",
         "relevance_score": 0.6, "candidate_signals": {}}
    ]
    top = compute_top_missed_chunks(diags)
    assert "chunk_id" in top[0]
    assert "document_name" in top[0]
    assert "signal_strength" in top[0]
    assert "failure_stage" in top[0]
    assert "failure_reason" in top[0]


def test_compute_top_missed_chunks_respects_n():
    diags = [
        {"chunk_id": f"c{i}", "document_name": "d.pdf", "signal_strength": i,
         "failure_stage": EMPTY_EXTRACTION, "failure_reason": "",
         "relevance_score": 0.5, "candidate_signals": {}}
        for i in range(20)
    ]
    top = compute_top_missed_chunks(diags, n=5)
    assert len(top) == 5


# ---------------------------------------------------------------------------
# analyze_document_failures
# ---------------------------------------------------------------------------


def test_analyze_document_failures_returns_required_keys():
    diags = [
        {"document_name": "doc.pdf", "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": "test"},
        {"document_name": "doc.pdf", "failure_stage": CROSS_CHUNK, "failure_reason": "other"},
    ]
    result = analyze_document_failures("doc.pdf", diags)
    assert result["document_name"] == "doc.pdf"
    assert "chunks" in result
    assert "evidence_created" in result
    assert "most_common_failure_stage" in result
    assert "most_common_failure_reason" in result
    assert "stage_breakdown" in result


def test_analyze_document_failures_most_common_stage():
    diags = [
        {"document_name": "doc.pdf", "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": "a"},
        {"document_name": "doc.pdf", "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": "a"},
        {"document_name": "doc.pdf", "failure_stage": EMPTY_EXTRACTION, "failure_reason": "b"},
    ]
    result = analyze_document_failures("doc.pdf", diags)
    assert result["most_common_failure_stage"] == ATTRIBUTION_FAILURE


def test_analyze_document_failures_filters_by_doc():
    diags = [
        {"document_name": "other.pdf", "failure_stage": EMPTY_EXTRACTION, "failure_reason": ""},
        {"document_name": "doc.pdf", "failure_stage": ATTRIBUTION_FAILURE, "failure_reason": ""},
    ]
    result = analyze_document_failures("doc.pdf", diags)
    assert result["stage_breakdown"].get(EMPTY_EXTRACTION, 0) == 0
    assert result["stage_breakdown"].get(ATTRIBUTION_FAILURE, 0) == 1


def test_analyze_document_failures_empty_diags():
    result = analyze_document_failures("missing.pdf", [])
    assert result["failed_chunks"] == 0
    assert result["most_common_failure_stage"] == UNKNOWN


# ---------------------------------------------------------------------------
# ALL_FAILURE_STAGES constant
# ---------------------------------------------------------------------------


def test_all_failure_stages_contains_spec_stages():
    required = {
        "NO_LLM_OUTPUT", "EMPTY_EXTRACTION", "PARSER_FAILURE",
        "SCHEMA_VALIDATION_FAILURE", "QUALITY_THRESHOLD_REJECTION",
        "DUPLICATE_SUPPRESSION", "POST_PROCESSING_REJECTION", "UNKNOWN",
    }
    assert required.issubset(set(ALL_FAILURE_STAGES))


def test_spec_failure_stages_contains_all_required():
    required = {
        "NO_LLM_OUTPUT", "EMPTY_EXTRACTION", "PARSER_FAILURE",
        "SCHEMA_VALIDATION_FAILURE", "QUALITY_THRESHOLD_REJECTION",
        "DUPLICATE_SUPPRESSION", "POST_PROCESSING_REJECTION", "UNKNOWN",
    }
    assert required == set(SPEC_FAILURE_STAGES)


def test_failure_summary_contains_all_spec_keys():
    """failure_summary must always emit every spec-required key."""
    from research_agent.extraction_diagnostics import build_failure_summary
    summary = build_failure_summary([])
    for stage in SPEC_FAILURE_STAGES:
        assert stage in summary, f"spec stage missing from summary: {stage}"


def test_failure_summary_duplicate_suppression_increments():
    diags = [
        {"failure_stage": DUPLICATE_SUPPRESSION},
        {"failure_stage": DUPLICATE_SUPPRESSION},
    ]
    summary = build_failure_summary(diags)
    assert summary[DUPLICATE_SUPPRESSION] == 2


def test_all_failure_stages_contains_attribution_stages():
    assert ATTRIBUTION_FAILURE in ALL_FAILURE_STAGES
    assert CROSS_CHUNK in ALL_FAILURE_STAGES
    assert BUDGET_EXCLUSION in ALL_FAILURE_STAGES
