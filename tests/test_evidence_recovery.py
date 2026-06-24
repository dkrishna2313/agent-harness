"""Tests for research_agent.evidence_recovery (JH1a)."""

from __future__ import annotations

from pathlib import Path

import pytest

from research_agent.evidence_recovery import (
    RECOVERY_MIN_OVERALL_SCORE,
    RecoveryResult,
    _classify_recovery_reason,
    _split_sentences,
    attribute_evidence_to_chunks,
    compute_zero_yield_documents,
    find_recovery_eligible_chunks,
    recover_evidence_from_chunk,
    run_recovery_pass,
)
from research_agent.schemas import (
    Chunk,
    ChunkDiagnostic,
    EvidenceItem,
    SourceDocument,
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
    chunk_type: str = "evidence_dense",
    relevance_score: float = 0.7,
    sent_to_claude: bool = True,
    doc: str = "doc.pdf",
    candidate_signals: dict | None = None,
) -> ChunkDiagnostic:
    return ChunkDiagnostic(
        chunk_id=chunk_id,
        document_name=doc,
        chunk_size=200,
        relevance_score=relevance_score,
        evidence_candidate_count=2,
        sent_to_claude=sent_to_claude,
        evidence_items_created=0,
        extraction_decision="rejected",
        rejection_reason="no evidence extracted",
        chunk_type=chunk_type,
        extraction_priority="high" if chunk_type == "evidence_dense" else "medium",
        candidate_signals=candidate_signals or {},
    )


def _item(
    evidence_id: str = "E001",
    doc: str = "doc.pdf",
    snippet: str = "The grid needs 500 MW of new capacity by 2030.",
    chunk_id: str = "",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        claim="Grid capacity claim.",
        source_document=doc,
        source_chunk_id=chunk_id,
        evidence_snippet=snippet,
        category="power",
        relevance="Directly relevant.",
        confidence="high",
    )


def _source_doc(name: str = "doc.pdf", text: str = "x" * 1000) -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_filters_short():
    sentences = _split_sentences("Hi. Short. This is a much longer sentence about grid capacity.")
    assert all(len(s) >= 40 for s in sentences)


def test_split_sentences_returns_empty_for_blank():
    assert _split_sentences("") == []


def test_split_sentences_splits_on_period():
    text = (
        "NVIDIA Rubin NVL72 rack power is rated at 120 kW per cabinet. "
        "The liquid cooling system handles 35°C inlet temperature."
    )
    sentences = _split_sentences(text)
    assert len(sentences) == 2


# ---------------------------------------------------------------------------
# _classify_recovery_reason
# ---------------------------------------------------------------------------


def test_classify_recovery_reason_numeric():
    assert _classify_recovery_reason("The data center requires 500 MW of power.") == "numeric_claim"


def test_classify_recovery_reason_policy():
    assert _classify_recovery_reason(
        "FERC Order 2023 mandates faster interconnection processing."
    ) == "policy_claim"


def test_classify_recovery_reason_timeline():
    assert _classify_recovery_reason(
        "The project is expected to complete commissioning in 2027."
    ) == "timeline_claim"


def test_classify_recovery_reason_technical_spec():
    assert _classify_recovery_reason(
        "Each rack is rated at 100kW and houses 72 GPUs in 4U chassis."
    ) == "numeric_claim"  # numeric match comes first


def test_classify_recovery_reason_market_projection():
    s = "The market is expected to reach $50 billion by 2028."
    # market projection overlaps with numeric — numeric comes first
    reason = _classify_recovery_reason(s)
    assert reason in ("numeric_claim", "market_projection")


def test_classify_recovery_reason_grid_constraint():
    assert _classify_recovery_reason(
        "Transmission congestion causes significant curtailment in the MISO region."
    ) in ("policy_claim", "grid_constraint")


def test_classify_recovery_reason_other_for_vague():
    assert _classify_recovery_reason("The team is working on the project.") == "other"


# ---------------------------------------------------------------------------
# attribute_evidence_to_chunks
# ---------------------------------------------------------------------------


def test_attribute_evidence_sets_chunk_id_when_snippet_in_chunk():
    snippet = "The grid needs 500 MW of new capacity by 2030."
    chunk = _chunk("doc_C001", "Some preamble. " + snippet + " More text follows here.")
    item = _item(snippet=snippet, doc="doc.pdf", chunk_id="")
    result = attribute_evidence_to_chunks([item], [chunk])
    assert len(result) == 1
    assert result[0].source_chunk_id == "doc_C001"


def test_attribute_evidence_skips_already_attributed():
    snippet = "The grid needs 500 MW of new capacity by 2030."
    chunk = _chunk("doc_C001", snippet)
    item = _item(snippet=snippet, chunk_id="doc_C999")
    result = attribute_evidence_to_chunks([item], [chunk])
    assert result[0].source_chunk_id == "doc_C999"  # unchanged


def test_attribute_evidence_no_match_leaves_id_empty():
    chunk = _chunk("doc_C001", "Some other text not matching the snippet at all.")
    item = _item(snippet="completely different text about CAISO", chunk_id="")
    result = attribute_evidence_to_chunks([item], [chunk])
    assert result[0].source_chunk_id == ""


def test_attribute_evidence_matches_correct_document():
    snippet = "ERCOT grid demand peaked at 85 GW in summer 2023."
    chunk_a = _chunk("docA_C001", snippet, doc="docA.pdf")
    chunk_b = _chunk("docB_C001", "Unrelated content about reactor design.", doc="docB.pdf")
    item = _item(snippet=snippet, doc="docA.pdf", chunk_id="")
    result = attribute_evidence_to_chunks([item], [chunk_a, chunk_b])
    assert result[0].source_chunk_id == "docA_C001"


def test_attribute_evidence_returns_same_length_list():
    items = [_item(evidence_id=f"E{i:03d}") for i in range(5)]
    chunks: list[Chunk] = []
    result = attribute_evidence_to_chunks(items, chunks)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# find_recovery_eligible_chunks
# ---------------------------------------------------------------------------


def test_find_eligible_chunks_excludes_chunks_with_evidence():
    chunk = _chunk("doc_C001", "500 MW grid capacity.")
    diag = _diag("doc_C001")
    item = _item(chunk_id="doc_C001")
    eligible = find_recovery_eligible_chunks([chunk], [item], [diag])
    assert eligible == []


def test_find_eligible_chunks_includes_zero_evidence_dense():
    chunk = _chunk("doc_C001", "500 MW grid capacity.")
    diag = _diag("doc_C001", chunk_type="evidence_dense", relevance_score=0.6)
    eligible = find_recovery_eligible_chunks([chunk], [], [diag])
    assert len(eligible) == 1
    assert eligible[0].chunk_id == "doc_C001"


def test_find_eligible_chunks_excludes_boilerplate():
    chunk = _chunk("doc_C001", "Table of Contents\n1. Introduction\n2. Methodology")
    diag = _diag("doc_C001", chunk_type="boilerplate")
    eligible = find_recovery_eligible_chunks([chunk], [], [diag])
    assert eligible == []


def test_find_eligible_chunks_excludes_low_relevance():
    chunk = _chunk("doc_C001", "Some grid content about 500 MW capacity.")
    diag = _diag("doc_C001", chunk_type="evidence_dense", relevance_score=0.1)
    eligible = find_recovery_eligible_chunks([chunk], [], [diag])
    assert eligible == []


def test_find_eligible_chunks_excludes_context_type():
    chunk = _chunk("doc_C001", "Background context section.")
    diag = _diag("doc_C001", chunk_type="context", relevance_score=0.8)
    eligible = find_recovery_eligible_chunks([chunk], [], [diag])
    assert eligible == []


# ---------------------------------------------------------------------------
# recover_evidence_from_chunk
# ---------------------------------------------------------------------------


_SIGNAL_TEXT = (
    "FERC Order 2023 requires utilities to process interconnection requests within 150 days. "
    "The MISO queue exceeded 2,000 GW of pending capacity in 2023. "
    "Each data center rack consumes 120kW at full load with PUE of 1.3. "
    "By 2030, the total market is projected to reach $200 billion. "
    "Transmission curtailment increased by 35% due to grid congestion in Q3 2023."
)


def test_recover_evidence_from_chunk_returns_evidence_items():
    chunk = _chunk("doc_C001", _SIGNAL_TEXT)
    items = recover_evidence_from_chunk(chunk, "grid capacity constraints 2030")
    assert len(items) > 0


def test_recovered_items_are_marked_recovered():
    chunk = _chunk("doc_C001", _SIGNAL_TEXT)
    items = recover_evidence_from_chunk(chunk, "grid capacity constraints 2030")
    for item in items:
        assert item.recovered is True


def test_recovered_items_have_recovery_reason():
    chunk = _chunk("doc_C001", _SIGNAL_TEXT)
    items = recover_evidence_from_chunk(chunk, "grid")
    for item in items:
        assert item.recovery_reason != ""
        assert item.recovery_reason != "other"


def test_recovered_items_have_source_chunk_id():
    chunk = _chunk("doc_C001", _SIGNAL_TEXT)
    items = recover_evidence_from_chunk(chunk, "grid")
    for item in items:
        assert item.source_chunk_id == "doc_C001"


def test_recover_from_vague_chunk_returns_empty():
    chunk = _chunk("doc_C001", "The team is working on improving their communication processes.")
    items = recover_evidence_from_chunk(chunk, "grid power")
    assert items == []


def test_recover_respects_max_per_chunk():
    # Long text with many signals
    text = (_SIGNAL_TEXT + " ") * 20
    chunk = _chunk("doc_C001", text)
    from research_agent.evidence_recovery import _MAX_RECOVERED_PER_CHUNK
    items = recover_evidence_from_chunk(chunk, "grid capacity constraints")
    assert len(items) <= _MAX_RECOVERED_PER_CHUNK


# ---------------------------------------------------------------------------
# RecoveryResult dataclass
# ---------------------------------------------------------------------------


def test_recovery_result_defaults():
    r = RecoveryResult()
    assert r.recovered_items == []
    assert r.missed_chunk_queue == []
    assert r.recovery_metrics == {}
    assert r.yield_before == {}
    assert r.yield_after == {}


# ---------------------------------------------------------------------------
# run_recovery_pass
# ---------------------------------------------------------------------------


def _make_chunks_and_diags(text: str, doc: str = "doc.pdf"):
    chunk = _chunk(f"{doc.replace('.', '_')}_C001", text, doc=doc)
    diag = _diag(chunk.chunk_id, chunk_type="evidence_dense", relevance_score=0.7, doc=doc)
    return chunk, diag


def test_run_recovery_pass_returns_recovery_result():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    assert isinstance(result, RecoveryResult)


def test_run_recovery_pass_populates_yield_before():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    assert "chunks_selected" in result.yield_before
    assert "evidence_items_created" in result.yield_before


def test_run_recovery_pass_populates_yield_after():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    assert "chunks_selected" in result.yield_after
    assert "evidence_items_created" in result.yield_after


def test_run_recovery_pass_yield_after_gte_before():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    assert result.yield_after["evidence_items_created"] >= result.yield_before["evidence_items_created"]


def test_run_recovery_pass_metrics_keys():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    m = result.recovery_metrics
    assert "eligible_chunks" in m
    assert "recovery_attempted" in m
    assert "chunks_recovered" in m
    assert "recovered_evidence_items" in m
    assert "recovery_yield" in m


def test_run_recovery_pass_no_eligible_gives_empty():
    chunk = _chunk("doc_C001", _SIGNAL_TEXT)
    # Boilerplate → not eligible
    diag = _diag("doc_C001", chunk_type="boilerplate")
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid",
    )
    assert result.recovery_metrics["eligible_chunks"] == 0
    assert result.recovered_items == []


def test_run_recovery_pass_missed_chunk_queue_populated():
    chunk, diag = _make_chunks_and_diags(_SIGNAL_TEXT)
    result = run_recovery_pass(
        [chunk], [chunk], [],
        [diag],
        question="grid capacity constraints",
    )
    assert isinstance(result.missed_chunk_queue, list)
    if result.recovery_metrics["eligible_chunks"] > 0:
        assert len(result.missed_chunk_queue) > 0
        q = result.missed_chunk_queue[0]
        assert "chunk_id" in q
        assert "document_name" in q
        assert "recovered_evidence_count" in q


# ---------------------------------------------------------------------------
# compute_zero_yield_documents
# ---------------------------------------------------------------------------


def test_compute_zero_yield_documents_flags_zero_evidence_docs():
    chunk = _chunk("doc_C001", "x" * 500, doc="zero.pdf")
    docs = [_source_doc("zero.pdf")]
    result = compute_zero_yield_documents([chunk], [], docs)
    assert len(result) == 1
    assert result[0]["document_name"] == "zero.pdf"
    assert result[0]["evidence_items"] == 0


def test_compute_zero_yield_documents_excludes_doc_with_evidence():
    chunk = _chunk("doc_C001", "x" * 500, doc="has_ev.pdf")
    item = _item(doc="has_ev.pdf")
    docs = [_source_doc("has_ev.pdf")]
    result = compute_zero_yield_documents([chunk], [item], docs)
    assert all(r["document_name"] != "has_ev.pdf" for r in result)


def test_compute_zero_yield_documents_recommendation_field():
    chunks = [
        _chunk(f"doc_C{i:03d}", "x" * 100, doc="multi.pdf", chunk_number=i)
        for i in range(5)
    ]
    docs = [_source_doc("multi.pdf")]
    result = compute_zero_yield_documents(chunks, [], docs)
    assert len(result) == 1
    assert result[0]["recommendation"] == "inspect_parser_output"


def test_compute_zero_yield_documents_sorted_by_chunks_desc():
    chunks_a = [_chunk(f"a_C{i:03d}", "x" * 50, doc="a.pdf", chunk_number=i) for i in range(5)]
    chunks_b = [_chunk(f"b_C{i:03d}", "x" * 50, doc="b.pdf", chunk_number=i) for i in range(2)]
    docs = [_source_doc("a.pdf"), _source_doc("b.pdf")]
    result = compute_zero_yield_documents(chunks_a + chunks_b, [], docs)
    assert result[0]["document_name"] == "a.pdf"


def test_compute_zero_yield_documents_returns_empty_when_all_have_evidence():
    chunk = _chunk("doc_C001", "x" * 100, doc="doc.pdf")
    item = _item(doc="doc.pdf")
    docs = [_source_doc("doc.pdf")]
    result = compute_zero_yield_documents([chunk], [item], docs)
    assert result == []
