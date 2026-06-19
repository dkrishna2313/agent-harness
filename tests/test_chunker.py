"""Tests for research_agent.chunker."""

from pathlib import Path

import pytest

from research_agent.chunker import (
    CHUNK_MAX,
    CHUNK_TARGET,
    CHUNK_SELECTION_BUDGET,
    _chunk_id,
    _extract_question_terms,
    chunk_document,
    chunk_documents,
    compute_chunk_diagnostics,
    count_evidence_candidates,
    score_chunk_relevance,
    select_relevant_chunks,
)
from research_agent.agent import DcPowerAgent
from research_agent.claude_client import MockClaudeClient
from research_agent.schemas import Chunk, ChunkDiagnostic, EvidenceItem, SourceDocument
from research_agent.trace import build_trace


def _make_doc(text: str, name: str = "test.txt") -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


def _long_text(n_chars: int) -> str:
    """Generate plausible multi-sentence text of approximately n_chars."""
    sentence = "NVIDIA Rubin NVL72 rack systems use liquid cooling and high-power networking. "
    repeat = (n_chars // len(sentence)) + 2
    return (sentence * repeat)[:n_chars]


# --- chunk_document ---


def test_chunk_document_empty_returns_no_chunks():
    doc = _make_doc("")
    assert chunk_document(doc) == []


def test_chunk_document_short_text_returns_single_chunk():
    doc = _make_doc("Short text about power.")
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text about power."
    assert chunks[0].start_offset == 0
    assert chunks[0].end_offset == len("Short text about power.")


def test_chunk_document_large_text_produces_multiple_chunks_within_size_bounds():
    text = _long_text(CHUNK_MAX * 3)
    doc = _make_doc(text)
    chunks = chunk_document(doc, target_size=CHUNK_TARGET, max_size=CHUNK_MAX)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.char_count <= CHUNK_MAX, f"Chunk {chunk.chunk_id} exceeds CHUNK_MAX"


def test_chunk_document_offsets_are_contiguous():
    text = _long_text(CHUNK_MAX * 3)
    doc = _make_doc(text)
    chunks = chunk_document(doc)
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(text)
    for i in range(len(chunks) - 1):
        assert chunks[i].end_offset == chunks[i + 1].start_offset


def test_chunk_document_ids_are_sequential():
    text = _long_text(CHUNK_MAX * 3)
    doc = _make_doc(text)
    chunks = chunk_document(doc)
    for i, chunk in enumerate(chunks, start=1):
        assert chunk.chunk_number == i
        assert f"C{i:03d}" in chunk.chunk_id


def test_chunk_document_text_covers_document():
    text = _long_text(CHUNK_MAX * 3)
    doc = _make_doc(text)
    chunks = chunk_document(doc)
    reconstructed = "".join(c.text for c in chunks)
    assert reconstructed == text


# --- chunk_documents ---


def test_chunk_documents_includes_all_documents():
    doc1 = _make_doc("Short doc one.", "doc1.txt")
    doc2 = _make_doc("Short doc two.", "doc2.txt")
    chunks = chunk_documents([doc1, doc2])
    names = {c.document_name for c in chunks}
    assert "doc1.txt" in names
    assert "doc2.txt" in names


# --- _chunk_id ---


def test_chunk_id_contains_document_stem():
    cid = _chunk_id("rubin_spec.pdf", 1)
    assert "rubin_spec" in cid
    assert "C001" in cid


# --- schema ---


def test_evidence_item_accepts_source_chunk_id():
    item = EvidenceItem(
        claim="Rubin rack power.",
        source_document="rubin.pdf",
        source_chunk_id="rubin_pdf_C001",
        evidence_snippet="NVIDIA Rubin NVL72 rack systems use 120kW.",
        category="power",
        relevance="Directly relevant.",
        confidence="high",
    )
    assert item.source_chunk_id == "rubin_pdf_C001"
    data = item.model_dump()
    assert data["source_chunk_id"] == "rubin_pdf_C001"


# --- agent integration ---


def test_agent_mock_memo_metadata_includes_chunk_fields():
    text = (
        "NVIDIA Rubin NVL72 rack architecture uses liquid cooling and high-power distribution. "
        "Power distribution systems must handle 120kW rack density. "
        "Networking uses NVLink and InfiniBand switching fabrics. "
        "Thermal management and cooling infrastructure are critical for AI factories. "
        "Rack-scale systems require careful facility power planning."
    )
    doc = _make_doc(text, "rubin.md")
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Explain Rubin power", [doc])

    assert memo.metadata.get("chunk_count", 0) > 0
    assert "chunks_per_document" in memo.metadata
    assert "evidence_per_chunk" in memo.metadata


# --- trace ---


def test_trace_output_includes_chunk_metadata():
    text = "NVIDIA Rubin NVL72 rack power and cooling infrastructure."
    doc = _make_doc(text, "rubin.md")
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Explain Rubin power", [doc])

    trace = build_trace(
        question="Explain Rubin power",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )

    assert trace["chunk_count"] >= 0
    assert "chunks_per_document" in trace
    assert "chunk_diagnostics" in trace


# --- select_relevant_chunks ---


def test_select_relevant_chunks_prefers_relevant_content():
    relevant_text = (
        "NVIDIA Rubin NVL72 rack power distribution requires 120kW cooling infrastructure. "
        "The liquid cooling system must handle high thermal loads from 72 GPUs. "
        "Power delivery systems include CDU and in-row cooling units."
    )
    irrelevant_text = (
        "This document covers general software engineering practices. "
        "The team uses agile methodology for sprint planning. "
        "Code reviews are conducted weekly."
    )
    doc_relevant = _make_doc(relevant_text, "rubin_power.txt")
    doc_irrelevant = _make_doc(irrelevant_text, "agile.txt")
    chunks = chunk_documents([doc_relevant, doc_irrelevant])

    selected, scores = select_relevant_chunks(chunks, "Rubin NVL72 rack power cooling")

    relevant_ids = {c.chunk_id for c in chunks if c.document_name == "rubin_power.txt"}
    selected_ids = {c.chunk_id for c in selected}
    assert relevant_ids.issubset(selected_ids), "All relevant chunks should be selected"


def test_select_relevant_chunks_respects_char_budget():
    text = _long_text(CHUNK_MAX * 4)
    doc = _make_doc(text, "large.txt")
    chunks = chunk_documents([doc])
    total_chars = sum(c.char_count for c in chunks)

    # Budget of 0 means nothing selected
    selected_none, _ = select_relevant_chunks(chunks, "power cooling", max_total_chars=0)
    assert selected_none == []

    # Full budget means everything selected
    selected_all, _ = select_relevant_chunks(chunks, "power cooling", max_total_chars=total_chars)
    assert len(selected_all) == len(chunks)


def test_select_relevant_chunks_returns_scores_for_all_chunks():
    doc1 = _make_doc("Rubin rack power cooling.", "doc1.txt")
    doc2 = _make_doc("Unrelated content here.", "doc2.txt")
    chunks = chunk_documents([doc1, doc2])

    _, scores = select_relevant_chunks(chunks, "Rubin power")

    assert len(scores) == len(chunks)
    for chunk in chunks:
        assert chunk.chunk_id in scores


def test_score_chunk_relevance_zero_for_no_overlap():
    chunk = Chunk(
        chunk_id="test_C001", document_name="test.txt", chunk_number=1,
        text="No matching terms here at all.", start_offset=0, end_offset=30,
    )
    score = score_chunk_relevance(chunk, {"nvidia", "rubin", "power"})
    assert score == 0.0


def test_score_chunk_relevance_nonzero_for_overlap():
    chunk = Chunk(
        chunk_id="test_C001", document_name="test.txt", chunk_number=1,
        text="NVIDIA Rubin power systems require liquid cooling.", start_offset=0, end_offset=49,
    )
    score = score_chunk_relevance(chunk, {"nvidia", "rubin", "power"})
    assert score > 0.0


def test_count_evidence_candidates_counts_matching_sentences():
    chunk = Chunk(
        chunk_id="test_C001", document_name="test.txt", chunk_number=1,
        text="Power distribution is critical. Cooling matters too. Nothing else here.",
        start_offset=0, end_offset=70,
    )
    count = count_evidence_candidates(chunk, {"power", "cooling"})
    assert count == 2


# --- compute_chunk_diagnostics ---


def test_compute_chunk_diagnostics_accepted_when_evidence_produced():
    chunk = Chunk(
        chunk_id="rubin_C001", document_name="rubin.txt", chunk_number=1,
        text="Rubin NVL72 power.", start_offset=0, end_offset=18,
    )
    evidence = [EvidenceItem(
        claim="Power claim.",
        source_document="rubin.txt",
        source_chunk_id="rubin_C001",
        evidence_snippet="Rubin NVL72 power.",
        category="power",
        relevance="Relevant.",
        confidence="high",
    )]
    diags = compute_chunk_diagnostics(
        [chunk], [chunk], evidence, {"rubin_C001": (0.8, 2)}
    )
    assert len(diags) == 1
    assert diags[0].extraction_decision == "accepted"
    assert diags[0].evidence_items_created == 1
    assert diags[0].rejection_reason is None
    assert diags[0].sent_to_claude is True


def test_compute_chunk_diagnostics_rejected_when_sent_but_no_evidence():
    chunk = Chunk(
        chunk_id="rubin_C001", document_name="rubin.txt", chunk_number=1,
        text="Rubin NVL72 power.", start_offset=0, end_offset=18,
    )
    diags = compute_chunk_diagnostics(
        [chunk], [chunk], [], {"rubin_C001": (0.5, 1)}
    )
    assert diags[0].extraction_decision == "rejected"
    assert diags[0].rejection_reason == "no evidence extracted"
    assert diags[0].sent_to_claude is True


def test_compute_chunk_diagnostics_not_sent_when_excluded_by_budget():
    chunk = Chunk(
        chunk_id="rubin_C001", document_name="rubin.txt", chunk_number=1,
        text="Rubin NVL72 power.", start_offset=0, end_offset=18,
    )
    # Not in selected_chunks
    diags = compute_chunk_diagnostics(
        [chunk], [], [], {"rubin_C001": (0.5, 1)}
    )
    assert diags[0].extraction_decision == "not_sent"
    assert diags[0].rejection_reason == "excluded by character budget"
    assert diags[0].sent_to_claude is False


def test_compute_chunk_diagnostics_not_relevant_when_score_zero():
    chunk = Chunk(
        chunk_id="agile_C001", document_name="agile.txt", chunk_number=1,
        text="Sprint planning and retrospectives.", start_offset=0, end_offset=34,
    )
    diags = compute_chunk_diagnostics(
        [chunk], [], [], {"agile_C001": (0.0, 0)}
    )
    assert diags[0].extraction_decision == "not_sent"
    assert diags[0].rejection_reason == "not relevant to question"


def test_compute_chunk_diagnostics_totals_are_consistent():
    chunks = [
        Chunk(chunk_id=f"doc_C{i:03d}", document_name="doc.txt", chunk_number=i,
              text=f"Chunk {i} text.", start_offset=i*10, end_offset=(i+1)*10)
        for i in range(1, 6)
    ]
    selected = chunks[:3]
    evidence = [EvidenceItem(
        claim="Claim.", source_document="doc.txt", source_chunk_id="doc_C001",
        evidence_snippet="Chunk 1 text.", category="power", relevance="Rel.", confidence="high",
    )]
    scores = {c.chunk_id: (0.5, 1) for c in chunks}
    diags = compute_chunk_diagnostics(chunks, selected, evidence, scores)

    assert len(diags) == 5
    accepted = [d for d in diags if d.extraction_decision == "accepted"]
    rejected = [d for d in diags if d.extraction_decision == "rejected"]
    not_sent = [d for d in diags if d.extraction_decision == "not_sent"]
    assert len(accepted) == 1
    assert len(rejected) == 2
    assert len(not_sent) == 2


def test_trace_includes_chunk_diagnostics():
    text = "NVIDIA Rubin NVL72 rack power and cooling infrastructure for AI factories."
    doc = _make_doc(text, "rubin.md")
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Explain Rubin power", [doc])

    trace = build_trace(
        question="Explain Rubin power",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )

    assert "chunk_diagnostics" in trace
    assert isinstance(trace["chunk_diagnostics"], list)
    # Each diagnostic must have required fields
    for diag in trace["chunk_diagnostics"]:
        assert "chunk_id" in diag
        assert "extraction_decision" in diag
        assert "sent_to_claude" in diag
        assert diag["extraction_decision"] in ("accepted", "rejected", "not_sent")


def test_trace_rejection_reasons_are_captured():
    # Two docs: one relevant, one not
    relevant = _make_doc(
        "Rubin NVL72 rack power cooling liquid CDU thermal.", "relevant.txt"
    )
    irrelevant = _make_doc(
        "General business strategy and market analysis.", "irrelevant.txt"
    )
    memo = DcPowerAgent(client=MockClaudeClient()).analyze(
        "Rubin NVL72 power cooling", [relevant, irrelevant]
    )
    trace = build_trace(
        question="Rubin NVL72 power cooling",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[relevant, irrelevant],
        memo=memo,
        mock_mode=True,
    )

    reasons = {d.get("rejection_reason") for d in trace["chunk_diagnostics"]}
    # At least one chunk should have a reason (the irrelevant doc should have "not relevant")
    non_none_reasons = {r for r in reasons if r is not None}
    assert non_none_reasons, f"Expected at least one rejection reason, got: {reasons}"


def test_accepted_rejected_not_sent_counts_sum_to_total():
    text = _long_text(CHUNK_MAX * 2)
    doc = _make_doc(text, "big.txt")
    memo = DcPowerAgent(client=MockClaudeClient()).analyze("Rubin power cooling", [doc])

    diagnostics = memo.metadata.get("chunk_diagnostics", [])
    total = memo.metadata.get("chunk_count", 0)

    if diagnostics and total:
        counted = sum(
            1 for d in diagnostics
            if d.get("extraction_decision") in ("accepted", "rejected", "not_sent")
        )
        assert counted == total
