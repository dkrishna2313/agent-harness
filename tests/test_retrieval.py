"""Tests for dc_power_agent.retrieval."""

from __future__ import annotations

from pathlib import Path

import pytest

from dc_power_agent.retrieval import (
    DEFAULT_TOP_CHUNKS,
    classify_document_priority,
    score_retrieval,
    select_top_chunks,
)
from dc_power_agent.schemas import Chunk, SourceDocument
from dc_power_agent.agent import DcPowerAgent
from dc_power_agent.claude_client import MockClaudeClient
from dc_power_agent.trace import build_trace


def _make_chunk(text: str, doc_name: str = "test.txt", chunk_number: int = 1) -> Chunk:
    stem = doc_name[:24].replace(".", "_")
    return Chunk(
        chunk_id=f"{stem}_C{chunk_number:03d}",
        document_name=doc_name,
        chunk_number=chunk_number,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _make_doc(text: str, name: str = "test.txt") -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


# --- score_retrieval ---

def test_score_retrieval_keyword_score_nonzero_for_relevant_chunk():
    chunk = _make_chunk("The power distribution unit supplies voltage to the rack.")
    question = "What is the power distribution strategy?"
    question_terms = {"power", "distribution", "strategy"}
    rs = score_retrieval(chunk, question, question_terms, set())
    assert rs.keyword_score > 0


def test_score_retrieval_keyword_score_zero_for_irrelevant_chunk():
    chunk = _make_chunk("The quick brown fox jumps over the lazy dog.")
    question = "What is the power distribution strategy?"
    question_terms = {"power", "distribution", "strategy"}
    rs = score_retrieval(chunk, question, question_terms, set())
    assert rs.keyword_score == 0.0


def test_score_retrieval_topic_match_score_for_cooling_chunk():
    chunk = _make_chunk("Liquid cooling systems use CDU units to manage thermal loads.")
    question = "How does liquid cooling work?"
    rs = score_retrieval(chunk, question, {"liquid", "cooling"}, {"cooling"})
    assert rs.topic_match_score > 0


def test_score_retrieval_document_priority_nvidia_primary():
    chunk = _make_chunk("text", doc_name="nvidia_blackwell_technical_blog.pdf")
    rs = score_retrieval(chunk, "question", set(), set())
    assert rs.document_priority_score >= 0.8


def test_score_retrieval_document_priority_general():
    chunk = _make_chunk("text", doc_name="general_commentary.pdf")
    rs = score_retrieval(chunk, "question", set(), set())
    assert rs.document_priority_score <= 0.6


# --- select_top_chunks ---

def test_select_top_chunks_returns_at_most_top_n():
    chunks = [_make_chunk(f"chunk text {i}", chunk_number=i) for i in range(1, 11)]
    selected, _ = select_top_chunks(chunks, "power rack", top_n=5)
    assert len(selected) <= 5


def test_select_top_chunks_returns_all_retrieval_scores():
    chunks = [_make_chunk(f"chunk text {i}", chunk_number=i) for i in range(1, 8)]
    _, retrieval_scores = select_top_chunks(chunks, "power cooling", top_n=3)
    assert len(retrieval_scores) == len(chunks)


def test_select_top_chunks_prefers_relevant_chunks():
    relevant = _make_chunk(
        "Power distribution and cooling infrastructure for NVIDIA racks.",
        doc_name="relevant.txt",
        chunk_number=1,
    )
    irrelevant = _make_chunk(
        "The cat sat on the mat near the fireplace.",
        doc_name="irrelevant.txt",
        chunk_number=1,
    )
    selected, scores = select_top_chunks(
        [irrelevant, relevant], "power cooling rack", top_n=1
    )
    assert len(selected) == 1
    # The relevant chunk should have a higher overall score
    score_map = {rs.chunk_id: rs.overall_retrieval_score for rs in scores}
    assert score_map[relevant.chunk_id] > score_map[irrelevant.chunk_id]


# --- trace integration ---

def test_trace_includes_retrieval_ranking():
    doc = _make_doc("NVIDIA Blackwell power cooling networking rack architecture data.", "nvidia.txt")
    agent = DcPowerAgent(client=MockClaudeClient())
    memo = agent.analyze("What is the power and cooling strategy?", [doc])

    trace = build_trace(
        question="What is the power and cooling strategy?",
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=[doc],
        memo=memo,
        mock_mode=True,
    )
    assert "retrieval_ranking" in trace
    assert isinstance(trace["retrieval_ranking"], list)
    assert "selected_chunk_ids" in trace
    assert "rejected_chunk_ids" in trace


def test_top_chunks_cli_option_limits_selection():
    # Build a doc large enough to produce multiple chunks
    sentence = "The NVIDIA rack uses power and liquid cooling and networking infrastructure. "
    text = sentence * 200  # ~14 800 chars → at least 2 chunks
    doc = _make_doc(text, "nvidia_rack.txt")

    agent = DcPowerAgent(client=MockClaudeClient(), top_chunks=1)
    memo = agent.analyze("What is the power strategy?", [doc])

    assert memo.metadata.get("chunks_selected", 0) <= 1


# ---------------------------------------------------------------------------
# Coverage guarantee: every relevant document gets at least one chunk selected
# ---------------------------------------------------------------------------


def test_coverage_guarantee_selects_short_synthetic_documents():
    """
    A short single-chunk synthetic doc must be selected even when competing
    against many higher-priority multi-chunk NVIDIA documents.

    This is the retrieval-layer guarantee that enables contradiction detection
    to work when test_power_a.txt / test_power_b.txt are loaded alongside real
    NVIDIA source PDFs.
    """
    from dc_power_agent.retrieval import MIN_COVERAGE_SCORE, select_top_chunks

    # Build many high-priority NVIDIA-named docs to fill the top-N slots
    nvidia_docs = [
        _make_doc(
            "NVIDIA Rubin NVL72 rack power cooling liquid networking infrastructure architecture. " * 20,
            f"nvidia_rubin_technical_blog_{i}.pdf",
        )
        for i in range(6)
    ]
    # Short synthetic test documents that would normally be pushed out
    test_doc_a = _make_doc(
        "The NVIDIA NVL72 rack draws 120 kW under peak load.",
        "test_power_a.txt",
    )
    test_doc_b = _make_doc(
        "The NVIDIA NVL72 rack requires 180 kW of DC power at full utilisation.",
        "test_power_b.txt",
    )

    from dc_power_agent.chunker import chunk_documents

    all_chunks = chunk_documents(nvidia_docs + [test_doc_a, test_doc_b])
    question = "What are the DC power implications of the NVIDIA NVL72 rack?"

    selected, scores = select_top_chunks(all_chunks, question)
    selected_docs = {c.document_name for c in selected}

    assert "test_power_a.txt" in selected_docs, (
        "test_power_a.txt must be selected by coverage guarantee"
    )
    assert "test_power_b.txt" in selected_docs, (
        "test_power_b.txt must be selected by coverage guarantee"
    )


def test_coverage_guarantee_excludes_irrelevant_documents():
    """
    The coverage guarantee must NOT include documents whose best chunk scores
    below MIN_COVERAGE_SCORE (genuinely irrelevant documents).

    We need enough chunks from high-priority docs to fill top_n=15 completely,
    so that the agile doc is excluded from phase-1 selection and the coverage
    guarantee is actually exercised.
    """
    from dc_power_agent.retrieval import DEFAULT_TOP_CHUNKS, MIN_COVERAGE_SCORE, select_top_chunks
    from dc_power_agent.chunker import CHUNK_MAX, chunk_documents

    # Each NVIDIA doc produces ~2 chunks; 10 docs → ~20 chunks > top_n=15
    sentence = "NVIDIA Rubin NVL72 rack power cooling liquid networking infrastructure architecture. "
    nvidia_docs = [
        _make_doc(sentence * ((CHUNK_MAX * 2) // len(sentence) + 1), f"nvidia_rubin_tech_{i}.pdf")
        for i in range(10)
    ]
    irrelevant_doc = _make_doc(
        "Sprint retrospectives and agile ceremonies. Team velocity metrics.",
        "agile_process.txt",
    )

    all_chunks = chunk_documents(nvidia_docs + [irrelevant_doc])
    assert len(all_chunks) > DEFAULT_TOP_CHUNKS, (
        "Need more chunks than top_n to exercise the coverage guarantee"
    )
    question = "What is the DC power draw of the NVIDIA NVL72 rack?"

    selected, scores = select_top_chunks(all_chunks, question)
    selected_docs = {c.document_name for c in selected}

    # Confirm the agile doc's best score is below the threshold
    irrelevant_scores = [s for s in scores if s.document_name == "agile_process.txt"]
    assert irrelevant_scores, "agile_process.txt must have a retrieval score"
    best_irrelevant = max(s.overall_retrieval_score for s in irrelevant_scores)
    assert best_irrelevant < MIN_COVERAGE_SCORE, (
        f"Test precondition failed: agile doc scored {best_irrelevant:.3f}, "
        f"expected < {MIN_COVERAGE_SCORE}"
    )

    assert "agile_process.txt" not in selected_docs, (
        "Irrelevant document scoring below MIN_COVERAGE_SCORE must not be force-included"
    )
