"""Regression tests: synthesis must complete without truncation.

G3 regression requirement: given a corpus of documents that triggers research
gap detection, the mock synthesize_memo call must return a memo whose
evaluation_warnings do NOT contain any max_tokens / truncation error, and the
memo must have a non-empty executive_summary.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dc_power_agent.agent import DcPowerAgent, MAX_SYNTHESIS_EVIDENCE
from dc_power_agent.claude_client import MockClaudeClient
from dc_power_agent.schemas import SourceDocument
from dc_power_agent.trace import build_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(text: str, name: str) -> SourceDocument:
    return SourceDocument(
        path=Path(f"sources/{name}"),
        title=name,
        extension=Path(name).suffix,
        text=text,
    )


def _g3_corpus() -> list[SourceDocument]:
    """Multi-document corpus that exercises G3 gap detection.

    Covers power/cooling topics partially so that some subtopics remain
    uncovered and research gaps are emitted, while providing enough evidence
    for synthesis to proceed.
    """
    docs = [
        _doc(
            "The NVIDIA Rubin NVL72 rack draws up to 132 kW of power. "
            "The rack requires a 480 V three-phase power feed. "
            "Power delivery uses a busway overhead distribution system. "
            "[Source: rubin_power.md, Evidence: E001]",
            "rubin_power.md",
        ),
        _doc(
            "The NVL72 uses direct liquid cooling (DLC) with rear-door CDUs. "
            "Coolant water supply temperature must be below 25 °C. "
            "The rack produces significant heat requiring facility-level heat rejection. "
            "[Source: rubin_cooling.md, Evidence: E002]",
            "rubin_cooling.md",
        ),
        _doc(
            "NVLink bandwidth between GPUs reaches 1.8 TB/s. "
            "The spine-leaf network topology uses high-radix switches. "
            "400G optical transceivers connect compute and storage tiers. "
            "[Source: rubin_networking.md, Evidence: E003]",
            "rubin_networking.md",
        ),
        _doc(
            "Commissioning the NVL72 requires validation of power and cooling. "
            "Ongoing monitoring tracks rack inlet temperatures and PDU loads. "
            "[Source: rubin_operations.md, Evidence: E004]",
            "rubin_operations.md",
        ),
    ]
    return docs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_synthesis_completes_without_truncation():
    """G3 regression: synthesis must not be truncated (no max_tokens error)."""
    docs = _g3_corpus()
    question = (
        "What are the power, cooling, and networking requirements "
        "for deploying NVIDIA NVL72 racks in a data center?"
    )

    memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, docs)

    # No synthesis truncation warning should be present
    assert not any("max_tokens" in w or "truncat" in w for w in memo.evaluation_warnings)

    # Mock client always returns a non-empty executive summary
    assert memo.executive_summary.strip(), "executive_summary must not be empty"


def test_synthesis_evidence_capped_at_max():
    """Evidence passed to synthesis must not exceed MAX_SYNTHESIS_EVIDENCE."""
    docs = _g3_corpus()
    question = (
        "What are the power, cooling, and networking requirements "
        "for deploying NVIDIA NVL72 racks in a data center?"
    )

    memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, docs)

    used = memo.metadata.get("evidence_items_used_for_synthesis", 0)
    assert used <= MAX_SYNTHESIS_EVIDENCE, (
        f"evidence_items_used_for_synthesis={used} exceeds cap {MAX_SYNTHESIS_EVIDENCE}"
    )


def test_synthesis_trace_fields_populated():
    """Trace must include all four new synthesis bookkeeping fields."""
    docs = _g3_corpus()
    question = (
        "What are the power, cooling, and networking requirements "
        "for deploying NVIDIA NVL72 racks in a data center?"
    )

    memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, docs)

    trace = build_trace(
        question=question,
        source_directory=Path("sources"),
        output_path=Path("outputs/memo.md"),
        documents=docs,
        memo=memo,
        mock_mode=True,
    )

    assert "synthesis_input_tokens" in trace
    assert "evidence_passed_to_synthesis" in trace
    assert "contradictions_passed_to_synthesis" in trace
    assert "research_gaps_passed_to_synthesis" in trace

    assert isinstance(trace["synthesis_input_tokens"], int)
    assert isinstance(trace["evidence_passed_to_synthesis"], int)


def test_g3_corpus_produces_research_gaps_and_synthesis_succeeds():
    """G3 + synthesis: research gaps are detected AND memo is produced."""
    docs = _g3_corpus()
    question = (
        "What are the power, cooling, and networking requirements "
        "for deploying NVIDIA NVL72 racks in a data center?"
    )

    memo = DcPowerAgent(client=MockClaudeClient()).analyze(question, docs)

    # Research gaps should be present (corpus is deliberately incomplete)
    assert "research_gaps" in memo.metadata

    # Synthesis must have produced a valid memo
    evidence = memo.source_notes or memo.evidence
    assert len(evidence) > 0, "At least one evidence item expected"
    assert memo.executive_summary.strip(), "executive_summary must be non-empty"
