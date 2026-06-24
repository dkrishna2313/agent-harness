"""JH1 – ChunkClassifier and evidence yield tests.

Covers:
  - CandidateSignals computation (numeric, units, dates, comparative, policy)
  - is_boilerplate() detection
  - is_reference_section() detection
  - is_high_priority_section() detection
  - classify_chunk() full classification
  - compute_evidence_yield_metrics()
  - chunker.select_relevant_chunks() skips boilerplate
  - ChunkDiagnostic has JH1 fields
"""

from __future__ import annotations

import pytest

from research_agent.chunk_classifier import (
    CandidateSignals,
    ChunkClassification,
    PRIORITY_BOOST,
    classify_chunk,
    compute_candidate_signals,
    is_boilerplate,
    is_high_priority_section,
    is_reference_section,
)
from research_agent.chunker import (
    compute_chunk_diagnostics,
    compute_evidence_yield_metrics,
    select_relevant_chunks,
)
from research_agent.schemas import Chunk, ChunkDiagnostic, EvidenceItem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str, doc: str = "doc.pdf", n: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_name=doc,
        chunk_number=n,
        text=text,
        start_offset=0,
        end_offset=len(text),
    )


def _evidence_item(chunk_id: str = "") -> EvidenceItem:
    return EvidenceItem(
        evidence_id="E001",
        claim="Test claim",
        source_document="doc.pdf",
        source_chunk_id=chunk_id,
        evidence_snippet="Test snippet",
        category="power",
        relevance="High relevance",
        confidence="high",
    )


_EVIDENCE_TEXT = (
    "The NVIDIA GB200 NVL72 system provides 1,440 B200 Tensor Core GPUs "
    "in a 72-GPU rack configuration. Each rack requires 120 kW of power "
    "and delivers 1.4 exaflops of AI compute. Rack-scale liquid cooling "
    "operates at 35°C inlet temperature. The system supports 1,800 GB/s "
    "NVLink bandwidth per GPU. Deployment cost is approximately $10 million "
    "per rack at Q1 2025 pricing. ASHRAE A2 compliance is required."
)

_TOC_TEXT = (
    "Table of Contents\n"
    "1. Executive Summary ....... 3\n"
    "2. Market Analysis ......... 7\n"
    "3. Technology Overview ..... 12\n"
    "4. Financial Projections ... 18\n"
    "5. Conclusions ............. 24\n"
    "6. References .............. 28\n"
)

_LEGAL_TEXT = (
    "© 2024 ACME Corporation. All rights reserved. "
    "No part of this publication may be reproduced without prior written permission. "
    "Confidential and Proprietary. "
    "This document is provided for informational purposes only. "
    "The information herein is subject to change without notice. "
    "ACME Corporation makes no representations or warranties, express or implied."
)

_REFERENCE_TEXT = (
    "References\n"
    "[1] Smith, J. (2023). AI Infrastructure Analysis. doi: 10.1234/ai.2023.001\n"
    "[2] Jones, M. et al. (2024). Grid Integration Studies. doi: 10.5678/grid.2024.002\n"
    "[3] Brown, K. (2022). Data Center Power. doi: 10.9012/dc.2022.003\n"
    "[4] White, R. (2023). Transmission Planning. doi: 10.3456/tp.2023.004\n"
)

_EXEC_SUMMARY_TEXT = (
    "Executive Summary\n\n"
    "This report analyses the strategic implications of AI infrastructure investment "
    "over the next decade. Key findings indicate that transmission capacity will be "
    "the binding constraint for large-scale GPU deployments. We recommend a grid-first "
    "site selection strategy. Power demand from AI data centers is projected to reach "
    "50 GW by 2030, up from 12 GW in 2024."
)


# ---------------------------------------------------------------------------
# compute_candidate_signals
# ---------------------------------------------------------------------------

def test_signals_numeric_count():
    signals = compute_candidate_signals(_EVIDENCE_TEXT)
    assert signals.numeric_claim_count >= 3


def test_signals_unit_count():
    signals = compute_candidate_signals(_EVIDENCE_TEXT)
    assert signals.unit_count >= 2


def test_signals_policy_terms():
    text = "FERC Order 2023 requires PJM and MISO to process interconnection requests."
    signals = compute_candidate_signals(text)
    assert signals.policy_or_standard_terms >= 2


def test_signals_date_count():
    text = "In 2023, demand grew. By Q1 2025 pricing had risen. Since 2020, capacity doubled."
    signals = compute_candidate_signals(text)
    assert signals.date_count >= 2


def test_signals_comparative_count():
    text = "The system is more than 2x faster than previous generation. Compared to air cooling, it is more efficient."
    signals = compute_candidate_signals(text)
    assert signals.comparative_claim_count >= 1


def test_signals_total_is_sum():
    signals = compute_candidate_signals(_EVIDENCE_TEXT)
    expected = (
        signals.numeric_claim_count
        + signals.named_entity_count
        + signals.date_count
        + signals.unit_count
        + signals.comparative_claim_count
        + signals.policy_or_standard_terms
    )
    assert signals.total == expected


def test_signals_to_dict_has_all_keys():
    signals = compute_candidate_signals(_EVIDENCE_TEXT)
    d = signals.to_dict()
    assert set(d.keys()) == {
        "numeric_claim_count", "named_entity_count", "date_count",
        "unit_count", "comparative_claim_count", "policy_or_standard_terms",
    }


def test_signal_score_0_to_1():
    signals = compute_candidate_signals(_EVIDENCE_TEXT)
    assert 0.0 <= signals.signal_score <= 1.0


def test_signal_score_caps_at_1():
    # Very rich text should not exceed 1.0
    rich = _EVIDENCE_TEXT * 3
    signals = compute_candidate_signals(rich)
    assert signals.signal_score <= 1.0


def test_empty_text_zero_signals():
    signals = compute_candidate_signals("")
    assert signals.total == 0
    assert signals.signal_score == 0.0


# ---------------------------------------------------------------------------
# is_boilerplate
# ---------------------------------------------------------------------------

def test_is_boilerplate_toc():
    assert is_boilerplate(_TOC_TEXT) is True


def test_is_boilerplate_legal():
    assert is_boilerplate(_LEGAL_TEXT) is True


def test_is_boilerplate_evidence_text():
    assert is_boilerplate(_EVIDENCE_TEXT) is False


def test_is_boilerplate_short_text():
    # Very short texts are treated as boilerplate
    assert is_boilerplate("Cover page") is True


def test_is_boilerplate_exec_summary():
    assert is_boilerplate(_EXEC_SUMMARY_TEXT) is False


# ---------------------------------------------------------------------------
# is_reference_section
# ---------------------------------------------------------------------------

def test_is_reference_section_true():
    long_ref = (
        "References\n"
        "[1] Smith, J. (2023). AI Infrastructure Analysis. doi: 10.1234/ai.2023.001\n"
        "[2] Jones, M. et al. (2024). Grid Integration Studies. doi: 10.5678/grid.2024.002\n"
        "[3] Brown, K. (2022). Data Center Power Trends in Hyperscale Facilities. doi: 10.9012/dc.2022.003\n"
        "[4] White, R. (2023). Transmission Planning and Interconnection. doi: 10.3456/tp.2023.004\n"
        "[5] Green, L. (2024). Policy Implications of FERC Order 2023. doi: 10.7890/ferc.2024.005\n"
    )
    assert is_reference_section(long_ref) is True


def test_is_reference_section_false_for_evidence():
    assert is_reference_section(_EVIDENCE_TEXT) is False


# ---------------------------------------------------------------------------
# is_high_priority_section
# ---------------------------------------------------------------------------

def test_is_high_priority_exec_summary():
    assert is_high_priority_section(_EXEC_SUMMARY_TEXT) is True


def test_is_high_priority_key_findings():
    text = "Key Findings\nOur analysis shows that demand will grow 40% by 2030."
    assert is_high_priority_section(text) is True


def test_is_high_priority_recommendations():
    text = "Recommendations\nWe recommend a grid-first site selection strategy."
    assert is_high_priority_section(text) is True


def test_not_high_priority_middle_text():
    text = "This section provides background on the history of data centers."
    assert is_high_priority_section(text) is False


# ---------------------------------------------------------------------------
# classify_chunk
# ---------------------------------------------------------------------------

def test_classify_evidence_dense():
    clf = classify_chunk("C001", _EVIDENCE_TEXT)
    assert clf.chunk_type == "evidence_dense"
    assert clf.extraction_priority in ("high", "medium")


def test_classify_boilerplate():
    clf = classify_chunk("C002", _TOC_TEXT)
    assert clf.chunk_type == "boilerplate"
    assert clf.extraction_priority == "skip"


def test_classify_legal_boilerplate():
    clf = classify_chunk("C003", _LEGAL_TEXT)
    assert clf.chunk_type == "boilerplate"
    assert clf.extraction_priority == "skip"


def test_classify_reference_section():
    clf = classify_chunk("C004", _REFERENCE_TEXT)
    # Both "reference" and "boilerplate" map to skip — both are correct
    assert clf.extraction_priority == "skip"


def test_classify_exec_summary_high_priority():
    clf = classify_chunk("C005", _EXEC_SUMMARY_TEXT)
    assert clf.extraction_priority == "high"


def test_classify_returns_chunk_classification():
    clf = classify_chunk("C001", _EVIDENCE_TEXT)
    assert isinstance(clf, ChunkClassification)


def test_classify_includes_candidate_signals():
    clf = classify_chunk("C001", _EVIDENCE_TEXT)
    assert isinstance(clf.candidate_signals, CandidateSignals)


def test_classify_has_classification_reason():
    clf = classify_chunk("C001", _EVIDENCE_TEXT)
    assert isinstance(clf.classification_reason, str) and clf.classification_reason


def test_classify_chunk_id_preserved():
    clf = classify_chunk("MYID", _EVIDENCE_TEXT)
    assert clf.chunk_id == "MYID"


# ---------------------------------------------------------------------------
# PRIORITY_BOOST values
# ---------------------------------------------------------------------------

def test_priority_boost_high_greater_than_medium():
    assert PRIORITY_BOOST["high"] > PRIORITY_BOOST["medium"]


def test_priority_boost_skip_is_negative():
    assert PRIORITY_BOOST["skip"] < 0


def test_priority_boost_keys():
    assert set(PRIORITY_BOOST.keys()) == {"high", "medium", "low", "skip"}


# ---------------------------------------------------------------------------
# select_relevant_chunks – boilerplate exclusion
# ---------------------------------------------------------------------------

def test_select_chunks_excludes_boilerplate():
    chunks = [
        _chunk("C001", _EVIDENCE_TEXT, n=1),
        _chunk("C002", _TOC_TEXT, n=2),
        _chunk("C003", _LEGAL_TEXT, n=3),
    ]
    selected, _ = select_relevant_chunks(chunks, "GPU power cooling data center")
    selected_ids = {c.chunk_id for c in selected}
    assert "C002" not in selected_ids
    assert "C003" not in selected_ids


def test_select_chunks_includes_evidence_dense():
    chunks = [
        _chunk("C001", _EVIDENCE_TEXT, n=1),
        _chunk("C002", _TOC_TEXT, n=2),
    ]
    selected, _ = select_relevant_chunks(chunks, "GPU power cooling data center")
    assert any(c.chunk_id == "C001" for c in selected)


def test_select_chunks_scores_returned_for_all():
    chunks = [
        _chunk("C001", _EVIDENCE_TEXT, n=1),
        _chunk("C002", _TOC_TEXT, n=2),
    ]
    _, scores = select_relevant_chunks(chunks, "GPU power cooling")
    assert "C001" in scores
    assert "C002" in scores


# ---------------------------------------------------------------------------
# compute_chunk_diagnostics – JH1 fields
# ---------------------------------------------------------------------------

def test_chunk_diagnostic_has_jh1_fields():
    chunk = _chunk("C001", _EVIDENCE_TEXT)
    ev = _evidence_item("C001")
    _, scores = select_relevant_chunks([chunk], "GPU power")
    diagnostics = compute_chunk_diagnostics([chunk], [chunk], [ev], scores)
    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert hasattr(d, "chunk_type") or isinstance(d, ChunkDiagnostic)
    # Model dump includes JH1 fields
    data = d.model_dump()
    assert "chunk_type" in data
    assert "extraction_priority" in data
    assert "candidate_signals" in data


def test_chunk_diagnostic_boilerplate_chunk_type():
    chunk = _chunk("C001", _TOC_TEXT)
    _, scores = select_relevant_chunks([chunk], "GPU power")
    diagnostics = compute_chunk_diagnostics([chunk], [], [], scores)
    assert len(diagnostics) == 1
    d = diagnostics[0]
    assert d.chunk_type == "boilerplate"
    assert d.extraction_priority == "skip"


def test_chunk_diagnostic_evidence_dense_type():
    chunk = _chunk("C001", _EVIDENCE_TEXT)
    ev = _evidence_item("C001")
    _, scores = select_relevant_chunks([chunk], "GPU power cooling kW")
    diagnostics = compute_chunk_diagnostics([chunk], [chunk], [ev], scores)
    d = diagnostics[0]
    assert d.chunk_type == "evidence_dense"


def test_chunk_diagnostic_rejection_reason_for_boilerplate():
    chunk = _chunk("C001", _TOC_TEXT)
    _, scores = select_relevant_chunks([chunk], "GPU power")
    diagnostics = compute_chunk_diagnostics([chunk], [], [], scores)
    d = diagnostics[0]
    assert d.rejection_reason is not None
    assert "boilerplate" in d.rejection_reason or "skip" in d.rejection_reason


# ---------------------------------------------------------------------------
# compute_evidence_yield_metrics
# ---------------------------------------------------------------------------

def test_yield_metrics_has_required_keys():
    chunks = [_chunk("C001", _EVIDENCE_TEXT)]
    ev = [_evidence_item("C001")]
    metrics = compute_evidence_yield_metrics(chunks, chunks, ev, documents_loaded=1)
    required = {
        "documents_loaded", "chunks_total", "chunks_selected",
        "chunks_with_evidence", "evidence_items_created",
        "zero_evidence_chunks", "zero_evidence_selected_chunks",
        "skipped_boilerplate_chunks", "yield_per_selected_chunk",
        "yield_per_total_chunk",
    }
    assert required <= set(metrics.keys())


def test_yield_metrics_basic_values():
    chunks = [_chunk("C001", _EVIDENCE_TEXT), _chunk("C002", _EVIDENCE_TEXT)]
    ev = [_evidence_item("C001")]
    metrics = compute_evidence_yield_metrics(chunks, chunks[:1], ev, documents_loaded=1)
    assert metrics["chunks_total"] == 2
    assert metrics["chunks_selected"] == 1
    assert metrics["evidence_items_created"] == 1
    assert metrics["chunks_with_evidence"] == 1


def test_yield_metrics_yield_per_selected_chunk():
    chunks = [_chunk("C001", _EVIDENCE_TEXT)]
    ev = [_evidence_item("C001"), _evidence_item("C001")]
    metrics = compute_evidence_yield_metrics(chunks, chunks, ev, documents_loaded=1)
    # 2 evidence items from 1 selected chunk = 2.0
    assert metrics["yield_per_selected_chunk"] == 2.0


def test_yield_metrics_zero_chunks():
    metrics = compute_evidence_yield_metrics([], [], [], documents_loaded=0)
    assert metrics["yield_per_selected_chunk"] == 0.0
    assert metrics["yield_per_total_chunk"] == 0.0


def test_yield_metrics_skipped_boilerplate_count():
    chunks = [
        _chunk("C001", _EVIDENCE_TEXT),
        _chunk("C002", _TOC_TEXT),
        _chunk("C003", _LEGAL_TEXT),
    ]
    metrics = compute_evidence_yield_metrics(chunks, [chunks[0]], [], documents_loaded=1)
    assert metrics["skipped_boilerplate_chunks"] >= 2


def test_yield_metrics_zero_evidence_selected():
    chunk = _chunk("C001", _EVIDENCE_TEXT)
    # Selected but no evidence extracted
    metrics = compute_evidence_yield_metrics([chunk], [chunk], [], documents_loaded=1)
    assert metrics["zero_evidence_selected_chunks"] == 1
