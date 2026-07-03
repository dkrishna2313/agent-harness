"""Tests for the Report input boundary (PH2.5 Part B)."""

from __future__ import annotations

import pytest

from functional_agents.report_boundary import (
    ReportInput,
    finalize_report_input,
    normalize_report_input,
    validate_report_input,
    ReportBoundaryError,
    ReportNormalizationError,
    ReportValidationError,
)


class _Memo:
    """Stand-in for a ResearchMemo object."""


def test_valid_inputs_produce_typed_report_input():
    memo = _Memo()
    out, diag = finalize_report_input({
        "question": "What are the power constraints?",
        "memo": memo,
        "documents": [1, 2],
        "plan": {"research_type": "RESEARCH"},
        "evidence_note": {"evidence_items": []},
        "recommendation_count": 3,
    })
    assert isinstance(out, ReportInput)
    assert out.memo is memo          # live object passed through untouched
    assert out.documents == [1, 2]
    assert out.recommendation_count == 3
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}
    assert diag["failed_stage"] is None


def test_missing_memo_fails_validation():
    with pytest.raises(ReportValidationError) as ei:
        finalize_report_input({"question": "q", "memo": None})
    assert ei.value.diagnostics["failed_stage"] == "validation"
    assert any("memo" in e for e in ei.value.diagnostics["errors"])


def test_missing_question_fails_validation():
    with pytest.raises(ReportValidationError):
        finalize_report_input({"question": "", "memo": _Memo()})


def test_non_mapping_fails_normalization():
    with pytest.raises(ReportNormalizationError) as ei:
        finalize_report_input("not a mapping")
    assert ei.value.diagnostics["failed_stage"] == "normalization"


def test_normalization_coerces_wrong_types():
    norm, _ = normalize_report_input({
        "question": "q", "memo": _Memo(),
        "documents": "not-a-list", "plan": None, "recommendation_count": None,
    })
    assert norm["documents"] == []
    assert norm["plan"] == {}
    assert norm["recommendation_count"] == 0


def test_boundary_error_hierarchy():
    assert issubclass(ReportNormalizationError, ReportBoundaryError)
    assert issubclass(ReportValidationError, ReportBoundaryError)


def test_validate_returns_diagnostics():
    _, diag = validate_report_input({"question": "q", "memo": _Memo(), "documents": [1],
                                     "plan": {}, "evidence_note": {}, "recommendation_count": 2})
    assert diag["passed"] is True
    assert diag["documents"] == 1
    assert diag["recommendations"] == 2
