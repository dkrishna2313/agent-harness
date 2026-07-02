"""Tests for the Evidence LLM boundary (PH2.2)."""

from __future__ import annotations

import json

import pytest

from functional_agents.evidence_boundary import (
    EvidenceOutput,
    finalize_evidence,
    normalize_evidence_note,
    validate_evidence_output,
    EvidenceBoundaryError,
    EvidenceNormalizationError,
    EvidenceValidationError,
)


def _note() -> dict:
    return {
        "evidence_items": [
            {"evidence_id": "E1", "claim": "c1", "source_document": "d.md"},
            {"evidence_id": "E2", "claim": "c2", "source_document": "d.md"},
        ],
        "evidence_by_subquestion": {"q1": ["E1", "E2"]},
        "evidence_by_area": {"Power": ["E1"]},
        "coverage_by_subquestion": {"q1": {"coverage": "STRONG"}},
        "evidence_summary": {"total_evidence_items": 2},
        "profile_coverage_by_profile": {"ai_data_centers": {"evidence_count": 2}},
        "profiles_requested": ["ai_data_centers"],
        "profiles_contributing": ["ai_data_centers"],
        "profiles_missing": [],
    }


# ---------------------------------------------------------------------------
# Valid path — behavior + citation preservation
# ---------------------------------------------------------------------------

def test_valid_note_round_trips_identically():
    note = _note()
    out, diag = finalize_evidence(note, plan={"subquestions": ["q1"]})
    assert isinstance(out, EvidenceOutput)
    assert out.as_note() == note          # byte-identical for well-formed input
    assert diag["failed_stage"] is None
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}


def test_citation_integrity_reported():
    out, diag = finalize_evidence(_note())
    assert diag["validation"]["citations_present"] == 2
    assert diag["validation"]["citations_missing"] == 0


def test_planner_alignment_reported():
    _, diag = finalize_evidence(_note(), plan={"subquestions": ["q1"]})
    assert diag["validation"]["planner_aligned"] is True
    _, diag2 = finalize_evidence(_note(), plan={"subquestions": ["OTHER"]})
    assert diag2["validation"]["planner_aligned"] is False  # non-fatal diagnostic


def test_stringified_json_note_recovered():
    out, diag = finalize_evidence(json.dumps(_note()))
    assert len(out.evidence_items) == 2
    assert diag["stages"]["normalization"] == "ok"


# ---------------------------------------------------------------------------
# Normalization recovery (safe repair only)
# ---------------------------------------------------------------------------

def test_malformed_items_dropped():
    note = _note()
    note["evidence_items"] = [
        {"evidence_id": "E1", "claim": "c1"}, "not-a-dict", {"no_id": 1}, 42,
    ]
    # mapping referenced E2 which is now gone → dangling repaired
    note["evidence_by_subquestion"] = {"q1": ["E1", "E2"]}
    out, diag = finalize_evidence(note)
    assert [i["evidence_id"] for i in out.evidence_items] == ["E1"]
    assert out.evidence_by_subquestion["q1"] == ["E1"]
    assert any("malformed evidence item" in r for r in diag["normalization"]["repairs"])


def test_dangling_mapping_references_repaired():
    note = _note()
    note["evidence_by_subquestion"] = {"q1": ["E1", "MISSING"]}
    note["evidence_by_area"] = {"Power": ["ALSO_MISSING"]}
    out, diag = finalize_evidence(note)
    assert out.evidence_by_subquestion["q1"] == ["E1"]
    assert out.evidence_by_area["Power"] == []
    assert any("mapping reference" in r for r in diag["normalization"]["repairs"])


def test_scalar_mapping_value_coerced():
    note = _note()
    note["evidence_by_subquestion"] = {"q1": "E1"}  # scalar, not list
    out, _ = finalize_evidence(note)
    assert out.evidence_by_subquestion["q1"] == ["E1"]


def test_missing_optional_collections_default_empty():
    note = {"evidence_items": [{"evidence_id": "E1", "source_document": "d"}]}
    out, _ = finalize_evidence(note)
    assert out.evidence_by_subquestion == {}
    assert out.profiles_requested == []


# ---------------------------------------------------------------------------
# Deterministic failures
# ---------------------------------------------------------------------------

def test_non_object_note_fails_normalization():
    with pytest.raises(EvidenceNormalizationError) as ei:
        finalize_evidence("not an object")
    assert ei.value.diagnostics["failed_stage"] == "normalization"


def test_empty_note_is_valid_zero_evidence():
    """Zero evidence is a valid (degenerate) outcome — not a failure."""
    out, diag = finalize_evidence({"evidence_items": []})
    assert out.evidence_items == []
    assert diag["failed_stage"] is None


def test_boundary_error_hierarchy():
    assert issubclass(EvidenceNormalizationError, EvidenceBoundaryError)
    assert issubclass(EvidenceValidationError, EvidenceBoundaryError)


def test_validation_catches_dangling_if_normalization_bypassed():
    """Directly validating a note with dangling refs fails deterministically."""
    bad = {
        "evidence_items": [{"evidence_id": "E1"}],
        "evidence_by_subquestion": {"q": ["E1", "GHOST"]},
    }
    with pytest.raises(EvidenceValidationError) as ei:
        validate_evidence_output(bad)
    assert ei.value.diagnostics["failed_stage"] == "validation"


def test_normalize_then_validate_are_consistent():
    norm, _ = normalize_evidence_note(_note())
    out, vdiag = validate_evidence_output(norm, plan={"subquestions": ["q1"]})
    assert vdiag["passed"] is True
    assert len(out.evidence_items) == 2
