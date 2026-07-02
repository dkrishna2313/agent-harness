"""Tests for the Hypothesis LLM boundary (PH2.3)."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield


from functional_agents.context import AgentContext
from functional_agents.hypothesis_agent import HypothesisAgent
from functional_agents.hypothesis_boundary import (
    HypothesisOutput,
    finalize_hypotheses,
    normalize_hypothesis_payload,
    validate_hypothesis_output,
    HypothesisBoundaryError,
    HypothesisNormalizationError,
    HypothesisValidationError,
    VALID_CONFIDENCE,
)


def _raw() -> dict:
    return {
        "hypotheses": [
            {"id": "H1", "title": "t1", "summary": "s1",
             "supporting_evidence": ["E1", "E2"], "confidence": "high",
             "confidence_rationale": "r1", "decision_implications": ["do X"]},
            {"id": "H2", "title": "t2", "summary": "s2",
             "supporting_evidence": ["E3"], "confidence": "medium"},
        ],
        "synthesis_note": "two competing hypotheses",
    }


# ---------------------------------------------------------------------------
# Valid path
# ---------------------------------------------------------------------------

def test_valid_payload_typed_output():
    out, diag = finalize_hypotheses(_raw())
    assert isinstance(out, HypothesisOutput)
    assert len(out.hypotheses) == 2
    assert out.synthesis_note == "two competing hypotheses"
    assert diag["failed_stage"] is None
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}


def test_as_dicts_preserves_schema_and_references():
    out, _ = finalize_hypotheses(_raw())
    d = out.as_dicts()[0]
    assert d["id"] == "H1"
    assert d["supporting_evidence"] == ["E1", "E2"]  # evidence references preserved
    for key in ("id", "title", "summary", "type", "confidence", "confidence_rationale",
                "supporting_evidence", "contradicting_evidence", "evidence_gaps",
                "decision_implications", "disconfirming_evidence_needed"):
        assert key in d


def test_stringified_json_recovered():
    out, diag = finalize_hypotheses(json.dumps(_raw()))
    assert len(out.hypotheses) == 2
    assert diag["stages"]["normalization"] == "ok"


# ---------------------------------------------------------------------------
# Normalization recovery
# ---------------------------------------------------------------------------

def test_confidence_casing_normalized():
    raw = {"hypotheses": [{"id": "H1", "title": "t", "summary": "s", "confidence": " HIGH "}]}
    norm, diag = normalize_hypothesis_payload(raw)
    assert norm["hypotheses"][0]["confidence"] == "high"
    assert any("confidence casing" in r for r in diag["repairs"])


def test_scalar_evidence_ref_coerced_to_list():
    raw = {"hypotheses": [{"id": "H1", "title": "t", "summary": "s",
                           "supporting_evidence": "E1"}]}
    norm, _ = normalize_hypothesis_payload(raw)
    assert norm["hypotheses"][0]["supporting_evidence"] == ["E1"]


def test_malformed_hypotheses_dropped():
    raw = {"hypotheses": [
        {"id": "H1", "title": "t", "summary": "s"},
        {"id": "H2", "title": "no summary"},   # missing summary → dropped
        "not-a-dict",
    ]}
    out, diag = finalize_hypotheses(raw)
    assert [h.id for h in out.hypotheses] == ["H1"]
    assert any("malformed hypothesis" in r for r in diag["normalization"]["repairs"])


# ---------------------------------------------------------------------------
# Deterministic failures
# ---------------------------------------------------------------------------

def test_non_object_fails_normalization():
    with pytest.raises(HypothesisNormalizationError) as ei:
        finalize_hypotheses("not json")
    assert ei.value.diagnostics["failed_stage"] == "normalization"


def test_invalid_confidence_enum_fails_validation():
    raw = {"hypotheses": [{"id": "H1", "title": "t", "summary": "s", "confidence": "certain"}]}
    with pytest.raises(HypothesisValidationError) as ei:
        finalize_hypotheses(raw)
    assert ei.value.diagnostics["failed_stage"] == "validation"


def test_all_valid_confidence_enums_accepted():
    for c in VALID_CONFIDENCE:
        raw = {"hypotheses": [{"id": "H1", "title": "t", "summary": "s", "confidence": c}]}
        out, _ = finalize_hypotheses(raw)
        assert out.hypotheses[0].confidence == c


def test_empty_hypotheses_is_valid():
    out, diag = finalize_hypotheses({"hypotheses": [], "synthesis_note": "none"})
    assert out.hypotheses == []
    assert diag["failed_stage"] is None


def test_boundary_error_hierarchy():
    assert issubclass(HypothesisNormalizationError, HypothesisBoundaryError)
    assert issubclass(HypothesisValidationError, HypothesisBoundaryError)


def test_validate_flags_missing_evidence_reference_types():
    # supporting_evidence not a list → validation error
    with pytest.raises(HypothesisValidationError):
        validate_hypothesis_output({"hypotheses": [
            {"id": "H1", "title": "t", "summary": "s", "confidence": "high",
             "supporting_evidence": "E1"}  # str, not list (normalization bypassed)
        ]})


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def _ctx() -> AgentContext:
    ctx = AgentContext(
        question="q", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"research_id": "R-HB", "contradictions": []}, run_id="hb001",
    )
    ctx.evidence_notes = [{
        "evidence_items": [{"evidence_id": "E1", "claim": "c", "source_document": "d"}],
        "profile_coverage_by_profile": {},
    }]
    return ctx


def test_agent_records_boundary_and_produces_hypotheses():
    ctx = _ctx()
    HypothesisAgent().run(ctx)  # no client → deterministic mock payload, valid
    assert ctx.hypotheses
    boundary = ctx.trace["_hypothesis_boundary"]
    assert boundary["failed_stage"] is None
    assert boundary["stages"]["validation"] == "ok"


class _MalformedHypClient:
    is_mock = False
    def generate_hypotheses_raw(self, *a, **k):
        return {"hypotheses": [{"id": "H1", "title": "t", "summary": "s", "confidence": "certain"}]}


def test_agent_deterministic_failure_on_bad_enum():
    ctx = _ctx()
    with pytest.raises(HypothesisBoundaryError):
        HypothesisAgent(client=_MalformedHypClient()).run(ctx)
    assert ctx.trace["_hypothesis_boundary"]["failed_stage"] == "validation"
    assert not ctx.hypotheses  # business logic did not run


class _GenBoomHypClient:
    is_mock = False
    def generate_hypotheses_raw(self, *a, **k):
        raise RuntimeError("api down")


def test_generation_failure_classified():
    from functional_agents.hypothesis_boundary import HypothesisGenerationError
    ctx = _ctx()
    with pytest.raises(HypothesisGenerationError):
        HypothesisAgent(client=_GenBoomHypClient()).run(ctx)
    assert ctx.trace["_hypothesis_boundary"]["failed_stage"] == "generation"
