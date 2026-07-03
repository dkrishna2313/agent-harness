"""Tests for the Recommendation LLM boundary (PH2.4)."""

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
from functional_agents.recommendation_agent import RecommendationAgent
from functional_agents.recommendation_boundary import (
    RecommendationOutput,
    finalize_recommendations,
    normalize_recommendation_payload,
    validate_recommendation_output,
    RecommendationBoundaryError,
    RecommendationNormalizationError,
    RecommendationValidationError,
    VALID_PRIORITY,
    VALID_CONFIDENCE,
    VALID_TIME_HORIZON,
)


def _raw() -> dict:
    return {
        "recommendations": [
            {"id": "R1", "title": "t1", "summary": "s1", "priority": "high",
             "time_horizon": "near_term", "confidence": "high",
             "supporting_evidence": ["E1"], "supported_by_hypotheses": ["H1"],
             "key_risks": ["risk"], "trigger_conditions": ["trigger"]},
            {"id": "R2", "title": "t2", "summary": "s2", "priority": "low",
             "time_horizon": "long_term", "confidence": "low"},
        ],
        "recommendation_portfolio": {"near_term": ["R1"], "medium_term": [], "long_term": ["R2"]},
        "synthesis_note": "two recommendations",
    }


# ---------------------------------------------------------------------------
# Valid path — behaviour + portfolio preservation
# ---------------------------------------------------------------------------

def test_valid_payload_typed_output():
    out, diag = finalize_recommendations(_raw())
    assert isinstance(out, RecommendationOutput)
    assert len(out.recommendations) == 2
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}


def test_as_dicts_and_portfolio_preserved():
    out, _ = finalize_recommendations(_raw())
    d = out.as_dicts()[0]
    assert d["id"] == "R1"
    assert d["supporting_evidence"] == ["E1"]
    assert d["supported_by_hypotheses"] == ["H1"]
    for key in ("id", "recommendation_id", "title", "summary", "priority", "time_horizon",
                "supported_by_hypotheses", "supporting_evidence", "key_risks",
                "trigger_conditions", "confidence"):
        assert key in d
    assert out.portfolio_dict() == {"near_term": ["R1"], "medium_term": [], "long_term": ["R2"]}


def test_stringified_json_recovered():
    out, diag = finalize_recommendations(json.dumps(_raw()))
    assert len(out.recommendations) == 2
    assert diag["stages"]["normalization"] == "ok"


# ---------------------------------------------------------------------------
# Normalization recovery (safe repair)
# ---------------------------------------------------------------------------

def test_enum_casing_normalized():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s",
                                "priority": "HIGH", "confidence": "Medium",
                                "time_horizon": "NEAR_TERM"}]}
    norm, diag = normalize_recommendation_payload(raw)
    r = norm["recommendations"][0]
    assert r["priority"] == "high" and r["confidence"] == "medium" and r["time_horizon"] == "near_term"
    assert any("casing" in rp for rp in diag["repairs"])


def test_scalar_refs_coerced():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s",
                               "supporting_evidence": "E1", "key_risks": "risk"}]}
    norm, _ = normalize_recommendation_payload(raw)
    r = norm["recommendations"][0]
    assert r["supporting_evidence"] == ["E1"]
    assert r["key_risks"] == ["risk"]


def test_malformed_recommendations_dropped():
    raw = {"recommendations": [
        {"id": "R1", "title": "t", "summary": "s"},
        {"id": "R2", "title": "no summary"},   # missing summary → dropped
        "not-a-dict",
    ]}
    out, diag = finalize_recommendations(raw)
    assert [r.id for r in out.recommendations] == ["R1"]
    assert any("malformed recommendation" in r for r in diag["normalization"]["repairs"])


def test_missing_portfolio_defaults_empty():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s"}]}
    out, _ = finalize_recommendations(raw)
    assert out.portfolio_dict() == {"near_term": [], "medium_term": [], "long_term": []}


# ---------------------------------------------------------------------------
# Deterministic failures
# ---------------------------------------------------------------------------

def test_non_object_fails_normalization():
    with pytest.raises(RecommendationNormalizationError) as ei:
        finalize_recommendations("not json")
    assert ei.value.diagnostics["failed_stage"] == "normalization"


def test_invalid_priority_fails_validation():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s", "priority": "urgent"}]}
    with pytest.raises(RecommendationValidationError) as ei:
        finalize_recommendations(raw)
    assert ei.value.diagnostics["failed_stage"] == "validation"


def test_invalid_confidence_fails_validation():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s", "confidence": "certain"}]}
    with pytest.raises(RecommendationValidationError):
        finalize_recommendations(raw)


def test_invalid_time_horizon_fails_validation():
    raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s", "time_horizon": "someday"}]}
    with pytest.raises(RecommendationValidationError):
        finalize_recommendations(raw)


def test_all_valid_enums_accepted():
    for p in VALID_PRIORITY:
        for c in VALID_CONFIDENCE:
            for th in VALID_TIME_HORIZON:
                raw = {"recommendations": [{"id": "R1", "title": "t", "summary": "s",
                                            "priority": p, "confidence": c, "time_horizon": th}]}
                out, _ = finalize_recommendations(raw)
                r = out.recommendations[0]
                assert (r.priority, r.confidence, r.time_horizon) == (p, c, th)


def test_empty_recommendations_valid():
    out, diag = finalize_recommendations({"recommendations": []})
    assert out.recommendations == []
    assert diag["failed_stage"] is None


def test_boundary_error_hierarchy():
    assert issubclass(RecommendationNormalizationError, RecommendationBoundaryError)
    assert issubclass(RecommendationValidationError, RecommendationBoundaryError)


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

_HYPS = [{"id": "H1", "title": "h", "summary": "s", "supporting_evidence": ["E1"], "confidence": "medium"}]


def _ctx() -> AgentContext:
    ctx = AgentContext(
        question="q", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"research_id": "R-RB", "contradictions": [], "validated_contradictions": []},
        run_id="rb001",
    )
    ctx.hypotheses = _HYPS
    ctx.surviving_hypotheses = [{"hypothesis_id": "H1", "survival_status": "strong", "reason": "ok"}]
    ctx.hypothesis_challenges = [{"hypothesis_id": "H1", "challenge_summary": "c", "robustness": "high"}]
    ctx.evidence_notes = [{"evidence_items": [{"evidence_id": "E1", "claim": "c", "source_document": "d"}],
                           "profile_coverage_by_profile": {}}]
    return ctx


def test_agent_records_boundary_and_produces_recommendations():
    ctx = _ctx()
    RecommendationAgent().run(ctx)  # no client → deterministic mock payload, valid
    assert ctx.recommendations
    boundary = ctx.trace["_recommendation_boundary"]
    assert boundary["failed_stage"] is None
    assert boundary["stages"]["validation"] == "ok"


class _BadRecClient:
    is_mock = False
    def generate_recommendations_raw(self, *a, **k):
        return {"recommendations": [{"id": "R1", "title": "t", "summary": "s", "priority": "urgent"}]}


def test_agent_deterministic_failure_on_bad_priority():
    ctx = _ctx()
    with pytest.raises(RecommendationBoundaryError):
        RecommendationAgent(client=_BadRecClient()).run(ctx)
    assert ctx.trace["_recommendation_boundary"]["failed_stage"] == "validation"
    assert not ctx.recommendations


class _GenBoomRecClient:
    is_mock = False
    def generate_recommendations_raw(self, *a, **k):
        raise RuntimeError("api down")


def test_generation_failure_classified():
    from functional_agents.recommendation_boundary import RecommendationGenerationError
    ctx = _ctx()
    with pytest.raises(RecommendationGenerationError):
        RecommendationAgent(client=_GenBoomRecClient()).run(ctx)
    assert ctx.trace["_recommendation_boundary"]["failed_stage"] == "generation"
