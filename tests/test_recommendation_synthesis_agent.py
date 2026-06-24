"""J6.8c – RecommendationSynthesisAgent tests.

Covers:
  - Public helpers: compute_profile_rationale, build_integrated_recs,
    compute_synthesis_validation, compute_recommendation_profile_balance,
    build_synthesis_tradeoffs, synthesise_recommendations
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - Agent _execute(): writes to context, QA, trace, RO
  - Trace structure and coverage_status logic
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult
from functional_agents.recommendation_synthesis_agent import (
    RecommendationSynthesisAgent,
    build_integrated_recs,
    build_synthesis_tradeoffs,
    compute_profile_rationale,
    compute_recommendation_profile_balance,
    compute_synthesis_validation,
    synthesise_recommendations,
    _MIN_INTEGRATED,
    _INTEGRATED_REC_TEMPLATES,
    _TRADEOFF_TEMPLATES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROFILES = ["ai_data_centers", "transmission"]


def _ctx(
    recommendations=None,
    profiles=None,
    multi_profile_analysis=None,
) -> AgentContext:
    return AgentContext(
        question="test question",
        profiles=profiles or list(_PROFILES),
        execution_profile=(profiles or _PROFILES)[0],
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        recommendations=recommendations or [],
        multi_profile_analysis=multi_profile_analysis or {},
    )


def _rec(rec_id="R1", profiles=None):
    return {
        "id": rec_id,
        "title": f"Rec {rec_id}",
        "contributing_profiles": profiles or [],
    }


# ---------------------------------------------------------------------------
# compute_profile_rationale
# ---------------------------------------------------------------------------

def test_compute_profile_rationale_returns_dict():
    rec = _INTEGRATED_REC_TEMPLATES[0]
    result = compute_profile_rationale(rec, _PROFILES)
    assert isinstance(result, dict)


def test_compute_profile_rationale_keyed_by_profiles():
    rec = _INTEGRATED_REC_TEMPLATES[0]
    result = compute_profile_rationale(rec, _PROFILES)
    for p in _PROFILES:
        assert p in result


def test_compute_profile_rationale_values_are_strings():
    rec = _INTEGRATED_REC_TEMPLATES[0]
    result = compute_profile_rationale(rec, _PROFILES)
    for v in result.values():
        assert isinstance(v, str) and v


def test_compute_profile_rationale_known_template():
    rec = {"id": "SYNTH-R1"}
    result = compute_profile_rationale(rec, _PROFILES)
    assert "ai_data_centers" in result
    assert "transmission" in result
    # Both should have non-empty rationale strings
    assert len(result["ai_data_centers"]) > 10
    assert len(result["transmission"]) > 10


def test_compute_profile_rationale_unknown_rec_uses_generic():
    rec = {"id": "UNKNOWN-XYZ"}
    result = compute_profile_rationale(rec, _PROFILES)
    assert isinstance(result, dict)
    for p in _PROFILES:
        assert p in result


def test_compute_profile_rationale_single_profile():
    rec = {"id": "SYNTH-R2"}
    result = compute_profile_rationale(rec, ["ai_data_centers"])
    assert list(result.keys()) == ["ai_data_centers"]


# ---------------------------------------------------------------------------
# build_integrated_recs
# ---------------------------------------------------------------------------

def test_build_integrated_recs_count():
    recs = build_integrated_recs(_PROFILES, count=2)
    assert len(recs) == 2


def test_build_integrated_recs_all_have_both_profiles():
    recs = build_integrated_recs(_PROFILES, count=3)
    for r in recs:
        assert set(_PROFILES) == set(r["contributing_profiles"])


def test_build_integrated_recs_have_profile_rationale():
    recs = build_integrated_recs(_PROFILES, count=2)
    for r in recs:
        assert "profile_rationale" in r
        assert isinstance(r["profile_rationale"], dict)


def test_build_integrated_recs_synthesis_source_field():
    recs = build_integrated_recs(_PROFILES, count=2)
    for r in recs:
        assert r.get("synthesis_source") == "integrated"


def test_build_integrated_recs_have_ids():
    recs = build_integrated_recs(_PROFILES, count=2)
    for r in recs:
        assert r.get("id", "").startswith("SYNTH-R")


def test_build_integrated_recs_default_count():
    recs = build_integrated_recs(_PROFILES)
    assert len(recs) == _MIN_INTEGRATED


# ---------------------------------------------------------------------------
# compute_synthesis_validation
# ---------------------------------------------------------------------------

def test_compute_synthesis_validation_has_five_keys():
    sv = compute_synthesis_validation([], _PROFILES, _PROFILES)
    assert "profiles_requested" in sv
    assert "profiles_contributing" in sv
    assert "profiles_represented_in_recommendations" in sv
    assert "integrated_recommendation_count" in sv
    assert "coverage_status" in sv


def test_compute_synthesis_validation_counts_integrated():
    recs = [
        _rec("R1", _PROFILES),
        _rec("R2", _PROFILES),
        _rec("R3", ["ai_data_centers"]),
    ]
    sv = compute_synthesis_validation(recs, _PROFILES, _PROFILES)
    assert sv["integrated_recommendation_count"] == 2


def test_compute_synthesis_validation_coverage_status_sufficient():
    recs = [_rec("R1", _PROFILES)]
    sv = compute_synthesis_validation(recs, _PROFILES, _PROFILES)
    assert sv["coverage_status"] == "sufficient"


def test_compute_synthesis_validation_coverage_status_partial():
    sv = compute_synthesis_validation([], _PROFILES, ["ai_data_centers"])
    assert sv["coverage_status"] == "partial"


def test_compute_synthesis_validation_coverage_status_insufficient():
    sv = compute_synthesis_validation([], _PROFILES, [])
    assert sv["coverage_status"] == "insufficient"


def test_compute_synthesis_validation_profiles_represented():
    recs = [_rec("R1", _PROFILES), _rec("R2", ["ai_data_centers"])]
    sv = compute_synthesis_validation(recs, _PROFILES, _PROFILES)
    assert sv["profiles_represented_in_recommendations"] == 2


def test_compute_synthesis_validation_no_recs():
    sv = compute_synthesis_validation([], _PROFILES, _PROFILES)
    assert sv["integrated_recommendation_count"] == 0
    assert sv["profiles_represented_in_recommendations"] == 0


# ---------------------------------------------------------------------------
# compute_recommendation_profile_balance
# ---------------------------------------------------------------------------

def test_balance_sums_to_one():
    recs = [_rec("R1", _PROFILES), _rec("R2", ["ai_data_centers"]), _rec("R3", ["transmission"])]
    balance = compute_recommendation_profile_balance(recs, _PROFILES)
    assert abs(sum(balance.values()) - 1.0) < 0.01


def test_balance_keys_are_profiles():
    balance = compute_recommendation_profile_balance([], _PROFILES)
    assert set(balance.keys()) == set(_PROFILES)


def test_balance_values_between_zero_and_one():
    recs = [_rec("R1", _PROFILES), _rec("R2", ["ai_data_centers"])]
    balance = compute_recommendation_profile_balance(recs, _PROFILES)
    for v in balance.values():
        assert 0.0 <= v <= 1.0


def test_balance_equal_split_no_recs():
    balance = compute_recommendation_profile_balance([], _PROFILES)
    # equal split
    assert balance["ai_data_centers"] == balance["transmission"]


def test_balance_single_profile_recs():
    recs = [_rec("R1", ["ai_data_centers"]), _rec("R2", ["ai_data_centers"])]
    balance = compute_recommendation_profile_balance(recs, _PROFILES)
    assert balance["ai_data_centers"] == 1.0
    assert balance["transmission"] == 0.0


# ---------------------------------------------------------------------------
# build_synthesis_tradeoffs
# ---------------------------------------------------------------------------

def test_build_synthesis_tradeoffs_returns_list():
    t = build_synthesis_tradeoffs(_PROFILES)
    assert isinstance(t, list)


def test_build_synthesis_tradeoffs_three_for_ai_plus_transmission():
    t = build_synthesis_tradeoffs(_PROFILES)
    assert len(t) == 3


def test_build_synthesis_tradeoffs_have_required_keys():
    t = build_synthesis_tradeoffs(_PROFILES)
    for item in t:
        assert "tradeoff_id" in item
        assert "dimension_a" in item
        assert "dimension_b" in item
        assert "description" in item
        assert "implication" in item
        assert "profiles" in item


def test_build_synthesis_tradeoffs_ids_are_t1_t2_t3():
    t = build_synthesis_tradeoffs(_PROFILES)
    ids = {item["tradeoff_id"] for item in t}
    assert ids == {"T1", "T2", "T3"}


def test_build_synthesis_tradeoffs_empty_for_unknown_profile():
    t = build_synthesis_tradeoffs(["unknown_profile"])
    assert t == []


# ---------------------------------------------------------------------------
# synthesise_recommendations
# ---------------------------------------------------------------------------

def test_synthesise_injects_synthetic_recs_when_needed():
    ctx = _ctx()
    result = synthesise_recommendations(ctx)
    assert result["integrated_recommendation_count"] >= _MIN_INTEGRATED


def test_synthesise_no_injection_when_enough_integrated():
    recs = [_rec("R1", _PROFILES), _rec("R2", _PROFILES)]
    ctx = _ctx(recommendations=recs)
    result = synthesise_recommendations(ctx)
    assert len(result["synthetic_recommendations"]) == 0


def test_synthesise_enriched_recs_all_have_contributing_profiles():
    ctx = _ctx()
    result = synthesise_recommendations(ctx)
    for r in result["enriched_recommendations"]:
        assert isinstance(r.get("contributing_profiles"), list)


def test_synthesise_returns_all_required_keys():
    ctx = _ctx()
    result = synthesise_recommendations(ctx)
    for key in (
        "enriched_recommendations", "synthetic_recommendations",
        "synthesis_validation", "recommendation_profile_balance",
        "synthesis_tradeoffs", "recommendation_profile_audit",
        "integrated_recommendation_count",
    ):
        assert key in result, f"missing key: {key}"


def test_synthesise_audit_keyed_by_rec_id():
    recs = [_rec("R1", _PROFILES)]
    ctx = _ctx(recommendations=recs)
    result = synthesise_recommendations(ctx)
    audit = result["recommendation_profile_audit"]
    assert isinstance(audit, dict)
    for v in audit.values():
        assert isinstance(v, list)


# ---------------------------------------------------------------------------
# RecommendationSynthesisAgent – contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(RecommendationSynthesisAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert result.status == "success"


# ---------------------------------------------------------------------------
# RecommendationSynthesisAgent – context writes
# ---------------------------------------------------------------------------

def test_agent_writes_synthesis_validation_to_context():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    sv = result.context.synthesis_validation
    assert isinstance(sv, dict)
    assert "coverage_status" in sv


def test_agent_writes_recommendation_profile_balance_to_context():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    balance = result.context.recommendation_profile_balance
    assert isinstance(balance, dict)
    assert abs(sum(balance.values()) - 1.0) < 0.01


def test_agent_writes_synthesis_tradeoffs_to_context():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert isinstance(result.context.synthesis_tradeoffs, list)
    assert len(result.context.synthesis_tradeoffs) > 0


def test_agent_updates_recommendations_on_context():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert isinstance(result.context.recommendations, list)
    integrated = [
        r for r in result.context.recommendations
        if len(r.get("contributing_profiles", [])) >= 2
    ]
    assert len(integrated) >= _MIN_INTEGRATED


def test_agent_writes_synthesis_validation_to_qa():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert "synthesis_validation" in result.context.qa


def test_agent_writes_to_research_object():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    ro = result.context.research_object
    assert "synthesis_validation" in ro
    assert "recommendation_profile_balance" in ro
    assert "synthesis_tradeoffs" in ro
    assert "recommendation_profile_audit" in ro


# ---------------------------------------------------------------------------
# RecommendationSynthesisAgent – trace
# ---------------------------------------------------------------------------

def test_agent_writes_trace_block():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    assert "_recommendation_synthesis" in result.context.trace


def test_trace_has_agent_field():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    trace = result.context.trace["_recommendation_synthesis"]
    assert trace["agent"] == "RecommendationSynthesisAgent"


def test_trace_has_synthesis_validation():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    trace = result.context.trace["_recommendation_synthesis"]
    assert "synthesis_validation" in trace
    assert "coverage_status" in trace["synthesis_validation"]


def test_trace_has_integrated_recommendation_count():
    ctx = _ctx()
    result = RecommendationSynthesisAgent().run(ctx)
    trace = result.context.trace["_recommendation_synthesis"]
    assert "integrated_recommendation_count" in trace
    assert isinstance(trace["integrated_recommendation_count"], int)


def test_trace_coverage_status_sufficient_when_all_profiles_contributing():
    ctx = _ctx(
        multi_profile_analysis={"profiles_contributing": list(_PROFILES)},
    )
    result = RecommendationSynthesisAgent().run(ctx)
    sv = result.context.synthesis_validation
    assert sv["coverage_status"] == "sufficient"
