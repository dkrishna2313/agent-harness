"""Tests for J5.6a MultiProfileAgent.

Covers:
- build_evidence_profile_map() correctly maps evidence IDs to profiles
- attribute_findings() adds contributing_profiles to hypotheses
- attribute_recommendations() adds contributing_profiles to recommendations
- compute_profile_influence() counts per-profile evidence/findings/recs
- diagnose_missing_profiles() returns correct diagnostic entries
- build_multi_profile_analysis() assembles full analysis dict
- MultiProfileAgent._execute() writes all required context fields
- Single-profile context handled without errors
- Multi-profile context with missing profile produces diagnostics
- WorkflowState.MULTI_PROFILE exists
- AgentContext.multi_profile_analysis field exists
- Orchestrator accepts multi_profile_factory parameter
- Contract validator includes MultiProfileAgent
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from functional_agents.multi_profile_agent import (
    MultiProfileAgent,
    build_evidence_profile_map,
    attribute_findings,
    attribute_recommendations,
    compute_profile_influence,
    diagnose_missing_profiles,
    build_multi_profile_analysis,
)
from functional_agents.context import AgentContext, WorkflowState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ctx(
    profiles=None,
    execution_profile="ai_data_centers",
    evidence_items=None,
    hypotheses=None,
    recommendations=None,
    profile_coverage=None,
    profiles_contributing=None,
    profiles_missing=None,
) -> AgentContext:
    ctx = AgentContext(goal="test")
    ctx.profiles = profiles or ["ai_data_centers", "power", "transmission"]
    ctx.execution_profile = execution_profile
    ctx.research_object = {}

    note: dict = {
        "evidence_items": evidence_items or [],
        "profile_coverage_by_profile": profile_coverage or {},
        "profiles_contributing": profiles_contributing or [],
        "profiles_missing": profiles_missing or [],
    }
    ctx.evidence_notes = [note]
    ctx.hypotheses = hypotheses or []
    ctx.recommendations = recommendations or []
    return ctx


def _ev(eid, profile):
    return {"evidence_id": eid, "source_profile": profile, "claim": f"claim {eid}"}


def _hyp(hid, ev_ids=None):
    return {
        "hypothesis_id": hid,
        "statement": f"Hypothesis {hid}",
        "supporting_evidence": ev_ids or [],
    }


def _rec(rid, ev_ids=None, hyp_ids=None):
    return {
        "id": rid,
        "title": f"Recommendation {rid}",
        "supporting_evidence": ev_ids or [],
        "supported_by_hypotheses": hyp_ids or [],
    }


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_multi_profile_exists():
    assert WorkflowState.MULTI_PROFILE == "MULTI_PROFILE"


# ---------------------------------------------------------------------------
# AgentContext field
# ---------------------------------------------------------------------------

def test_context_multi_profile_analysis_default():
    ctx = AgentContext(goal="test")
    assert isinstance(ctx.multi_profile_analysis, dict)
    assert ctx.multi_profile_analysis == {}


# ---------------------------------------------------------------------------
# build_evidence_profile_map()
# ---------------------------------------------------------------------------

def test_ev_map_basic():
    ctx = _ctx(
        evidence_items=[_ev("E001", "power"), _ev("E002", "transmission")],
    )
    ev_map = build_evidence_profile_map(ctx)
    assert ev_map == {"E001": "power", "E002": "transmission"}


def test_ev_map_falls_back_to_execution_profile():
    ctx = _ctx(
        execution_profile="ai_data_centers",
        evidence_items=[{"evidence_id": "E003", "claim": "no profile field"}],
    )
    ev_map = build_evidence_profile_map(ctx)
    assert ev_map["E003"] == "ai_data_centers"


def test_ev_map_uses_profile_alias():
    ctx = _ctx(
        evidence_items=[{"evidence_id": "E004", "profile": "power", "claim": "x"}],
    )
    ev_map = build_evidence_profile_map(ctx)
    assert ev_map["E004"] == "power"


def test_ev_map_skips_empty_ids():
    ctx = _ctx(evidence_items=[{"evidence_id": "", "source_profile": "power"}])
    ev_map = build_evidence_profile_map(ctx)
    assert "" not in ev_map


def test_ev_map_empty_notes():
    ctx = AgentContext(goal="test")
    ctx.profiles = ["ai_data_centers"]
    ctx.execution_profile = "ai_data_centers"
    ctx.evidence_notes = []
    ev_map = build_evidence_profile_map(ctx)
    assert ev_map == {}


# ---------------------------------------------------------------------------
# attribute_findings()
# ---------------------------------------------------------------------------

def test_attribute_findings_single_profile():
    ev_map = {"E001": "power", "E002": "power"}
    hyps = [_hyp("H1", ["E001", "E002"])]
    result = attribute_findings(hyps, ev_map)
    assert result[0]["contributing_profiles"] == ["power"]


def test_attribute_findings_multi_profile():
    ev_map = {"E001": "power", "E002": "transmission"}
    hyps = [_hyp("H1", ["E001", "E002"])]
    result = attribute_findings(hyps, ev_map)
    assert set(result[0]["contributing_profiles"]) == {"power", "transmission"}


def test_attribute_findings_no_evidence():
    hyps = [_hyp("H1", [])]
    result = attribute_findings(hyps, {})
    assert result[0]["contributing_profiles"] == []


def test_attribute_findings_unknown_evidence_id():
    ev_map = {"E001": "power"}
    hyps = [_hyp("H1", ["E999"])]
    result = attribute_findings(hyps, ev_map)
    assert result[0]["contributing_profiles"] == []


def test_attribute_findings_preserves_original_fields():
    ev_map = {"E001": "power"}
    hyps = [{"hypothesis_id": "H1", "statement": "test", "supporting_evidence": ["E001"]}]
    result = attribute_findings(hyps, ev_map)
    assert result[0]["hypothesis_id"] == "H1"
    assert result[0]["statement"] == "test"


# ---------------------------------------------------------------------------
# attribute_recommendations()
# ---------------------------------------------------------------------------

def test_attribute_recs_from_evidence():
    ev_map = {"E001": "power"}
    recs = [_rec("R1", ev_ids=["E001"])]
    result = attribute_recommendations(recs, {}, ev_map)
    assert "power" in result[0]["contributing_profiles"]


def test_attribute_recs_from_hypotheses():
    hyp_map = {"H1": ["transmission", "ai_data_centers"]}
    recs = [_rec("R1", hyp_ids=["H1"])]
    result = attribute_recommendations(recs, hyp_map, {})
    assert set(result[0]["contributing_profiles"]) == {"transmission", "ai_data_centers"}


def test_attribute_recs_unions_both_sources():
    ev_map = {"E001": "power"}
    hyp_map = {"H1": ["transmission"]}
    recs = [_rec("R1", ev_ids=["E001"], hyp_ids=["H1"])]
    result = attribute_recommendations(recs, hyp_map, ev_map)
    assert set(result[0]["contributing_profiles"]) == {"power", "transmission"}


def test_attribute_recs_no_attribution():
    recs = [_rec("R1")]
    result = attribute_recommendations(recs, {}, {})
    assert result[0]["contributing_profiles"] == []


def test_attribute_recs_preserves_original_fields():
    recs = [{"id": "R1", "title": "test", "supporting_evidence": [], "supported_by_hypotheses": []}]
    result = attribute_recommendations(recs, {}, {})
    assert result[0]["id"] == "R1"
    assert result[0]["title"] == "test"


# ---------------------------------------------------------------------------
# compute_profile_influence()
# ---------------------------------------------------------------------------

def test_influence_counts_evidence():
    items = [_ev("E1", "power"), _ev("E2", "power"), _ev("E3", "transmission")]
    influence = compute_profile_influence(["power", "transmission"], items, [], [])
    assert influence["power"]["evidence"] == 2
    assert influence["transmission"]["evidence"] == 1


def test_influence_counts_findings():
    items = []
    findings = [
        {"contributing_profiles": ["power", "transmission"]},
        {"contributing_profiles": ["power"]},
    ]
    influence = compute_profile_influence(["power", "transmission"], items, findings, [])
    assert influence["power"]["findings"] == 2
    assert influence["transmission"]["findings"] == 1


def test_influence_counts_recommendations():
    recs = [{"contributing_profiles": ["ai_data_centers"]}]
    influence = compute_profile_influence(["ai_data_centers", "power"], [], [], recs)
    assert influence["ai_data_centers"]["recommendations"] == 1
    assert influence["power"]["recommendations"] == 0


def test_influence_unknown_profile_not_in_result():
    items = [_ev("E1", "unknown_profile")]
    influence = compute_profile_influence(["power"], items, [], [])
    assert "unknown_profile" not in influence


def test_influence_empty_inputs():
    influence = compute_profile_influence(["power"], [], [], [])
    assert influence["power"] == {"evidence": 0, "findings": 0, "recommendations": 0}


# ---------------------------------------------------------------------------
# diagnose_missing_profiles()
# ---------------------------------------------------------------------------

def test_diagnose_missing_returns_entry_for_missing():
    diagnostics = diagnose_missing_profiles(
        ["power", "transmission"],
        ["power"],
        {"transmission": {"evidence_count": 0}},
    )
    assert len(diagnostics) == 1
    assert diagnostics[0]["profile"] == "transmission"
    assert diagnostics[0]["status"] == "missing"
    assert "no evidence retrieved" in diagnostics[0]["reason"]


def test_diagnose_missing_no_missing():
    diagnostics = diagnose_missing_profiles(
        ["power", "transmission"],
        ["power", "transmission"],
        {},
    )
    assert diagnostics == []


def test_diagnose_missing_reason_for_nonzero_count():
    diagnostics = diagnose_missing_profiles(
        ["transmission"],
        [],
        {"transmission": {"evidence_count": 5}},
    )
    assert len(diagnostics) == 1
    assert "insufficient" in diagnostics[0]["reason"]


# ---------------------------------------------------------------------------
# build_multi_profile_analysis()
# ---------------------------------------------------------------------------

def test_build_analysis_has_all_keys():
    ctx = _ctx(
        evidence_items=[_ev("E1", "power")],
        hypotheses=[_hyp("H1", ["E1"])],
        recommendations=[_rec("R1", ["E1"], ["H1"])],
        profiles_contributing=["power"],
        profiles_missing=["transmission"],
    )
    analysis = build_multi_profile_analysis(ctx)
    for key in (
        "profiles_requested", "profiles_contributing", "profiles_missing",
        "coverage_status", "profile_coverage", "profile_influence",
        "missing_profile_diagnostics", "attributed_findings", "attributed_recommendations",
    ):
        assert key in analysis, f"Missing key: {key}"


def test_build_analysis_coverage_sufficient_when_all_contribute():
    ctx = _ctx(
        evidence_items=[
            _ev("E1", "ai_data_centers"), _ev("E2", "power"), _ev("E3", "transmission")
        ],
        profiles_contributing=["ai_data_centers", "power", "transmission"],
        profiles_missing=[],
    )
    analysis = build_multi_profile_analysis(ctx)
    assert analysis["coverage_status"] == "sufficient"


def test_build_analysis_coverage_insufficient_when_none_contribute():
    ctx = _ctx(
        evidence_items=[],
        profiles_contributing=[],
        profiles_missing=["ai_data_centers", "power", "transmission"],
    )
    analysis = build_multi_profile_analysis(ctx)
    assert analysis["coverage_status"] == "insufficient"


def test_build_analysis_attributed_findings_have_contributing_profiles():
    ctx = _ctx(
        evidence_items=[_ev("E1", "power")],
        hypotheses=[_hyp("H1", ["E1"])],
    )
    analysis = build_multi_profile_analysis(ctx)
    assert "contributing_profiles" in analysis["attributed_findings"][0]


def test_build_analysis_attributed_recs_have_contributing_profiles():
    ctx = _ctx(
        evidence_items=[_ev("E1", "power")],
        recommendations=[_rec("R1", ["E1"])],
    )
    analysis = build_multi_profile_analysis(ctx)
    assert "contributing_profiles" in analysis["attributed_recommendations"][0]


# ---------------------------------------------------------------------------
# MultiProfileAgent contract
# ---------------------------------------------------------------------------

def test_agent_writes_multi_profile_analysis():
    ctx = _ctx(
        evidence_items=[_ev("E1", "power"), _ev("E2", "transmission")],
        hypotheses=[_hyp("H1", ["E1"])],
        recommendations=[_rec("R1", ["E1", "E2"])],
        profiles_contributing=["power", "transmission"],
    )
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert isinstance(result_ctx.multi_profile_analysis, dict)
    assert "profiles_requested" in result_ctx.multi_profile_analysis


def test_agent_writes_qa_multi_profile_validation():
    ctx = _ctx(profiles_contributing=["power"])
    result_ctx = MultiProfileAgent()._execute(ctx)
    qa_block = result_ctx.qa.get("multi_profile_validation", {})
    assert "requested_profiles" in qa_block
    assert "contributing_profiles" in qa_block
    assert "coverage_status" in qa_block


def test_agent_writes_research_object():
    ctx = _ctx()
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert "multi_profile_analysis" in result_ctx.research_object


def test_agent_writes_trace():
    ctx = _ctx()
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert "_multi_profile" in result_ctx.trace
    trace_block = result_ctx.trace["_multi_profile"]
    assert "multi_profile_validation" in trace_block


def test_agent_enriches_hypotheses():
    ctx = _ctx(
        evidence_items=[_ev("E1", "power")],
        hypotheses=[_hyp("H1", ["E1"])],
    )
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert "contributing_profiles" in result_ctx.hypotheses[0]


def test_agent_enriches_recommendations():
    ctx = _ctx(
        evidence_items=[_ev("E1", "transmission")],
        recommendations=[_rec("R1", ["E1"])],
    )
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert "contributing_profiles" in result_ctx.recommendations[0]


def test_agent_single_profile_no_error():
    ctx = AgentContext(goal="test")
    ctx.profiles = ["ai_data_centers"]
    ctx.execution_profile = "ai_data_centers"
    ctx.research_object = {}
    ctx.evidence_notes = [{"evidence_items": [_ev("E1", "ai_data_centers")],
                           "profile_coverage_by_profile": {},
                           "profiles_contributing": ["ai_data_centers"],
                           "profiles_missing": []}]
    ctx.hypotheses = []
    ctx.recommendations = []
    result_ctx = MultiProfileAgent()._execute(ctx)
    assert result_ctx.multi_profile_analysis["coverage_status"] == "sufficient"


def test_agent_qa_validation_format():
    ctx = _ctx(profiles=["ai_data_centers", "power", "transmission"],
               profiles_contributing=["ai_data_centers", "power", "transmission"])
    result_ctx = MultiProfileAgent()._execute(ctx)
    qa = result_ctx.qa["multi_profile_validation"]
    assert qa["requested_profiles"] == 3
    assert qa["contributing_profiles"] == 3
    assert qa["coverage_status"] == "sufficient"


# ---------------------------------------------------------------------------
# Orchestrator wiring
# ---------------------------------------------------------------------------

def test_orchestrator_accepts_multi_profile_factory():
    from functional_agents.orchestrator import AgentOrchestrator
    dummy = lambda: MagicMock()
    orch = AgentOrchestrator(
        planner_factory=dummy,
        evidence_factory=dummy,
        qa_factory=dummy,
        report_factory=dummy,
        multi_profile_factory=dummy,
    )
    assert orch._multi_profile_factory is not None


# ---------------------------------------------------------------------------
# Contract validator
# ---------------------------------------------------------------------------

def test_contract_includes_multi_profile_agent():
    from functional_agents.contract import validate_all_classes
    results = validate_all_classes()
    assert "MultiProfileAgent" in results
    r = results["MultiProfileAgent"]
    assert r["inherits_base_agent"] is True
    assert r["implements_run"] is True
