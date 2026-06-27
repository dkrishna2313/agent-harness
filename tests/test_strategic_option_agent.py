"""J7.5 – StrategicOptionAgent tests.

Covers:
  - WorkflowState.STRATEGIC_OPTIONS constant
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - No-op when assumptions absent
  - _execute(): writes strategic_options, preferred_option, RO, trace
  - Exactly one option has recommended=True
  - Trace fields match J7.5 spec
  - options reference J7 graph IDs
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.strategic_option_agent import StrategicOptionAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assumption(a_id="A-001", rec_ids=None):
    return {
        "assumption_id": a_id,
        "statement": f"Assumption {a_id} must hold",
        "importance": "Critical",
        "evidence_ids": ["EV-001"],
        "supported_recommendation_ids": rec_ids or ["REC-001"],
        "conflicts_with": [],
    }


def _risk(r_id="RSK-001", a_ids=None):
    return {
        "risk_id": r_id,
        "statement": f"Risk {r_id} could materialise",
        "severity": "High",
        "related_assumption_ids": a_ids or ["A-001"],
        "affected_recommendation_ids": ["REC-001"],
    }


def _opportunity(o_id="OPP-001", a_ids=None):
    return {
        "opportunity_id": o_id,
        "statement": f"Opportunity {o_id} becomes available",
        "impact": "High",
        "related_assumption_ids": a_ids or ["A-001"],
        "enabled_recommendation_ids": ["REC-001"],
    }


def _recommendation(rec_id="REC-001"):
    return {
        "recommendation_id": rec_id,
        "title": f"Recommendation {rec_id}",
        "supported_assumption_ids": ["A-001"],
    }


def _ctx(*, with_assumptions=True) -> AgentContext:
    assumptions = [_assumption("A-001"), _assumption("A-002"), _assumption("A-003")] if with_assumptions else []
    return AgentContext(
        question="Should we invest in SMR?",
        profiles=["SMR"],
        execution_profile="SMR",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        assumptions=assumptions,
        risks=[_risk("RSK-001"), _risk("RSK-002")],
        opportunities=[_opportunity("OPP-001"), _opportunity("OPP-002")],
        recommendations=[_recommendation("REC-001"), _recommendation("REC-002")],
    )


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_strategic_options_constant():
    assert WorkflowState.STRATEGIC_OPTIONS == "STRATEGIC_OPTIONS"


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(StrategicOptionAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = StrategicOptionAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = StrategicOptionAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# No-op when assumptions absent
# ---------------------------------------------------------------------------

def test_agent_skips_without_assumptions():
    result = StrategicOptionAgent().run(_ctx(with_assumptions=False))
    assert result.status == "skipped"


def test_agent_skip_writes_trace():
    result = StrategicOptionAgent().run(_ctx(with_assumptions=False))
    trace = result.context.trace.get("_strategic_options", {})
    assert trace.get("skipped") is True


def test_agent_skip_leaves_options_empty():
    result = StrategicOptionAgent().run(_ctx(with_assumptions=False))
    assert result.context.strategic_options == []


# ---------------------------------------------------------------------------
# Context writes
# ---------------------------------------------------------------------------

def test_agent_writes_strategic_options_list():
    result = StrategicOptionAgent().run(_ctx())
    assert isinstance(result.context.strategic_options, list)
    assert len(result.context.strategic_options) >= 3


def test_agent_writes_preferred_option():
    result = StrategicOptionAgent().run(_ctx())
    pref = result.context.preferred_option
    assert isinstance(pref, dict)
    assert "option_id" in pref


def test_exactly_one_recommended_option():
    result = StrategicOptionAgent().run(_ctx())
    recommended = [o for o in result.context.strategic_options if o.get("recommended")]
    assert len(recommended) == 1


def test_recommended_option_matches_preferred_option():
    result = StrategicOptionAgent().run(_ctx())
    pref_id = result.context.preferred_option.get("option_id")
    recommended = [o for o in result.context.strategic_options if o.get("recommended")]
    assert recommended[0]["option_id"] == pref_id


# ---------------------------------------------------------------------------
# Option schema fields
# ---------------------------------------------------------------------------

def test_options_have_required_fields():
    required = {
        "option_id", "title", "description", "strategic_objective",
        "expected_outcomes", "supporting_assumption_ids",
        "associated_risk_ids", "associated_opportunity_ids",
        "supporting_recommendation_ids", "advantages", "disadvantages",
        "implementation_complexity", "estimated_time_horizon",
        "capital_intensity", "confidence", "recommended", "rationale",
    }
    result = StrategicOptionAgent().run(_ctx())
    for opt in result.context.strategic_options:
        missing = required - set(opt.keys())
        assert not missing, f"{opt.get('option_id')} missing: {missing}"


def test_options_have_distinct_ids():
    result = StrategicOptionAgent().run(_ctx())
    ids = [o["option_id"] for o in result.context.strategic_options]
    assert len(ids) == len(set(ids))


def test_options_have_non_empty_titles():
    result = StrategicOptionAgent().run(_ctx())
    for opt in result.context.strategic_options:
        assert opt.get("title"), f"{opt.get('option_id')} has empty title"


def test_options_reference_assumptions():
    result = StrategicOptionAgent().run(_ctx())
    for opt in result.context.strategic_options:
        assert isinstance(opt.get("supporting_assumption_ids"), list)


# ---------------------------------------------------------------------------
# Research Object writes
# ---------------------------------------------------------------------------

def test_agent_writes_strategic_options_to_ro():
    result = StrategicOptionAgent().run(_ctx())
    ro = result.context.research_object
    assert "strategic_options" in ro
    assert len(ro["strategic_options"]) >= 3


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------

def test_agent_writes_trace_block():
    result = StrategicOptionAgent().run(_ctx())
    assert "_strategic_options" in result.context.trace


def test_trace_has_option_count():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "option_count" in trace
    assert trace["option_count"] >= 3


def test_trace_has_recommended_option():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "recommended_option" in trace
    assert trace["recommended_option"] is not None


def test_trace_has_average_metrics():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "average_assumptions_per_option" in trace
    assert "average_risks_per_option" in trace
    assert "average_opportunities_per_option" in trace


def test_trace_has_persistence_flags():
    result = StrategicOptionAgent().run(_ctx())
    trace = result.context.trace["_strategic_options"]
    assert "dm_persisted" in trace
    assert "ro_persisted" in trace
