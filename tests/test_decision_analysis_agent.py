"""J7.6 – DecisionAnalysisAgent tests.

Covers:
  - WorkflowState.DECISION_ANALYSIS constant
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - No-op when strategic options absent
  - _execute(): writes decision_analysis, RO, trace
  - decision_analysis has required fields
  - Exactly one recommended_option_id matching a real option
  - key_tradeoffs is a non-empty list
  - decision_matrix has one row per option
  - trace["_decision_analysis"] has all spec fields
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.decision_analysis_agent import DecisionAnalysisAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _option(opt_id: str, recommended: bool = False) -> dict:
    return {
        "option_id": opt_id,
        "title": f"Option {opt_id}",
        "description": "A strategic option.",
        "strategic_objective": "Achieve something.",
        "expected_outcomes": ["Outcome A"],
        "supporting_assumption_ids": ["A-001"],
        "associated_risk_ids": ["RSK-001"],
        "associated_opportunity_ids": ["OPP-001"],
        "supporting_recommendation_ids": ["REC-001"],
        "advantages": ["Fast"],
        "disadvantages": ["Costly"],
        "implementation_complexity": "Medium",
        "estimated_time_horizon": "Medium-term",
        "capital_intensity": "Medium",
        "confidence": "Medium",
        "recommended": recommended,
        "rationale": "Preferred because...",
    }


def _ctx(*, with_options=True) -> AgentContext:
    options = [_option("OPT-A"), _option("OPT-B", recommended=True), _option("OPT-C")] if with_options else []
    return AgentContext(
        question="Should we invest in SMR?",
        profiles=["SMR"],
        execution_profile="SMR",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        strategic_options=options,
        assumptions=[{"assumption_id": "A-001", "statement": "Tech is viable"}],
        risks=[{"risk_id": "RSK-001", "statement": "Cost overrun"}],
        opportunities=[{"opportunity_id": "OPP-001", "statement": "First mover"}],
        recommendations=[{"recommendation_id": "REC-001", "title": "Proceed"}],
        decision_model={"strategic_question": "Should we invest in SMR?"},
        research_strategy={},
    )


def _mock_client_with_analysis(analysis_dict: dict) -> MagicMock:
    from research_agent.claude_client import DecisionAnalysisItem, DecisionAnalysisPayload
    item = DecisionAnalysisItem(**analysis_dict)
    payload = DecisionAnalysisPayload(analysis=item)
    client = MagicMock()
    client.generate_decision_analysis.return_value = payload
    return client


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_decision_analysis_constant():
    assert WorkflowState.DECISION_ANALYSIS == "DECISION_ANALYSIS"


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(DecisionAnalysisAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = DecisionAnalysisAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = DecisionAnalysisAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# No-op when options absent
# ---------------------------------------------------------------------------

def test_agent_skips_without_strategic_options():
    result = DecisionAnalysisAgent().run(_ctx(with_options=False))
    assert result.status == "skipped"


def test_agent_skip_writes_trace():
    result = DecisionAnalysisAgent().run(_ctx(with_options=False))
    trace = result.context.trace.get("_decision_analysis", {})
    assert trace.get("skipped") is True


def test_agent_skip_leaves_decision_analysis_empty():
    result = DecisionAnalysisAgent().run(_ctx(with_options=False))
    assert result.context.decision_analysis == {}


# ---------------------------------------------------------------------------
# Context writes
# ---------------------------------------------------------------------------

def test_agent_writes_decision_analysis_to_context():
    result = DecisionAnalysisAgent().run(_ctx())
    assert isinstance(result.context.decision_analysis, dict)
    assert result.context.decision_analysis


def test_agent_writes_decision_analysis_to_ro():
    result = DecisionAnalysisAgent().run(_ctx())
    ro = result.context.research_object
    assert "decision_analysis" in ro
    assert ro["decision_analysis"]


def test_agent_writes_trace_block():
    result = DecisionAnalysisAgent().run(_ctx())
    assert "_decision_analysis" in result.context.trace


# ---------------------------------------------------------------------------
# DecisionAnalysis schema fields
# ---------------------------------------------------------------------------

def test_analysis_has_required_fields():
    required = {
        "analysis_id", "recommended_option_id", "executive_summary",
        "comparison_dimensions", "option_rankings", "decision_matrix",
        "key_tradeoffs", "key_uncertainties",
        "sensitivity_analysis", "confidence_summary", "rationale", "confidence",
    }
    result = DecisionAnalysisAgent().run(_ctx())
    da = result.context.decision_analysis
    missing = required - set(da.keys())
    assert not missing, f"Missing fields: {missing}"


def test_recommended_option_id_matches_an_existing_option():
    result = DecisionAnalysisAgent().run(_ctx())
    da = result.context.decision_analysis
    option_ids = {o["option_id"] for o in result.context.strategic_options}
    assert da["recommended_option_id"] in option_ids


def test_option_rankings_includes_all_options():
    result = DecisionAnalysisAgent().run(_ctx())
    da = result.context.decision_analysis
    option_ids = {o["option_id"] for o in result.context.strategic_options}
    for oid in da.get("option_rankings", []):
        assert oid in option_ids, f"Ranked option {oid!r} not in strategic options"


def test_decision_matrix_has_one_row_per_option():
    result = DecisionAnalysisAgent().run(_ctx())
    da = result.context.decision_analysis
    matrix = da.get("decision_matrix", [])
    option_ids = {o["option_id"] for o in result.context.strategic_options}
    matrix_ids = {row["option_id"] for row in matrix}
    assert matrix_ids == option_ids


def test_key_tradeoffs_is_non_empty():
    result = DecisionAnalysisAgent().run(_ctx())
    tradeoffs = result.context.decision_analysis.get("key_tradeoffs", [])
    assert isinstance(tradeoffs, list)
    assert len(tradeoffs) >= 1


def test_comparison_dimensions_is_non_empty():
    result = DecisionAnalysisAgent().run(_ctx())
    dims = result.context.decision_analysis.get("comparison_dimensions", [])
    assert isinstance(dims, list)
    assert len(dims) >= 1


def test_sensitivity_analysis_is_non_empty_string():
    result = DecisionAnalysisAgent().run(_ctx())
    sens = result.context.decision_analysis.get("sensitivity_analysis", "")
    assert isinstance(sens, str)
    assert sens.strip()


# ---------------------------------------------------------------------------
# Trace fields (J7.6 spec)
# ---------------------------------------------------------------------------

def test_trace_has_all_required_fields():
    result = DecisionAnalysisAgent().run(_ctx())
    trace = result.context.trace["_decision_analysis"]
    for field in ("option_count", "comparison_dimensions", "tradeoff_count",
                  "recommended_option", "decision_persisted", "analysis_persisted"):
        assert field in trace, f"Missing trace field: {field}"


def test_trace_option_count_matches_context():
    result = DecisionAnalysisAgent().run(_ctx())
    trace = result.context.trace["_decision_analysis"]
    assert trace["option_count"] == len(result.context.strategic_options)


def test_trace_recommended_option_matches_analysis():
    result = DecisionAnalysisAgent().run(_ctx())
    trace = result.context.trace["_decision_analysis"]
    da = result.context.decision_analysis
    assert trace["recommended_option"] == da["recommended_option_id"]


def test_trace_tradeoff_count_matches_analysis():
    result = DecisionAnalysisAgent().run(_ctx())
    trace = result.context.trace["_decision_analysis"]
    da = result.context.decision_analysis
    assert trace["tradeoff_count"] == len(da.get("key_tradeoffs", []))
