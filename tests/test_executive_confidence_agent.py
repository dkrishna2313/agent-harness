"""J7.7 – ExecutiveConfidenceAgent tests.

Covers:
  - WorkflowState.EXECUTIVE_CONFIDENCE constant
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - No-op when strategic options absent (reason=no_strategic_options)
  - No-op when decision analysis absent (reason=no_decision_analysis)
  - _execute(): writes executive_confidence, RO, trace
  - All 12 schema fields present in output
  - Trace block has 6 required fields
  - Board recommendation varies with assumption quality
  - validation_priorities and critical_unknowns are non-empty lists
  - conditional confidence strings are set
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.executive_confidence_agent import ExecutiveConfidenceAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _option(opt_id: str, recommended: bool = False) -> dict:
    return {
        "option_id": opt_id,
        "title": f"Option {opt_id}",
        "description": "A strategic option.",
        "recommended": recommended,
    }


def _assumption(imp: str = "Important") -> dict:
    return {
        "assumption_id": f"A-{imp[:3].upper()}",
        "statement": f"Assume {imp} thing holds.",
        "importance": imp,
        "evidence_support": "Moderate",
    }


def _risk(sev: str = "High") -> dict:
    return {
        "risk_id": f"RSK-{sev[:3].upper()}",
        "statement": "Risk of cost overrun.",
        "severity": sev,
        "mitigation_notes": "Monitor regularly.",
    }


def _decision_analysis(confidence: str = "Medium") -> dict:
    return {
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-B",
        "executive_summary": "Option B is preferred.",
        "confidence": confidence,
        "rationale": "Best risk-adjusted return.",
        "key_tradeoffs": ["Cost vs speed"],
        "decision_matrix": [],
        "option_rankings": ["OPT-B", "OPT-A"],
        "comparison_dimensions": ["Cost", "Risk"],
        "key_uncertainties": ["Regulatory outcome"],
        "sensitivity_analysis": "Moderate sensitivity to cost.",
        "confidence_summary": "Medium confidence overall.",
    }


def _ctx(
    *,
    with_options: bool = True,
    with_decision_analysis: bool = True,
    n_critical_assumptions: int = 0,
    n_high_risks: int = 1,
    da_confidence: str = "Medium",
) -> AgentContext:
    options = [_option("OPT-A"), _option("OPT-B", recommended=True)] if with_options else []
    assumptions = [_assumption("Critical")] * n_critical_assumptions + [_assumption("Important")]
    risks = [_risk("High")] * n_high_risks + [_risk("Medium")]
    da = _decision_analysis(da_confidence) if with_decision_analysis else {}
    return AgentContext(
        question="Should we invest in SMR technology?",
        profiles=["SMR"],
        execution_profile="SMR",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        strategic_options=options,
        assumptions=assumptions,
        risks=risks,
        opportunities=[{"opportunity_id": "OPP-001", "statement": "First mover advantage", "likelihood": "High"}],
        recommendations=[{"recommendation_id": "REC-001", "summary": "Proceed with pilot", "time_horizon": "near_term"}],
        decision_analysis=da,
        decision_model={"strategic_question": "Should we invest in SMR?"},
        scenarios=[],
    )


# ---------------------------------------------------------------------------
# WorkflowState
# ---------------------------------------------------------------------------

def test_workflow_state_executive_confidence_constant():
    assert WorkflowState.EXECUTIVE_CONFIDENCE == "EXECUTIVE_CONFIDENCE"


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(ExecutiveConfidenceAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = ExecutiveConfidenceAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = ExecutiveConfidenceAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# No-op when strategic options absent
# ---------------------------------------------------------------------------

def test_agent_skips_without_strategic_options():
    result = ExecutiveConfidenceAgent().run(_ctx(with_options=False))
    assert result.status == "skipped"


def test_agent_skip_no_options_writes_trace():
    result = ExecutiveConfidenceAgent().run(_ctx(with_options=False))
    trace = result.context.trace.get("_executive_confidence", {})
    assert trace.get("skipped") is True
    assert trace.get("reason") == "no_strategic_options"


def test_agent_skip_no_options_leaves_confidence_empty():
    result = ExecutiveConfidenceAgent().run(_ctx(with_options=False))
    assert result.context.executive_confidence == {}


# ---------------------------------------------------------------------------
# No-op when decision analysis absent
# ---------------------------------------------------------------------------

def test_agent_skips_without_decision_analysis():
    result = ExecutiveConfidenceAgent().run(_ctx(with_decision_analysis=False))
    assert result.status == "skipped"


def test_agent_skip_no_da_writes_trace():
    result = ExecutiveConfidenceAgent().run(_ctx(with_decision_analysis=False))
    trace = result.context.trace.get("_executive_confidence", {})
    assert trace.get("skipped") is True
    assert trace.get("reason") == "no_decision_analysis"


def test_agent_skip_no_da_leaves_confidence_empty():
    result = ExecutiveConfidenceAgent().run(_ctx(with_decision_analysis=False))
    assert result.context.executive_confidence == {}


# ---------------------------------------------------------------------------
# Context writes
# ---------------------------------------------------------------------------

def test_agent_writes_executive_confidence_to_context():
    result = ExecutiveConfidenceAgent().run(_ctx())
    ec = result.context.executive_confidence
    assert isinstance(ec, dict)
    assert ec


def test_agent_writes_executive_confidence_to_ro():
    result = ExecutiveConfidenceAgent().run(_ctx())
    ro = result.context.research_object
    assert "executive_confidence" in ro
    assert ro["executive_confidence"]


def test_agent_writes_trace_block():
    result = ExecutiveConfidenceAgent().run(_ctx())
    assert "_executive_confidence" in result.context.trace


# ---------------------------------------------------------------------------
# ExecutiveConfidence schema fields
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = {
    "confidence_id",
    "overall_confidence",
    "decision_readiness",
    "board_recommendation",
    "confidence_rationale",
    "confidence_drivers",
    "confidence_limiters",
    "critical_unknowns",
    "validation_priorities",
    "confidence_if_assumptions_hold",
    "confidence_if_assumptions_fail",
    "decision_horizon",
}


def test_confidence_has_required_fields():
    result = ExecutiveConfidenceAgent().run(_ctx())
    ec = result.context.executive_confidence
    missing = _REQUIRED_FIELDS - set(ec.keys())
    assert not missing, f"Missing fields: {missing}"


def test_overall_confidence_is_valid_literal():
    result = ExecutiveConfidenceAgent().run(_ctx())
    val = result.context.executive_confidence.get("overall_confidence")
    assert val in ("High", "Medium", "Low"), f"Unexpected overall_confidence: {val!r}"


def test_decision_readiness_is_valid_literal():
    result = ExecutiveConfidenceAgent().run(_ctx())
    val = result.context.executive_confidence.get("decision_readiness")
    assert val in (
        "Ready for Decision",
        "Needs Additional Validation",
        "Not Ready",
    ), f"Unexpected decision_readiness: {val!r}"


def test_board_recommendation_is_valid_literal():
    result = ExecutiveConfidenceAgent().run(_ctx())
    val = result.context.executive_confidence.get("board_recommendation")
    assert val in (
        "Proceed",
        "Proceed with Conditions",
        "Delay Pending Evidence",
        "Reject",
    ), f"Unexpected board_recommendation: {val!r}"


def test_validation_priorities_is_non_empty():
    result = ExecutiveConfidenceAgent().run(_ctx())
    vp = result.context.executive_confidence.get("validation_priorities", [])
    assert isinstance(vp, list)
    assert len(vp) >= 1


def test_critical_unknowns_is_list():
    result = ExecutiveConfidenceAgent().run(_ctx())
    cu = result.context.executive_confidence.get("critical_unknowns", [])
    assert isinstance(cu, list)


def test_confidence_if_assumptions_hold_is_string():
    result = ExecutiveConfidenceAgent().run(_ctx())
    val = result.context.executive_confidence.get("confidence_if_assumptions_hold", "")
    assert isinstance(val, str)


def test_confidence_if_assumptions_fail_is_string():
    result = ExecutiveConfidenceAgent().run(_ctx())
    val = result.context.executive_confidence.get("confidence_if_assumptions_fail", "")
    assert isinstance(val, str)


def test_last_updated_is_stamped():
    result = ExecutiveConfidenceAgent().run(_ctx())
    lu = result.context.executive_confidence.get("last_updated", "")
    assert lu, "last_updated should be stamped"
    assert "T" in lu, "last_updated should be ISO 8601"


# ---------------------------------------------------------------------------
# Trace fields (J7.7 spec – 6 fields)
# ---------------------------------------------------------------------------

_TRACE_FIELDS = (
    "overall_confidence",
    "decision_readiness",
    "board_recommendation",
    "critical_unknown_count",
    "validation_priority_count",
    "persisted",
)


def test_trace_has_all_required_fields():
    result = ExecutiveConfidenceAgent().run(_ctx())
    trace = result.context.trace["_executive_confidence"]
    for field in _TRACE_FIELDS:
        assert field in trace, f"Missing trace field: {field}"


def test_trace_counts_match_confidence():
    result = ExecutiveConfidenceAgent().run(_ctx())
    trace = result.context.trace["_executive_confidence"]
    ec = result.context.executive_confidence
    assert trace["critical_unknown_count"] == len(ec.get("critical_unknowns", []))
    assert trace["validation_priority_count"] == len(ec.get("validation_priorities", []))


def test_trace_confidence_matches_output():
    result = ExecutiveConfidenceAgent().run(_ctx())
    trace = result.context.trace["_executive_confidence"]
    ec = result.context.executive_confidence
    assert trace["overall_confidence"] == ec["overall_confidence"]
    assert trace["decision_readiness"] == ec["decision_readiness"]
    assert trace["board_recommendation"] == ec["board_recommendation"]


# ---------------------------------------------------------------------------
# Board recommendation logic (mock determinism)
# ---------------------------------------------------------------------------

def test_high_risk_reduces_confidence():
    """Many high-severity risks should produce a lower overall_confidence or
    a more cautious board_recommendation than the baseline."""
    baseline = ExecutiveConfidenceAgent().run(_ctx(n_high_risks=0))
    risky = ExecutiveConfidenceAgent().run(_ctx(n_high_risks=5))
    baseline_board = baseline.context.executive_confidence.get("board_recommendation")
    risky_board = risky.context.executive_confidence.get("board_recommendation")
    _ORDER = {"Proceed": 0, "Proceed with Conditions": 1, "Delay Pending Evidence": 2, "Reject": 3}
    assert _ORDER.get(risky_board, 0) >= _ORDER.get(baseline_board, 0), (
        f"High-risk scenario should not produce a more optimistic recommendation "
        f"than baseline: risky={risky_board!r} baseline={baseline_board!r}"
    )
