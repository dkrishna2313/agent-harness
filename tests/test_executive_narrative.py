"""Tests for the J12.0 Executive Narrative Layer.

Covers: ExecutiveNarrative construction and serialisation, ExecutiveNarrativeBuilder
extraction logic, AgentContext field, trace capture, no-reasoning-mutation
invariant, graceful degradation, and AST constraint verification.
"""

from __future__ import annotations

import ast
import inspect
import json

import pytest

from functional_agents.context import AgentContext
from functional_agents.narrative import ExecutiveNarrative, ExecutiveNarrativeBuilder
from functional_agents.pipeline_trace import build_canonical_trace


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="Should we invest?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={
            "research_id": "R-EN-001",
            "decision_architecture": {"decision_statement": "Invest now or defer?"},
        },
        decision_architecture={"decision_statement": "Invest now or defer?"},
        strategic_synthesis={"executive_summary": "AI investment is strategic and timely."},
        decision_analysis={
            "recommended_option_id": "OPT-001",
            "rationale": "OPT-001 dominates on strategic fit with manageable risk.",
            "comparison_dimensions": ["Strategic Fit", "Execution Risk", "Capital Efficiency"],
            "option_rankings": ["OPT-001", "OPT-002"],
        },
        strategic_options=[
            {
                "option_id": "OPT-001", "title": "Full Build",
                "description": "Build end-to-end AI platform.",
                "estimated_time_horizon": "near_term",
                "capital_intensity": "high", "confidence": "high",
            },
            {
                "option_id": "OPT-002", "title": "Partnership",
                "description": "Partner with hyperscaler.",
                "estimated_time_horizon": "medium_term",
                "capital_intensity": "moderate", "confidence": "medium",
            },
        ],
        preferred_option={"option_id": "OPT-001", "title": "Full Build"},
        risks=[
            {"risk_id": "R-001", "statement": "Cost overruns likely.",
             "severity": "high", "likelihood": "medium"},
            {"risk_id": "R-002", "statement": "Talent shortage.",
             "severity": "critical", "likelihood": "high"},
        ],
        assumptions=[
            {"assumption_id": "A-001", "statement": "Demand grows 30% YoY.",
             "importance": "critical", "confidence": "high"},
            {"assumption_id": "A-002", "statement": "Regulatory approval secured.",
             "importance": "important", "confidence": "medium"},
        ],
        opportunities=[
            {"opportunity_id": "OPP-001", "title": "First-mover advantage",
             "impact": "high", "description": "Lead the market."},
        ],
        executive_confidence={
            "overall_confidence": "high",
            "decision_readiness": "ready",
            "board_recommendation": "proceed",
            "confidence_rationale": "Strong fundamentals.",
            "validation_priorities": ["Validate cost model", "Confirm talent pipeline"],
        },
        recommendations=[
            {"id": "REC-001", "recommendation_id": "REC-001", "title": "Approve Phase 1 budget"},
            {"id": "REC-002", "recommendation_id": "REC-002", "title": "Hire AI lead"},
        ],
        recommendation_portfolio={
            "near_term": ["REC-001", "REC-002"],
            "medium_term": [],
            "long_term": [],
        },
    )


def _empty_ctx() -> AgentContext:
    return AgentContext(
        question="Q?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R-EMPTY"},
    )


# ---------------------------------------------------------------------------
# ExecutiveNarrative — construction and defaults
# ---------------------------------------------------------------------------

def test_executive_narrative_default_fields():
    en = ExecutiveNarrative()
    assert en.decision == ""
    assert en.executive_summary == ""
    assert en.recommended_option == {}
    assert en.why_this_option == ""
    assert en.key_tradeoffs == []
    assert en.key_risks == []
    assert en.key_opportunities == []
    assert en.critical_assumptions == []
    assert en.executive_confidence == {}
    assert en.immediate_actions == []
    assert en.validation_priorities == []


def test_executive_narrative_to_dict_includes_all_eleven_fields():
    en = ExecutiveNarrative()
    d = en.to_dict()
    expected_keys = {
        "decision", "executive_summary", "recommended_option", "why_this_option",
        "key_tradeoffs", "key_risks", "key_opportunities", "critical_assumptions",
        "executive_confidence", "immediate_actions", "validation_priorities",
    }
    assert set(d.keys()) == expected_keys


def test_executive_narrative_to_dict_is_json_serialisable():
    en = ExecutiveNarrative(
        decision="Invest now.",
        key_risks=[{"risk_id": "R-1", "statement": "Cost risk.", "severity": "high"}],
    )
    json.dumps(en.to_dict())


def test_executive_narrative_from_dict_round_trips():
    original = ExecutiveNarrative(
        decision="Invest now.",
        executive_summary="Strong fundamentals.",
        recommended_option={"option_id": "OPT-001", "title": "Full Build"},
        why_this_option="Dominates on strategic fit.",
        key_tradeoffs=["Cost vs Speed"],
        key_risks=[{"risk_id": "R-1", "severity": "high"}],
        validation_priorities=["Validate cost model"],
    )
    restored = ExecutiveNarrative.from_dict(original.to_dict())
    assert restored.decision == original.decision
    assert restored.executive_summary == original.executive_summary
    assert restored.recommended_option == original.recommended_option
    assert restored.key_tradeoffs == original.key_tradeoffs
    assert restored.validation_priorities == original.validation_priorities


def test_executive_narrative_from_dict_handles_none():
    en = ExecutiveNarrative.from_dict(None)
    assert en == ExecutiveNarrative()


def test_executive_narrative_from_dict_handles_empty_dict():
    en = ExecutiveNarrative.from_dict({})
    assert en == ExecutiveNarrative()


# ---------------------------------------------------------------------------
# ExecutiveNarrativeBuilder — extraction correctness
# ---------------------------------------------------------------------------

def test_builder_extracts_decision_from_decision_architecture():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.decision == "Invest now or defer?"


def test_builder_extracts_executive_summary_from_strategic_synthesis():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.executive_summary == "AI investment is strategic and timely."


def test_builder_falls_back_to_decision_analysis_executive_summary():
    ctx = _full_ctx()
    ctx.strategic_synthesis = {}
    ctx.decision_analysis["executive_summary"] = "Fallback summary."
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.executive_summary == "Fallback summary."


def test_builder_extracts_recommended_option_with_real_schema_fields():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    rec = narrative.recommended_option
    assert rec["option_id"] == "OPT-001"
    assert rec["title"] == "Full Build"
    assert rec["description"] == "Build end-to-end AI platform."
    assert "estimated_time_horizon" in rec
    assert "capital_intensity" in rec
    assert "confidence" in rec


def test_builder_extracts_why_this_option_from_decision_analysis():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert "strategic fit" in narrative.why_this_option


def test_builder_extracts_key_tradeoffs_from_comparison_dimensions():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert "Strategic Fit" in narrative.key_tradeoffs
    assert "Execution Risk" in narrative.key_tradeoffs


def test_builder_prefers_strategic_synthesis_key_tradeoffs():
    ctx = _full_ctx()
    ctx.strategic_synthesis["key_tradeoffs"] = ["Speed vs Capital"]
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.key_tradeoffs == ["Speed vs Capital"]


def test_builder_extracts_key_risks_sorted_by_severity():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.key_risks) == 2
    assert narrative.key_risks[0]["risk_id"] == "R-002"  # critical before high
    assert narrative.key_risks[1]["risk_id"] == "R-001"


def test_builder_extracts_key_risks_with_correct_fields():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    for r in narrative.key_risks:
        assert "risk_id" in r
        assert "statement" in r
        assert "severity" in r
        assert "likelihood" in r


def test_builder_extracts_key_risks_top_5_max():
    ctx = _full_ctx()
    ctx.risks = [
        {"risk_id": f"R-{i}", "statement": f"Risk {i}.", "severity": "medium",
         "likelihood": "medium"}
        for i in range(10)
    ]
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.key_risks) == 5


def test_builder_extracts_key_opportunities():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.key_opportunities) == 1
    assert narrative.key_opportunities[0]["opportunity_id"] == "OPP-001"
    assert narrative.key_opportunities[0]["title"] == "First-mover advantage"


def test_builder_extracts_critical_assumptions_sorted_by_importance():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.critical_assumptions[0]["assumption_id"] == "A-001"  # critical
    assert narrative.critical_assumptions[1]["assumption_id"] == "A-002"  # important


def test_builder_extracts_executive_confidence_summary():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    ec = narrative.executive_confidence
    assert ec["overall_confidence"] == "high"
    assert ec["decision_readiness"] == "ready"
    assert ec["board_recommendation"] == "proceed"
    assert "confidence_rationale" in ec


def test_builder_extracts_immediate_actions_from_near_term_portfolio():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.immediate_actions) == 2
    action_ids = [a["id"] for a in narrative.immediate_actions]
    assert "REC-001" in action_ids
    assert "REC-002" in action_ids


def test_builder_extracts_validation_priorities():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert "Validate cost model" in narrative.validation_priorities
    assert "Confirm talent pipeline" in narrative.validation_priorities


# ---------------------------------------------------------------------------
# AgentContext population
# ---------------------------------------------------------------------------

def test_builder_sets_context_executive_narrative():
    ctx = _full_ctx()
    ExecutiveNarrativeBuilder().build(ctx)
    assert ctx.executive_narrative
    assert isinstance(ctx.executive_narrative, dict)


def test_builder_context_narrative_matches_returned_narrative():
    ctx = _full_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert ctx.executive_narrative == narrative.to_dict()


def test_agentcontext_has_executive_narrative_field():
    ctx = AgentContext(
        question="Q?", profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R"},
    )
    assert hasattr(ctx, "executive_narrative")
    assert ctx.executive_narrative == {}


def test_context_remains_json_serialisable_after_build():
    import dataclasses
    ctx = _full_ctx()
    ExecutiveNarrativeBuilder().build(ctx)
    json.dumps(dataclasses.asdict(ctx), default=str)


# ---------------------------------------------------------------------------
# No mutation of Strategic Reasoning Graph
# ---------------------------------------------------------------------------

def test_builder_does_not_extend_agent_history():
    ctx = _full_ctx()
    ctx.agent_history.append({"agent": "FakeAgent", "status": "success"})
    before = len(ctx.agent_history)
    ExecutiveNarrativeBuilder().build(ctx)
    assert len(ctx.agent_history) == before


def test_builder_does_not_mutate_risks():
    ctx = _full_ctx()
    risks_before = [dict(r) for r in ctx.risks]
    ExecutiveNarrativeBuilder().build(ctx)
    assert ctx.risks == risks_before


def test_builder_does_not_mutate_assumptions():
    ctx = _full_ctx()
    assumptions_before = [dict(a) for a in ctx.assumptions]
    ExecutiveNarrativeBuilder().build(ctx)
    assert ctx.assumptions == assumptions_before


def test_builder_does_not_mutate_decision_analysis():
    ctx = _full_ctx()
    da_before = dict(ctx.decision_analysis)
    ExecutiveNarrativeBuilder().build(ctx)
    assert ctx.decision_analysis == da_before


# ---------------------------------------------------------------------------
# Stable contracts
# ---------------------------------------------------------------------------

def test_builder_never_imports_functional_agents():
    """Builder must not import any *Agent class or FunctionalAgent (J12.0 constraint)."""
    from functional_agents.narrative import builder as builder_module
    tree = ast.parse(inspect.getsource(builder_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    bad = [s for s in imported_symbols if s.endswith("Agent")]
    assert not bad, f"builder.py must not import Agent classes: {bad}"
    assert "FunctionalAgent" not in imported_symbols


def test_executive_narrative_never_imports_functional_agents():
    from functional_agents.narrative import executive_narrative as en_module
    tree = ast.parse(inspect.getsource(en_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    bad = [s for s in imported_symbols if s.endswith("Agent")]
    assert not bad, f"executive_narrative.py must not import Agent classes: {bad}"


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_builder_degrades_gracefully_on_empty_context():
    ctx = _empty_ctx()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert isinstance(narrative, ExecutiveNarrative)
    assert narrative.decision == ""
    assert narrative.key_risks == []
    assert narrative.immediate_actions == []
    assert ctx.executive_narrative == narrative.to_dict()


# ---------------------------------------------------------------------------
# Trace capture
# ---------------------------------------------------------------------------

def test_canonical_trace_includes_executive_narrative_key_after_build():
    ctx = _full_ctx()
    ExecutiveNarrativeBuilder().build(ctx)
    trace = build_canonical_trace(ctx)
    assert "executive_narrative" in trace
    assert trace["executive_narrative"] == {"generated": True}


def test_canonical_trace_executive_narrative_is_none_before_build():
    ctx = _full_ctx()
    trace = build_canonical_trace(ctx)
    assert "executive_narrative" in trace
    assert trace["executive_narrative"] is None
