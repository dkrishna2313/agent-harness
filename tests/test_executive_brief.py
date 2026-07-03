"""Tests for the J11.1 Executive Brief Generator.

Covers: ExecutiveBriefGenerator registration alongside MarkdownReportGenerator,
executive brief content assembly against the REAL production strategic-option
schema (StrategicOptionItem — description/estimated_time_horizon/capital_intensity/
confidence, not the legacy posture/required_capabilities shape used elsewhere in
the test suite), graceful degradation on missing sections, and confirmation
that adding a second generator does not change ReportAgent's default (markdown)
output or invoke any Functional Agent.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from functional_agents.context import AgentContext
from functional_agents.deliverables import (
    DeliverableRequest,
    ExecutiveBriefGenerator,
    MarkdownReportGenerator,
    default_registry,
)
from functional_agents.deliverables import executive_brief as executive_brief_module
from functional_agents.deliverables.executive_brief import build_executive_brief_content
from functional_agents.run_agent import run_agent

_FIXTURES = "fixtures"


def _make_executive_context() -> AgentContext:
    """AgentContext populated with the REAL StrategicOptionItem schema.

    Deliberately does NOT reuse test_report_agent_j7.py's `_make_context()` —
    that fixture uses the legacy posture/required_capabilities/dependencies
    shape the J7 report renderer expects, which does not match what
    StrategicOptionAgent actually produces in production (description,
    estimated_time_horizon, capital_intensity, confidence, advantages,
    disadvantages). Using the real shape here validates executive_brief.py
    against genuine production data, not a mismatched legacy fixture.
    """
    strategic_options = [
        {
            "option_id": "OPT-A",
            "title": "Phased Deployment Preserving Optionality",
            "description": "Deploy in stages, securing grid interconnection early while deferring full capital commitment.",
            "strategic_objective": "Balance speed-to-market with capital discipline.",
            "expected_outcomes": ["Faster time-to-power", "Reduced stranded capital risk"],
            "advantages": ["Lower downside risk", "Preserves flexibility"],
            "disadvantages": ["Slower full-scale ramp"],
            "implementation_complexity": "Medium",
            "estimated_time_horizon": "near_term",
            "capital_intensity": "Medium",
            "confidence": "High",
            "recommended": True,
            "rationale": "Best balances risk-adjusted returns with capital discipline.",
        },
        {
            "option_id": "OPT-B",
            "title": "Full Speed Build-Out",
            "description": "Commit to full-scale GPU deployment immediately.",
            "estimated_time_horizon": "immediate",
            "capital_intensity": "High",
            "confidence": "Medium",
            "recommended": False,
            "rationale": "Higher upside but greater exposure to grid delay risk.",
        },
    ]

    return AgentContext(
        question="What is the strategic outlook for AI data center power demand?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        strategic_options=strategic_options,
        preferred_option=strategic_options[0],
        decision_analysis={
            "analysis_id": "DA-001",
            "recommended_option_id": "OPT-A",
            "executive_summary": "OPT-A balances risk-adjusted returns with capital discipline.",
            "comparison_dimensions": ["Capital Intensity", "Time to Value", "Risk Exposure"],
            "option_rankings": ["OPT-A", "OPT-B"],
            "rationale": "OPT-A wins on strategic fit and manages downside risk better than OPT-B.",
            "confidence": "High",
        },
        assumptions=[
            {"assumption_id": "A-001", "statement": "Grid interconnection secured within 24 months",
             "importance": "Critical", "confidence": "Medium", "evidence_support": "Moderate"},
            {"assumption_id": "A-002", "statement": "GPU pricing remains stable through 2027",
             "importance": "Important", "confidence": "Low", "evidence_support": "Weak"},
            {"assumption_id": "A-003", "statement": "Cooling vendor capacity is sufficient",
             "importance": "Supporting", "confidence": "High", "evidence_support": "Strong"},
        ],
        risks=[
            {"risk_id": "RSK-001", "statement": "Grid interconnection delay beyond 24 months",
             "severity": "Critical", "likelihood": "Medium", "mitigation_notes": "Secure queue position early"},
            {"risk_id": "RSK-002", "statement": "GPU supply shortage",
             "severity": "High", "likelihood": "Medium", "mitigation_notes": "Diversify suppliers"},
            {"risk_id": "RSK-003", "statement": "Cooling capacity shortfall",
             "severity": "Medium", "likelihood": "Low", "mitigation_notes": "Pre-negotiate capacity"},
        ],
        executive_confidence={
            "overall_confidence": "High",
            "decision_readiness": "Ready",
            "board_recommendation": "Proceed",
            "confidence_rationale": "Strong evidence base across profiles.",
            "confidence_drivers": ["Broad evidence coverage", "Aligned stakeholder incentives"],
            "confidence_limiters": ["Regulatory timeline uncertainty"],
            "validation_priorities": ["Confirm grid queue position", "Validate GPU supply contracts"],
            "critical_unknowns": ["Final interconnection cost"],
            "decision_horizon": "0-3 months",
        },
        recommendations=[
            {"id": "REC-001", "title": "Initiate grid interconnection filing", "time_horizon": "near_term", "priority": "high"},
            {"id": "REC-002", "title": "Negotiate GPU supply contracts", "time_horizon": "medium_term", "priority": "medium"},
        ],
        recommendation_portfolio={"near_term": ["REC-001"], "medium_term": ["REC-002"], "long_term": []},
        strategic_synthesis={
            "executive_summary": "Cross-domain analysis favors a phased deployment that preserves optionality.",
        },
        research_object={
            "decision_architecture": {"decision_statement": "Decide the AI data center power procurement strategy."},
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_executive_brief_generator_declares_its_type():
    assert ExecutiveBriefGenerator.deliverable_type == "executive_brief"


def test_default_registry_has_both_generators_registered():
    assert isinstance(default_registry.get("markdown"), MarkdownReportGenerator)
    assert isinstance(default_registry.get("executive_brief"), ExecutiveBriefGenerator)


def test_registry_dispatches_executive_brief_by_type(tmp_path):
    ctx = _make_executive_context()
    request = DeliverableRequest(type="executive_brief")

    artifact = default_registry.generate(ctx, request, tmp_path / "brief.md")

    assert artifact.type == "executive_brief"
    assert (tmp_path / "brief.md").exists()


# ---------------------------------------------------------------------------
# Content assembly — real production schema
# ---------------------------------------------------------------------------

def test_executive_brief_contains_all_section_headers():
    content = build_executive_brief_content(_make_executive_context())
    for i in range(1, 10):
        assert f"## {i}." in content, f"missing section {i}"


def test_executive_brief_executive_decision_and_summary():
    content = build_executive_brief_content(_make_executive_context())
    assert "Decide the AI data center power procurement strategy." in content
    assert "Phased Deployment Preserving Optionality" in content
    assert "Cross-domain analysis favors a phased deployment" in content


def test_executive_brief_recommended_option_uses_real_schema_fields():
    content = build_executive_brief_content(_make_executive_context())
    assert "OPT-A: Phased Deployment Preserving Optionality" in content
    assert "Deploy in stages, securing grid interconnection early" in content
    assert "**Capital Intensity:** Medium" in content
    assert "**Confidence:** High" in content


def test_executive_brief_top_risks_sorted_by_severity():
    content = build_executive_brief_content(_make_executive_context())
    idx_critical = content.index("RSK-001")
    idx_high = content.index("RSK-002")
    idx_medium = content.index("RSK-003")
    assert idx_critical < idx_high < idx_medium


def test_executive_brief_limits_risks_to_top_5():
    ctx = _make_executive_context()
    ctx.risks = [
        {"risk_id": f"RSK-{i:03d}", "statement": f"Risk {i}", "severity": "Medium", "likelihood": "Low"}
        for i in range(8)
    ]
    content = build_executive_brief_content(ctx)
    assert sum(content.count(f"RSK-{i:03d}") for i in range(8)) == 5


def test_executive_brief_critical_assumptions_sorted_by_importance():
    content = build_executive_brief_content(_make_executive_context())
    idx_critical = content.index("A-001")
    idx_important = content.index("A-002")
    idx_supporting = content.index("A-003")
    assert idx_critical < idx_important < idx_supporting


def test_executive_brief_executive_confidence_section():
    content = build_executive_brief_content(_make_executive_context())
    assert "**Overall Confidence:** High" in content
    assert "**Decision Readiness:** Ready" in content
    assert "**Board Recommendation:** Proceed" in content
    assert "Broad evidence coverage" in content


def test_executive_brief_immediate_decisions_from_recommendation_portfolio():
    content = build_executive_brief_content(_make_executive_context())
    # near_term only — REC-001, not the medium_term REC-002.
    assert "REC-001" in content
    section_8 = content.split("## 8.")[1].split("## 9.")[0]
    assert "REC-002" not in section_8


def test_executive_brief_validation_priorities():
    content = build_executive_brief_content(_make_executive_context())
    assert "Confirm grid queue position" in content
    assert "Validate GPU supply contracts" in content


def test_executive_brief_appendix_includes_profiles_and_unknowns():
    content = build_executive_brief_content(_make_executive_context())
    assert "## 10. Appendix" in content
    assert "ai_data_centers" in content
    assert "Final interconnection cost" in content


def test_executive_brief_appendix_omitted_when_empty():
    ctx = _make_executive_context()
    ctx.profiles = []
    ctx.executive_confidence["critical_unknowns"] = []
    content = build_executive_brief_content(ctx)
    assert "## 10. Appendix" not in content


def test_executive_brief_degrades_gracefully_on_empty_context():
    ctx = AgentContext(question="Q?", profiles=["ai_data_centers"], execution_profile="ai_data_centers")
    content = build_executive_brief_content(ctx)  # must not raise
    assert "# Executive Strategic Brief" in content
    assert "No decision statement" in content
    assert "No executive summary" in content
    assert "No risks recorded" in content
    assert "No assumptions recorded" in content
    assert "not available for this run" in content  # executive confidence


# ---------------------------------------------------------------------------
# J11.1 constraint: never invoke a Functional Agent, no new reasoning
# ---------------------------------------------------------------------------

def test_executive_brief_generator_never_touches_functional_agents():
    tree = ast.parse(inspect.getsource(executive_brief_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert not any(name.endswith("Agent") for name in imported_symbols)
    assert "FunctionalAgent" not in imported_symbols


# ---------------------------------------------------------------------------
# Registering a second generator must not change ReportAgent's default output
# ---------------------------------------------------------------------------

def test_report_agent_still_defaults_to_markdown_only(tmp_path):
    out = tmp_path / "report.md"
    res = run_agent("report", f"{_FIXTURES}/report_start.json", no_llm=True, out_path=str(out))

    ctx = res["context"]
    assert len(ctx["deliverables"]) == 1
    assert ctx["deliverables"][0]["type"] == "markdown"
    assert ctx["deliverable_request"]["type"] == "markdown"
