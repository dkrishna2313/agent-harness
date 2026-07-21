"""J12.5 — Narrative-Driven Executive Report tests.

Validates that _build_j7_executive_report consumes ExecutiveNarrative for all
executive-facing prose sections (§1 summary, §3 recommended option, §4 why
this option wins, §5 executive confidence, §11 key tradeoffs) while keeping
structured sections (tables, assumptions, risks, evidence, appendices) reading
from AgentContext directly.
"""

from __future__ import annotations

import ast
import inspect

import pytest

import functional_agents.report_agent as report_agent_module
from functional_agents.context import AgentContext
from functional_agents.report_agent import _build_j7_executive_report


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_narrative_context() -> AgentContext:
    """Production-schema fixture with strategic_synthesis and executive_confidence.

    Uses description/estimated_time_horizon (production schema) so that
    narrative fields are non-empty and distinguishable from da values.
    strategic_synthesis.executive_summary is deliberately different from
    da.executive_summary to verify the narrative source is preferred.
    """
    return AgentContext(
        question="Should we invest in SMR technology?",
        profiles=["smr"],
        execution_profile="smr",
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Phased Deployment",
                "description": "Deploy in stages to preserve optionality.",
                "estimated_time_horizon": "near_term",
                "capital_intensity": "Medium",
                "confidence": "High",
                "advantages": ["Lower downside risk", "Preserves flexibility"],
                "recommended": True,
            },
            {
                "option_id": "OPT-B",
                "title": "Aggressive Build",
                "description": "Full deployment immediately.",
                "estimated_time_horizon": "immediate",
                "capital_intensity": "High",
                "confidence": "Medium",
                "recommended": False,
            },
        ],
        preferred_option={"option_id": "OPT-A", "title": "Phased Deployment"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "executive_summary": "DA summary — superseded by strategic_synthesis.",
            "comparison_dimensions": ["Capital Intensity", "Time to Value"],
            "option_rankings": ["OPT-A", "OPT-B"],
            "rationale": "OPT-A wins on strategic fit.",
            "key_tradeoffs": ["Speed vs. Capital Efficiency", "Risk vs. Return"],
            "confidence": "High",
            "confidence_summary": "Strong evidence base.",
            "sensitivity_analysis": "If grid delays exceed 18 months, OPT-B preferred.",
            "key_uncertainties": ["Regulatory timeline"],
        },
        strategic_synthesis={
            "executive_summary": "Cross-domain synthesis favors phased deployment with capital discipline.",
        },
        executive_confidence={
            "overall_confidence": "High",
            "decision_readiness": "Ready",
            "board_recommendation": "Proceed",
            "confidence_rationale": "Strong evidence across profiles.",
            "confidence_drivers": ["Broad coverage"],
            "confidence_limiters": ["Regulatory uncertainty"],
            "validation_priorities": ["Confirm grid queue position"],
            "critical_unknowns": ["Final interconnection cost"],
            "decision_horizon": "0–3 months",
        },
        assumptions=[
            {
                "assumption_id": "A-001",
                "statement": "Grid connection secured in 24 months",
                "importance": "Critical",
                "confidence": "Medium",
                "evidence_support": "Moderate",
            },
        ],
        risks=[
            {
                "risk_id": "RSK-001",
                "statement": "Grid interconnection delay",
                "severity": "High",
                "likelihood": "Medium",
                "mitigation_notes": "Secure queue position early",
                "related_assumption_ids": ["A-001"],
                "affected_recommendation_ids": ["REC-001"],
            },
        ],
        recommendations=[
            {
                "recommendation_id": "REC-001",
                "title": "File grid interconnection application",
                "time_horizon": "near_term",
                "priority": "high",
                "summary": "Start the grid process now to maintain schedule.",
            },
        ],
        recommendation_portfolio={
            "near_term": ["REC-001"],
            "medium_term": [],
            "long_term": [],
        },
        research_object={
            "evidence_summary": {"total_evidence_items": 10, "citation_count": 8},
            "profiles": ["smr"],
        },
    )


# ---------------------------------------------------------------------------
# Static contract
# ---------------------------------------------------------------------------

def test_report_agent_module_imports_executive_narrative_builder():
    """report_agent.py must import ExecutiveNarrativeBuilder (J12.5 contract)."""
    tree = ast.parse(inspect.getsource(report_agent_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "ExecutiveNarrativeBuilder" in imported_symbols
    assert "ExecutiveNarrative" in imported_symbols


# ---------------------------------------------------------------------------
# Side effect: context.executive_narrative is populated
# ---------------------------------------------------------------------------

def test_report_sets_executive_narrative_on_context():
    """_build_j7_executive_report sets context.executive_narrative as side effect."""
    ctx = _make_narrative_context()
    assert ctx.executive_narrative == {}
    _build_j7_executive_report(ctx)
    assert ctx.executive_narrative != {}
    assert "decision" in ctx.executive_narrative
    assert ctx.executive_narrative.get("version") == "1.1"


# ---------------------------------------------------------------------------
# Section 1 — Executive Summary from narrative
# ---------------------------------------------------------------------------

def test_section1_executive_summary_from_narrative():
    """§1 is recommendation-led; raw DA executive_summary is not shown (P1.1)."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_1 = report.split("## 2.")[0]
    # Section 1 is now recommendation-led; the raw DA executive_summary must not appear
    assert "DA summary" not in section_1
    # The recommendation from narrative is present
    assert "Phased Deployment" in section_1


def test_section1_recommended_option_title_from_narrative():
    """§1 recommendation line uses narrative.recommended_option title (PH6.0 prose format)."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_1 = report.split("## 2.")[0]
    # PH6.0: no "**Recommendation:**" label — plain prose opener "Pursue {title}"
    assert "Phased Deployment" in section_1


# ---------------------------------------------------------------------------
# Section 3 — Recommended Option prose from narrative
# ---------------------------------------------------------------------------

def test_section3_uses_narrative_recommended_option_fields():
    """§3 renders title and description from narrative.recommended_option (no option_id in prose — PH5.x)."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_3 = report.split("## 3.")[1].split("## 4.")[0]
    assert "Phased Deployment" in section_3
    assert "OPT-A" not in section_3
    assert "Deploy in stages to preserve optionality." in section_3


def test_section3_time_horizon_from_narrative():
    """§3 renders estimated_time_horizon from narrative.recommended_option."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_3 = report.split("## 3.")[1].split("## 4.")[0]
    assert "near term" in section_3


# ---------------------------------------------------------------------------
# Section 4 — Why This Option Wins from narrative (with composer enrichment)
# ---------------------------------------------------------------------------

def test_section4_why_this_option_from_narrative():
    """§4 rationale comes from narrative.why_this_option (sourced from da.rationale).

    PH5.x: the option_id is stripped from prose; only the reasoning phrase survives.
    """
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_4 = report.split("## 4.")[1].split("## 5.")[0]
    assert "wins on strategic fit." in section_4


def test_section4_why_enriched_with_recommended_option_advantages():
    """Composer enriches why_this_option with OPT-A advantages; they appear in §4."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_4 = report.split("## 4.")[1].split("## 5.")[0]
    assert "Lower downside risk" in section_4
    assert "Preserves flexibility" in section_4


def test_section4_option_rankings_from_narrative():
    """§4 option rankings resolve option_ids to titles (PH5.x — no IDs in prose)."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_4 = report.split("## 4.")[1].split("## 5.")[0]
    assert "Option Rankings" in section_4
    # Rankings must show titles, not raw option_ids
    assert "Phased Deployment" in section_4
    assert "Aggressive Build" in section_4


# ---------------------------------------------------------------------------
# Section 5 — Executive Confidence from narrative
# ---------------------------------------------------------------------------

def test_section5_overall_confidence_from_narrative():
    """§5 confidence summary header comes from narrative.executive_confidence."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_5 = report.split("## 5.")[1].split("## 6.")[0]
    assert "**Overall Confidence:** High" in section_5
    assert "**Decision Readiness:** Ready" in section_5
    assert "**Board Recommendation:** Proceed" in section_5


def test_section5_confidence_rationale_from_narrative():
    """§5 confidence rationale prose comes from narrative.executive_confidence."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_5 = report.split("## 5.")[1].split("## 6.")[0]
    assert "Strong evidence across profiles." in section_5


def test_section5_confidence_drivers_from_narrative():
    """§5 confidence drivers rendered from narrative.executive_confidence."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_5 = report.split("## 5.")[1].split("## 6.")[0]
    assert "Broad coverage" in section_5


def test_section5_validation_priorities_from_narrative():
    """§5 validation priorities come from narrative.validation_priorities."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_5 = report.split("## 5.")[1].split("## 6.")[0]
    assert "Confirm grid queue position" in section_5


def test_section5_critical_unknowns_from_narrative():
    """§5 critical unknowns come from narrative.critical_unknowns."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_5 = report.split("## 5.")[1].split("## 6.")[0]
    assert "Final interconnection cost" in section_5


# ---------------------------------------------------------------------------
# Section 11 — Key Tradeoffs from narrative
# ---------------------------------------------------------------------------

def test_section4_key_tradeoffs_from_narrative():
    """§4 key tradeoffs come from narrative.key_tradeoffs (da.key_tradeoffs path)."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    section_4 = report.split("## 4.")[1].split("## 5.")[0]
    assert "Speed vs. Capital Efficiency" in section_4
    assert "Risk vs. Return" in section_4


def test_section4_key_tradeoffs_prefer_strategic_synthesis():
    """§4 prefers strategic_synthesis.key_tradeoffs over da.key_tradeoffs."""
    ctx = _make_narrative_context()
    ctx.strategic_synthesis = {
        "executive_summary": "Cross-domain synthesis.",
        "key_tradeoffs": ["Strategic tradeoff from synthesis"],
    }
    report = _build_j7_executive_report(ctx)
    section_4 = report.split("## 4.")[1].split("## 5.")[0]
    assert "Strategic tradeoff from synthesis" in section_4
    assert "Speed vs. Capital Efficiency" not in section_4


# ---------------------------------------------------------------------------
# Structured sections preserved (no regression)
# ---------------------------------------------------------------------------

def test_structured_sections_still_from_agentcontext():
    """Structured tables (assumptions, risks, options) still render from AgentContext."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    # Assumptions present (Section 6 exec summary + Appendix B)
    assert "A-001" in report
    assert "Grid connection secured in 24 months" in report
    # Risks present (Section 7 exec summary + Appendix C)
    assert "RSK-001" in report
    assert "Grid interconnection delay" in report
    # Strategic options detail in Appendix A (Section 10)
    appendix = report.split("## 10.")[1]
    assert "OPT-B" in appendix
    assert "Aggressive Build" in appendix


def test_all_10_sections_present_with_narrative_context():
    """All 10 report sections render correctly with the production-schema fixture."""
    ctx = _make_narrative_context()
    report = _build_j7_executive_report(ctx)
    for i in range(1, 11):
        assert f"## {i}." in report, f"Missing section {i}"


def test_no_regression_on_empty_executive_confidence():
    """§5 degrades gracefully when context.executive_confidence is empty."""
    ctx = _make_narrative_context()
    ctx.executive_confidence = {}
    report = _build_j7_executive_report(ctx)
    assert "Executive confidence assessment not available for this run." in report
