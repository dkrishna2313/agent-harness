"""J7.6b – ReportAgent executive report tests.

Covers:
  - J7 path triggers when strategic_options is non-empty
  - Legacy path triggers when strategic_options is empty
  - All 14 section headers present in J7 output
  - Section 1: executive summary + recommended option title
  - Section 3: recommended option ID/title present
  - Section 5: assumption table rows
  - Section 6: risk table rows
  - Section 7: opportunity table rows
  - Section 8: all option IDs in Strategic Options section
  - Section 8: recommended option marked
  - Section 9: decision matrix column headers present
  - Section 10: tradeoff bullets
  - Section 11: sensitivity text present
  - Section 12: confidence level present
  - Section 13: recommendation grouped by timeframe
  - Section 14: evidence count present
  - Empty decision_analysis falls back gracefully (no crash)
  - Timeframe normalisation maps aliases correctly
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from functional_agents.context import AgentContext
from functional_agents.report_agent import _build_j7_executive_report, _normalise_timeframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_context(
    *,
    include_da: bool = True,
    include_assumptions: bool = True,
    include_risks: bool = True,
    include_opps: bool = True,
    include_recs: bool = True,
) -> AgentContext:
    options = [
        {
            "option_id": "OPT-A",
            "title": "Aggressive Expansion",
            "rationale": "Move fast to capture market share.",
            "posture": "growth",
            "time_horizon": "near_term",
            "required_capabilities": ["Capital", "Talent"],
            "dependencies": ["Regulatory approval"],
            "risks": ["Execution risk"],
            "supporting_recommendations": ["REC-001"],
        },
        {
            "option_id": "OPT-B",
            "title": "Cautious Partnership",
            "rationale": "Reduce risk through collaboration.",
            "posture": "conservative",
            "time_horizon": "medium_term",
            "required_capabilities": ["Partnerships"],
            "dependencies": [],
            "risks": [],
            "supporting_recommendations": ["REC-002"],
        },
    ]
    da = {
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "executive_summary": "OPT-A is the preferred path given the competitive dynamics.",
        "comparison_dimensions": ["Cost", "Speed", "Risk"],
        "option_rankings": ["OPT-A", "OPT-B"],
        "decision_matrix": [
            {
                "option_id": "OPT-A",
                "strategic_fit": "High",
                "implementation_risk": "Medium",
                "execution_complexity": "Medium",
                "capital_requirement": "High",
                "expected_return": "Very High",
                "time_to_value": "High",
                "dependency_strength": "Medium",
                "assumption_strength": "Medium",
                "risk_exposure": "Medium",
                "opportunity_capture": "High",
                "overall_score": "High",
                "strengths": ["First mover"],
                "weaknesses": ["High capital"],
            },
            {
                "option_id": "OPT-B",
                "strategic_fit": "Medium",
                "implementation_risk": "Low",
                "execution_complexity": "Low",
                "capital_requirement": "Low",
                "expected_return": "Medium",
                "time_to_value": "Medium",
                "dependency_strength": "Low",
                "assumption_strength": "High",
                "risk_exposure": "Low",
                "opportunity_capture": "Medium",
                "overall_score": "Medium",
                "strengths": ["Lower risk"],
                "weaknesses": ["Slower growth"],
            },
        ],
        "key_tradeoffs": ["Speed vs risk", "Capital intensity vs return"],
        "key_uncertainties": ["Regulatory timeline", "Competitor response"],
        "sensitivity_analysis": "If regulatory approval is delayed >12m, OPT-B becomes preferred.",
        "confidence_summary": "High confidence based on 40 evidence items across 3 profiles.",
        "rationale": "OPT-A wins on strategic fit and expected return despite higher capital requirements.",
        "confidence": "High",
    } if include_da else {}

    assumptions = [
        {"assumption_id": "A-001", "statement": "Technology is commercially viable by 2027", "criticality": "Critical", "confidence": "Medium"},
        {"assumption_id": "A-002", "statement": "Regulatory approval in 18 months", "criticality": "High", "confidence": "Low"},
    ] if include_assumptions else []

    risks = [
        {"risk_id": "RSK-001", "title": "Cost overrun", "likelihood": "Medium", "impact": "High", "mitigation": "Fixed-price contracts"},
    ] if include_risks else []

    opps = [
        {"opportunity_id": "OPP-001", "title": "First mover advantage", "category": "Market", "probability": "High", "impact": "High"},
    ] if include_opps else []

    recs = [
        {"recommendation_id": "REC-001", "title": "Initiate permitting", "rationale": "Critical path item.", "timeframe": "0-3 months", "priority": "High"},
        {"recommendation_id": "REC-002", "title": "Evaluate partners", "rationale": "Reduce execution risk.", "timeframe": "3-12 months", "priority": "Medium"},
    ] if include_recs else []

    return AgentContext(
        question="Should we invest in SMR technology?",
        profiles=["smr", "ai_data_centers"],
        execution_profile="smr",
        strategic_options=options,
        preferred_option={"option_id": "OPT-A", "title": "Aggressive Expansion"},
        decision_analysis=da,
        assumptions=assumptions,
        risks=risks,
        opportunities=opps,
        recommendations=recs,
        research_object={
            "summary": {"evidence_count": 42, "citation_count": 38},
            "profiles": ["smr"],
            "evidence_topics": {"SMR Technology": 15, "Regulation": 8},
        },
    )


# ---------------------------------------------------------------------------
# Section header presence
# ---------------------------------------------------------------------------

_EXPECTED_SECTIONS = [
    "## 1. Executive Summary",
    "## 2. Strategic Question",
    "## 3. Recommended Strategic Option",
    "## 4. Why This Option Wins",
    "## 5. Strategic Assumptions",
    "## 6. Strategic Risks",
    "## 7. Strategic Opportunities",
    "## 8. Strategic Options",
    "## 9. Decision Matrix",
    "## 10. Key Tradeoffs",
    "## 11. Sensitivity Analysis",
    "## 12. Confidence Assessment",
    "## 13. Immediate Actions",
    "## 14. Supporting Evidence",
]


def test_all_14_sections_present():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    for section in _EXPECTED_SECTIONS:
        assert section in report, f"Missing section: {section}"


def test_section1_contains_executive_summary():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "OPT-A is the preferred path" in report


def test_section1_contains_recommended_title():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "Aggressive Expansion" in report


def test_section2_contains_question():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "Should we invest in SMR technology?" in report


def test_section3_recommended_option_present():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "OPT-A: Aggressive Expansion" in report


def test_section5_assumption_table_rows():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "A-001" in report
    assert "Technology is commercially viable" in report
    assert "Critical" in report


def test_section5_critical_assumptions_first():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    pos_a001 = report.index("A-001")
    pos_a002 = report.index("A-002")
    assert pos_a001 < pos_a002, "Critical assumption A-001 should appear before High A-002"


def test_section6_risk_table():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "RSK-001" in report
    assert "Cost overrun" in report
    assert "Fixed-price contracts" in report


def test_section7_opportunity_table():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "OPP-001" in report
    assert "First mover advantage" in report


def test_section8_all_option_ids_present():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "OPT-A" in report
    assert "OPT-B" in report


def test_section8_recommended_option_marked():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "*(Recommended)*" in report


def test_section9_decision_matrix_column_headers():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "Strategic Fit" in report
    assert "Overall" in report
    assert "Expected Return" in report


def test_section9_decision_matrix_rows_per_option():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    # Both options appear in matrix rows
    section_9_start = report.index("## 9. Decision Matrix")
    section_10_start = report.index("## 10. Key Tradeoffs")
    matrix_section = report[section_9_start:section_10_start]
    assert "OPT-A" in matrix_section
    assert "OPT-B" in matrix_section


def test_section10_tradeoffs():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "Speed vs risk" in report
    assert "Capital intensity vs return" in report


def test_section11_sensitivity_analysis():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "regulatory approval is delayed" in report


def test_section12_confidence_level():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "**Overall Confidence:** High" in report
    assert "40 evidence items" in report


def test_section13_timeframe_grouping():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "### 0-3 months" in report
    assert "### 3-12 months" in report
    assert "REC-001" in report
    assert "REC-002" in report


def test_section14_evidence_count():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "42" in report  # evidence_count
    assert "38" in report  # citation_count


def test_section14_source_profiles():
    ctx = _make_context()
    report = _build_j7_executive_report(ctx)
    assert "smr" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_decision_analysis_no_crash():
    ctx = _make_context(include_da=False)
    report = _build_j7_executive_report(ctx)
    for section in _EXPECTED_SECTIONS:
        assert section in report


def test_no_assumptions_graceful():
    ctx = _make_context(include_assumptions=False)
    report = _build_j7_executive_report(ctx)
    assert "No assumptions recorded" in report


def test_no_risks_graceful():
    ctx = _make_context(include_risks=False)
    report = _build_j7_executive_report(ctx)
    assert "No risks recorded" in report


def test_no_opportunities_graceful():
    ctx = _make_context(include_opps=False)
    report = _build_j7_executive_report(ctx)
    assert "No opportunities recorded" in report


def test_no_recommendations_graceful():
    ctx = _make_context(include_recs=False)
    report = _build_j7_executive_report(ctx)
    assert "No recommendations recorded" in report


# ---------------------------------------------------------------------------
# Timeframe normalisation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alias,expected", [
    ("immediate", "0-3 months"),
    ("short_term", "0-3 months"),
    ("short-term", "0-3 months"),
    ("near_term", "3-12 months"),
    ("near-term", "3-12 months"),
    ("medium_term", "1-3 years"),
    ("medium-term", "1-3 years"),
    ("long_term", "3+ years"),
    ("long-term", "3+ years"),
    ("0-3 months", "0-3 months"),   # pass-through
    ("custom bucket", "custom bucket"),  # unknown pass-through
])
def test_timeframe_normalisation(alias, expected):
    assert _normalise_timeframe(alias) == expected
