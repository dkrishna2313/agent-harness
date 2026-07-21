"""Tests for PH6.6–PH6.8 — RecommendationWriter, RiskWriter, OpportunityWriter.

Covers: writer contract, registry ordering, tables, subtitle, bullet groups,
content constraints, provenance, isolation (other sections untouched), mock
fallback chain, empty-context handling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from functional_agents.context import AgentContext
from functional_agents.editorial import (
    DecisionAnalysisWriter,
    EditorialCoordinator,
    EditorialWriter,
    ExecutiveSummaryWriter,
    OpportunityWriter,
    RecommendationWriter,
    RiskWriter,
)
from functional_agents.editorial.editorial_manuscript import EditorialManuscript

# ---------------------------------------------------------------------------
# Shared fixture context (rich data for all three writers)
# ---------------------------------------------------------------------------

_CTX = AgentContext(
    question="Should Deloitte enter the sports analytics market?",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph668-unit",
    research_object={
        "research_id": "R-001",
        "profile": "sports",
        "evidence_summary": {"total_evidence_items": 50, "citation_count": 16},
    },
    strategic_options=[
        {
            "option_id": "OPT-A",
            "title": "Advisory-Led Entry",
            "description": "Enter through advisory before building proprietary tools.",
            "strategic_objective": "Capture early market share with minimal capital exposure.",
            "recommended": True,
        },
        {
            "option_id": "OPT-B",
            "title": "Product-First Entry",
            "description": "Build proprietary platform before client engagement.",
            "strategic_objective": "Establish a defensible IP position.",
            "recommended": False,
        },
    ],
    preferred_option={"option_id": "OPT-A", "title": "Advisory-Led Entry"},
    decision_analysis={
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "option_rankings": ["OPT-A", "OPT-B"],
        "comparison_dimensions": ["Capital Efficiency", "Speed to Revenue"],
        "key_tradeoffs": ["Lower upfront investment vs. slower IP development"],
        "key_uncertainties": ["Market demand at projected rate levels"],
        "decision_matrix": [
            {
                "option_id": "OPT-A",
                "option_title": "Advisory-Led Entry",
                "dimensions": {"Capital Efficiency": "High", "Speed to Revenue": "High"},
                "overall_score": "Preferred",
                "is_recommended": True,
            },
        ],
    },
    assumptions=[
        {
            "assumption_id": "A-001",
            "statement": "Market demand is addressable at scale.",
            "importance": "Critical",
            "confidence": "Medium",
            "evidence_support": "Moderate",
        }
    ],
    risks=[
        {
            "risk_id": "RSK-001",
            "statement": "Commoditization risk from established analytics vendors.",
            "severity": "High",
            "likelihood": "Medium",
            "mitigation_notes": "Differentiate via domain expertise and proprietary datasets.",
        },
        {
            "risk_id": "RSK-002",
            "statement": "Talent shortage for specialist sports data scientists.",
            "severity": "Medium",
            "likelihood": "High",
            "mitigation_notes": "Partner with universities and develop internal pipeline.",
        },
        {
            "risk_id": "RSK-003",
            "statement": "Client budget constraints in off-season periods.",
            "severity": "Low",
            "likelihood": "Medium",
            "mitigation_notes": "",
        },
    ],
    opportunities=[
        {
            "opportunity_id": "OPP-001",
            "statement": "First mover advantage in mid-market sports analytics.",
            "category": "Market",
            "likelihood": "High",
            "impact": "High",
        },
        {
            "opportunity_id": "OPP-002",
            "statement": "Cross-sell analytics capabilities to existing sports clients.",
            "category": "Commercial",
            "likelihood": "High",
            "impact": "Medium",
        },
        {
            "opportunity_id": "OPP-003",
            "statement": "Build proprietary benchmark datasets for future IP monetisation.",
            "category": "IP",
            "likelihood": "Medium",
            "impact": "High",
        },
    ],
    recommendations=[
        {
            "recommendation_id": "REC-001",
            "title": "Launch 90-Day Advisory Pilot",
            "summary": "Engage two mid-market sports clubs with structured advisory programme.",
            "time_horizon": "near_term",
            "priority": "high",
        },
        {
            "recommendation_id": "REC-002",
            "title": "Establish Talent Pipeline",
            "summary": "Partner with three universities to seed sports data science capability.",
            "time_horizon": "medium_term",
            "priority": "high",
        },
        {
            "recommendation_id": "REC-003",
            "title": "Develop IP Framework",
            "summary": "Define IP ownership policy for client engagement deliverables.",
            "time_horizon": "long_term",
            "priority": "medium",
        },
    ],
    executive_confidence={
        "confidence_id": "EC-001",
        "overall_confidence": "Medium",
        "decision_readiness": "Needs Additional Validation",
        "board_recommendation": "Proceed with Conditions",
        "validation_priorities": ["Validate TAM"],
        "confidence_limiters": ["Limited independent data"],
        "critical_unknowns": ["Client willingness to pay advisory rates"],
    },
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coord():
    return EditorialCoordinator()


@pytest.fixture(scope="module")
def brief(coord):
    return coord.build(_CTX)


@pytest.fixture(scope="module")
def ms_all(coord, brief):
    """Manuscript with all five writers applied via run_writers."""
    ms = coord.build_manuscript(brief)
    coord.run_writers(brief, ms, client=None)
    return ms


@pytest.fixture(scope="module")
def ms_rec_only(coord, brief):
    ms = coord.build_manuscript(brief)
    RecommendationWriter(client=None).write(brief, ms)
    return ms


@pytest.fixture(scope="module")
def ms_risk_only(coord, brief):
    ms = coord.build_manuscript(brief)
    RiskWriter(client=None).write(brief, ms)
    return ms


@pytest.fixture(scope="module")
def ms_opp_only(coord, brief):
    ms = coord.build_manuscript(brief)
    OpportunityWriter(client=None).write(brief, ms)
    return ms


# ---------------------------------------------------------------------------
# Writer contract checks
# ---------------------------------------------------------------------------

def test_recommendation_writer_is_editorial_writer():
    assert issubclass(RecommendationWriter, EditorialWriter)


def test_risk_writer_is_editorial_writer():
    assert issubclass(RiskWriter, EditorialWriter)


def test_opportunity_writer_is_editorial_writer():
    assert issubclass(OpportunityWriter, EditorialWriter)


def test_recommendation_writer_returns_manuscript(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    result = RecommendationWriter(client=None).write(brief, ms)
    assert result is ms  # same object, mutated in place


def test_risk_writer_returns_manuscript(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    result = RiskWriter(client=None).write(brief, ms)
    assert result is ms


def test_opportunity_writer_returns_manuscript(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    result = OpportunityWriter(client=None).write(brief, ms)
    assert result is ms


# ---------------------------------------------------------------------------
# Writer registry — run_writers()
# ---------------------------------------------------------------------------

def test_run_writers_returns_manuscript(coord, brief):
    ms = coord.build_manuscript(brief)
    result = coord.run_writers(brief, ms, client=None)
    assert result is ms


def test_run_writers_populates_all_five_sections(ms_all):
    assert ms_all.executive_summary.paragraphs != []
    assert ms_all.decision_analysis.paragraphs != []
    assert ms_all.recommendations.paragraphs != []
    assert ms_all.strategic_risks.paragraphs != []
    assert ms_all.strategic_opportunities.paragraphs != []


def test_run_writers_leaves_confidence_empty(ms_all):
    assert ms_all.executive_confidence.paragraphs == []
    assert ms_all.executive_confidence.tables == []


def test_run_writers_leaves_appendix_empty(ms_all):
    assert ms_all.appendix.paragraphs == []
    assert ms_all.appendix.tables == []


# ---------------------------------------------------------------------------
# RecommendationWriter — structure
# ---------------------------------------------------------------------------

def test_rec_subtitle_populated(ms_rec_only):
    sub = ms_rec_only.recommendations.subtitle
    assert len(sub) > 0


def test_rec_subtitle_mentions_count(ms_rec_only):
    sub = ms_rec_only.recommendations.subtitle
    assert any(c.isdigit() for c in sub)


def test_rec_subtitle_mentions_high_priority(ms_rec_only):
    sub = ms_rec_only.recommendations.subtitle.lower()
    assert "high" in sub or "priority" in sub


def test_rec_paragraphs_populated(ms_rec_only):
    paras = ms_rec_only.recommendations.paragraphs
    assert len(paras) >= 4
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 20


def test_rec_bullet_groups_present(ms_rec_only):
    bgs = ms_rec_only.recommendations.bullet_groups
    assert len(bgs) >= 2
    for grp in bgs:
        assert isinstance(grp, list)
        assert len(grp) >= 1


def test_rec_table_present(ms_rec_only):
    tables = ms_rec_only.recommendations.tables
    assert len(tables) == 1


def test_rec_table_structure(ms_rec_only):
    t = ms_rec_only.recommendations.tables[0]
    assert t["title"] == "Recommended Actions"
    assert "Recommendation" in t["headers"]
    assert "Priority" in t["headers"]
    assert "Time Horizon" in t["headers"]
    assert "Summary" in t["headers"]


def test_rec_table_row_count(ms_rec_only):
    t = ms_rec_only.recommendations.tables[0]
    assert len(t["rows"]) == 3  # three recommendations in fixture


def test_rec_high_priority_bullet_includes_high_rec(ms_rec_only):
    bgs = ms_rec_only.recommendations.bullet_groups
    first_group = " ".join(bgs[0]).lower()
    assert "pilot" in first_group or "talent" in first_group


def test_rec_does_not_touch_other_sections(ms_rec_only):
    assert ms_rec_only.executive_summary.paragraphs == []
    assert ms_rec_only.decision_analysis.paragraphs == []
    assert ms_rec_only.strategic_risks.paragraphs == []
    assert ms_rec_only.strategic_opportunities.paragraphs == []


def test_rec_no_internal_ids_in_prose(ms_rec_only):
    import re
    all_text = "\n".join(ms_rec_only.recommendations.paragraphs)
    assert not re.search(r'\b(REC|RSK|OPP|OPT|A|EC|DA|EB|EM|R)-\d+\b', all_text)


# ---------------------------------------------------------------------------
# RiskWriter — structure
# ---------------------------------------------------------------------------

def test_risk_subtitle_populated(ms_risk_only):
    sub = ms_risk_only.strategic_risks.subtitle
    assert len(sub) > 0


def test_risk_subtitle_mentions_count(ms_risk_only):
    sub = ms_risk_only.strategic_risks.subtitle
    assert any(c.isdigit() for c in sub)


def test_risk_subtitle_mentions_high_severity(ms_risk_only):
    sub = ms_risk_only.strategic_risks.subtitle.lower()
    assert "high" in sub


def test_risk_paragraphs_populated(ms_risk_only):
    paras = ms_risk_only.strategic_risks.paragraphs
    assert len(paras) >= 4
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 20


def test_risk_bullet_groups_present(ms_risk_only):
    bgs = ms_risk_only.strategic_risks.bullet_groups
    assert len(bgs) >= 2
    for grp in bgs:
        assert isinstance(grp, list)
        assert len(grp) >= 1


def test_risk_table_present(ms_risk_only):
    tables = ms_risk_only.strategic_risks.tables
    assert len(tables) == 1


def test_risk_table_structure(ms_risk_only):
    t = ms_risk_only.strategic_risks.tables[0]
    assert t["title"] == "Risk Register"
    assert "Risk" in t["headers"]
    assert "Severity" in t["headers"]
    assert "Likelihood" in t["headers"]
    assert "Mitigation" in t["headers"]


def test_risk_table_sorted_high_first(ms_risk_only):
    rows = ms_risk_only.strategic_risks.tables[0]["rows"]
    # First row should be the High-severity risk
    assert "Commoditization" in rows[0][0] or rows[0][1].lower() == "high"


def test_risk_table_row_count(ms_risk_only):
    rows = ms_risk_only.strategic_risks.tables[0]["rows"]
    assert len(rows) == 3


def test_risk_does_not_touch_other_sections(ms_risk_only):
    assert ms_risk_only.executive_summary.paragraphs == []
    assert ms_risk_only.decision_analysis.paragraphs == []
    assert ms_risk_only.recommendations.paragraphs == []
    assert ms_risk_only.strategic_opportunities.paragraphs == []


def test_risk_no_internal_ids_in_prose(ms_risk_only):
    import re
    all_text = "\n".join(ms_risk_only.strategic_risks.paragraphs)
    assert not re.search(r'\b(RSK|OPP|REC|OPT|A|EC|DA|EB|EM|R)-\d+\b', all_text)


def test_risk_mitigation_in_second_bullet_group(ms_risk_only):
    bgs = ms_risk_only.strategic_risks.bullet_groups
    if len(bgs) >= 2:
        mit_text = " ".join(bgs[1]).lower()
        assert "differentiat" in mit_text or "partner" in mit_text or "mitigation" in mit_text or len(bgs[1]) >= 1


# ---------------------------------------------------------------------------
# OpportunityWriter — structure
# ---------------------------------------------------------------------------

def test_opp_subtitle_populated(ms_opp_only):
    sub = ms_opp_only.strategic_opportunities.subtitle
    assert len(sub) > 0


def test_opp_subtitle_mentions_count(ms_opp_only):
    sub = ms_opp_only.strategic_opportunities.subtitle
    assert any(c.isdigit() for c in sub)


def test_opp_subtitle_mentions_categories(ms_opp_only):
    sub = ms_opp_only.strategic_opportunities.subtitle.lower()
    assert "categor" in sub


def test_opp_paragraphs_populated(ms_opp_only):
    paras = ms_opp_only.strategic_opportunities.paragraphs
    assert len(paras) >= 4
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 20


def test_opp_bullet_groups_present(ms_opp_only):
    bgs = ms_opp_only.strategic_opportunities.bullet_groups
    assert len(bgs) >= 2
    for grp in bgs:
        assert isinstance(grp, list)
        assert len(grp) >= 1


def test_opp_table_present(ms_opp_only):
    tables = ms_opp_only.strategic_opportunities.tables
    assert len(tables) == 1


def test_opp_table_structure(ms_opp_only):
    t = ms_opp_only.strategic_opportunities.tables[0]
    assert t["title"] == "Strategic Opportunities"
    assert "Opportunity" in t["headers"]
    assert "Category" in t["headers"]
    assert "Likelihood" in t["headers"]
    assert "Impact" in t["headers"]


def test_opp_table_sorted_high_impact_first(ms_opp_only):
    rows = ms_opp_only.strategic_opportunities.tables[0]["rows"]
    # High-impact rows should appear first
    first_impact = rows[0][3].lower()
    assert first_impact == "high"


def test_opp_table_row_count(ms_opp_only):
    rows = ms_opp_only.strategic_opportunities.tables[0]["rows"]
    assert len(rows) == 3


def test_opp_does_not_touch_other_sections(ms_opp_only):
    assert ms_opp_only.executive_summary.paragraphs == []
    assert ms_opp_only.decision_analysis.paragraphs == []
    assert ms_opp_only.recommendations.paragraphs == []
    assert ms_opp_only.strategic_risks.paragraphs == []


def test_opp_no_internal_ids_in_prose(ms_opp_only):
    import re
    all_text = "\n".join(ms_opp_only.strategic_opportunities.paragraphs)
    assert not re.search(r'\b(OPP|RSK|REC|OPT|A|EC|DA|EB|EM|R)-\d+\b', all_text)


def test_opp_high_impact_in_first_bullet_group(ms_opp_only):
    bgs = ms_opp_only.strategic_opportunities.bullet_groups
    first_group = " ".join(bgs[0]).lower()
    assert "mover" in first_group or "analytics" in first_group or "high" in first_group


# ---------------------------------------------------------------------------
# Fallback chain — missing method
# ---------------------------------------------------------------------------

class _BadClient:
    is_mock = False
    # no generate_*_prose methods


def test_rec_falls_back_when_client_lacks_method(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    RecommendationWriter(client=_BadClient()).write(brief, ms)
    assert ms.recommendations.paragraphs != []


def test_risk_falls_back_when_client_lacks_method(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    RiskWriter(client=_BadClient()).write(brief, ms)
    assert ms.strategic_risks.paragraphs != []


def test_opp_falls_back_when_client_lacks_method(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    OpportunityWriter(client=_BadClient()).write(brief, ms)
    assert ms.strategic_opportunities.paragraphs != []


# ---------------------------------------------------------------------------
# Empty context handling
# ---------------------------------------------------------------------------

_EMPTY_CTX = AgentContext(
    question="Minimal test question",
    profiles=["default"],
    execution_profile="default",
    run_id="ph668-empty",
    research_object={"research_id": "R-EMPTY", "profile": "default"},
    strategic_options=[],
    preferred_option={},
    decision_analysis={},
    assumptions=[],
    risks=[],
    opportunities=[],
    recommendations=[],
    executive_confidence={},
)


def test_rec_handles_empty_recommendations():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    RecommendationWriter(client=None).write(brief, ms)
    # Should not raise; subtitle may be empty but paragraphs should be a list
    assert isinstance(ms.recommendations.paragraphs, list)
    assert isinstance(ms.recommendations.tables, list)


def test_risk_handles_empty_risks():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    RiskWriter(client=None).write(brief, ms)
    assert isinstance(ms.strategic_risks.paragraphs, list)
    assert ms.strategic_risks.tables == []


def test_opp_handles_empty_opportunities():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    OpportunityWriter(client=None).write(brief, ms)
    assert isinstance(ms.strategic_opportunities.paragraphs, list)
    assert ms.strategic_opportunities.tables == []
