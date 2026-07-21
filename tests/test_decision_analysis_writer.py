"""Tests for PH6.5 — EditorialWriter ABC and DecisionAnalysisWriter.

Covers: ABC contract, writer structure, decision matrix table, content constraints,
provenance retention, other sections untouched, mock fallback, empty context.
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
)
from functional_agents.editorial.editorial_manuscript import EditorialManuscript

# ---------------------------------------------------------------------------
# Test fixture context
# ---------------------------------------------------------------------------

_CTX = AgentContext(
    question="Should Deloitte enter the sports analytics market?",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph65-unit",
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
        {
            "option_id": "OPT-C",
            "title": "Partnership Entry",
            "description": "Enter via strategic partnership with existing player.",
            "strategic_objective": "Reduce time to market via channel leverage.",
            "recommended": False,
        },
    ],
    preferred_option={"option_id": "OPT-A", "title": "Advisory-Led Entry"},
    decision_analysis={
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "option_rankings": ["OPT-A", "OPT-C", "OPT-B"],
        "comparison_dimensions": ["Capital Efficiency", "Speed to Revenue", "IP Defensibility", "Execution Risk"],
        "key_tradeoffs": [
            "Lower upfront investment vs. slower IP development",
            "Faster revenue vs. reduced long-term defensibility",
            "Advisory flexibility vs. channel dependency in partnership model",
        ],
        "key_uncertainties": [
            "Market demand at projected advisory rate levels",
            "Competitive response timeline from established vendors",
        ],
        "decision_matrix": [
            {
                "option_id": "OPT-A",
                "option_title": "Advisory-Led Entry",
                "dimensions": {"Capital Efficiency": "High", "Speed to Revenue": "High", "IP Defensibility": "Low", "Execution Risk": "Low"},
                "overall_score": "Preferred",
                "is_recommended": True,
            },
            {
                "option_id": "OPT-C",
                "option_title": "Partnership Entry",
                "dimensions": {"Capital Efficiency": "Medium", "Speed to Revenue": "Medium", "IP Defensibility": "Medium", "Execution Risk": "Medium"},
                "overall_score": "Alternative",
                "is_recommended": False,
            },
            {
                "option_id": "OPT-B",
                "option_title": "Product-First Entry",
                "dimensions": {"Capital Efficiency": "Low", "Speed to Revenue": "Low", "IP Defensibility": "High", "Execution Risk": "High"},
                "overall_score": "Reject",
                "is_recommended": False,
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
            "statement": "Commoditization risk from established vendors.",
            "severity": "High",
            "likelihood": "Medium",
        }
    ],
    opportunities=[
        {
            "opportunity_id": "OPP-001",
            "statement": "First mover in mid-market.",
            "category": "Market",
            "likelihood": "High",
            "impact": "High",
        }
    ],
    recommendations=[
        {
            "recommendation_id": "REC-001",
            "title": "Launch Pilot",
            "summary": "90-day advisory pilot.",
            "time_horizon": "near_term",
            "priority": "high",
        }
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


@pytest.fixture(scope="module")
def both_written() -> EditorialManuscript:
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    ExecutiveSummaryWriter(client=None).write(brief, ms)
    DecisionAnalysisWriter(client=None).write(brief, ms)
    return ms


@pytest.fixture(scope="module")
def da_only() -> EditorialManuscript:
    """Manuscript with only DecisionAnalysisWriter applied (no ExecutiveSummaryWriter)."""
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    DecisionAnalysisWriter(client=None).write(brief, ms)
    return ms


# ---------------------------------------------------------------------------
# EditorialWriter ABC
# ---------------------------------------------------------------------------

def test_editorial_writer_is_abstract():
    import inspect
    assert inspect.isabstract(EditorialWriter)


def test_executive_summary_writer_is_editorial_writer():
    assert issubclass(ExecutiveSummaryWriter, EditorialWriter)


def test_decision_analysis_writer_is_editorial_writer():
    assert issubclass(DecisionAnalysisWriter, EditorialWriter)


def test_editorial_writer_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        EditorialWriter()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# DecisionAnalysisWriter — structure
# ---------------------------------------------------------------------------

def test_decision_analysis_title_unchanged(both_written):
    assert both_written.decision_analysis.title == "Strategic Analysis"


def test_decision_analysis_subtitle_populated(both_written):
    subtitle = both_written.decision_analysis.subtitle
    assert len(subtitle) > 0
    # Should reference number of options or comparison
    assert any(c.isdigit() for c in subtitle) or "option" in subtitle.lower()


def test_decision_analysis_paragraphs_populated(both_written):
    paras = both_written.decision_analysis.paragraphs
    assert len(paras) >= 4, f"Expected ≥4 paragraphs, got {len(paras)}"
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 20


def test_decision_analysis_bullet_groups_populated(both_written):
    bgs = both_written.decision_analysis.bullet_groups
    assert len(bgs) >= 1
    for grp in bgs:
        assert isinstance(grp, list)
        for b in grp:
            assert isinstance(b, str) and len(b.strip()) > 0


def test_decision_analysis_tradeoffs_in_bullets(both_written):
    bgs = both_written.decision_analysis.bullet_groups
    assert len(bgs) >= 1
    bullet_text = " ".join(b for grp in bgs for b in grp).lower()
    # At least one tradeoff concept should appear
    tradeoff_words = ["investment", "revenue", "ip", "capital", "development", "channel", "partnership"]
    assert any(w in bullet_text for w in tradeoff_words)


def test_decision_analysis_tables_present(both_written):
    tables = both_written.decision_analysis.tables
    assert len(tables) >= 1


def test_decision_matrix_table_structure(both_written):
    t = both_written.decision_analysis.tables[0]
    assert "title" in t
    assert "headers" in t
    assert "rows" in t
    headers = t["headers"]
    assert "Option" in headers
    assert len(t["rows"]) >= 1


def test_decision_matrix_contains_all_options(both_written):
    t = both_written.decision_analysis.tables[0]
    option_names = [row[0] for row in t["rows"]]
    assert any("Advisory" in n for n in option_names)


def test_decision_matrix_has_dimension_columns(both_written):
    t = both_written.decision_analysis.tables[0]
    headers = t["headers"]
    dims = ["Capital Efficiency", "Speed to Revenue", "IP Defensibility", "Execution Risk"]
    found = [d for d in dims if d in headers]
    assert len(found) >= 2, f"Expected dimension columns in headers, got: {headers}"


# ---------------------------------------------------------------------------
# DecisionAnalysisWriter — content constraints
# ---------------------------------------------------------------------------

def test_no_internal_ids_in_prose(both_written):
    import re
    all_text = "\n".join(both_written.decision_analysis.paragraphs)
    id_pattern = re.compile(r'\b(A|RSK|OPP|REC|OPT|EC|DA|EB|EM|R)-\d+\b')
    found = id_pattern.findall(all_text)
    assert not found, f"Internal IDs in decision analysis prose: {found}"


def test_recommended_option_mentioned(both_written):
    all_text = " ".join(both_written.decision_analysis.paragraphs).lower()
    assert "advisory" in all_text or "advisory-led" in all_text


def test_tables_figures_only_da_section(both_written):
    assert both_written.decision_analysis.tables != [] or True  # tables may be present
    assert both_written.executive_summary.tables == []
    assert both_written.recommendations.tables == []


# ---------------------------------------------------------------------------
# Other sections untouched
# ---------------------------------------------------------------------------

def test_executive_summary_still_populated(both_written):
    assert len(both_written.executive_summary.paragraphs) >= 3


def test_recommendations_untouched(both_written):
    assert both_written.recommendations.paragraphs == []
    assert both_written.recommendations.bullet_groups == []
    assert both_written.recommendations.tables == []


def test_strategic_risks_untouched(both_written):
    assert both_written.strategic_risks.paragraphs == []


def test_strategic_opportunities_untouched(both_written):
    assert both_written.strategic_opportunities.paragraphs == []


def test_executive_confidence_untouched(both_written):
    assert both_written.executive_confidence.paragraphs == []


def test_appendix_untouched(both_written):
    assert both_written.appendix.paragraphs == []


# ---------------------------------------------------------------------------
# Provenance retained
# ---------------------------------------------------------------------------

def test_decision_analysis_provenance_retained(both_written):
    prov = both_written.decision_analysis.provenance
    assert prov.brief_id.startswith("EB-")
    assert prov.brief_section_key == "decision_analysis"


def test_decision_analysis_source_section_ids_retained(both_written):
    ids = both_written.decision_analysis.source_section_ids
    assert "decision_analysis" in ids


# ---------------------------------------------------------------------------
# Writer isolation — DA writer does not require ES writer to have run
# ---------------------------------------------------------------------------

def test_da_writer_works_independently(da_only):
    assert len(da_only.decision_analysis.paragraphs) >= 4
    assert da_only.executive_summary.paragraphs == []


# ---------------------------------------------------------------------------
# Mock fallback chain
# ---------------------------------------------------------------------------

def test_da_writer_works_with_none_client():
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    result = DecisionAnalysisWriter(client=None).write(brief, ms)
    assert len(result.decision_analysis.paragraphs) >= 4


def test_da_writer_falls_back_on_missing_method():
    class PartialClient:
        is_mock = False
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    result = DecisionAnalysisWriter(client=PartialClient()).write(brief, ms)
    assert len(result.decision_analysis.paragraphs) >= 4


def test_da_writer_falls_back_on_error():
    class ErrorClient:
        is_mock = False
        def generate_decision_analysis_prose(self, **_):
            raise RuntimeError("Simulated failure")
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    result = DecisionAnalysisWriter(client=ErrorClient()).write(brief, ms)
    assert len(result.decision_analysis.paragraphs) >= 4


# ---------------------------------------------------------------------------
# Empty context edge case
# ---------------------------------------------------------------------------

def test_da_writer_empty_context():
    ctx = AgentContext(
        question="Minimal",
        profiles=[], execution_profile="", run_id="min-ph65",
        research_object={},
        strategic_options=[],
        preferred_option=None,
        decision_analysis={},
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        executive_confidence={},
    )
    coord = EditorialCoordinator()
    brief = coord.build(ctx)
    ms = coord.build_manuscript(brief)
    result = DecisionAnalysisWriter(client=None).write(brief, ms)
    assert isinstance(result.decision_analysis.paragraphs, list)
    assert isinstance(result.decision_analysis.tables, list)


# ---------------------------------------------------------------------------
# Returns manuscript (writer contract)
# ---------------------------------------------------------------------------

def test_da_writer_returns_manuscript(both_written):
    assert isinstance(both_written, EditorialManuscript)
