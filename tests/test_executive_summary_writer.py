"""Tests for PH6.4 — ExecutiveSummaryWriter.

Covers: writer output structure, content constraints, provenance retention,
other sections untouched, mock fallback, empty-context edge case.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from functional_agents.context import AgentContext
from functional_agents.editorial import (
    EditorialCoordinator,
    ExecutiveSummaryWriter,
)
from functional_agents.editorial.editorial_manuscript import EditorialManuscript

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_CTX = AgentContext(
    question="Should Deloitte enter the sports analytics market?",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph64-unit",
    research_object={
        "research_id": "R-001",
        "profile": "sports",
        "evidence_summary": {"total_evidence_items": 50, "citation_count": 16},
    },
    strategic_options=[
        {
            "option_id": "OPT-A",
            "title": "Advisory-Led Entry",
            "description": "Enter through advisory services before building proprietary analytics tools.",
            "strategic_objective": "Capture early market share with minimal capital exposure.",
            "recommended": True,
        },
        {
            "option_id": "OPT-B",
            "title": "Product-First Entry",
            "description": "Build proprietary sports analytics platform before client engagement.",
            "strategic_objective": "Establish a defensible IP position.",
            "recommended": False,
        },
    ],
    preferred_option={"option_id": "OPT-A", "title": "Advisory-Led Entry"},
    decision_analysis={
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "option_rankings": ["OPT-A", "OPT-B"],
        "key_tradeoffs": [
            "Lower upfront investment vs. slower IP development",
            "Faster revenue vs. reduced long-term defensibility",
        ],
    },
    assumptions=[
        {
            "assumption_id": "A-001",
            "statement": "Market demand for sports analytics advisory is addressable at scale.",
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
            "statement": "First mover advantage in underpenetrated mid-market segment.",
            "category": "Market",
            "likelihood": "High",
            "impact": "High",
        }
    ],
    recommendations=[
        {
            "recommendation_id": "REC-001",
            "title": "Launch Pilot",
            "summary": "Conduct a 90-day advisory pilot with two anchor clients.",
            "time_horizon": "near_term",
            "priority": "high",
        }
    ],
    executive_confidence={
        "confidence_id": "EC-001",
        "overall_confidence": "Medium",
        "decision_readiness": "Needs Additional Validation",
        "board_recommendation": "Proceed with Conditions",
        "validation_priorities": ["Validate total addressable market size"],
        "confidence_limiters": ["Limited independent market data"],
        "critical_unknowns": ["Whether mid-market clients will pay advisory rates"],
    },
)


@pytest.fixture(scope="module")
def written_manuscript() -> EditorialManuscript:
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    manuscript = coord.build_manuscript(brief)
    writer = ExecutiveSummaryWriter(client=None)  # mock client
    return writer.write(brief, manuscript)


# ---------------------------------------------------------------------------
# Executive Summary section populated
# ---------------------------------------------------------------------------

def test_executive_summary_title_unchanged(written_manuscript):
    assert written_manuscript.executive_summary.title == "Executive Summary"


def test_executive_summary_subtitle_populated(written_manuscript):
    subtitle = written_manuscript.executive_summary.subtitle
    assert len(subtitle) > 0
    assert "Proceed with Conditions" in subtitle or "Needs Additional Validation" in subtitle


def test_executive_summary_paragraphs_populated(written_manuscript):
    paras = written_manuscript.executive_summary.paragraphs
    assert len(paras) >= 3, f"Expected ≥3 paragraphs, got {len(paras)}"
    for p in paras:
        assert isinstance(p, str)
        assert len(p.strip()) > 20, f"Paragraph too short: {p!r}"


def test_executive_summary_paragraphs_are_strings(written_manuscript):
    for p in written_manuscript.executive_summary.paragraphs:
        assert isinstance(p, str)


def test_executive_summary_bullet_groups_populated(written_manuscript):
    bgs = written_manuscript.executive_summary.bullet_groups
    assert len(bgs) >= 1, "Expected at least 1 bullet group"
    for grp in bgs:
        assert isinstance(grp, list)
        for b in grp:
            assert isinstance(b, str)
            assert len(b.strip()) > 0


def test_executive_summary_no_internal_ids_in_prose(written_manuscript):
    """Writer must not echo structured IDs (A-*, RSK-*, OPT-*, REC-*) into prose."""
    import re
    all_text = "\n".join(written_manuscript.executive_summary.paragraphs)
    id_pattern = re.compile(r'\b(A|RSK|OPP|REC|OPT|EC|DA)-\d+\b')
    found = id_pattern.findall(all_text)
    assert not found, f"Internal IDs found in executive summary prose: {found}"


def test_executive_summary_recommended_direction_mentioned(written_manuscript):
    all_text = " ".join(written_manuscript.executive_summary.paragraphs)
    assert "Advisory-Led" in all_text or "advisory" in all_text.lower()


def test_executive_summary_tables_figures_empty(written_manuscript):
    assert written_manuscript.executive_summary.tables == []
    assert written_manuscript.executive_summary.figures == []


# ---------------------------------------------------------------------------
# Provenance retained
# ---------------------------------------------------------------------------

def test_executive_summary_provenance_retained(written_manuscript):
    prov = written_manuscript.executive_summary.provenance
    assert prov.brief_id.startswith("EB-")
    assert prov.brief_section_key == "executive_summary"


def test_executive_summary_source_section_ids_retained(written_manuscript):
    assert "executive_summary" in written_manuscript.executive_summary.source_section_ids


# ---------------------------------------------------------------------------
# Other sections remain empty (writer must not touch them)
# ---------------------------------------------------------------------------

def test_decision_analysis_section_untouched(written_manuscript):
    sec = written_manuscript.decision_analysis
    assert sec.paragraphs == [], "decision_analysis.paragraphs should be empty"
    assert sec.bullet_groups == []
    assert sec.tables == []
    assert sec.figures == []


def test_recommendations_section_untouched(written_manuscript):
    sec = written_manuscript.recommendations
    assert sec.paragraphs == []
    assert sec.bullet_groups == []


def test_strategic_risks_section_untouched(written_manuscript):
    assert written_manuscript.strategic_risks.paragraphs == []


def test_strategic_opportunities_section_untouched(written_manuscript):
    assert written_manuscript.strategic_opportunities.paragraphs == []


def test_executive_confidence_section_untouched(written_manuscript):
    assert written_manuscript.executive_confidence.paragraphs == []


def test_appendix_section_untouched(written_manuscript):
    assert written_manuscript.appendix.paragraphs == []


# ---------------------------------------------------------------------------
# Mock client fallback
# ---------------------------------------------------------------------------

def test_writer_works_with_none_client():
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    writer = ExecutiveSummaryWriter(client=None)
    result = writer.write(brief, ms)
    assert len(result.executive_summary.paragraphs) >= 3


def test_writer_works_with_mock_client_flag():
    """Client with is_mock=True should trigger deterministic fallback."""
    class FakeMock:
        is_mock = True
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    writer = ExecutiveSummaryWriter(client=FakeMock())
    result = writer.write(brief, ms)
    assert len(result.executive_summary.paragraphs) >= 3


def test_writer_falls_back_on_client_without_method():
    """Client that lacks generate_executive_summary_prose falls back gracefully."""
    class PartialClient:
        is_mock = False
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    writer = ExecutiveSummaryWriter(client=PartialClient())
    result = writer.write(brief, ms)
    assert len(result.executive_summary.paragraphs) >= 3


def test_writer_falls_back_on_client_error():
    """Client that raises on generate_executive_summary_prose falls back gracefully."""
    class ErrorClient:
        is_mock = False
        def generate_executive_summary_prose(self, **_kwargs):
            raise RuntimeError("Simulated API failure")
    coord = EditorialCoordinator()
    brief = coord.build(_CTX)
    ms = coord.build_manuscript(brief)
    writer = ExecutiveSummaryWriter(client=ErrorClient())
    result = writer.write(brief, ms)
    assert len(result.executive_summary.paragraphs) >= 3


# ---------------------------------------------------------------------------
# Empty context edge case
# ---------------------------------------------------------------------------

def test_writer_with_empty_options():
    ctx = AgentContext(
        question="Minimal",
        profiles=[], execution_profile="", run_id="min-ph64",
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
    writer = ExecutiveSummaryWriter(client=None)
    result = writer.write(brief, ms)
    assert isinstance(result.executive_summary.paragraphs, list)
    assert len(result.executive_summary.paragraphs) >= 1


# ---------------------------------------------------------------------------
# Manuscript is returned (writer returns the manuscript object)
# ---------------------------------------------------------------------------

def test_writer_returns_manuscript(written_manuscript):
    assert isinstance(written_manuscript, EditorialManuscript)
