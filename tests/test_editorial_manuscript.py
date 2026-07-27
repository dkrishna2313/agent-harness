"""Tests for PH6.3 — EditorialManuscript scaffold.

Validates: structure, empty content, provenance chain,
JSON serialisation, and persistence.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from functional_agents.context import AgentContext
from functional_agents.editorial import (
    AppendixManuscriptSection,
    ConfidenceManuscriptSection,
    DecisionAnalysisManuscriptSection,
    EditorialCoordinator,
    EditorialManuscript,
    ExecutiveSummaryManuscriptSection,
    ManuscriptMetadata,
    ManuscriptProvenance,
    ManuscriptSection,
    OpportunityManuscriptSection,
    RecommendationManuscriptSection,
    RiskManuscriptSection,
)

# ---------------------------------------------------------------------------
# Minimal context fixture
# ---------------------------------------------------------------------------

_MINIMAL_CTX = AgentContext(
    question="Sports analytics market entry",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph63-unit",
    research_object={
        "research_id": "R-001",
        "profile": "sports",
        "evidence_summary": {"total_evidence_items": 50, "citation_count": 16},
    },
    strategic_options=[
        {
            "option_id": "OPT-A",
            "title": "Advisory-Led",
            "description": "Advisory first.",
            "strategic_objective": "Capture share.",
            "recommended": True,
        }
    ],
    preferred_option={"option_id": "OPT-A", "title": "Advisory-Led"},
    decision_analysis={
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "option_rankings": ["OPT-A"],
        "key_tradeoffs": ["Speed vs risk"],
    },
    assumptions=[
        {
            "assumption_id": "A-001",
            "statement": "Market is addressable.",
            "importance": "Critical",
            "confidence": "Medium",
            "evidence_support": "Moderate",
        }
    ],
    risks=[
        {
            "risk_id": "RSK-001",
            "statement": "Execution risk.",
            "severity": "High",
            "likelihood": "Medium",
        }
    ],
    opportunities=[
        {
            "opportunity_id": "OPP-001",
            "statement": "First mover.",
            "category": "Market",
            "likelihood": "High",
            "impact": "High",
        }
    ],
    recommendations=[
        {
            "recommendation_id": "REC-001",
            "title": "Pilot",
            "summary": "90-day pilot.",
            "time_horizon": "near_term",
            "priority": "high",
        }
    ],
    executive_confidence={
        "confidence_id": "EC-001",
        "overall_confidence": "Low",
        "decision_readiness": "Needs Additional Validation",
        "board_recommendation": "Proceed with Conditions",
        "validation_priorities": ["Validate market size"],
        "confidence_limiters": ["Limited client data"],
    },
)

_REQUIRED_SECTION_FIELDS = {
    "title",
    "subtitle",
    "paragraphs",
    "bullet_groups",
    "tables",
    "figures",
    "provenance",
    "source_section_ids",
}


@pytest.fixture(scope="module")
def manuscript() -> EditorialManuscript:
    coord = EditorialCoordinator()
    brief = coord.build(_MINIMAL_CTX)
    return coord.build_manuscript(brief)


@pytest.fixture(scope="module")
def manuscript_dict(manuscript) -> dict:
    return manuscript.to_dict()


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------

def test_manuscript_has_metadata(manuscript):
    assert isinstance(manuscript.metadata, ManuscriptMetadata)
    assert manuscript.metadata.manuscript_id.startswith("EM-")
    assert manuscript.metadata.brief_id.startswith("EB-")


def test_manuscript_has_all_seven_sections(manuscript):
    assert isinstance(manuscript.executive_summary, ExecutiveSummaryManuscriptSection)
    assert isinstance(manuscript.decision_analysis, DecisionAnalysisManuscriptSection)
    assert isinstance(manuscript.recommendations, RecommendationManuscriptSection)
    assert isinstance(manuscript.strategic_risks, RiskManuscriptSection)
    assert isinstance(manuscript.strategic_opportunities, OpportunityManuscriptSection)
    assert isinstance(manuscript.executive_confidence, ConfidenceManuscriptSection)
    assert isinstance(manuscript.appendix, AppendixManuscriptSection)


def test_all_sections_inherit_manuscript_section(manuscript):
    sections = [
        manuscript.executive_summary,
        manuscript.decision_analysis,
        manuscript.recommendations,
        manuscript.strategic_risks,
        manuscript.strategic_opportunities,
        manuscript.executive_confidence,
        manuscript.appendix,
    ]
    for sec in sections:
        assert isinstance(sec, ManuscriptSection), f"{type(sec)} does not inherit ManuscriptSection"


def test_all_sections_have_required_fields(manuscript):
    sections = [
        ("executive_summary", manuscript.executive_summary),
        ("decision_analysis", manuscript.decision_analysis),
        ("recommendations", manuscript.recommendations),
        ("strategic_risks", manuscript.strategic_risks),
        ("strategic_opportunities", manuscript.strategic_opportunities),
        ("executive_confidence", manuscript.executive_confidence),
        ("appendix", manuscript.appendix),
    ]
    for name, sec in sections:
        actual = {f.name for f in dataclasses.fields(sec)}
        missing = _REQUIRED_SECTION_FIELDS - actual
        assert not missing, f"Section {name!r} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Titles
# ---------------------------------------------------------------------------

def test_section_titles(manuscript):
    assert manuscript.executive_summary.title == "Executive Summary"
    assert manuscript.decision_analysis.title == "Strategic Analysis"
    assert manuscript.recommendations.title == "Immediate Actions"
    assert manuscript.strategic_risks.title == "Key Risks"
    assert manuscript.strategic_opportunities.title == "Strategic Opportunities"
    assert manuscript.executive_confidence.title == "Decision Readiness"
    assert manuscript.appendix.title == "Supporting Evidence"


# ---------------------------------------------------------------------------
# Empty content (scaffold only — writers populate in PH6.4+)
# ---------------------------------------------------------------------------

def test_all_section_content_empty(manuscript):
    sections = [
        manuscript.executive_summary,
        manuscript.decision_analysis,
        manuscript.recommendations,
        manuscript.strategic_risks,
        manuscript.strategic_opportunities,
        manuscript.executive_confidence,
        manuscript.appendix,
    ]
    for sec in sections:
        assert sec.paragraphs == [], f"{type(sec).__name__}.paragraphs is not empty"
        assert sec.bullet_groups == [], f"{type(sec).__name__}.bullet_groups is not empty"
        assert sec.tables == [], f"{type(sec).__name__}.tables is not empty"
        assert sec.figures == [], f"{type(sec).__name__}.figures is not empty"


# ---------------------------------------------------------------------------
# No prose / markdown / formatting
# ---------------------------------------------------------------------------

def test_no_markdown_in_serialised_manuscript(manuscript_dict):
    import re
    text = json.dumps(manuscript_dict)
    # Exclude title fields from markdown scan (they contain plain section names)
    # We look for typical markdown: fenced headers, bold, table pipes
    data_values_only = json.dumps({
        k: v for k, v in manuscript_dict.items() if k != "metadata"
    })
    markers = re.findall(r'#{1,6}\s|\*\*|\|\s*\w+\s*\|', data_values_only)
    assert not markers, f"Markdown markers found in manuscript: {markers}"


def test_no_docx_pptx_references_in_serialised_manuscript(manuscript_dict):
    text = json.dumps(manuscript_dict).lower()
    assert "docx" not in text
    assert "pptx" not in text
    assert "run_bold" not in text
    assert "paragraph_format" not in text


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_all_sections_have_provenance(manuscript):
    sections = [
        manuscript.executive_summary,
        manuscript.decision_analysis,
        manuscript.recommendations,
        manuscript.strategic_risks,
        manuscript.strategic_opportunities,
        manuscript.executive_confidence,
        manuscript.appendix,
    ]
    for sec in sections:
        assert hasattr(sec, "provenance"), f"{type(sec).__name__} missing provenance"
        assert isinstance(sec.provenance, ManuscriptProvenance)


def test_provenance_has_brief_id(manuscript):
    brief_id = manuscript.metadata.brief_id
    sections = [
        manuscript.executive_summary,
        manuscript.decision_analysis,
        manuscript.recommendations,
        manuscript.strategic_risks,
        manuscript.strategic_opportunities,
        manuscript.executive_confidence,
        manuscript.appendix,
    ]
    for sec in sections:
        assert sec.provenance.brief_id == brief_id, (
            f"{type(sec).__name__}.provenance.brief_id expected {brief_id!r}, "
            f"got {sec.provenance.brief_id!r}"
        )


def test_provenance_has_section_key(manuscript):
    expected_keys = {
        "executive_summary": "executive_summary",
        "decision_analysis": "decision_analysis",
        "recommendations": "recommendations",
        "strategic_risks": "strategic_risks",
        "strategic_opportunities": "strategic_opportunities",
        "executive_confidence": "executive_confidence",
        "appendix": "appendix",
    }
    for attr, expected_key in expected_keys.items():
        sec = getattr(manuscript, attr)
        assert sec.provenance.brief_section_key == expected_key, (
            f"{attr}.provenance.brief_section_key expected {expected_key!r}, "
            f"got {sec.provenance.brief_section_key!r}"
        )


def test_decision_analysis_source_section_ids(manuscript):
    assert set(manuscript.decision_analysis.source_section_ids) >= {"decision_analysis", "strategic_options"}


def test_executive_confidence_source_section_ids(manuscript):
    assert set(manuscript.executive_confidence.source_section_ids) >= {"executive_confidence", "validation_priorities"}


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_returns_dict(manuscript):
    d = manuscript.to_dict()
    assert isinstance(d, dict)


def test_to_dict_has_all_top_level_keys(manuscript_dict):
    expected_keys = {
        "metadata",
        "executive_summary",
        "decision_analysis",
        "recommendations",
        "strategic_risks",
        "strategic_opportunities",
        "executive_confidence",
        "appendix",
        "strategic_direction",  # PH11.4: optional strategy section scaffold
    }
    assert expected_keys == set(manuscript_dict.keys())


def test_to_dict_is_json_serialisable(manuscript):
    raw = json.dumps(manuscript.to_dict())
    parsed = json.loads(raw)
    assert parsed["metadata"]["manuscript_id"] == manuscript.metadata.manuscript_id


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_persist_creates_versioned_file(manuscript):
    coord = EditorialCoordinator()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        path = coord.persist_manuscript(manuscript, base=base)
        assert path.exists()
        assert path.name == f"{manuscript.metadata.manuscript_id}.json"
        data = json.loads(path.read_text())
        assert data["metadata"]["manuscript_id"] == manuscript.metadata.manuscript_id


def test_persist_creates_latest_file(manuscript):
    coord = EditorialCoordinator()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        coord.persist_manuscript(manuscript, base=base, write_latest=True)
        latest = base / "latest_editorial_manuscript.json"
        assert latest.exists()
        data = json.loads(latest.read_text())
        assert data["metadata"]["manuscript_id"] == manuscript.metadata.manuscript_id


def test_persist_no_latest_when_disabled(manuscript):
    coord = EditorialCoordinator()
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        coord.persist_manuscript(manuscript, base=base, write_latest=False)
        latest = base / "latest_editorial_manuscript.json"
        assert not latest.exists()


# ---------------------------------------------------------------------------
# Empty context edge case
# ---------------------------------------------------------------------------

def test_empty_optional_fields_in_context():
    """Coordinator must not raise when optional context fields are absent."""
    ctx = AgentContext(
        question="Minimal test",
        profiles=[],
        execution_profile="",
        run_id="min-001",
        research_object={},
        strategic_options=[],
        preferred_option=None,
        decision_analysis={},
        assumptions=[],
        risks=[],
        opportunities=[],
        recommendations=[],
        executive_confidence={},
    )
    coord = EditorialCoordinator()
    brief = coord.build(ctx)
    manuscript = coord.build_manuscript(brief)
    assert isinstance(manuscript, EditorialManuscript)
    assert len(manuscript.metadata.manuscript_id) > 0
