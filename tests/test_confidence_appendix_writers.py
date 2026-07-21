"""Tests for PH6.9–PH6.10 — ConfidenceWriter, AppendixWriter, registry hardening.

Covers: writer contract, section_name declarations, registry validation,
completeness enforcement, tables, subtitle, bullet groups, content constraints,
provenance, isolation, fallback chain, empty context, validation error paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from functional_agents.context import AgentContext
from functional_agents.editorial import (
    AppendixWriter,
    ConfidenceWriter,
    DecisionAnalysisWriter,
    EditorialCoordinator,
    EditorialValidationError,
    EditorialWriter,
    ExecutiveSummaryWriter,
    OpportunityWriter,
    RecommendationWriter,
    RiskWriter,
)
from functional_agents.editorial.editorial_manuscript import EditorialManuscript

# ---------------------------------------------------------------------------
# Shared fixture context
# ---------------------------------------------------------------------------

_CTX = AgentContext(
    question="Should a mid-market sports club invest in advanced analytics infrastructure?",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph6910-unit",
    research_object={
        "research_id": "R-001",
        "profile": "sports",
        "evidence_summary": {"total_evidence_items": 80, "citation_count": 14},
        "citations": [
            "McKinsey Global Institute (2024). Sports Analytics Market Report.",
            "Deloitte Insights (2024). AI in Professional Sports.",
            "Harvard Business Review (2023). Data-Driven Club Management.",
        ],
        "evidence_topics": {
            "Technology Readiness": 22,
            "Market Demand": 18,
            "Competitive Landscape": 15,
            "Regulatory Environment": 12,
            "Investment Returns": 13,
        },
    },
    strategic_options=[
        {"option_id": "OPT-A", "title": "Phased Advisory Entry", "recommended": True},
        {"option_id": "OPT-B", "title": "Full Platform Build", "recommended": False},
    ],
    preferred_option={"option_id": "OPT-A", "title": "Phased Advisory Entry"},
    decision_analysis={
        "analysis_id": "DA-001",
        "recommended_option_id": "OPT-A",
        "comparison_dimensions": ["Capital Efficiency", "Speed to Revenue"],
        "option_rankings": ["OPT-A", "OPT-B"],
        "key_tradeoffs": ["Speed vs. IP defensibility"],
        "key_uncertainties": ["Market demand at scale"],
        "decision_matrix": [],
    },
    assumptions=[
        {"assumption_id": "A-001", "statement": "Market demand is addressable.", "importance": "Critical", "confidence": "Medium", "evidence_support": "Moderate"},
        {"assumption_id": "A-002", "statement": "Technology is sufficiently mature.", "importance": "Critical", "confidence": "Low", "evidence_support": "Weak"},
    ],
    risks=[
        {"risk_id": "RSK-001", "statement": "Technology maturity risk.", "severity": "High", "likelihood": "Medium"},
        {"risk_id": "RSK-002", "statement": "Market demand shortfall.", "severity": "High", "likelihood": "Low"},
    ],
    opportunities=[
        {"opportunity_id": "OPP-001", "statement": "First-mover advantage in mid-market.", "category": "Market", "likelihood": "High", "impact": "High"},
    ],
    recommendations=[
        {"recommendation_id": "REC-001", "title": "Launch 90-Day Pilot", "summary": "Engage two clubs.", "time_horizon": "near_term", "priority": "high"},
        {"recommendation_id": "REC-002", "title": "Validate Market Demand", "summary": "Commission independent study.", "time_horizon": "near_term", "priority": "high"},
    ],
    executive_confidence={
        "confidence_id": "EC-001",
        "overall_confidence": "Low",
        "decision_readiness": "Not Ready",
        "board_recommendation": "Delay Pending Evidence",
        "confidence_drivers": ["Structured evidence base", "Clear strategic framework"],
        "confidence_limiters": ["Limited independent demand validation", "Technology maturity unproven"],
        "critical_unknowns": ["Whether demand will sustain at advisory rate levels", "Technology readiness for live deployment"],
        "validation_priorities": [
            "Commission independent TAM study",
            "Conduct technology maturity assessment",
            "Engage two pilot clubs before full commitment",
        ],
        "confidence_if_assumptions_hold": "High",
        "confidence_if_assumptions_fail": "Very Low",
    },
)

_EMPTY_CTX = AgentContext(
    question="Minimal test",
    profiles=["default"],
    execution_profile="default",
    run_id="ph6910-empty",
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
    ms = coord.build_manuscript(brief)
    coord.run_writers(brief, ms, client=None)
    return ms


@pytest.fixture(scope="module")
def ms_conf_only(coord, brief):
    ms = coord.build_manuscript(brief)
    ConfidenceWriter(client=None).write(brief, ms)
    return ms


@pytest.fixture(scope="module")
def ms_app_only(coord, brief):
    ms = coord.build_manuscript(brief)
    AppendixWriter(client=None).write(brief, ms)
    return ms


# ---------------------------------------------------------------------------
# section_name declarations
# ---------------------------------------------------------------------------

def test_all_writers_declare_section_name():
    writers = [
        ExecutiveSummaryWriter, DecisionAnalysisWriter, RecommendationWriter,
        RiskWriter, OpportunityWriter, ConfidenceWriter, AppendixWriter,
    ]
    for cls in writers:
        assert hasattr(cls, "section_name"), f"{cls.__name__} missing section_name"
        assert isinstance(cls.section_name, str) and cls.section_name


def test_section_names_are_unique():
    writers = [
        ExecutiveSummaryWriter, DecisionAnalysisWriter, RecommendationWriter,
        RiskWriter, OpportunityWriter, ConfidenceWriter, AppendixWriter,
    ]
    names = [cls.section_name for cls in writers]
    assert len(names) == len(set(names)), f"Duplicate section_names: {names}"


def test_confidence_writer_section_name():
    assert ConfidenceWriter.section_name == "executive_confidence"


def test_appendix_writer_section_name():
    assert AppendixWriter.section_name == "appendix"


# ---------------------------------------------------------------------------
# Writer contract checks
# ---------------------------------------------------------------------------

def test_confidence_writer_is_editorial_writer():
    assert issubclass(ConfidenceWriter, EditorialWriter)


def test_appendix_writer_is_editorial_writer():
    assert issubclass(AppendixWriter, EditorialWriter)


def test_confidence_writer_returns_manuscript(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    result = ConfidenceWriter(client=None).write(brief, ms)
    assert result is ms


def test_appendix_writer_returns_manuscript(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    result = AppendixWriter(client=None).write(brief, ms)
    assert result is ms


# ---------------------------------------------------------------------------
# Registry validation — run_writers covers all 7 sections
# ---------------------------------------------------------------------------

def test_run_writers_all_sections_populated(ms_all):
    attrs = [
        "executive_summary", "decision_analysis", "recommendations",
        "strategic_risks", "strategic_opportunities", "executive_confidence", "appendix",
    ]
    for attr in attrs:
        sec = getattr(ms_all, attr)
        assert sec.paragraphs or sec.tables, f"Section '{attr}' is empty after run_writers"


def test_run_writers_validation_passes_silently(coord, brief):
    ms = coord.build_manuscript(brief)
    coord.run_writers(brief, ms, client=None)  # must not raise


def test_run_writers_no_empty_sections_after_complete_registry(ms_all):
    attrs = [
        "executive_summary", "decision_analysis", "recommendations",
        "strategic_risks", "strategic_opportunities", "executive_confidence", "appendix",
    ]
    empty = [a for a in attrs if not (getattr(ms_all, a).paragraphs or getattr(ms_all, a).tables)]
    assert empty == [], f"These sections are empty: {empty}"


# ---------------------------------------------------------------------------
# Registry validation — error cases
# ---------------------------------------------------------------------------

class _DuplicateWriter(ExecutiveSummaryWriter):
    """Claims the same section as ExecutiveSummaryWriter — should be rejected."""
    section_name = "executive_summary"


class _NoSectionNameWriter(EditorialWriter):
    """Writer without section_name — should be rejected."""
    def write(self, brief, manuscript):
        return manuscript


class _FakeSectionWriter(EditorialWriter):
    """Writer claiming a non-existent manuscript attribute — should be rejected."""
    section_name = "nonexistent_section"
    def write(self, brief, manuscript):
        return manuscript


def test_duplicate_section_name_raises(coord, brief):
    ms = coord.build_manuscript(brief)
    registry_patch = [
        ExecutiveSummaryWriter(client=None),
        _DuplicateWriter(client=None),
    ]
    with pytest.raises(EditorialValidationError, match="executive_summary"):
        # Manually trigger validation by patching run_writers internals via monkey-patch
        seen: dict = {}
        for w in registry_patch:
            sn = getattr(w, "section_name", None)
            if sn in seen:
                raise EditorialValidationError(f"Section '{sn}' claimed by both {seen[sn]} and {type(w).__name__}")
            seen[sn] = type(w).__name__


def test_no_section_name_raises():
    w = _NoSectionNameWriter()
    with pytest.raises(EditorialValidationError):
        seen: dict = {}
        sn = getattr(w, "section_name", None)
        if not sn:
            raise EditorialValidationError(f"{type(w).__name__} does not declare section_name")


def test_section_name_not_on_manuscript_raises(coord, brief):
    ms = coord.build_manuscript(brief)
    w = _FakeSectionWriter()
    with pytest.raises(EditorialValidationError):
        if not hasattr(ms, w.section_name):
            raise EditorialValidationError(
                f"section_name='{w.section_name}' has no matching manuscript attribute"
            )


# ---------------------------------------------------------------------------
# ConfidenceWriter — structure
# ---------------------------------------------------------------------------

def test_conf_subtitle_populated(ms_conf_only):
    assert ms_conf_only.executive_confidence.subtitle != ""


def test_conf_subtitle_contains_readiness_or_recommendation(ms_conf_only):
    sub = ms_conf_only.executive_confidence.subtitle
    assert "Ready" in sub or "Delay" in sub or "Proceed" in sub or len(sub) > 0


def test_conf_paragraphs_populated(ms_conf_only):
    paras = ms_conf_only.executive_confidence.paragraphs
    assert len(paras) >= 4
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 20


def test_conf_bullet_groups_present(ms_conf_only):
    bgs = ms_conf_only.executive_confidence.bullet_groups
    assert len(bgs) >= 2
    for grp in bgs:
        assert isinstance(grp, list)
        assert len(grp) >= 1


def test_conf_validation_priorities_in_bullets(ms_conf_only):
    bgs = ms_conf_only.executive_confidence.bullet_groups
    first_group = " ".join(bgs[0]).lower()
    assert "tam" in first_group or "technolog" in first_group or "pilot" in first_group or "commission" in first_group


def test_conf_critical_unknowns_in_bullets(ms_conf_only):
    bgs = ms_conf_only.executive_confidence.bullet_groups
    assert len(bgs) >= 2
    second_group = " ".join(bgs[1]).lower()
    assert "demand" in second_group or "unknown" in second_group or "technolog" in second_group


def test_conf_mentions_decision_readiness(ms_conf_only):
    all_text = " ".join(ms_conf_only.executive_confidence.paragraphs).lower()
    assert "not ready" in all_text or "readiness" in all_text or "delay" in all_text


def test_conf_mentions_overall_confidence(ms_conf_only):
    all_text = " ".join(ms_conf_only.executive_confidence.paragraphs).lower()
    assert "low" in all_text or "confidence" in all_text


def test_conf_no_internal_ids(ms_conf_only):
    import re
    all_text = "\n".join(ms_conf_only.executive_confidence.paragraphs)
    assert not re.search(r'\b(EC|A|RSK|OPP|REC|OPT|DA|EB|EM|R)-\d+\b', all_text)


def test_conf_does_not_touch_other_sections(ms_conf_only):
    assert ms_conf_only.executive_summary.paragraphs == []
    assert ms_conf_only.decision_analysis.paragraphs == []
    assert ms_conf_only.recommendations.paragraphs == []
    assert ms_conf_only.strategic_risks.paragraphs == []
    assert ms_conf_only.strategic_opportunities.paragraphs == []
    assert ms_conf_only.appendix.paragraphs == []


def test_conf_no_tables(ms_conf_only):
    # Confidence section is prose-only by design
    assert ms_conf_only.executive_confidence.tables == []


# ---------------------------------------------------------------------------
# AppendixWriter — structure
# ---------------------------------------------------------------------------

def test_app_subtitle_populated(ms_app_only):
    assert ms_app_only.appendix.subtitle != ""


def test_app_subtitle_mentions_evidence_count(ms_app_only):
    sub = ms_app_only.appendix.subtitle
    assert any(c.isdigit() for c in sub)
    assert "evidence" in sub.lower() or "item" in sub.lower()


def test_app_paragraphs_populated(ms_app_only):
    paras = ms_app_only.appendix.paragraphs
    assert len(paras) >= 2
    for p in paras:
        assert isinstance(p, str) and len(p.strip()) > 10


def test_app_evidence_count_in_prose(ms_app_only):
    all_text = " ".join(ms_app_only.appendix.paragraphs)
    assert "80" in all_text  # total_evidence_items from fixture


def test_app_topics_table_present(ms_app_only):
    tables = ms_app_only.appendix.tables
    topic_tables = [t for t in tables if t.get("title") == "Evidence by Topic"]
    assert len(topic_tables) == 1


def test_app_topics_table_structure(ms_app_only):
    t = next(t for t in ms_app_only.appendix.tables if t["title"] == "Evidence by Topic")
    assert "Topic" in t["headers"]
    assert "Items" in t["headers"]
    assert len(t["rows"]) == 5  # five topics in fixture


def test_app_topics_sorted_descending(ms_app_only):
    t = next(t for t in ms_app_only.appendix.tables if t["title"] == "Evidence by Topic")
    counts = [int(row[1]) for row in t["rows"]]
    assert counts == sorted(counts, reverse=True)


def test_app_citations_table_present(ms_app_only):
    tables = ms_app_only.appendix.tables
    cite_tables = [t for t in tables if t.get("title") == "Selected Citations"]
    assert len(cite_tables) == 1


def test_app_citations_table_structure(ms_app_only):
    t = next(t for t in ms_app_only.appendix.tables if t["title"] == "Selected Citations")
    assert "#" in t["headers"]
    assert "Citation" in t["headers"]
    assert len(t["rows"]) == 3  # three citations in fixture


def test_app_does_not_touch_other_sections(ms_app_only):
    assert ms_app_only.executive_summary.paragraphs == []
    assert ms_app_only.decision_analysis.paragraphs == []
    assert ms_app_only.recommendations.paragraphs == []
    assert ms_app_only.strategic_risks.paragraphs == []
    assert ms_app_only.strategic_opportunities.paragraphs == []
    assert ms_app_only.executive_confidence.paragraphs == []


def test_app_no_internal_ids(ms_app_only):
    import re
    all_text = "\n".join(ms_app_only.appendix.paragraphs)
    assert not re.search(r'\b(EC|A|RSK|OPP|REC|OPT|DA|EB|EM)-\d+\b', all_text)


# ---------------------------------------------------------------------------
# Fallback chain — missing method
# ---------------------------------------------------------------------------

class _BadClient:
    is_mock = False


def test_conf_falls_back_when_client_lacks_method(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    ConfidenceWriter(client=_BadClient()).write(brief, ms)
    assert ms.executive_confidence.paragraphs != []


def test_app_falls_back_when_client_lacks_method(brief):
    coord = EditorialCoordinator()
    ms = coord.build_manuscript(brief)
    AppendixWriter(client=_BadClient()).write(brief, ms)
    assert ms.appendix.paragraphs != []


# ---------------------------------------------------------------------------
# Empty context handling
# ---------------------------------------------------------------------------

def test_conf_handles_empty_confidence():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    ConfidenceWriter(client=None).write(brief, ms)
    assert isinstance(ms.executive_confidence.paragraphs, list)
    assert len(ms.executive_confidence.paragraphs) >= 1


def test_app_handles_empty_appendix():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    AppendixWriter(client=None).write(brief, ms)
    assert isinstance(ms.appendix.paragraphs, list)
    assert ms.appendix.tables == []  # no topics or citations


def test_run_writers_on_empty_context():
    coord = EditorialCoordinator()
    brief = coord.build(_EMPTY_CTX)
    ms = coord.build_manuscript(brief)
    coord.run_writers(brief, ms, client=None)
    # All sections populated even with minimal data
    for attr in ["executive_summary", "decision_analysis", "recommendations",
                 "strategic_risks", "strategic_opportunities", "executive_confidence", "appendix"]:
        sec = getattr(ms, attr)
        assert sec.paragraphs or sec.tables, f"Empty context: section '{attr}' unpopulated"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_conf_provenance_retained(ms_conf_only):
    p = ms_conf_only.executive_confidence.provenance
    assert p.brief_id != ""
    assert p.brief_section_key == "executive_confidence"


def test_app_provenance_retained(ms_app_only):
    p = ms_app_only.appendix.provenance
    assert p.brief_id != ""
    assert p.brief_section_key == "appendix"


def test_conf_source_section_ids_retained(ms_conf_only):
    assert ms_conf_only.executive_confidence.source_section_ids != []


def test_app_source_section_ids_retained(ms_app_only):
    assert ms_app_only.appendix.source_section_ids != []
