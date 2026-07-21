"""PH6.2b — EditorialBrief schema and persistence tests.

Covers:
  - Prose fields absent from to_dict() output (structural contract)
  - Required structured fields present in every section
  - Provenance populated for every section
  - Section item counts match source data
  - top_risk_id resolved to highest-severity risk
  - critical_count matches Critical assumptions
  - option_rankings duplicate absent from StrategicOptionsSection
  - to_dict() is JSON-serialisable
  - Persistence: versioned file and latest_editorial_brief.json created
  - Empty context: coordinator does not raise
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from functional_agents.context import AgentContext
from functional_agents.editorial import EditorialBrief, EditorialCoordinator
from functional_agents.editorial.editorial_brief import SectionProvenance


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_context() -> AgentContext:
    return AgentContext(
        question="Should we enter the sports analytics market?",
        profiles=["sports", "ai_data_centers"],
        execution_profile="sports",
        run_id="test-run-ph62b",
        decision_model={"decision_model_id": "DM-001"},
        research_object={
            "research_id": "R-001",
            "profile": "sports",
            "evidence_summary": {"total_evidence_items": 50, "citation_count": 16},
            "evidence_topics": {"Sports Analytics": 20, "Data Platforms": 10},
        },
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Advisory-Led Entry",
                "description": "Lead with advisory services.",
                "strategic_objective": "Capture advisory market share.",
                "expected_outcomes": ["Margin expansion", "Client acquisition"],
                "advantages": ["Low capital"],
                "disadvantages": ["Slow scale"],
                "implementation_complexity": "Low",
                "estimated_time_horizon": "Near-term",
                "capital_intensity": "Low",
                "confidence": "High",
                "recommended": True,
                "rationale": "This is a prose paragraph that should NOT appear in the brief.",
                "supporting_assumption_ids": ["A-001"],
                "associated_risk_ids": ["RSK-001"],
                "associated_opportunity_ids": ["OPP-001"],
                "supporting_recommendation_ids": ["REC-001"],
            },
            {
                "option_id": "OPT-B",
                "title": "Platform-First",
                "description": "Build proprietary data platform first.",
                "strategic_objective": "Own the data layer.",
                "recommended": False,
                "rationale": "Another prose paragraph that should NOT appear in the brief.",
            },
        ],
        preferred_option={"option_id": "OPT-A", "title": "Advisory-Led Entry"},
        decision_analysis={
            "analysis_id": "DA-001",
            "recommended_option_id": "OPT-A",
            "executive_summary": "OPT-A is preferred. This is a prose paragraph.",
            "comparison_dimensions": ["Capital", "Speed", "Risk"],
            "option_rankings": ["OPT-A", "OPT-B"],
            "key_tradeoffs": ["Speed vs capital", "Risk vs return"],
            "key_uncertainties": ["Competitor response", "Regulatory path"],
            "sensitivity_analysis": "This is a sensitivity paragraph that should NOT appear.",
            "confidence_summary": "This is a confidence-summary paragraph that should NOT appear.",
            "rationale": "This is a rationale paragraph that should NOT appear.",
            "decision_matrix": [
                {"option_id": "OPT-A", "strategic_fit": "High", "overall_score": "High"},
            ],
        },
        assumptions=[
            {
                "assumption_id": "A-001",
                "statement": "Market is addressable at scale.",
                "importance": "Critical",
                "confidence": "Medium",
                "evidence_support": "Moderate",
            },
            {
                "assumption_id": "A-002",
                "statement": "Regulatory environment stays stable.",
                "importance": "Important",
                "confidence": "High",
                "evidence_support": "Strong",
            },
        ],
        risks=[
            {
                "risk_id": "RSK-001",
                "statement": "Execution risk from talent gaps.",
                "severity": "High",
                "likelihood": "Medium",
                "mitigation_notes": "Hire specialist team.",
                "related_assumption_ids": ["A-001"],
                "affected_recommendation_ids": ["REC-001"],
            },
            {
                "risk_id": "RSK-002",
                "statement": "Market adoption slower than forecast.",
                "severity": "Medium",
                "likelihood": "High",
                "mitigation_notes": "Phased rollout with pilots.",
            },
        ],
        opportunities=[
            {
                "opportunity_id": "OPP-001",
                "statement": "First mover advantage in sports analytics.",
                "category": "Market",
                "likelihood": "High",
                "impact": "High",
                "rationale": "This is a rationale paragraph that should NOT appear in the brief.",
                "related_assumption_ids": ["A-001"],
                "enabled_recommendation_ids": ["REC-001"],
            },
        ],
        recommendations=[
            {
                "recommendation_id": "REC-001",
                "title": "Initiate advisory pilot",
                "summary": "Run a 90-day pilot with two anchor clients.",
                "time_horizon": "near_term",
                "priority": "high",
                "supported_assumption_ids": ["A-001"],
                "affected_risk_ids": ["RSK-001"],
            },
        ],
        executive_confidence={
            "confidence_id": "EC-001",
            "overall_confidence": "Low",
            "decision_readiness": "Needs Additional Validation",
            "board_recommendation": "Proceed with Conditions",
            "confidence_rationale": "This is a rationale paragraph that should NOT appear.",
            "confidence_drivers": ["Strong advisory heritage"],
            "confidence_limiters": ["Limited client validation to date"],
            "critical_unknowns": ["Competitor response timeline"],
            "validation_priorities": ["Validate market size with 15 client interviews"],
            "confidence_if_assumptions_hold": "High",
            "confidence_if_assumptions_fail": "Low",
            "decision_horizon": "Q4 2026",
        },
    )


@pytest.fixture
def ctx() -> AgentContext:
    return _make_context()


@pytest.fixture
def brief(ctx) -> EditorialBrief:
    return EditorialCoordinator().build(ctx)


# ---------------------------------------------------------------------------
# Prose field exclusion
# ---------------------------------------------------------------------------

def test_no_prose_fields_in_dict(brief):
    """Prose paragraphs must be absent from every section's serialised dict."""
    d = brief.to_dict()

    # ExecutiveSummarySection: these prose fields were removed in PH6.2b
    es = d["executive_summary"]
    assert "why_this_option" not in es, "why_this_option (prose) must not appear in ExecutiveSummarySection"
    assert "executive_summary_prose" not in es, "executive_summary_prose must not appear in ExecutiveSummarySection"

    # DecisionAnalysisSection: prose paragraphs removed
    da = d["decision_analysis"]
    assert "executive_summary" not in da, "executive_summary paragraph must not appear in DecisionAnalysisSection"
    assert "sensitivity_analysis" not in da, "sensitivity_analysis paragraph must not appear in DecisionAnalysisSection"
    assert "confidence_summary" not in da, "confidence_summary paragraph must not appear in DecisionAnalysisSection"
    assert "rationale" not in da, "rationale paragraph must not appear in DecisionAnalysisSection"

    # StrategicOptionEntry: rationale removed
    for opt in d["strategic_options"]["options"]:
        assert "rationale" not in opt, f"rationale must not appear in StrategicOptionEntry ({opt.get('option_id')})"

    # OpportunityEntry: rationale removed
    for opp in d["strategic_opportunities"]["opportunities"]:
        assert "rationale" not in opp, f"rationale must not appear in OpportunityEntry ({opp.get('opportunity_id')})"

    # ConfidenceSection: confidence_rationale removed
    conf = d["executive_confidence"]
    assert "confidence_rationale" not in conf, "confidence_rationale must not appear in ConfidenceSection"


# ---------------------------------------------------------------------------
# Structural presence
# ---------------------------------------------------------------------------

def test_all_top_level_sections_present(brief):
    d = brief.to_dict()
    expected = {
        "metadata", "executive_summary", "decision_analysis", "strategic_options",
        "recommendations", "strategic_assumptions", "strategic_risks",
        "strategic_opportunities", "executive_confidence", "validation_priorities",
        "appendix",
    }
    assert set(d.keys()) == expected


def test_executive_summary_fields(brief):
    es = brief.executive_summary
    assert es.recommended_option_id == "OPT-A"
    assert es.recommended_option_title == "Advisory-Led Entry"
    assert es.board_recommendation == "Proceed with Conditions"
    assert es.decision_readiness == "Needs Additional Validation"
    assert es.overall_confidence == "Low"
    assert es.key_conditions == ["Limited client validation to date"]
    assert es.critical_unknowns == ["Competitor response timeline"]
    assert "REC-001" in es.supporting_recommendation_ids


def test_decision_analysis_fields(brief):
    da = brief.decision_analysis
    assert da.analysis_id == "DA-001"
    assert da.recommended_option_id == "OPT-A"
    assert da.option_rankings == ["OPT-A", "OPT-B"]
    assert "Speed vs capital" in da.key_tradeoffs
    assert "Competitor response" in da.key_uncertainties
    assert len(da.decision_matrix) == 1
    assert "Capital" in da.comparison_dimensions


def test_strategic_options_no_rankings_field(brief):
    d = brief.to_dict()
    assert "option_rankings" not in d["strategic_options"], (
        "StrategicOptionsSection should not duplicate option_rankings — "
        "authoritative copy is in decision_analysis"
    )


def test_strategic_options_count(brief):
    assert len(brief.strategic_options.options) == 2


def test_recommended_option_flagged(brief):
    opts = {o.option_id: o for o in brief.strategic_options.options}
    assert opts["OPT-A"].recommended is True
    assert opts["OPT-B"].recommended is False


def test_option_rationale_absent(brief):
    for opt in brief.strategic_options.options:
        d = {f.name: getattr(opt, f.name) for f in opt.__dataclass_fields__.values()}
        assert "rationale" not in d, "StrategicOptionEntry must not have rationale field"


def test_risks_count(brief):
    assert len(brief.strategic_risks.risks) == 2


def test_top_risk_is_high_severity(brief):
    assert brief.strategic_risks.top_risk_id == "RSK-001"


def test_assumptions_critical_count(brief):
    assert brief.strategic_assumptions.critical_count == 1


def test_opportunities_count(brief):
    assert len(brief.strategic_opportunities.opportunities) == 1


def test_opportunity_rationale_absent(brief):
    for opp in brief.strategic_opportunities.opportunities:
        d = {f.name: getattr(opp, f.name) for f in opp.__dataclass_fields__.values()}
        assert "rationale" not in d, "OpportunityEntry must not have rationale field"


def test_recommendations_count(brief):
    assert len(brief.recommendations.recommendations) == 1


def test_validation_priorities(brief):
    assert "Validate market size" in brief.validation_priorities.priorities[0]


def test_confidence_rationale_absent(brief):
    d = {
        f.name: getattr(brief.executive_confidence, f.name)
        for f in brief.executive_confidence.__dataclass_fields__.values()
    }
    assert "confidence_rationale" not in d


def test_appendix_evidence_counts(brief):
    app = brief.appendix
    assert app.research_object_id == "R-001"
    assert app.total_evidence_items == 50
    assert app.citation_count == 16
    assert "Sports Analytics" in app.evidence_topics


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_executive_summary_provenance(brief):
    prov = brief.executive_summary.provenance
    assert isinstance(prov, SectionProvenance)
    assert prov.analysis_id == "DA-001"
    assert prov.confidence_id == "EC-001"
    assert "REC-001" in prov.recommendation_ids


def test_decision_analysis_provenance(brief):
    prov = brief.decision_analysis.provenance
    assert prov.analysis_id == "DA-001"
    assert "OPT-A" in prov.option_ids
    assert "OPT-B" in prov.option_ids


def test_risks_provenance(brief):
    prov = brief.strategic_risks.provenance
    assert "RSK-001" in prov.risk_ids
    assert "RSK-002" in prov.risk_ids
    assert "A-001" in prov.assumption_ids


def test_opportunities_provenance(brief):
    prov = brief.strategic_opportunities.provenance
    assert "OPP-001" in prov.opportunity_ids


def test_assumptions_provenance(brief):
    prov = brief.strategic_assumptions.provenance
    assert "A-001" in prov.assumption_ids
    assert "A-002" in prov.assumption_ids


def test_confidence_provenance(brief):
    prov = brief.executive_confidence.provenance
    assert prov.confidence_id == "EC-001"
    assert "RSK-001" in prov.risk_ids


def test_appendix_provenance(brief):
    prov = brief.appendix.provenance
    assert prov.research_object_id == "R-001"


def test_all_sections_have_provenance(brief):
    sections = [
        brief.executive_summary,
        brief.decision_analysis,
        brief.strategic_options,
        brief.recommendations,
        brief.strategic_assumptions,
        brief.strategic_risks,
        brief.strategic_opportunities,
        brief.executive_confidence,
        brief.validation_priorities,
        brief.appendix,
    ]
    for section in sections:
        assert hasattr(section, "provenance"), (
            f"{type(section).__name__} is missing provenance field"
        )
        assert isinstance(section.provenance, SectionProvenance)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def test_to_dict_is_json_serialisable(brief):
    d = brief.to_dict()
    text = json.dumps(d)
    assert len(text) > 100
    loaded = json.loads(text)
    assert loaded["metadata"]["brief_id"].startswith("EB-")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_persist_creates_versioned_and_latest(ctx):
    coord = EditorialCoordinator()
    brief = coord.build(ctx)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        path = coord.persist(brief, base=base)
        assert path.exists()
        assert path.name.startswith("EB-")
        latest = base / "latest_editorial_brief.json"
        assert latest.exists()
        data = json.loads(latest.read_text())
        assert data["metadata"]["brief_id"] == brief.metadata.brief_id


def test_persist_write_latest_false(ctx):
    coord = EditorialCoordinator()
    brief = coord.build(ctx)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        coord.persist(brief, base=base, write_latest=False)
        assert not (base / "latest_editorial_brief.json").exists()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_context_does_not_raise():
    ctx = AgentContext(
        question="Edge case question",
        profiles=["sports"],
        execution_profile="sports",
        research_object={"research_id": "R-000", "profile": "sports"},
    )
    brief = EditorialCoordinator().build(ctx)
    assert brief.strategic_risks.top_risk_id == ""
    assert brief.strategic_assumptions.critical_count == 0
    assert brief.validation_priorities.priorities == []
    # Must still serialise cleanly
    json.dumps(brief.to_dict())
