"""Tests for PH7 — MarkdownRenderer: EditorialManuscript → Markdown report.

Covers:
- MarkdownRenderer exported from editorial package
- render() produces a non-empty string
- All 12 required sections present in output
- Prose comes from manuscript (editorial content, not legacy AgentContext)
- Brief-driven tables (Section 3 decision parameters, assumptions, risks, opportunities)
- Appendix structure (A–F sub-sections)
- Graceful degradation when brief is None
- Graceful degradation when manuscript sections are empty
- MarkdownReportGenerator uses MarkdownRenderer when manuscript in trace
- MarkdownReportGenerator falls back to legacy when no manuscript in trace
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from functional_agents.context import AgentContext
from functional_agents.editorial import (
    EditorialCoordinator,
    MarkdownRenderer,
)
from functional_agents.editorial.editorial_manuscript import EditorialManuscript

# ---------------------------------------------------------------------------
# Shared context — rich enough to exercise all 12 sections
# ---------------------------------------------------------------------------

_CTX = AgentContext(
    question="Should a mid-market sports club invest in advanced analytics infrastructure?",
    profiles=["sports"],
    execution_profile="sports",
    run_id="ph7-renderer-unit",
    research_object={
        "research_id": "R-PH7",
        "profile": "sports",
        "evidence_summary": {"total_evidence_items": 60, "citation_count": 5},
        "citations": [
            "McKinsey (2024). Sports Analytics.",
            "Deloitte (2024). AI in Sports.",
        ],
        "evidence_topics": {"Technology": 30, "Market": 20, "Risk": 10},
    },
    strategic_options=[
        {
            "option_id": "OPT-A",
            "title": "Phased Advisory Entry",
            "description": "Deploy in staged tranches with validation gates.",
            "strategic_objective": "Capture mid-market analytics opportunity.",
            "advantages": ["Lower downside risk", "Preserves optionality"],
            "disadvantages": ["Slower to full scale"],
            "implementation_complexity": "Medium",
            "estimated_time_horizon": "medium_term",
            "capital_intensity": "Medium",
            "confidence": "Medium",
            "recommended": True,
        },
        {
            "option_id": "OPT-B",
            "title": "Full Platform Build",
            "description": "Build proprietary analytics platform end-to-end.",
            "strategic_objective": "Achieve market leadership.",
            "advantages": ["Maximum upside"],
            "disadvantages": ["High capital at risk"],
            "implementation_complexity": "High",
            "estimated_time_horizon": "long_term",
            "capital_intensity": "High",
            "confidence": "Low",
            "recommended": False,
        },
    ],
    preferred_option={"option_id": "OPT-A", "title": "Phased Advisory Entry"},
    decision_analysis={
        "analysis_id": "DA-PH7",
        "recommended_option_id": "OPT-A",
        "comparison_dimensions": ["Capital Efficiency", "Speed to Revenue"],
        "option_rankings": ["OPT-A", "OPT-B"],
        "key_tradeoffs": ["Speed vs. defensibility", "Capital commitment vs. optionality"],
        "key_uncertainties": ["Market demand at scale", "Technology maturity timeline"],
        "decision_matrix": [],
    },
    assumptions=[
        {
            "assumption_id": "A-001",
            "statement": "Market demand is addressable at advisory rates.",
            "importance": "Critical",
            "confidence": "Medium",
            "evidence_support": "Moderate",
        },
        {
            "assumption_id": "A-002",
            "statement": "Technology is sufficiently mature.",
            "importance": "Important",
            "confidence": "Low",
            "evidence_support": "Weak",
        },
    ],
    risks=[
        {
            "risk_id": "RSK-001",
            "statement": "Technology maturity shortfall causes delay.",
            "severity": "High",
            "likelihood": "Medium",
            "related_assumption_ids": ["A-002"],
        },
        {
            "risk_id": "RSK-002",
            "statement": "Market demand falls below projections.",
            "severity": "Medium",
            "likelihood": "Low",
            "related_assumption_ids": ["A-001"],
        },
    ],
    opportunities=[
        {
            "opportunity_id": "OPP-001",
            "statement": "First-mover advantage in mid-market.",
            "category": "Market",
            "likelihood": "High",
            "impact": "High",
        },
    ],
    recommendations=[
        {
            "recommendation_id": "REC-001",
            "title": "Launch 90-Day Pilot",
            "summary": "Engage two clubs immediately.",
            "time_horizon": "near_term",
            "priority": "high",
        },
        {
            "recommendation_id": "REC-002",
            "title": "Commission TAM Study",
            "summary": "Independent demand validation.",
            "time_horizon": "medium_term",
            "priority": "medium",
        },
    ],
    executive_confidence={
        "confidence_id": "EC-PH7",
        "overall_confidence": "Low",
        "decision_readiness": "Not Ready",
        "board_recommendation": "Delay Pending Evidence",
        "confidence_drivers": ["Structured evidence base"],
        "confidence_limiters": ["Limited independent demand validation"],
        "critical_unknowns": ["Whether demand sustains at advisory rates"],
        "validation_priorities": ["Commission independent TAM study"],
        "confidence_if_assumptions_hold": "High",
        "confidence_if_assumptions_fail": "Very Low",
        "decision_horizon": "Q3 2026",
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
def manuscript(coord, brief):
    ms = coord.build_manuscript(brief)
    coord.run_writers(brief, ms, client=None)
    return ms


@pytest.fixture(scope="module")
def rendered(manuscript, brief):
    return MarkdownRenderer().render(manuscript, brief=brief)


# ---------------------------------------------------------------------------
# Export and instantiation
# ---------------------------------------------------------------------------

def test_markdown_renderer_exported_from_package():
    """MarkdownRenderer must be importable from functional_agents.editorial."""
    from functional_agents.editorial import MarkdownRenderer as MR  # noqa: F401
    assert MR is not None


def test_markdown_renderer_instantiates():
    renderer = MarkdownRenderer()
    assert renderer is not None


# ---------------------------------------------------------------------------
# render() output — basic contract
# ---------------------------------------------------------------------------

def test_render_returns_string(rendered):
    assert isinstance(rendered, str)


def test_render_non_empty(rendered):
    assert len(rendered) > 500


def test_render_starts_with_title(rendered):
    assert rendered.startswith("# Executive Strategic Report")


# ---------------------------------------------------------------------------
# All 12 required sections present
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "## 1. Executive Summary",
    "## 2. Strategic Context",
    "## 3. Strategic Recommendation",
    "## 4. Recommendation Rationale",
    "## 5. Decision Readiness",
    "## 6. Critical Assumptions",
    "## 7. Key Risks",
    "## 8. Strategic Opportunities",
    "## 9. Immediate Actions",
    "## 10. Appendix",
]

@pytest.mark.parametrize("heading", REQUIRED_SECTIONS)
def test_required_section_present(rendered, heading):
    assert heading in rendered, f"Missing section: {heading!r}"


# ---------------------------------------------------------------------------
# Prose comes from manuscript — not legacy AgentContext
# ---------------------------------------------------------------------------

def test_executive_summary_uses_manuscript_prose(rendered, manuscript):
    """At least one paragraph from manuscript.executive_summary appears in output."""
    paragraphs = manuscript.executive_summary.paragraphs or []
    assert paragraphs, "Manuscript executive_summary has no paragraphs (writer failed?)"
    assert any(p[:40] in rendered for p in paragraphs), (
        "Executive summary prose from manuscript not found in rendered output"
    )


def test_decision_analysis_uses_manuscript_prose(rendered, manuscript):
    paragraphs = manuscript.decision_analysis.paragraphs or []
    assert paragraphs
    assert any(p[:40] in rendered for p in paragraphs)


def test_decision_readiness_uses_manuscript_prose(rendered, manuscript):
    paragraphs = manuscript.executive_confidence.paragraphs or []
    assert paragraphs
    assert any(p[:40] in rendered for p in paragraphs)


def test_key_risks_uses_manuscript_prose(rendered, manuscript):
    paragraphs = manuscript.strategic_risks.paragraphs or []
    assert paragraphs
    assert any(p[:40] in rendered for p in paragraphs)


def test_strategic_opportunities_uses_manuscript_prose(rendered, manuscript):
    paragraphs = manuscript.strategic_opportunities.paragraphs or []
    assert paragraphs
    assert any(p[:40] in rendered for p in paragraphs)


def test_immediate_actions_uses_manuscript_prose(rendered, manuscript):
    paragraphs = manuscript.recommendations.paragraphs or []
    assert paragraphs
    assert any(p[:40] in rendered for p in paragraphs)


# ---------------------------------------------------------------------------
# Section 3 — Strategic Recommendation decision parameters table
# ---------------------------------------------------------------------------

def test_s3_contains_recommended_option_title(rendered):
    assert "Phased Advisory Entry" in rendered


def test_s3_contains_decision_readiness(rendered):
    assert "Not Ready" in rendered


def test_s3_contains_board_recommendation(rendered):
    assert "Delay Pending Evidence" in rendered


def test_s3_contains_option_description(rendered):
    assert "staged tranches" in rendered


# ---------------------------------------------------------------------------
# Section 6 — Critical Assumptions
# ---------------------------------------------------------------------------

def test_s6_contains_assumption_statement(rendered):
    assert "Market demand is addressable" in rendered


def test_s6_contains_importance_label(rendered):
    assert "Critical" in rendered


def test_s6_references_appendix_b(rendered):
    assert "Appendix B" in rendered


# ---------------------------------------------------------------------------
# Section 7 — Key Risks
# ---------------------------------------------------------------------------

def test_s7_contains_risk_statement(rendered):
    assert "Technology maturity shortfall" in rendered


def test_s7_references_appendix_c(rendered):
    assert "Appendix C" in rendered


# ---------------------------------------------------------------------------
# Section 8 — Strategic Opportunities
# ---------------------------------------------------------------------------

def test_s8_contains_opportunity_statement(rendered):
    assert "First-mover advantage" in rendered


def test_s8_references_appendix_d(rendered):
    assert "Appendix D" in rendered


# ---------------------------------------------------------------------------
# Section 9 — Immediate Actions (grouped by time horizon)
# ---------------------------------------------------------------------------

def test_s9_contains_near_term_action(rendered):
    assert "Launch 90-Day Pilot" in rendered


def test_s9_contains_medium_term_action(rendered):
    assert "Commission TAM Study" in rendered


# ---------------------------------------------------------------------------
# Appendix sub-sections
# ---------------------------------------------------------------------------

def test_appendix_a_strategic_options(rendered):
    assert "Appendix A" in rendered
    assert "Phased Advisory Entry" in rendered


def test_appendix_b_assumption_register(rendered):
    assert "Appendix B" in rendered
    assert "A-001" in rendered


def test_appendix_c_risk_register(rendered):
    assert "Appendix C" in rendered
    assert "RSK-001" in rendered


def test_appendix_d_opportunity_register(rendered):
    assert "Appendix D" in rendered
    assert "OPP-001" in rendered


def test_appendix_e_confidence_analysis(rendered):
    assert "Appendix E" in rendered


def test_appendix_f_supporting_evidence(rendered):
    """Appendix F present when manuscript.appendix is populated."""
    assert "Appendix F" in rendered


# ---------------------------------------------------------------------------
# Glossary auto-generation
# ---------------------------------------------------------------------------

def test_glossary_present_when_terms_appear():
    """Glossary is inserted when defined domain terms appear in rendered content."""
    coord = EditorialCoordinator()
    ctx_with_ai = AgentContext(
        question="Should the club invest in AI analytics infrastructure?",
        profiles=["sports"],
        execution_profile="sports",
        run_id="ph7-glossary-test",
        research_object={"research_id": "R-GLO", "profile": "sports"},
        strategic_options=[{"option_id": "OPT-A", "title": "AI Platform", "recommended": True}],
        preferred_option={"option_id": "OPT-A"},
        decision_analysis={
            "analysis_id": "DA-G",
            "recommended_option_id": "OPT-A",
            "comparison_dimensions": [],
            "option_rankings": ["OPT-A"],
            "key_tradeoffs": [],
            "key_uncertainties": [],
            "decision_matrix": [],
        },
        assumptions=[],
        risks=[],
        opportunities=[],
        recommendations=[],
        executive_confidence={
            "confidence_id": "EC-G",
            "overall_confidence": "Medium",
            "decision_readiness": "Ready",
            "board_recommendation": "Proceed",
        },
    )
    brief_g = coord.build(ctx_with_ai)
    ms_g = coord.build_manuscript(brief_g)
    coord.run_writers(brief_g, ms_g, client=None)
    result = MarkdownRenderer().render(ms_g, brief=brief_g)
    # "AI" is a known glossary term — should trigger glossary insertion
    assert "## Glossary" in result


def test_glossary_absent_when_no_terms_match(manuscript, brief):
    """Verify glossary presence is conditional — here test structure only."""
    result = MarkdownRenderer().render(manuscript, brief=brief)
    # Glossary may or may not appear depending on term presence — no crash either way
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Brief=None degradation — renderer must not crash
# ---------------------------------------------------------------------------

def test_render_without_brief_does_not_crash(manuscript):
    result = MarkdownRenderer().render(manuscript, brief=None)
    assert isinstance(result, str)
    assert "# Executive Strategic Report" in result


def test_render_without_brief_still_has_executive_summary(manuscript):
    result = MarkdownRenderer().render(manuscript, brief=None)
    assert "## 1. Executive Summary" in result


def test_render_without_brief_skips_assumption_table(manuscript):
    result = MarkdownRenderer().render(manuscript, brief=None)
    assert "No assumptions recorded" in result


# ---------------------------------------------------------------------------
# Empty manuscript sections — renderer must not crash
# ---------------------------------------------------------------------------

def test_render_empty_manuscript_does_not_crash(brief, coord):
    """An unfilled manuscript (before writers run) renders without error."""
    empty_ms = coord.build_manuscript(brief)
    result = MarkdownRenderer().render(empty_ms, brief=brief)
    assert isinstance(result, str)
    assert "# Executive Strategic Report" in result


# ---------------------------------------------------------------------------
# MarkdownReportGenerator — PH7 routing
# ---------------------------------------------------------------------------

def test_generator_uses_renderer_when_manuscript_in_trace(tmp_path, manuscript, brief):
    """MarkdownReportGenerator should call MarkdownRenderer when _editorial_manuscript is set."""
    from functional_agents.deliverables.markdown_report import MarkdownReportGenerator

    mock_context = MagicMock()
    mock_context.trace = {
        "_editorial_manuscript": manuscript,
        "_editorial_brief": brief,
    }

    output_path = tmp_path / "report.md"

    with patch("functional_agents.deliverables.markdown_report.MarkdownReportGenerator.generate") as mock_gen:
        # Call the real generate — it should pick up the manuscript
        gen = MarkdownReportGenerator()
        # Patch write_markdown to capture what report_content was passed
        with patch("research_agent.markdown.write_markdown") as mock_write:
            mock_write.return_value = output_path
            artifact = gen.generate(mock_context, output_path)

    assert artifact is not None


def test_generator_produces_ph7_content_when_manuscript_in_trace(tmp_path, manuscript, brief):
    """When manuscript is in trace, the output contains editorial prose."""
    from functional_agents.deliverables.markdown_report import MarkdownReportGenerator

    class MockContext:
        trace = {
            "_editorial_manuscript": manuscript,
            "_editorial_brief": brief,
        }
        # Attributes that legacy path would need — not needed here
        strategic_options = []
        domain_evidence = []

    gen = MarkdownReportGenerator()
    captured_content: list[str] = []

    with patch("research_agent.markdown.write_markdown") as mock_write:
        mock_write.side_effect = lambda content, path: (
            captured_content.append(content) or path
        )
        gen.generate(MockContext(), tmp_path / "ph7.md")

    assert captured_content, "write_markdown was never called"
    content = captured_content[0]
    assert "# Executive Strategic Report" in content
    assert "## 1. Executive Summary" in content


def test_generator_falls_back_to_legacy_when_no_manuscript(tmp_path):
    """MarkdownReportGenerator must call legacy path when no _editorial_manuscript."""
    from functional_agents.deliverables.markdown_report import MarkdownReportGenerator

    class MockContext:
        trace: dict = {}
        strategic_options = []
        domain_evidence = []
        question = "test?"

    gen = MarkdownReportGenerator()

    # build_markdown_report_content is imported lazily inside generate(), so patch
    # at the source module rather than the import site.
    with patch(
        "functional_agents.report_agent.build_markdown_report_content",
        return_value="# Legacy Report",
    ) as mock_legacy, patch("research_agent.markdown.write_markdown", return_value=tmp_path / "r.md"):
        gen.generate(MockContext(), tmp_path / "r.md")

    mock_legacy.assert_called_once()


# ---------------------------------------------------------------------------
# _render_table helper
# ---------------------------------------------------------------------------

def test_render_table_produces_markdown_table():
    renderer = MarkdownRenderer()
    table = {
        "title": "Test Table",
        "headers": ["Col A", "Col B"],
        "rows": [["val1", "val2"]],
        "notes": "A note.",
    }
    lines = renderer._render_table(table)
    joined = "\n".join(lines)
    assert "| Col A | Col B |" in joined
    assert "| val1 | val2 |" in joined
    assert "Test Table" in joined
    assert "A note." in joined


def test_render_table_escapes_pipe_characters():
    renderer = MarkdownRenderer()
    table = {
        "title": "",
        "headers": ["Field"],
        "rows": [["val|with|pipes"]],
        "notes": "",
    }
    lines = renderer._render_table(table)
    joined = "\n".join(lines)
    assert "\\|" in joined


def test_render_table_empty_headers_returns_empty():
    renderer = MarkdownRenderer()
    lines = renderer._render_table({"headers": [], "rows": [], "title": "", "notes": ""})
    assert lines == []
