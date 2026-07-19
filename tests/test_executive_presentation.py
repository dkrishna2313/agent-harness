"""Tests for ExecutivePresentationGenerator and render_markdown."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from functional_agents.context import AgentContext
from functional_agents.executive_presentation import (
    ExecutivePresentationGenerator,
    _bullets,
    _find_option,
    _normalise_horizon,
    _strip_ids,
    _truncate,
    render_markdown,
)
from functional_agents.presentation_model import Presentation, Slide


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_context(**overrides) -> AgentContext:
    defaults = dict(
        question="Should Deloitte invest in Sports consulting?",
        profiles=["sports"],
        execution_profile="sports",
        engagement={
            "title": "Sports Strategy",
            "client": "Deloitte",
        },
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Platform-Led Strategy",
                "advantages": ["First-mover advantage in data analytics"],
                "disadvantages": ["High capital requirement"],
                "estimated_time_horizon": "long_term",
                "capital_intensity": "Very High",
                "confidence": "Medium",
            },
            {
                "option_id": "OPT-B",
                "title": "Governing-Body Anchor",
                "advantages": [
                    "Leverages existing Deloitte multi-service breadth",
                    "Provides near-term revenue certainty",
                    "Avoids binary execution risk of platform option",
                ],
                "disadvantages": ["Lower ceiling than platform"],
                "estimated_time_horizon": "near_term",
                "capital_intensity": "Medium",
                "confidence": "High",
            },
        ],
        preferred_option={"option_id": "OPT-B", "title": "Governing-Body Anchor"},
        decision_analysis={
            "analysis_id": "DA-001",
            "recommended_option_id": "OPT-B",
            "executive_summary": "OPT-B is preferred because it plays to confirmed capabilities.",
            "comparison_dimensions": ["Strategic Fit", "Capital", "Risk"],
            "option_rankings": ["OPT-B", "OPT-A"],
            "decision_matrix": [
                {
                    "option_id": "OPT-B",
                    "strategic_fit": "High",
                    "implementation_risk": "Medium",
                    "capital_requirement": "Medium",
                    "expected_return": "High",
                    "time_to_value": "High",
                },
                {
                    "option_id": "OPT-A",
                    "strategic_fit": "High",
                    "implementation_risk": "High",
                    "capital_requirement": "Very High",
                    "expected_return": "Very High",
                    "time_to_value": "Low",
                },
            ],
            "key_tradeoffs": [
                "Higher capital (OPT-A) means longer payback",
                "OPT-B provides near-term revenue certainty",
            ],
            "rationale": "OPT-B wins on strategic fit and assumption strength despite lower ceiling.",
            "confidence": "Medium",
        },
        assumptions=[
            {
                "assumption_id": "A-001",
                "statement": "Sports organisations are willing to share data with Deloitte",
                "importance": "Critical",
                "confidence": "Low",
                "evidence_support": "Weak",
            },
            {
                "assumption_id": "A-002",
                "statement": "Deloitte has a demonstrable right-to-win in governing-body mandates",
                "importance": "Critical",
                "confidence": "Low",
                "evidence_support": "Weak",
            },
        ],
        risks=[
            {
                "risk_id": "RSK-001",
                "statement": "Sports organisations refuse to share proprietary data",
                "severity": "High",
                "likelihood": "Medium",
                "mitigation_notes": "Structure data-sharing through federated or anonymised models",
            },
            {
                "risk_id": "RSK-002",
                "statement": "Deloitte fails to establish credible right-to-win",
                "severity": "High",
                "likelihood": "Medium",
                "mitigation_notes": "Accelerate capability acquisition via targeted M&A",
            },
        ],
        opportunities=[
            {
                "opportunity_id": "OPP-001",
                "statement": "First-mover advantage in data analytics could create structural moat",
                "category": "Market",
                "likelihood": "Medium",
                "impact": "High",
            },
            {
                "opportunity_id": "OPP-002",
                "statement": "Women's sports market growing rapidly with early-mover opportunity",
                "category": "Customer",
                "likelihood": "Medium",
                "impact": "High",
            },
        ],
        recommendations=[
            {
                "recommendation_id": "REC-001",
                "title": "Conduct governing-body discovery interviews",
                "summary": "Validate data-sharing appetite with three to five governing bodies",
                "time_horizon": "near_term",
                "priority": "high",
            },
            {
                "recommendation_id": "REC-002",
                "title": "Commission independent market-sizing study",
                "summary": "Validate addressable market before capital commitment",
                "time_horizon": "near_term",
                "priority": "high",
            },
        ],
        executive_confidence={
            "confidence_id": "EC-001",
            "overall_confidence": "Low",
            "decision_readiness": "Needs Additional Validation",
            "board_recommendation": "Proceed with Conditions",
            "confidence_rationale": "Two critical assumptions carry weak evidence.",
            "confidence_drivers": ["Deloitte's multi-service breadth is confirmed"],
            "confidence_limiters": ["Both critical assumptions carry weak evidence"],
            "validation_priorities": [
                "Conduct discovery interviews with three to five governing bodies",
                "Commission independent market-sizing study",
                "Assess Deloitte's governing-body advisory credentials",
            ],
            "confidence_if_assumptions_hold": "Medium",
            "confidence_if_assumptions_fail": "Low",
            "decision_horizon": "Within 60–90 days",
        },
    )
    defaults.update(overrides)
    return AgentContext(**defaults)


@pytest.fixture()
def ctx() -> AgentContext:
    return _make_context()


@pytest.fixture()
def pres(ctx) -> Presentation:
    return ExecutivePresentationGenerator().generate(ctx)


# ---------------------------------------------------------------------------
# Presentation structure
# ---------------------------------------------------------------------------

def test_generates_presentation(pres):
    assert isinstance(pres, Presentation)


def test_slide_count_within_range(pres):
    assert 7 <= len(pres.slides) <= 11


def test_slide_numbers_sequential(pres):
    for i, slide in enumerate(pres.slides, start=1):
        assert slide.slide_number == i


def test_presentation_title_set(pres):
    assert pres.title
    assert len(pres.title) > 0


def test_presentation_subtitle_is_question(pres, ctx):
    assert pres.subtitle == ctx.question


# ---------------------------------------------------------------------------
# Slide 1 — Title
# ---------------------------------------------------------------------------

def test_title_slide_is_first(pres):
    assert pres.slides[0].slide_type == "title"


def test_title_slide_mentions_recommended_option(pres):
    title_slide = pres.slides[0]
    assert "Governing-Body Anchor" in title_slide.key_message


# ---------------------------------------------------------------------------
# Bullet content rules
# ---------------------------------------------------------------------------

def test_no_slide_has_more_than_5_bullets(pres):
    for slide in pres.slides:
        assert len(slide.bullets) <= 5, f"Slide {slide.slide_number} has {len(slide.bullets)} bullets"


def test_all_bullets_within_15_words(pres):
    for slide in pres.slides:
        for bullet in slide.bullets:
            word_count = len(bullet.split())
            assert word_count <= 16, (
                f"Slide {slide.slide_number} bullet too long ({word_count} words): {bullet!r}"
            )


def test_no_raw_ids_in_bullets(pres):
    for slide in pres.slides:
        for bullet in slide.bullets:
            assert "A-00" not in bullet
            assert "RSK-0" not in bullet
            assert "OPT-" not in bullet
            assert "EC-0" not in bullet
            assert "REC-0" not in bullet


def test_no_raw_ids_in_titles(pres):
    for slide in pres.slides:
        assert "A-00" not in slide.title
        assert "RSK-0" not in slide.title


def test_no_raw_ids_in_key_messages(pres):
    for slide in pres.slides:
        assert "A-00" not in slide.key_message
        assert "RSK-0" not in slide.key_message


# ---------------------------------------------------------------------------
# Titles are declarative (not generic labels)
# ---------------------------------------------------------------------------

def test_recommendation_slide_title_is_declarative(pres):
    rec_slide = next(s for s in pres.slides if s.slide_type == "content" and s.slide_number > 1)
    # Title should not just be a generic label
    assert rec_slide.title not in ("Strategic Options", "Recommendation", "Risks", "Assumptions")
    assert len(rec_slide.title.split()) >= 4  # at least 4 words = a statement


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def test_options_slide_has_table(pres):
    options_slide = next((s for s in pres.slides if s.slide_type == "comparison" and "Paths" in s.title), None)
    assert options_slide is not None
    assert options_slide.table is not None


def test_options_table_has_recommended_marker(pres):
    options_slide = next(s for s in pres.slides if "Paths" in s.title)
    cells = [cell for row in options_slide.table.rows for cell in row.cells]
    assert "✓" in cells


def test_risks_slide_has_table(pres):
    risk_slide = next((s for s in pres.slides if "Risk" in s.title), None)
    assert risk_slide is not None
    assert risk_slide.table is not None


def test_assumptions_slide_has_table(pres):
    assump_slide = next((s for s in pres.slides if "Assumption" in s.title), None)
    assert assump_slide is not None
    assert assump_slide.table is not None


def test_confidence_slide_has_scenario_table(pres):
    conf_slide = next((s for s in pres.slides if "Confidence" in s.title), None)
    assert conf_slide is not None
    assert conf_slide.table is not None


# ---------------------------------------------------------------------------
# Graceful degradation (minimal context)
# ---------------------------------------------------------------------------

def test_empty_context_produces_presentation():
    ctx = AgentContext()
    pres = ExecutivePresentationGenerator().generate(ctx)
    assert isinstance(pres, Presentation)
    assert len(pres.slides) >= 1  # at least a title slide


def test_no_options_skips_options_slide():
    ctx = _make_context(strategic_options=[], preferred_option={})
    pres = ExecutivePresentationGenerator().generate(ctx)
    titles = [s.title for s in pres.slides]
    assert not any("Paths" in t for t in titles)


def test_no_risks_skips_risk_slide():
    ctx = _make_context(risks=[])
    pres = ExecutivePresentationGenerator().generate(ctx)
    titles = [s.title for s in pres.slides]
    assert not any("Risk" in t for t in titles)


def test_no_assumptions_skips_assumptions_slide():
    ctx = _make_context(assumptions=[])
    pres = ExecutivePresentationGenerator().generate(ctx)
    titles = [s.title for s in pres.slides]
    assert not any("Assumption" in t for t in titles)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def test_render_markdown_produces_string(pres):
    md = render_markdown(pres)
    assert isinstance(md, str)
    assert len(md) > 100


def test_markdown_has_slide_headings(pres):
    md = render_markdown(pres)
    assert "## Slide 1" in md
    assert "## Slide 2" in md


def test_markdown_has_no_raw_ids(pres):
    md = render_markdown(pres)
    import re
    ids = re.findall(r"\b(?:A|RSK|OPP|REC|EC|DA)-[0-9]{3}\b", md)
    assert ids == [], f"Raw IDs found in markdown: {ids}"


def test_markdown_has_tables(pres):
    md = render_markdown(pres)
    assert "|" in md  # at least one pipe table


def test_markdown_has_blockquotes(pres):
    md = render_markdown(pres)
    assert "> " in md  # blockquote for key messages


def test_markdown_no_raw_option_ids_in_tables(pres):
    md = render_markdown(pres)
    # OPT-A and OPT-B should not appear directly (option_id should be replaced by title + rank)
    import re
    opt_ids = re.findall(r"\bOPT-[A-Z]\b", md)
    assert opt_ids == [], f"Raw option IDs found: {opt_ids}"


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------

def test_truncate_short_text():
    assert _truncate("short text") == "short text"


def test_truncate_long_text():
    long = "word " * 20
    result = _truncate(long.strip())
    assert len(result.split()) <= 16  # 15 words + ellipsis counts as one


def test_truncate_adds_ellipsis():
    long = " ".join([f"w{i}" for i in range(20)])
    result = _truncate(long)
    assert result.endswith("…")


def test_strip_ids_removes_assumption_ids():
    assert "A-001" not in _strip_ids("Assumption A-001 is critical")


def test_strip_ids_removes_risk_ids():
    assert "RSK-002" not in _strip_ids("Risk RSK-002 applies here")


def test_bullets_caps_at_five():
    result = _bullets(["item"] * 10)
    assert len(result) <= 5


def test_bullets_truncates_each():
    long_item = " ".join([f"word{i}" for i in range(30)])
    result = _bullets([long_item])
    assert len(result[0].split()) <= 16


def test_find_option_returns_correct():
    options = [{"option_id": "OPT-A", "title": "Alpha"}, {"option_id": "OPT-B", "title": "Beta"}]
    assert _find_option(options, "OPT-B")["title"] == "Beta"


def test_find_option_returns_empty_for_missing():
    assert _find_option([], "OPT-X") == {}


def test_normalise_horizon():
    assert _normalise_horizon("near_term") == "3–12 months"
    assert _normalise_horizon("medium_term") == "1–3 years"
    assert _normalise_horizon("long_term") == "3+ years"
    assert _normalise_horizon("custom") == "custom"
    assert _normalise_horizon("") == "—"


# ---------------------------------------------------------------------------
# Smoke test on real session (skipped if absent)
# ---------------------------------------------------------------------------

_SESSIONS_DIR = Path("outputs/sessions")


@pytest.mark.skipif(
    not _SESSIONS_DIR.exists() or not any(_SESSIONS_DIR.glob("SS-*.json")),
    reason="No session files found in outputs/sessions/",
)
def test_real_session_smoke(tmp_path):
    from functional_agents.executive_presentation import context_from_session, _latest_session

    session_path = _latest_session(_SESSIONS_DIR)
    assert session_path is not None

    ctx  = context_from_session(session_path)
    pres = ExecutivePresentationGenerator().generate(ctx)
    md   = render_markdown(pres)

    out = tmp_path / "presentation.md"
    out.write_text(md, encoding="utf-8")

    assert out.exists()
    assert len(pres.slides) >= 2
    assert "## Slide 1" in md
