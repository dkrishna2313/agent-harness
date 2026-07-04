"""Tests for StrategyDeckGenerator (J11.3 + J12.2).

J11.3 coverage: registration, all 12 slide headers, content correctness,
slide separator, no-duplicate-reasoning invariant, graceful degradation,
AST constraint, bundle with 3 deliverable types, ReportAgent regression.

J12.2 coverage: generator consumes ExecutiveNarrative, module imports
ExecutiveNarrativeBuilder, slide renderers have ExecutiveNarrative signature,
no direct reasoning field access in slide builders, supporting_evidence and
portfolio actions rendered, context.executive_narrative set as side effect.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from functional_agents.context import AgentContext
from functional_agents.deliverables import (
    DeliverableBundleGenerator,
    DeliverableRegistry,
    StrategyDeckGenerator,
    default_registry,
)
from functional_agents.deliverables.strategy_deck import build_strategy_deck_content
from research_agent.schemas import ResearchMemo

_SLIDE_HEADERS = [
    "# Slide 1 — Executive Decision",
    "# Slide 2 — Client Situation",
    "# Slide 3 — Executive Summary",
    "# Slide 4 — Strategic Options",
    "# Slide 5 — Decision Matrix",
    "# Slide 6 — Strategic Risks",
    "# Slide 7 — Strategic Opportunities",
    "# Slide 8 — Critical Assumptions",
    "# Slide 9 — Executive Confidence",
    "# Slide 10 — Immediate Actions",
    "# Slide 11 — Supporting Evidence",
]


def _full_ctx() -> AgentContext:
    ctx = AgentContext(
        question="Should we invest in AI infrastructure?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R-SD-001",
                         "decision_architecture": {"decision_statement": "Invest now or defer?"}},
        engagement={"title": "AI Strategy", "client": "Acme Corp", "industry": "Tech",
                    "decision_horizon": "12 months",
                    "constraints": ["Budget cap $500M", "18-month lead time"]},
        decision_architecture={"decision_statement": "Invest now or defer?",
                                "decision_horizon": "12 months"},
        strategic_synthesis={"executive_summary": "AI infrastructure investment is strategic."},
        decision_analysis={
            "recommended_option_id": "OPT-001",
            "executive_summary": "Invest now for competitive advantage.",
            "rationale": "Market timing is optimal.",
            "comparison_dimensions": ["Cost", "Speed", "Risk"],
            "option_rankings": ["OPT-001", "OPT-002", "OPT-003"],
        },
        strategic_options=[
            {"option_id": "OPT-001", "title": "Full Build",
             "description": "Build end-to-end AI platform.",
             "estimated_time_horizon": "near_term", "capital_intensity": "high",
             "confidence": "high",
             "advantages": ["Fast time to market", "Full control"],
             "disadvantages": ["High capex"]},
            {"option_id": "OPT-002", "title": "Partnership",
             "description": "Partner with hyperscaler.",
             "estimated_time_horizon": "medium_term", "capital_intensity": "moderate",
             "confidence": "medium",
             "advantages": ["Lower risk"], "disadvantages": ["Less control"]},
        ],
        preferred_option={"option_id": "OPT-001", "title": "Full Build"},
        risks=[
            {"risk_id": "R-001", "statement": "Cost overruns likely.", "severity": "high",
             "likelihood": "medium", "mitigation": "Fixed-price contracts."},
            {"risk_id": "R-002", "statement": "Talent shortage.", "severity": "critical",
             "likelihood": "high", "mitigation": "Hire early."},
        ],
        assumptions=[
            {"assumption_id": "A-001", "statement": "Demand grows 30% YoY.",
             "importance": "critical", "confidence": "high"},
            {"assumption_id": "A-002", "statement": "Regulatory approval secured.",
             "importance": "important", "confidence": "medium"},
        ],
        opportunities=[
            {"opportunity_id": "OPP-001", "title": "First-mover advantage",
             "description": "Lead the market before competitors.", "impact": "high"},
        ],
        executive_confidence={
            "overall_confidence": "high",
            "decision_readiness": "ready",
            "board_recommendation": "proceed",
            "confidence_rationale": "Strong fundamentals.",
            "validation_priorities": ["Validate cost model", "Confirm talent pipeline"],
            "critical_unknowns": ["Regulatory timeline"],
        },
        recommendations=[
            {"id": "REC-001", "recommendation_id": "REC-001", "title": "Approve Phase 1 budget"},
            {"id": "REC-002", "recommendation_id": "REC-002", "title": "Hire AI lead"},
            {"id": "REC-003", "recommendation_id": "REC-003", "title": "Partner evaluation"},
        ],
        recommendation_portfolio={
            "near_term": ["REC-001", "REC-002"],
            "medium_term": ["REC-003"],
            "long_term": [],
        },
        hypotheses=[
            {"id": "H-001", "title": "AI demand will grow 30% YoY",
             "confidence": "high"},
        ],
        surviving_hypotheses=[
            {"id": "H-001", "title": "AI demand will grow 30% YoY",
             "confidence": "high"},
        ],
    )
    ctx.trace["_report_memo"] = ResearchMemo(
        title="SD Report", question="Q?", executive_summary="Summary."
    )
    return ctx


def _empty_ctx() -> AgentContext:
    ctx = AgentContext(
        question="Q?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R-EMPTY"},
    )
    ctx.trace["_report_memo"] = ResearchMemo(
        title="T", question="Q?", executive_summary="S."
    )
    return ctx


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_strategy_deck_generator_registered_on_default_registry():
    gen = default_registry.get("strategy_deck")
    assert isinstance(gen, StrategyDeckGenerator)


def test_strategy_deck_generator_declares_its_type():
    assert StrategyDeckGenerator.deliverable_type == "strategy_deck"


def test_all_three_generators_registered_on_default_registry():
    assert default_registry.get("markdown")
    assert default_registry.get("executive_brief")
    assert default_registry.get("strategy_deck")


# ---------------------------------------------------------------------------
# Slide structure
# ---------------------------------------------------------------------------

def test_all_11_required_slide_headers_present():
    ctx = _full_ctx()
    content = build_strategy_deck_content(ctx)
    for header in _SLIDE_HEADERS:
        assert header in content, f"Missing slide: {header!r}"


def test_document_title_present():
    content = build_strategy_deck_content(_full_ctx())
    assert content.startswith("# Strategy Deck")


def test_slide_separator_present():
    content = build_strategy_deck_content(_full_ctx())
    assert "---" in content


def test_slides_in_correct_order():
    content = build_strategy_deck_content(_full_ctx())
    positions = [content.index(h) for h in _SLIDE_HEADERS]
    assert positions == sorted(positions), "Slides are out of order"


# ---------------------------------------------------------------------------
# Slide content correctness
# ---------------------------------------------------------------------------

def test_slide_01_contains_decision_statement():
    content = build_strategy_deck_content(_full_ctx())
    assert "Invest now or defer?" in content


def test_slide_01_contains_recommended_option():
    content = build_strategy_deck_content(_full_ctx())
    assert "OPT-001" in content


def test_slide_02_contains_client_and_industry():
    content = build_strategy_deck_content(_full_ctx())
    assert "Acme Corp" in content
    assert "Tech" in content


def test_slide_02_contains_decision_horizon():
    content = build_strategy_deck_content(_full_ctx())
    assert "12 months" in content


def test_slide_02_contains_constraints():
    content = build_strategy_deck_content(_full_ctx())
    assert "Budget cap $500M" in content


def test_slide_03_contains_executive_summary():
    content = build_strategy_deck_content(_full_ctx())
    assert "AI infrastructure investment is strategic." in content


def test_slide_04_contains_options_table():
    content = build_strategy_deck_content(_full_ctx())
    assert "Full Build" in content
    assert "Partnership" in content


def test_slide_04_marks_recommended_option():
    content = build_strategy_deck_content(_full_ctx())
    assert "✓" in content


def test_slide_05_contains_comparison_dimensions():
    content = build_strategy_deck_content(_full_ctx())
    assert "Cost" in content
    assert "Speed" in content
    assert "Risk" in content


def test_slide_06_contains_risks_sorted_by_severity():
    ctx = _full_ctx()
    content = build_strategy_deck_content(ctx)
    pos_critical = content.index("R-002")
    pos_high = content.index("R-001")
    assert pos_critical < pos_high, "Critical risk should appear before high risk"


def test_slide_06_contains_mitigation():
    content = build_strategy_deck_content(_full_ctx())
    assert "Fixed-price contracts" in content


def test_slide_07_contains_opportunities():
    content = build_strategy_deck_content(_full_ctx())
    assert "OPP-001" in content
    assert "First-mover advantage" in content


def test_slide_08_contains_assumptions_sorted_by_importance():
    ctx = _full_ctx()
    content = build_strategy_deck_content(ctx)
    pos_critical = content.index("A-001")
    pos_important = content.index("A-002")
    assert pos_critical < pos_important


def test_slide_09_contains_confidence_and_board_recommendation():
    content = build_strategy_deck_content(_full_ctx())
    assert "proceed" in content
    assert "Validate cost model" in content


def test_slide_10_contains_near_term_recommendations():
    content = build_strategy_deck_content(_full_ctx())
    assert "REC-001" in content
    assert "Approve Phase 1 budget" in content


def test_slide_10_contains_medium_term_recommendations():
    content = build_strategy_deck_content(_full_ctx())
    assert "REC-003" in content


def test_slide_11_uses_surviving_hypotheses_when_available():
    content = build_strategy_deck_content(_full_ctx())
    assert "H-001" in content
    assert "AI demand will grow 30% YoY" in content


# ---------------------------------------------------------------------------
# Appendix (optional)
# ---------------------------------------------------------------------------

def test_appendix_present_when_profiles_set():
    content = build_strategy_deck_content(_full_ctx())
    assert "# Slide 12 — Appendix" in content


def test_appendix_omitted_when_no_profiles_or_unknowns():
    # profiles=[] + no executive_confidence → appendix condition is false → omitted
    ctx = AgentContext(question="Q?", profiles=[], execution_profile="",
                       research_object={"research_id": "R"})
    content = build_strategy_deck_content(ctx)
    assert "# Slide 12 — Appendix" not in content


# ---------------------------------------------------------------------------
# Graceful degradation
# ---------------------------------------------------------------------------

def test_strategy_deck_degrades_gracefully_on_empty_context():
    ctx = _empty_ctx()
    content = build_strategy_deck_content(ctx)
    assert "# Strategy Deck" in content
    for header in _SLIDE_HEADERS:
        assert header in content, f"Empty-context degradation missing: {header!r}"
    assert "Not available for this run" in content or "No " in content


# ---------------------------------------------------------------------------
# No new reasoning — core invariant
# ---------------------------------------------------------------------------

def test_strategy_deck_generator_never_touches_functional_agents():
    """StrategyDeckGenerator must not import any *Agent class or FunctionalAgent."""
    from functional_agents.deliverables import strategy_deck as sd_module
    tree = ast.parse(inspect.getsource(sd_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    bad = [s for s in imported_symbols if s.endswith("Agent")]
    assert not bad, f"strategy_deck.py must not import Agent classes: {bad}"
    assert "FunctionalAgent" not in imported_symbols


def test_strategy_deck_generation_does_not_extend_agent_history(tmp_path):
    ctx = _full_ctx()
    ctx.agent_history.append({"agent": "FakeAgent", "status": "success"})
    before = len(ctx.agent_history)

    StrategyDeckGenerator().generate(ctx, tmp_path / "deck.md")

    assert len(ctx.agent_history) == before


def test_strategy_deck_generation_does_not_mutate_risks(tmp_path):
    ctx = _full_ctx()
    risks_before = [dict(r) for r in ctx.risks]

    StrategyDeckGenerator().generate(ctx, tmp_path / "deck.md")

    assert ctx.risks == risks_before


# ---------------------------------------------------------------------------
# Generator produce artefact
# ---------------------------------------------------------------------------

def test_strategy_deck_generator_produces_artifact(tmp_path):
    ctx = _full_ctx()
    artifact = StrategyDeckGenerator().generate(ctx, tmp_path / "deck.md")

    assert artifact.type == "strategy_deck"
    assert artifact.mime_type == "text/markdown"
    assert artifact.to_dict()["status"] == "generated"
    assert artifact.path == str(tmp_path / "deck.md")
    assert (tmp_path / "deck.md").exists()


def test_strategy_deck_generator_content_matches_build_function(tmp_path):
    ctx = _full_ctx()
    expected = build_strategy_deck_content(ctx)

    StrategyDeckGenerator().generate(ctx, tmp_path / "deck.md")

    assert (tmp_path / "deck.md").read_text(encoding="utf-8") == expected


# ---------------------------------------------------------------------------
# Bundle with all three deliverable types
# ---------------------------------------------------------------------------

def test_bundle_generates_all_three_deliverable_types(tmp_path):
    ctx = _full_ctx()
    bundle = DeliverableBundleGenerator().generate(
        ctx,
        ["markdown", "executive_brief", "strategy_deck"],
        tmp_path,
    )

    assert len(bundle.deliverables) == 3
    types = {a.type for a in bundle.deliverables}
    assert types == {"markdown", "executive_brief", "strategy_deck"}

    assert (tmp_path / "markdown.md").exists()
    assert (tmp_path / "executive-brief.md").exists()
    assert (tmp_path / "strategy-deck.md").exists()


def test_bundle_with_three_types_does_not_duplicate_reasoning(tmp_path):
    ctx = _full_ctx()
    before = len(ctx.agent_history)

    DeliverableBundleGenerator().generate(
        ctx,
        ["markdown", "executive_brief", "strategy_deck"],
        tmp_path,
    )

    assert len(ctx.agent_history) == before


def test_bundle_context_deliverable_bundle_contains_three_generated(tmp_path):
    ctx = _full_ctx()
    DeliverableBundleGenerator().generate(
        ctx,
        ["markdown", "executive_brief", "strategy_deck"],
        tmp_path,
    )
    generated_types = {g["type"] for g in ctx.deliverable_bundle["generated"]}
    assert generated_types == {"markdown", "executive_brief", "strategy_deck"}


# ---------------------------------------------------------------------------
# Existing markdown behaviour unchanged
# ---------------------------------------------------------------------------

def test_report_agent_still_defaults_to_markdown_only(tmp_path):
    from functional_agents.run_agent import run_agent
    res = run_agent("report", "fixtures/report_start.json",
                    no_llm=True, out_path=str(tmp_path / "report.md"))
    ctx = res["context"]
    assert len(ctx["deliverables"]) == 1
    assert ctx["deliverables"][0]["type"] == "markdown"


# ---------------------------------------------------------------------------
# J12.2 — narrative-driven architecture constraints
# ---------------------------------------------------------------------------

def test_strategy_deck_module_imports_executive_narrative_builder():
    """strategy_deck.py must import ExecutiveNarrativeBuilder (J12.2 contract)."""
    from functional_agents.deliverables import strategy_deck as sd_module
    tree = ast.parse(inspect.getsource(sd_module))
    imported_symbols = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "ExecutiveNarrativeBuilder" in imported_symbols
    assert "ExecutiveNarrative" in imported_symbols


def test_strategy_deck_slide_builders_have_no_direct_context_reads():
    """No _build_slide_* function body should reference 'context' as a Name node.

    The top-level build_strategy_deck_content reads metadata from context,
    then passes only narrative + metadata dicts to each slide builder.
    """
    from functional_agents.deliverables.strategy_deck import (
        _build_slide_01_executive_decision,
        _build_slide_03_executive_summary,
        _build_slide_04_strategic_options,
        _build_slide_05_decision_matrix,
        _build_slide_06_strategic_risks,
        _build_slide_07_strategic_opportunities,
        _build_slide_08_critical_assumptions,
        _build_slide_09_executive_confidence,
        _build_slide_10_immediate_actions,
        _build_slide_11_supporting_evidence,
    )
    import inspect as _inspect
    for fn in [
        _build_slide_01_executive_decision, _build_slide_03_executive_summary,
        _build_slide_04_strategic_options, _build_slide_05_decision_matrix,
        _build_slide_06_strategic_risks, _build_slide_07_strategic_opportunities,
        _build_slide_08_critical_assumptions, _build_slide_09_executive_confidence,
        _build_slide_10_immediate_actions, _build_slide_11_supporting_evidence,
    ]:
        src = _inspect.getsource(fn)
        tree = ast.parse(src)
        context_reads = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id == "context"
        ]
        assert not context_reads, (
            f"{fn.__name__} references 'context' directly — must consume ExecutiveNarrative only"
        )


def test_strategy_deck_sets_executive_narrative_on_context():
    """build_strategy_deck_content sets context.executive_narrative as a side effect."""
    ctx = _full_ctx()
    assert ctx.executive_narrative == {}
    build_strategy_deck_content(ctx)
    assert ctx.executive_narrative != {}
    assert "decision" in ctx.executive_narrative


def test_strategy_deck_slide_11_renders_supporting_evidence():
    """Slide 11 renders supporting evidence from narrative.supporting_evidence."""
    content = build_strategy_deck_content(_full_ctx())
    assert "H-001" in content
    assert "AI demand will grow 30% YoY" in content


def test_strategy_deck_slide_10_renders_medium_term_bucket():
    """Slide 10 renders medium-term portfolio from narrative.medium_term_actions."""
    content = build_strategy_deck_content(_full_ctx())
    assert "90 Days (Medium-Term)" in content
    assert "REC-003" in content


def test_strategy_deck_slide_06_renders_mitigation_from_narrative():
    """Slide 6 renders mitigation from narrative.key_risks (not context.risks directly)."""
    content = build_strategy_deck_content(_full_ctx())
    assert "Fixed-price contracts" in content


def test_strategy_deck_slide_04_marks_recommended_option_via_narrative():
    """Slide 4 marks the recommended option using narrative.recommended_option.option_id."""
    content = build_strategy_deck_content(_full_ctx())
    # Split on the slide separator (\n---\n\n), not bare --- (which appears in tables)
    slide_4 = content.split("# Slide 4")[1].split("\n---\n\n")[0]
    assert "✓" in slide_4
    assert "OPT-001" in slide_4
