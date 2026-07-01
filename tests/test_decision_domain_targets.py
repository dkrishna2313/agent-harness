"""Tests for Decision Domains producing Reasoning Targets (J10.3)."""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.planner_agent import PlannerAgent
from functional_agents.reasoning_target import (
    KIND_RESEARCH_QUESTION,
    KIND_DECISION_DOMAIN,
    reasoning_targets_diagnostics,
)


def _engagement_ctx() -> AgentContext:
    """Context resembling post-ProblemFraming state in engagement mode."""
    return AgentContext(
        question="What power procurement options fit a multi-GW build-out?",
        goal="Engagement brief …",
        engagement={"title": "AI DC Power Strategy", "objectives": ["O1"]},
        decision_architecture={
            "decision_statement": "Determine the optimal power strategy.",
            "decision_streams": [
                {"title": "Power Procurement", "executive_objective": "Choose model",
                 "research_questions": ["What PPA options exist?"], "expected_outputs": ["Ranked options"]},
                {"title": "Cooling Architecture", "executive_objective": "Choose cooling",
                 "research_questions": ["Liquid vs air?"], "expected_outputs": ["Recommendation"]},
                {"title": "Site Strategy", "executive_objective": "Choose siting",
                 "research_questions": [], "expected_outputs": ["Shortlist"]},
            ],
        },
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-DD"},
        run_id="dd001",
    )


def _goal_ctx() -> AgentContext:
    """Goal mode: builds a decision_architecture but no engagement."""
    return AgentContext(
        question="What are the power constraints for AI data centers?",
        goal="Analyze power",
        engagement={},  # goal mode → empty
        decision_architecture={
            "decision_streams": [
                {"title": "Power", "research_questions": ["q?"]},
            ]
        },
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-GOAL"},
        run_id="goal001",
    )


# ---------------------------------------------------------------------------
# Producer behaviour by mode
# ---------------------------------------------------------------------------

def test_goal_mode_single_research_question_target():
    ctx = _goal_ctx()
    targets = ctx.get_reasoning_targets()
    assert len(targets) == 1
    assert targets[0].kind == KIND_RESEARCH_QUESTION
    assert targets[0].question == ctx.question


def test_engagement_mode_one_target_per_decision_domain():
    ctx = _engagement_ctx()
    targets = ctx.get_reasoning_targets()
    assert len(targets) == 3
    assert all(t.kind == KIND_DECISION_DOMAIN for t in targets)
    assert [t.decision_domain_title for t in targets] == [
        "Power Procurement", "Cooling Architecture", "Site Strategy",
    ]
    assert [t.decision_domain_id for t in targets] == ["domain-1", "domain-2", "domain-3"]


def test_primary_target_is_deterministic_and_pins_question():
    """Primary target is targets[0]; its question == context.question (no drift)."""
    ctx = _engagement_ctx()
    t0 = ctx.get_reasoning_targets()[0]
    assert t0.id == "domain-1"
    assert t0.question == ctx.question  # planner plans identical question
    # deterministic across calls
    assert ctx.get_reasoning_targets()[0].to_dict() == t0.to_dict()


def test_secondary_domains_use_stream_questions():
    ctx = _engagement_ctx()
    targets = ctx.get_reasoning_targets()
    assert targets[1].question == "Liquid vs air?"          # from stream research_questions
    assert targets[2].question == "Choose siting"           # empty rqs → executive_objective


# ---------------------------------------------------------------------------
# Planner still plans exactly one (primary) target
# ---------------------------------------------------------------------------

def test_planner_plans_one_target_in_engagement_mode():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    diag = ctx.trace["_planner_reasoning"]
    assert diag["targets_received"] == 3
    assert diag["targets_planned"] == 1
    assert diag["primary_target_kind"] == KIND_DECISION_DOMAIN


def test_planner_output_unchanged_plans_context_question():
    """Engagement-mode Planner still plans context.question (byte-identical)."""
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    assert ctx.plan["question"] == ctx.question


def test_goal_mode_planner_kind_is_research_question():
    ctx = _goal_ctx()
    PlannerAgent().run(ctx)
    assert ctx.trace["_planner_reasoning"]["primary_target_kind"] == KIND_RESEARCH_QUESTION


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_reflect_multiple_targets():
    ctx = _engagement_ctx()
    diag = reasoning_targets_diagnostics(ctx.get_reasoning_targets(), source="decision_architecture")
    assert diag["count"] == 3
    assert diag["primary_kind"] == KIND_DECISION_DOMAIN
    assert diag["kinds"] == {"decision_domain": 3}


def test_engagement_without_streams_falls_back_to_single():
    """Engagement flag but no streams yet → legacy single target (no crash)."""
    ctx = _engagement_ctx()
    ctx.decision_architecture = {}  # streams not built yet
    targets = ctx.get_reasoning_targets()
    assert len(targets) == 1
    assert targets[0].kind == KIND_RESEARCH_QUESTION
