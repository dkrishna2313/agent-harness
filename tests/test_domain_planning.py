"""Tests for multi-domain planning (J10.4)."""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.planner_agent import PlannerAgent

_PLAN_SCHEMA_KEYS = {
    "question", "research_type", "subquestions",
    "investigation_areas", "profiles_used", "reasoning",
}


def _engagement_ctx() -> AgentContext:
    return AgentContext(
        question="What power procurement options fit a multi-GW build-out?",
        goal="Engagement brief …",
        engagement={"title": "AI DC Power Strategy", "objectives": ["O1"]},
        decision_model={
            "research_questions": ["What power procurement options fit a multi-GW build-out?"],
            "decision_areas": ["Power", "Cooling"],
        },
        decision_architecture={
            "decision_streams": [
                {"title": "Power Procurement", "research_questions": ["What PPA options exist?"]},
                {"title": "Cooling Architecture", "research_questions": ["Liquid vs air?"]},
                {"title": "Site Strategy", "research_questions": []},
            ],
        },
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-DP"},
        run_id="dp001",
    )


def _goal_ctx() -> AgentContext:
    return AgentContext(
        question="What are the power constraints for AI data centers?",
        goal="Analyze power",
        engagement={},  # goal mode
        decision_model={"research_questions": ["What are the power constraints for AI data centers?"],
                        "decision_areas": ["Power"]},
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-GP"},
        run_id="gp001",
    )


# ---------------------------------------------------------------------------
# Multi-domain planning
# ---------------------------------------------------------------------------

def test_one_plan_per_decision_domain():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    assert len(ctx.domain_plans) == 3
    assert [p["decision_domain_title"] for p in ctx.domain_plans] == [
        "Power Procurement", "Cooling Architecture", "Site Strategy",
    ]


def test_goal_mode_single_plan():
    ctx = _goal_ctx()
    PlannerAgent().run(ctx)
    assert len(ctx.domain_plans) == 1
    assert ctx.domain_plans[0]["target_kind"] == "research_question"


def test_primary_plan_deterministic_and_first():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    assert ctx.domain_plans[0]["is_primary"] is True
    assert all(p["is_primary"] is False for p in ctx.domain_plans[1:])
    # context.plan mirrors the primary domain plan's core fields.
    assert ctx.plan["question"] == ctx.domain_plans[0]["question"]


def test_context_plan_schema_unchanged():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    assert set(ctx.plan.keys()) == _PLAN_SCHEMA_KEYS  # no organizational metadata leaked


def test_downstream_plan_question_is_context_question():
    """Primary plan still plans context.question → EvidenceAgent unaffected."""
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    assert ctx.plan["question"] == ctx.question


def test_domain_plan_entries_contain_planning_object():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    for p in ctx.domain_plans:
        assert _PLAN_SCHEMA_KEYS.issubset(set(p.keys()))  # existing schema present
        assert "decision_domain_id" in p and "is_primary" in p  # + organization


def test_diagnostics_report_plan_counts():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    diag = ctx.trace["_planner_reasoning"]
    assert diag["targets_received"] == 3
    assert diag["plans_generated"] == 3
    assert diag["plans_executed"] == 1
    assert diag["primary_target_kind"] == "decision_domain"
    assert diag["targets_planned"] == 1  # retained J10.2 field


def test_goal_mode_diagnostics_single_plan():
    ctx = _goal_ctx()
    PlannerAgent().run(ctx)
    diag = ctx.trace["_planner_reasoning"]
    assert diag["plans_generated"] == 1
    assert diag["plans_executed"] == 1
    assert diag["primary_target_kind"] == "research_question"


def test_research_object_gets_primary_plan_fields():
    ctx = _engagement_ctx()
    PlannerAgent().run(ctx)
    ro = ctx.research_object
    assert ro["research_type"] == ctx.plan["research_type"]
    assert ro["subquestions"] == ctx.plan["subquestions"]
    assert ro["investigation_areas"] == ctx.plan["investigation_areas"]
