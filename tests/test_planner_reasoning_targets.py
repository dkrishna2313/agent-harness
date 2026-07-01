"""Tests for PlannerAgent consuming Reasoning Targets (J10.2)."""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.planner_agent import PlannerAgent
from functional_agents.reasoning_target import ReasoningTarget, KIND_RESEARCH_QUESTION


def _ctx(question: str = "What are the power constraints for AI data centers?") -> AgentContext:
    return AgentContext(
        question=question,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-PLAN"},
        run_id="plan001",
    )


def test_planner_consumes_reasoning_targets():
    """Planner records that it received/planned reasoning targets."""
    ctx = _ctx()
    PlannerAgent().run(ctx)  # no client → mock plan
    diag = ctx.trace.get("_planner_reasoning")
    assert diag is not None
    assert diag["targets_received"] == 1
    assert diag["targets_planned"] == 1
    assert diag["primary_target_kind"] == KIND_RESEARCH_QUESTION


def test_legacy_question_path_identical_planning():
    """Planning off the reasoning target must equal planning off context.question."""
    q = "What cooling is required for 130kW racks?"
    ctx = _ctx(question=q)
    PlannerAgent().run(ctx)
    # The planning question comes from the primary target, which == context.question.
    assert ctx.plan["question"] == q
    assert ctx.plan["research_type"]
    assert ctx.plan["subquestions"]
    assert ctx.plan["investigation_areas"]


def test_planner_plans_from_primary_when_multiple_targets(monkeypatch):
    """With multiple targets, Planner plans the primary (first) target for now."""
    ctx = _ctx(question="Primary question?")

    t1 = ReasoningTarget(id="primary", title="Primary question?", question="Primary question?")
    t2 = ReasoningTarget(id="secondary", title="Second question?", question="Second question?")

    # Simulate a future multi-target world without changing production behavior.
    monkeypatch.setattr(ctx, "get_reasoning_targets", lambda: [t1, t2])

    PlannerAgent().run(ctx)
    diag = ctx.trace["_planner_reasoning"]
    assert diag["targets_received"] == 2
    assert diag["targets_planned"] == 1
    # Planned the primary target's question.
    assert ctx.plan["question"] == "Primary question?"


def test_planner_handles_no_targets_gracefully():
    """Empty question → no targets → Planner still completes (falls back)."""
    ctx = AgentContext(
        goal="Analyze power",  # question empty until framing runs
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-PLAN2"},
        run_id="plan002",
    )
    assert ctx.get_reasoning_targets() == []
    result = PlannerAgent().run(ctx)
    assert result.status == "success"
    diag = ctx.trace["_planner_reasoning"]
    assert diag["targets_received"] == 0
    assert diag["targets_planned"] == 0


def test_planner_output_schema_unchanged():
    """context.plan keys are exactly the pre-J10.2 set (no schema drift)."""
    ctx = _ctx()
    PlannerAgent().run(ctx)
    assert set(ctx.plan.keys()) == {
        "question", "research_type", "subquestions",
        "investigation_areas", "profiles_used", "reasoning",
    }
