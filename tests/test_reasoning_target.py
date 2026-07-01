"""Tests for the Reasoning Target accessor seam (J10.1)."""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.reasoning_target import (
    ReasoningTarget,
    KIND_RESEARCH_QUESTION,
    reasoning_targets_diagnostics,
)


def _ctx(question: str = "What are the power constraints for AI data centers?") -> AgentContext:
    return AgentContext(
        question=question,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-RT"},
        run_id="rt001",
    )


# ---------------------------------------------------------------------------
# Legacy behaviour preserved
# ---------------------------------------------------------------------------

def test_context_question_still_works():
    ctx = _ctx()
    assert ctx.question == "What are the power constraints for AI data centers?"


def test_accessor_returns_single_target_in_legacy_mode():
    ctx = _ctx()
    targets = ctx.get_reasoning_targets()
    assert isinstance(targets, list)
    assert len(targets) == 1
    assert isinstance(targets[0], ReasoningTarget)


def test_target_contains_current_question():
    ctx = _ctx()
    t = ctx.get_reasoning_targets()[0]
    assert t.question == ctx.question
    assert t.title == ctx.question
    assert t.id == "primary"
    assert t.kind == KIND_RESEARCH_QUESTION


def test_property_mirrors_method():
    ctx = _ctx()
    assert ctx.reasoning_targets[0].to_dict() == ctx.get_reasoning_targets()[0].to_dict()


def test_no_decision_architecture_required():
    """Accessor works with no decision_architecture / decision_model present."""
    ctx = _ctx()
    assert ctx.decision_architecture == {}
    assert ctx.decision_model == {}
    targets = ctx.get_reasoning_targets()
    assert len(targets) == 1
    assert targets[0].decision_domain_id is None
    assert targets[0].evidence_requirements == []


def test_empty_question_yields_no_targets():
    """Goal-driven run before ProblemFraming populates the question."""
    ctx = AgentContext(
        goal="Analyze power strategy",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-RT2"},
        run_id="rt002",
    )
    assert ctx.get_reasoning_targets() == []


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_shape():
    ctx = _ctx()
    diag = reasoning_targets_diagnostics(ctx.get_reasoning_targets(), source="context.question")
    assert diag == {
        "count": 1,
        "primary_kind": "research_question",
        "source": "context.question",
        "kinds": {"research_question": 1},
    }


def test_diagnostics_empty():
    diag = reasoning_targets_diagnostics([], source="context.question")
    assert diag["count"] == 0
    assert diag["primary_kind"] is None
    assert diag["kinds"] == {}


def test_target_to_dict_has_future_fields():
    """The target shape already carries decision-domain fields for J10.2+."""
    t = ReasoningTarget(id="primary", title="q", question="q")
    d = t.to_dict()
    for key in ("id", "title", "kind", "question", "decision_domain_id",
                "decision_domain_title", "evidence_requirements"):
        assert key in d
