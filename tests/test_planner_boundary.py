"""Tests for the Planner LLM boundary (PH2.1)."""

from __future__ import annotations

import json

import pytest

from functional_agents.context import AgentContext
from functional_agents.planner_agent import PlannerAgent
from functional_agents.planner_boundary import (
    PlannerOutput,
    plan_from_raw,
    normalize_planner_payload,
    validate_planner_output,
    PlannerBoundaryError,
    PlannerNormalizationError,
    PlannerValidationError,
    VALID_RESEARCH_TYPES,
)

_VALID = {
    "research_type": "RESEARCH",
    "subquestions": ["q1", "q2"],
    "investigation_areas": ["Power", "Cooling"],
    "profiles_used": ["ai_data_centers"],
    "reasoning": "ok",
}


# ---------------------------------------------------------------------------
# Boundary: normalize → validate → typed
# ---------------------------------------------------------------------------

def test_valid_response_produces_typed_output():
    out, diag = plan_from_raw(_VALID)
    assert isinstance(out, PlannerOutput)
    assert out.research_type == "RESEARCH"
    assert out.subquestions == ["q1", "q2"]
    assert diag["failed_stage"] is None
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}


def test_stringified_json_recovered():
    out, diag = plan_from_raw(json.dumps(_VALID))
    assert out.research_type == "RESEARCH"
    assert diag["stages"]["normalization"] == "ok"


def test_malformed_non_object_fails_normalization():
    with pytest.raises(PlannerNormalizationError) as ei:
        plan_from_raw("not a json object")
    assert ei.value.diagnostics["failed_stage"] == "normalization"


def test_normalization_recovery_scalar_to_list():
    raw = {**_VALID, "subquestions": "only one question"}
    norm, diag = normalize_planner_payload(raw)
    assert norm["subquestions"] == ["only one question"]
    assert any("wrapped scalar string" in r for r in diag["repairs"])


def test_normalization_drops_non_string_items():
    raw = {**_VALID, "subquestions": ["good", 42, "", None, "also good"]}
    norm, diag = normalize_planner_payload(raw)
    assert norm["subquestions"] == ["good", "also good"]
    assert any("dropped" in r for r in diag["repairs"])


def test_research_type_casing_normalized():
    raw = {**_VALID, "research_type": "  research  "}
    norm, _ = normalize_planner_payload(raw)
    assert norm["research_type"] == "RESEARCH"


def test_missing_required_fields_fail_validation():
    raw = {"research_type": "RESEARCH"}  # no subquestions / investigation_areas
    with pytest.raises(PlannerValidationError) as ei:
        plan_from_raw(raw)
    errs = ei.value.diagnostics["errors"]
    assert any("subquestions" in e for e in errs)
    assert any("investigation_areas" in e for e in errs)


def test_invalid_enum_fails_validation():
    raw = {**_VALID, "research_type": "GUESSWORK"}
    with pytest.raises(PlannerValidationError) as ei:
        plan_from_raw(raw)
    assert ei.value.diagnostics["failed_stage"] == "validation"


def test_validate_accepts_all_valid_enums():
    for rt in VALID_RESEARCH_TYPES:
        out, _ = validate_planner_output({**_VALID, "research_type": rt})
        assert out.research_type == rt


def test_boundary_error_is_typed_hierarchy():
    assert issubclass(PlannerNormalizationError, PlannerBoundaryError)
    assert issubclass(PlannerValidationError, PlannerBoundaryError)


# ---------------------------------------------------------------------------
# Agent integration
# ---------------------------------------------------------------------------

def _ctx(question: str = "What are the power constraints for AI data centers?") -> AgentContext:
    return AgentContext(
        question=question,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"research_id": "R-PB"},
        run_id="pb001",
    )


def test_agent_records_boundary_stages_on_success():
    ctx = _ctx()
    PlannerAgent().run(ctx)  # no client → deterministic default, valid
    boundary = ctx.trace["_planner_boundary"]
    assert boundary["failed_stage"] is None
    assert boundary["stages"]["validation"] == "ok"
    # Business logic still produced the plan.
    assert ctx.plan["research_type"] in VALID_RESEARCH_TYPES


def test_agent_business_logic_receives_validated_output():
    """context.plan is only populated when the boundary passed."""
    ctx = _ctx()
    PlannerAgent().run(ctx)
    assert ctx.plan["subquestions"]
    assert ctx.plan["investigation_areas"]


class _MalformedClient:
    """Client whose raw planner output is structurally invalid."""
    is_mock = False

    def plan_research_question_raw(self, *a, **k):
        return {"research_type": "NONSENSE"}  # missing lists, bad enum


def test_agent_deterministic_failure_on_malformed_output():
    ctx = _ctx()
    with pytest.raises(PlannerBoundaryError):
        PlannerAgent(client=_MalformedClient()).run(ctx)
    # Failure stage recorded; business logic did not run (no plan produced).
    assert ctx.trace["_planner_boundary"]["failed_stage"] == "validation"
    assert not ctx.plan


class _GenBoomClient:
    is_mock = False
    def plan_research_question_raw(self, *a, **k):
        raise RuntimeError("api exploded")


def test_generation_failure_classified_distinctly():
    from functional_agents.planner_boundary import PlannerGenerationError
    ctx = _ctx()
    with pytest.raises(PlannerGenerationError):
        PlannerAgent(client=_GenBoomClient()).run(ctx)
    assert ctx.trace["_planner_boundary"]["failed_stage"] == "generation"


def test_existing_planner_behavior_preserved_valid_inputs():
    """Valid path yields the same plan schema as before the boundary."""
    ctx = _ctx()
    PlannerAgent().run(ctx)
    assert set(ctx.plan.keys()) == {
        "question", "research_type", "subquestions",
        "investigation_areas", "profiles_used", "reasoning",
    }
