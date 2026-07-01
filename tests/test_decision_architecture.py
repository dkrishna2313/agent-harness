"""Tests for the Decision Architecture (J9.2)."""

from __future__ import annotations

import pytest

from functional_agents.decision_architecture import (
    DecisionArchitecture,
    build_decision_architecture,
    architecture_trace_metadata,
)
from functional_agents.context import AgentContext
from functional_agents.problem_framing_agent import ProblemFramingAgent
from research_agent.claude_client import DecisionModelPayload
from research_agent.decision_model import (
    DecisionModel,
    create_decision_model,
    from_framing_payload,
)


def _payload() -> DecisionModelPayload:
    return DecisionModelPayload(
        objective="Determine the optimal power infrastructure strategy for multi-GW AI deployment.",
        decision_areas=["Power Procurement", "Cooling Architecture", "Site Strategy", "Capital Deployment"],
        critical_uncertainties=["Realized GB300 power draw", "Grid interconnection timelines"],
        research_questions=[
            "What power procurement options fit a multi-GW build-out?",
            "Which cooling architecture suits 130kW+ racks?",
            "Is SMR viable within the window?",
            "What siting minimizes grid delay?",
        ],
        evidence_requirements=["Power forecasts", "Cooling benchmarks"],
    )


_ENGAGEMENT = {
    "title": "AI Data Center Power Strategy",
    "objectives": ["Identify power strategies", "Determine cooling architecture"],
    "priorities": ["Speed to power", "Cost"],
    "success_criteria": ["Ranked strategies with trade-offs"],
    "decision_horizon": "24 months",
    "known_unknowns": ["Realized GB300 power draw at scale"],
    "constraints": ["24 month window"],
}


# ---------------------------------------------------------------------------
# Builder — with engagement
# ---------------------------------------------------------------------------

def test_builds_decision_statement():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert arch.decision_statement.startswith("Determine the optimal power")


def test_strategic_themes_from_areas_and_priorities():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert "Power Procurement" in arch.strategic_themes
    assert "Speed to power" in arch.strategic_themes


def test_decision_streams_created():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert 1 <= len(arch.decision_streams) <= 8
    for s in arch.decision_streams:
        assert s.title
        assert s.executive_objective
        assert s.expected_outputs


def test_all_research_questions_parented():
    """Every research question must become a child of some decision stream."""
    payload = _payload()
    arch = build_decision_architecture(payload, _ENGAGEMENT)
    parented = [q for s in arch.decision_streams for q in s.research_questions]
    assert sorted(parented) == sorted(payload.research_questions)


def test_executive_unknowns_prefer_engagement():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert "Realized GB300 power draw at scale" in arch.executive_unknowns


def test_board_decisions_identified():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert arch.board_decisions_required
    assert any("Power Procurement" in b for b in arch.board_decisions_required)


def test_success_definition_includes_horizon():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert any("24 months" in s for s in arch.success_definition)


def test_scope_in_scope_populated_out_scope_explicit_empty():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    assert arch.decision_scope.in_scope
    # Engagement contract carries no explicit exclusions — recorded empty, not invented.
    assert arch.decision_scope.out_of_scope == []
    assert arch.out_of_scope_items == []


def test_trace_metadata_counts():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    meta = architecture_trace_metadata(arch)
    assert meta["decision_stream_count"] == len(arch.decision_streams)
    assert meta["strategic_theme_count"] == len(arch.strategic_themes)
    assert meta["board_decision_count"] == len(arch.board_decisions_required)
    assert meta["research_questions_parented"] == 4


# ---------------------------------------------------------------------------
# Builder — without engagement (goal/question mode)
# ---------------------------------------------------------------------------

def test_builds_without_engagement():
    arch = build_decision_architecture(_payload(), None)
    assert arch.decision_statement
    assert arch.strategic_themes  # from decision_areas
    assert arch.decision_streams
    # No engagement → success definition may be empty (explicit, not invented).
    assert isinstance(arch.success_definition, list)


def test_empty_payload_is_safe():
    empty = DecisionModelPayload(objective="", decision_areas=[], research_questions=[])
    arch = build_decision_architecture(empty, None)
    assert isinstance(arch, DecisionArchitecture)
    assert arch.decision_streams == []


# ---------------------------------------------------------------------------
# Persistence — Decision Model v2
# ---------------------------------------------------------------------------

def test_decision_model_carries_architecture():
    arch = build_decision_architecture(_payload(), _ENGAGEMENT)
    dm = from_framing_payload(
        _payload(), strategic_question="Q", decision_architecture=arch.to_dict()
    )
    assert dm.decision_architecture
    assert dm.decision_architecture["strategic_themes"]


def test_decision_model_architecture_defaults_empty():
    dm = create_decision_model(strategic_question="Q")
    assert dm.decision_architecture == {}


# ---------------------------------------------------------------------------
# Integration — ProblemFramingAgent
# ---------------------------------------------------------------------------

def _goal_ctx(engagement: dict | None = None) -> AgentContext:
    return AgentContext(
        goal="Analyze AI data center power strategies.",
        engagement=engagement or {},
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-ARCH_TEST"},
        run_id="arch001",
    )


def test_framing_agent_produces_architecture_in_context():
    agent = ProblemFramingAgent()
    ctx = _goal_ctx(engagement=_ENGAGEMENT)
    result = agent.run(ctx)
    arch = result.context.decision_architecture
    assert arch
    assert arch["decision_statement"]
    assert arch["decision_streams"]


def test_framing_agent_persists_architecture_to_research_object():
    agent = ProblemFramingAgent()
    ctx = _goal_ctx(engagement=_ENGAGEMENT)
    result = agent.run(ctx)
    ro = result.context.research_object
    assert "decision_architecture" in ro
    assert ro["decision_architecture"]["strategic_themes"]


def test_framing_agent_records_architecture_trace_metadata():
    agent = ProblemFramingAgent()
    ctx = _goal_ctx(engagement=_ENGAGEMENT)
    result = agent.run(ctx)
    meta = result.context.trace.get("_decision_architecture_meta")
    assert meta is not None
    assert "decision_stream_count" in meta
    assert "strategic_theme_count" in meta
    assert "board_decision_count" in meta


def test_framing_agent_backwards_compatible_outputs_intact():
    """Existing framing outputs must remain unchanged (J9.2 is additive)."""
    agent = ProblemFramingAgent()
    ctx = _goal_ctx()
    result = agent.run(ctx)
    dm = result.context.decision_model
    for key in ("objective", "decision_areas", "critical_uncertainties",
                "research_questions", "evidence_requirements"):
        assert key in dm
