"""PH10.1 — StrategicChoiceGenerator unit tests.

Covers:
- StrategicChoiceGenerator.build(): return type, empty plan, dimensioned plan
- Choice fields extracted from research data
- StrategyCoordinator: _choice_set set after build(), is StrategicChoiceSet
- StrategyCoordinator: _choice_set does not influence StrategicPosition output
- Pipeline: existing StrategicPosition fields unchanged
- No LLM calls, no mutation of input objects
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategicChoice,
    StrategicChoiceGenerator,
    StrategicChoiceSet,
    StrategyConfig,
    StrategyCoordinator,
    StrategyObjectives,
    StrategyPlan,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _plan_no_dims() -> StrategyPlan:
    """StrategyPlan with no active dimensions (default executive config)."""
    return StrategyPlan(
        plan_id="SPLAN-TEST",
        framework="executive",
        active_dimensions=[],
        objectives=["Win"],
    )


def _plan_with_dims(*names: str) -> StrategyPlan:
    """StrategyPlan with the given active dimensions."""
    return StrategyPlan(
        plan_id="SPLAN-TEST",
        framework="executive",
        active_dimensions=list(names),
        objectives=["Win"],
    )


def _research(
    *,
    overall_confidence: str = "High",
    recommended_option_id: str = "OPT-A",
    rationale: str = "Strong evidence",
    assumptions: list | None = None,
) -> types.SimpleNamespace:
    """Minimal research stand-in (duck-typed AgentContext substitute)."""
    ns = types.SimpleNamespace()
    ns.executive_confidence = {"overall_confidence": overall_confidence}
    ns.decision_analysis = {
        "recommended_option_id": recommended_option_id,
        "rationale": rationale,
    }
    ns.preferred_option = {}
    ns.assumptions = assumptions or []
    return ns


def _full_ctx() -> AgentContext:
    """Fully-populated AgentContext for coordinator integration tests."""
    return AgentContext(
        question="What should we do?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST_001"},
        run_id="run001",
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Option A",
                "description": "Proceed aggressively.",
                "strategic_objective": "Maximise growth.",
                "expected_outcomes": ["Outcome 1"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Fast"],
                "disadvantages": ["Risky"],
                "implementation_complexity": "Medium",
                "estimated_time_horizon": "Near-term",
                "capital_intensity": "Medium",
                "confidence": "High",
                "recommended": True,
                "rationale": "Best risk-adjusted return.",
            }
        ],
        assumptions=[{"assumption_id": "A-001", "statement": "Market stable"}],
        risks=[],
        opportunities=[],
        recommendations=[],
        decision_model={"strategic_question": "What should we do?"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "rationale": "Best risk-adjusted return.",
            "key_tradeoffs": ["Speed vs. cost"],
            "decision_matrix": [],
        },
        executive_confidence={
            "overall_confidence": "High",
            "board_recommendation": "Proceed.",
            "decision_readiness": "Ready",
            "confidence_drivers": [],
            "confidence_limiters": [],
            "critical_unknowns": [],
            "validation_priorities": [],
        },
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — return type
# ---------------------------------------------------------------------------

class TestStrategicChoiceGeneratorReturnType:
    def test_returns_list(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())
        assert isinstance(result, list)

    def test_returns_strategic_choice_sets(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())
        assert all(isinstance(cs, StrategicChoiceSet) for cs in result)

    def test_first_set_is_frozen(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())
        with pytest.raises(Exception):
            result[0].id = "mutated"


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — empty plan (no active_dimensions)
# ---------------------------------------------------------------------------

class TestStrategicChoiceGeneratorNoDimensions:
    def test_zero_choices(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())[0]
        assert result.choices == []

    def test_completeness_is_one(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())[0]
        assert result.completeness == 1.0

    def test_no_internal_conflicts(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())[0]
        assert result.internal_conflicts == []

    def test_rationale_populated(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())[0]
        assert result.rationale != ""

    def test_overall_confidence_extracted(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research(overall_confidence="High"))[0]
        assert result.overall_confidence == "High"

    def test_id_set(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_no_dims(), _research())[0]
        assert result.id.startswith("SCS-")


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — plan with active_dimensions
# ---------------------------------------------------------------------------

class TestStrategicChoiceGeneratorWithDimensions:
    def test_one_choice_per_dimension(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market", "technology"), _research())[0]
        assert len(result.choices) == 2

    def test_choice_dimensions_match_plan(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market", "technology"), _research())[0]
        dims = {c.dimension for c in result.choices}
        assert dims == {"market", "technology"}

    def test_completeness_is_one_when_all_covered(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market"), _research())[0]
        assert result.completeness == 1.0

    def test_single_dimension(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("financial"), _research())[0]
        assert len(result.choices) == 1
        assert result.choices[0].dimension == "financial"

    def test_three_dimensions(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("a", "b", "c"), _research())[0]
        assert len(result.choices) == 3

    def test_choices_are_strategic_choice_instances(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market"), _research())[0]
        assert all(isinstance(c, StrategicChoice) for c in result.choices)


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — choice field extraction from research
# ---------------------------------------------------------------------------

class TestStrategicChoiceFieldExtraction:
    def test_selected_value_from_decision_analysis(self):
        gen = StrategicChoiceGenerator()
        # No strategic_options → falls back to da.recommended_option_id
        result = gen.build(
            _plan_with_dims("market"),
            _research(recommended_option_id="OPT-X"),
        )[0]
        assert result.choices[0].selected_value == "OPT-X"

    def test_confidence_from_executive_confidence(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(
            _plan_with_dims("market"),
            _research(overall_confidence="Medium"),
        )[0]
        assert result.choices[0].confidence == "Medium"

    def test_rationale_from_decision_analysis(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(
            _plan_with_dims("market"),
            _research(rationale="Strong evidence"),
        )[0]
        assert result.choices[0].rationale == "Strong evidence"

    def test_supporting_assumptions_extracted(self):
        gen = StrategicChoiceGenerator()
        research = _research(
            assumptions=[
                {"assumption_id": "A-001", "statement": "Market stable"},
                {"assumption_id": "A-002", "statement": "Tech proven"},
            ]
        )
        result = gen.build(_plan_with_dims("market"), research)[0]
        assert "Market stable" in result.choices[0].supporting_assumptions
        assert "Tech proven" in result.choices[0].supporting_assumptions

    def test_choice_id_contains_dimension(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("technology"), _research())[0]
        assert "technology" in result.choices[0].id

    def test_no_alternatives_considered(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market"), _research())[0]
        assert result.choices[0].alternatives_considered == []

    def test_requiredness_is_optional(self):
        gen = StrategicChoiceGenerator()
        result = gen.build(_plan_with_dims("market"), _research())[0]
        assert result.choices[0].requiredness == "optional"


# ---------------------------------------------------------------------------
# StrategicChoiceGenerator — degenerate / edge cases
# ---------------------------------------------------------------------------

class TestStrategicChoiceGeneratorEdgeCases:
    def test_empty_executive_confidence_uses_empty_string(self):
        gen = StrategicChoiceGenerator()
        research = types.SimpleNamespace(
            executive_confidence={},
            decision_analysis={},
            preferred_option={},
            assumptions=[],
        )
        result = gen.build(_plan_with_dims("market"), research)[0]
        assert result.choices[0].confidence == ""

    def test_none_executive_confidence_graceful(self):
        gen = StrategicChoiceGenerator()
        research = types.SimpleNamespace(
            executive_confidence=None,
            decision_analysis=None,
            preferred_option=None,
            assumptions=None,
        )
        result = gen.build(_plan_with_dims("market"), research)
        assert isinstance(result[0], StrategicChoiceSet)

    def test_preferred_option_selected_value_takes_priority_no_options(self):
        gen = StrategicChoiceGenerator()
        # No strategic_options → falls back to preferred/da; preferred wins
        research = types.SimpleNamespace(
            executive_confidence={"overall_confidence": "High"},
            decision_analysis={"recommended_option_id": "OPT-DA"},
            preferred_option={"option_id": "OPT-PREF"},
            assumptions=[],
        )
        result = gen.build(_plan_with_dims("market"), research)[0]
        assert result.choices[0].selected_value == "OPT-PREF"

    def test_does_not_mutate_plan(self):
        plan = _plan_with_dims("market", "technology")
        original_dims = list(plan.active_dimensions)
        StrategicChoiceGenerator().build(plan, _research())
        assert plan.active_dimensions == original_dims


# ---------------------------------------------------------------------------
# StrategyCoordinator — _choice_set wiring
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorChoiceSets:
    def test_choice_sets_empty_before_build(self):
        coord = StrategyCoordinator()
        assert coord._choice_sets == []

    def test_choice_sets_populated_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._choice_sets) > 0

    def test_choice_sets_are_strategic_choice_sets(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert all(isinstance(cs, StrategicChoiceSet) for cs in coord._choice_sets)

    def test_each_choice_set_completeness_valid(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert 0.0 <= cs.completeness <= 1.0

    def test_each_choice_set_no_internal_conflicts(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert cs.internal_conflicts == []

    def test_multiple_builds_each_update_choice_sets(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        first_ids = [cs.id for cs in coord._choice_sets]
        coord.build(_full_ctx())
        second_ids = [cs.id for cs in coord._choice_sets]
        assert all(i.startswith("SCS-") for i in first_ids)
        assert all(i.startswith("SCS-") for i in second_ids)


# ---------------------------------------------------------------------------
# StrategyCoordinator — StrategicPosition unchanged
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorPositionUnchanged:
    def test_position_has_run_id(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"

    def test_position_has_question(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.question == "What should we do?"

    def test_position_has_strategic_options(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert len(pos.strategic_options) == 1

    def test_position_theory_of_winning_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.theory_of_winning is not None

    def test_position_recommendation_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.recommendation is not None
        assert pos.recommendation.recommended_option_id == "OPT-A"

    def test_position_has_executive_confidence(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.executive_confidence.get("overall_confidence") == "High"


# ---------------------------------------------------------------------------
# Compatibility — default executive config produces vacuously complete set
# ---------------------------------------------------------------------------

class TestDefaultConfigCompatibility:
    def test_default_executive_config_no_active_dims(self):
        coord = StrategyCoordinator()
        assert coord._plan.active_dimensions == []

    def test_default_config_choice_sets_are_vacuously_complete(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert cs.completeness == 1.0
            assert cs.choices == []

    def test_custom_config_with_dimensions_produces_choices(self):
        from functional_agents.strategy import StrategyDimensions
        dims = StrategyDimensions()
        dims.add("market", "market entry strategy")
        dims.add("technology", "technology selection")
        cfg = StrategyConfig(
            dimensions=dims,
            objectives=StrategyObjectives(primary=["Win market share"]),
        )
        coord = StrategyCoordinator(config=cfg)
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert len(cs.choices) == 2
        covered = coord._choice_sets[0].dimensions_covered()
        assert "market" in covered
        assert "technology" in covered
