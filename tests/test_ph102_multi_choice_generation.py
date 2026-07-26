"""PH10.2 — Multi-choice generation unit tests.

Verifies that StrategicChoiceGenerator produces exactly three StrategicChoiceSets
with the required properties:
  - count: exactly three
  - completeness: 1.0 in every set
  - conflicts: none in any set
  - distinctness: sets differ from each other
  - dimensions: each set covers all active dimensions
  - posture ordering: recommended → alternative-a → alternative-b
  - option diversity: with three or more options, selected_value differs per set
  - determinism: same plan+research produces same structure
  - coordinator: _choice_sets stores the list; _choice_set no longer exists
  - StrategicPosition: unchanged by multi-set generation
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategicChoiceGenerator,
    StrategicChoiceSet,
    StrategyConfig,
    StrategyCoordinator,
    StrategyPlan,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _plan_no_dims() -> StrategyPlan:
    return StrategyPlan(plan_id="P-TEST", framework="executive", active_dimensions=[])


def _plan_with_dims(*names: str) -> StrategyPlan:
    return StrategyPlan(
        plan_id="P-TEST",
        framework="executive",
        active_dimensions=list(names),
    )


def _research_no_options(overall_confidence: str = "High") -> types.SimpleNamespace:
    """Research with no strategic_options — all sets will use da fallback."""
    ns = types.SimpleNamespace()
    ns.executive_confidence = {"overall_confidence": overall_confidence}
    ns.decision_analysis = {"recommended_option_id": "OPT-A", "rationale": "Best choice"}
    ns.preferred_option = {}
    ns.assumptions = []
    return ns


def _research_with_options(*option_ids: str) -> types.SimpleNamespace:
    """Research with the given option IDs as strategic_options."""
    ns = types.SimpleNamespace()
    ns.executive_confidence = {"overall_confidence": "Medium"}
    ns.decision_analysis = {"recommended_option_id": option_ids[0] if option_ids else "", "rationale": "R"}
    ns.preferred_option = {}
    ns.assumptions = []
    ns.strategic_options = [{"option_id": oid} for oid in option_ids]
    return ns


def _full_ctx() -> AgentContext:
    return AgentContext(
        question="What should we do?",
        profiles=["test"],
        execution_profile="test",
        research_object={"id": "R-TEST"},
        run_id="run001",
        strategic_options=[
            {
                "option_id": "OPT-A",
                "title": "Option A",
                "description": "First option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 1"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Fast"],
                "disadvantages": ["Risky"],
                "implementation_complexity": "Low",
                "estimated_time_horizon": "Near-term",
                "capital_intensity": "Low",
                "confidence": "High",
                "recommended": True,
                "rationale": "Best return.",
            },
            {
                "option_id": "OPT-B",
                "title": "Option B",
                "description": "Second option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 2"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Safe"],
                "disadvantages": ["Slow"],
                "implementation_complexity": "Low",
                "estimated_time_horizon": "Long-term",
                "capital_intensity": "Medium",
                "confidence": "Medium",
                "recommended": False,
                "rationale": "Lower risk.",
            },
            {
                "option_id": "OPT-C",
                "title": "Option C",
                "description": "Third option.",
                "strategic_objective": "Grow.",
                "expected_outcomes": ["Outcome 3"],
                "supporting_assumption_ids": [],
                "associated_risk_ids": [],
                "associated_opportunity_ids": [],
                "supporting_recommendation_ids": [],
                "advantages": ["Innovative"],
                "disadvantages": ["Unproven"],
                "implementation_complexity": "High",
                "estimated_time_horizon": "Long-term",
                "capital_intensity": "High",
                "confidence": "Low",
                "recommended": False,
                "rationale": "High upside.",
            },
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
# Return type and count
# ---------------------------------------------------------------------------

class TestMultiChoiceReturnType:
    def test_returns_list(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert isinstance(sets, list)

    def test_returns_exactly_three_sets(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert len(sets) == 3

    def test_all_elements_are_strategic_choice_sets(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert all(isinstance(cs, StrategicChoiceSet) for cs in sets)

    def test_returns_three_sets_with_dimensions(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market", "technology"), _research_no_options()
        )
        assert len(sets) == 3


# ---------------------------------------------------------------------------
# Completeness — every set is complete
# ---------------------------------------------------------------------------

class TestMultiChoiceCompleteness:
    def test_all_sets_complete_no_dimensions(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        for cs in sets:
            assert cs.completeness == 1.0

    def test_all_sets_complete_with_one_dimension(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market"), _research_no_options()
        )
        for cs in sets:
            assert cs.completeness == 1.0

    def test_all_sets_complete_with_multiple_dimensions(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market", "technology", "financial"), _research_no_options()
        )
        for cs in sets:
            assert cs.completeness == 1.0


# ---------------------------------------------------------------------------
# Conflicts — no set has internal conflicts
# ---------------------------------------------------------------------------

class TestMultiChoiceNoConflicts:
    def test_no_conflicts_no_dimensions(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        for cs in sets:
            assert cs.internal_conflicts == []

    def test_no_conflicts_with_dimensions(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market", "technology"), _research_no_options()
        )
        for cs in sets:
            assert cs.internal_conflicts == []


# ---------------------------------------------------------------------------
# Dimension coverage — each set has one choice per dimension
# ---------------------------------------------------------------------------

class TestMultiChoiceDimensionCoverage:
    def test_each_set_has_one_choice_per_dimension(self):
        plan = _plan_with_dims("market", "technology")
        sets = StrategicChoiceGenerator().build(plan, _research_no_options())
        for cs in sets:
            assert len(cs.choices) == 2

    def test_each_set_covers_all_dimension_names(self):
        plan = _plan_with_dims("market", "technology", "financial")
        sets = StrategicChoiceGenerator().build(plan, _research_no_options())
        for cs in sets:
            covered = set(cs.dimensions_covered())
            assert covered == {"market", "technology", "financial"}

    def test_single_dimension_each_set_has_one_choice(self):
        plan = _plan_with_dims("risk")
        sets = StrategicChoiceGenerator().build(plan, _research_no_options())
        for cs in sets:
            assert len(cs.choices) == 1
            assert cs.choices[0].dimension == "risk"


# ---------------------------------------------------------------------------
# Distinctness — sets differ from each other
# ---------------------------------------------------------------------------

class TestMultiChoiceDistinctness:
    def test_set_ids_are_all_different(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        ids = [cs.id for cs in sets]
        assert len(ids) == len(set(ids))

    def test_set_rationales_are_all_different(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        rationales = [cs.rationale for cs in sets]
        assert len(rationales) == len(set(rationales))

    def test_choice_ids_differ_across_sets(self):
        plan = _plan_with_dims("market")
        sets = StrategicChoiceGenerator().build(plan, _research_no_options())
        choice_ids = [cs.choices[0].id for cs in sets]
        assert len(choice_ids) == len(set(choice_ids))

    def test_three_options_produces_diverse_selected_values(self):
        plan = _plan_with_dims("market")
        research = _research_with_options("OPT-A", "OPT-B", "OPT-C")
        sets = StrategicChoiceGenerator().build(plan, research)
        values = [cs.choices[0].selected_value for cs in sets]
        # With 3 options, each set picks a different one
        assert values[0] == "OPT-A"
        assert values[1] == "OPT-B"
        assert values[2] == "OPT-C"

    def test_two_options_first_and_third_set_differ_from_second(self):
        plan = _plan_with_dims("market")
        research = _research_with_options("OPT-A", "OPT-B")
        sets = StrategicChoiceGenerator().build(plan, research)
        values = [cs.choices[0].selected_value for cs in sets]
        # index 0→OPT-A, 1→OPT-B, 2→OPT-A (2%2=0)
        assert values[0] == "OPT-A"
        assert values[1] == "OPT-B"
        assert values[2] == "OPT-A"

    def test_one_option_all_sets_same_selected_value_but_different_ids(self):
        plan = _plan_with_dims("market")
        research = _research_with_options("OPT-ONLY")
        sets = StrategicChoiceGenerator().build(plan, research)
        # Same selected_value (only one option), but different ids
        for cs in sets:
            assert cs.choices[0].selected_value == "OPT-ONLY"
        ids = [cs.id for cs in sets]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Posture ordering
# ---------------------------------------------------------------------------

class TestMultiChoicePostureOrdering:
    def test_first_set_is_recommended(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert "recommended" in sets[0].rationale.lower()

    def test_second_set_is_alternative_a(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert "alternative-a" in sets[1].rationale

    def test_third_set_is_alternative_b(self):
        sets = StrategicChoiceGenerator().build(_plan_no_dims(), _research_no_options())
        assert "alternative-b" in sets[2].rationale

    def test_first_set_choice_id_contains_recommended(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market"), _research_no_options()
        )
        assert "recommended" in sets[0].choices[0].id

    def test_second_set_choice_id_contains_alternative_a(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market"), _research_no_options()
        )
        assert "alternative-a" in sets[1].choices[0].id

    def test_third_set_choice_id_contains_alternative_b(self):
        sets = StrategicChoiceGenerator().build(
            _plan_with_dims("market"), _research_no_options()
        )
        assert "alternative-b" in sets[2].choices[0].id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestMultiChoiceDeterminism:
    def test_same_plan_same_research_same_count(self):
        plan = _plan_with_dims("market")
        research = _research_no_options()
        sets_a = StrategicChoiceGenerator().build(plan, research)
        sets_b = StrategicChoiceGenerator().build(plan, research)
        assert len(sets_a) == len(sets_b)

    def test_same_plan_same_research_same_posture_rationales(self):
        plan = _plan_no_dims()
        research = _research_no_options()
        sets_a = StrategicChoiceGenerator().build(plan, research)
        sets_b = StrategicChoiceGenerator().build(plan, research)
        for a, b in zip(sets_a, sets_b):
            assert a.rationale == b.rationale

    def test_same_plan_same_research_same_selected_values(self):
        plan = _plan_with_dims("market")
        research = _research_with_options("OPT-A", "OPT-B", "OPT-C")
        sets_a = StrategicChoiceGenerator().build(plan, research)
        sets_b = StrategicChoiceGenerator().build(plan, research)
        for a, b in zip(sets_a, sets_b):
            assert a.choices[0].selected_value == b.choices[0].selected_value

    def test_does_not_mutate_plan(self):
        plan = _plan_with_dims("market", "technology")
        original = list(plan.active_dimensions)
        StrategicChoiceGenerator().build(plan, _research_no_options())
        assert plan.active_dimensions == original


# ---------------------------------------------------------------------------
# StrategyCoordinator — _choice_sets
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorChoiceSetsAttribute:
    def test_choice_sets_empty_list_before_build(self):
        coord = StrategyCoordinator()
        assert coord._choice_sets == []

    def test_choice_sets_is_list_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert isinstance(coord._choice_sets, list)

    def test_choice_sets_has_three_elements(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._choice_sets) == 3

    def test_choice_sets_all_strategic_choice_sets(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert all(isinstance(cs, StrategicChoiceSet) for cs in coord._choice_sets)

    def test_choice_sets_all_complete(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert cs.completeness == 1.0

    def test_choice_sets_no_conflicts(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for cs in coord._choice_sets:
            assert cs.internal_conflicts == []

    def test_choice_set_singular_does_not_exist(self):
        coord = StrategyCoordinator()
        assert not hasattr(coord, "_choice_set")

    def test_three_options_in_ctx_produce_diverse_sets(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())  # _full_ctx has OPT-A, OPT-B, OPT-C
        # With configured dimensions this test is vacuous (default exec has none),
        # but coordinator should hold 3 sets regardless
        assert len(coord._choice_sets) == 3
        ids = [cs.id for cs in coord._choice_sets]
        assert len(set(ids)) == 3


# ---------------------------------------------------------------------------
# StrategyPosition unchanged
# ---------------------------------------------------------------------------

class TestStrategyPositionUnchanged:
    def test_position_run_id_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"

    def test_position_question_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.question == "What should we do?"

    def test_position_recommendation_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.recommendation.recommended_option_id == "OPT-A"

    def test_position_theory_of_winning_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.theory_of_winning is not None
        assert pos.theory_of_winning.recommended_option_id == "OPT-A"

    def test_position_does_not_contain_choice_sets(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        # StrategicChoiceSets are not (yet) on StrategicPosition
        assert not hasattr(pos, "choice_sets")
        assert not hasattr(pos, "_choice_sets")
