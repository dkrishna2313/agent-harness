"""PH10.6 — StrategySelector unit tests.

Covers:
- StrategySelector.select(): return type is TheoryOfWinning
- Primary selection: highest overall_score wins
- Tie-breaker 1: higher evaluation confidence wins
- Tie-breaker 2: fewer residual_risks wins
- Tie-breaker 3: earlier original order wins (stable)
- Single-theory input: that theory is always returned
- Validation: empty theories raises ValueError
- Validation: empty evaluations raises ValueError
- Validation: count mismatch raises ValueError
- Validation: unmatched evaluation theory_id raises ValueError (unique IDs)
- Selection metadata: _last_selection populated after select()
- StrategySelection: winner/runner-up IDs, scores, margin, tie_breaker_used
- StrategyCoordinator: _selected_theory None before build
- StrategyCoordinator: _selected_theory is TheoryOfWinning after build
- StrategyCoordinator: StrategicPosition.theory_of_winning == _selected_theory
- StrategyCoordinator: recommendation.recommended_option_id aligned with selected theory
- StrategyCoordinator: StrategicPosition structure otherwise unchanged
"""

from __future__ import annotations

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategyCoordinator,
    StrategySelection,
    StrategySelector,
)
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _plan() -> StrategyPlan:
    return StrategyPlan(plan_id="P-TEST", framework="executive", active_dimensions=[])


def _theory(
    *,
    option_id: str,
    theory_id: str = "",
    title: str = "",
    winning_position: str = "A position.",
    winning_mechanism: str = "A mechanism.",
    failure_modes: list | None = None,
    confidence: str = "High",
) -> TheoryOfWinning:
    return TheoryOfWinning(
        theory_id=theory_id or option_id,
        recommended_option_id=option_id,
        recommended_option_title=title or option_id,
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        strategic_choices=[],
        success_conditions=[],
        failure_modes=failure_modes or [],
        assumptions=[],
        evidence=[],
        confidence=confidence,
    )


def _eval(
    *,
    theory_id: str,
    overall_score: float,
    confidence: str = "High",
    residual_risks: list | None = None,
) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=theory_id,
        criteria_scores={
            "option_identified": CriterionScore(score=1.0, rationale="present", weight=1.0),
        },
        strengths=[],
        weaknesses=[],
        residual_risks=residual_risks or [],
        overall_score=overall_score,
        confidence=confidence,
        metadata={},
    )


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
            "key_tradeoffs": [],
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
# Return type
# ---------------------------------------------------------------------------

class TestStrategySelectorReturnType:
    def test_returns_theory_of_winning(self):
        theories = [_theory(option_id="A")]
        evals = [_eval(theory_id="A", overall_score=0.8)]
        result = StrategySelector().select(theories, evals, _plan())
        assert isinstance(result, TheoryOfWinning)

    def test_single_theory_always_returned(self):
        theory = _theory(option_id="ONLY")
        result = StrategySelector().select([theory], [_eval(theory_id="ONLY", overall_score=0.5)], _plan())
        assert result.recommended_option_id == "ONLY"

    def test_does_not_mutate_input_lists(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [_eval(theory_id="A", overall_score=0.9), _eval(theory_id="B", overall_score=0.5)]
        original_theory_ids = [t.recommended_option_id for t in theories]
        StrategySelector().select(theories, evals, _plan())
        assert [t.recommended_option_id for t in theories] == original_theory_ids


# ---------------------------------------------------------------------------
# Primary criterion: highest overall_score wins
# ---------------------------------------------------------------------------

class TestPrimaryScoreSelection:
    def test_highest_score_wins(self):
        theories = [
            _theory(option_id="LOW"),
            _theory(option_id="HIGH"),
            _theory(option_id="MED"),
        ]
        evals = [
            _eval(theory_id="LOW", overall_score=0.3),
            _eval(theory_id="HIGH", overall_score=0.9),
            _eval(theory_id="MED", overall_score=0.6),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "HIGH"

    def test_score_zero_vs_nonzero(self):
        theories = [_theory(option_id="ZERO"), _theory(option_id="NON")]
        evals = [
            _eval(theory_id="ZERO", overall_score=0.0),
            _eval(theory_id="NON", overall_score=0.01),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "NON"

    def test_score_one_always_wins(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=1.0),
            _eval(theory_id="B", overall_score=0.999),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "A"

    def test_three_distinct_scores(self):
        ids = ["X", "Y", "Z"]
        scores = [0.7, 0.5, 0.85]
        theories = [_theory(option_id=i) for i in ids]
        evals = [_eval(theory_id=i, overall_score=s) for i, s in zip(ids, scores)]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "Z"


# ---------------------------------------------------------------------------
# Tie-breaker 1: confidence (High > Medium > Low)
# ---------------------------------------------------------------------------

class TestTieBreakerConfidence:
    def test_high_confidence_beats_medium_on_score_tie(self):
        theories = [_theory(option_id="MED"), _theory(option_id="HIGH")]
        evals = [
            _eval(theory_id="MED",  overall_score=0.8, confidence="Medium"),
            _eval(theory_id="HIGH", overall_score=0.8, confidence="High"),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "HIGH"

    def test_medium_confidence_beats_low_on_score_tie(self):
        theories = [_theory(option_id="LOW"), _theory(option_id="MED")]
        evals = [
            _eval(theory_id="LOW", overall_score=0.5, confidence="Low"),
            _eval(theory_id="MED", overall_score=0.5, confidence="Medium"),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "MED"

    def test_high_confidence_beats_low_on_score_tie(self):
        theories = [_theory(option_id="LOW"), _theory(option_id="HIGH")]
        evals = [
            _eval(theory_id="LOW",  overall_score=0.7, confidence="Low"),
            _eval(theory_id="HIGH", overall_score=0.7, confidence="High"),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "HIGH"

    def test_tie_breaker_recorded_as_confidence(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.6, confidence="Medium"),
            _eval(theory_id="B", overall_score=0.6, confidence="High"),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.tie_breaker_used == "confidence"


# ---------------------------------------------------------------------------
# Tie-breaker 2: fewer residual_risks
# ---------------------------------------------------------------------------

class TestTieBreakerResidualRisks:
    def test_fewer_risks_wins_on_score_and_confidence_tie(self):
        theories = [_theory(option_id="RISKY"), _theory(option_id="SAFE")]
        evals = [
            _eval(
                theory_id="RISKY", overall_score=0.7, confidence="High",
                residual_risks=[{"r": 1}, {"r": 2}, {"r": 3}],
            ),
            _eval(
                theory_id="SAFE", overall_score=0.7, confidence="High",
                residual_risks=[{"r": 1}],
            ),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "SAFE"

    def test_zero_risks_beats_one_risk(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.8, confidence="High", residual_risks=[{"r": 1}]),
            _eval(theory_id="B", overall_score=0.8, confidence="High", residual_risks=[]),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "B"

    def test_tie_breaker_recorded_as_residual_risks(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.6, confidence="High", residual_risks=[{"r": 1}, {"r": 2}]),
            _eval(theory_id="B", overall_score=0.6, confidence="High", residual_risks=[{"r": 1}]),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.tie_breaker_used == "residual_risks"


# ---------------------------------------------------------------------------
# Tie-breaker 3: earlier original order
# ---------------------------------------------------------------------------

class TestTieBreakerOrder:
    def test_first_in_list_wins_when_all_tied(self):
        theories = [_theory(option_id="FIRST"), _theory(option_id="SECOND")]
        evals = [
            _eval(theory_id="FIRST",  overall_score=0.75, confidence="High", residual_risks=[]),
            _eval(theory_id="SECOND", overall_score=0.75, confidence="High", residual_risks=[]),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "FIRST"

    def test_order_tie_breaker_recorded(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.5, confidence="Medium", residual_risks=[]),
            _eval(theory_id="B", overall_score=0.5, confidence="Medium", residual_risks=[]),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.tie_breaker_used == "order"

    def test_three_way_tie_selects_index_zero(self):
        theories = [_theory(option_id="X"), _theory(option_id="Y"), _theory(option_id="Z")]
        evals = [
            _eval(theory_id="X", overall_score=0.6, confidence="Medium", residual_risks=[]),
            _eval(theory_id="Y", overall_score=0.6, confidence="Medium", residual_risks=[]),
            _eval(theory_id="Z", overall_score=0.6, confidence="Medium", residual_risks=[]),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.recommended_option_id == "X"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_empty_theories_raises(self):
        with pytest.raises(ValueError, match="theories list is empty"):
            StrategySelector().select([], [_eval(theory_id="A", overall_score=0.5)], _plan())

    def test_empty_evaluations_raises(self):
        with pytest.raises(ValueError, match="evaluations list is empty"):
            StrategySelector().select([_theory(option_id="A")], [], _plan())

    def test_count_mismatch_raises(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [_eval(theory_id="A", overall_score=0.8)]
        with pytest.raises(ValueError, match="counts differ"):
            StrategySelector().select(theories, evals, _plan())

    def test_unmatched_evaluation_theory_id_raises(self):
        # evaluation references a theory_id that doesn't exist
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.8),
            _eval(theory_id="GHOST", overall_score=0.5),  # no matching theory
        ]
        with pytest.raises(ValueError, match="GHOST"):
            StrategySelector().select(theories, evals, _plan())

    def test_count_mismatch_more_evaluations_than_theories(self):
        theories = [_theory(option_id="A")]
        evals = [
            _eval(theory_id="A", overall_score=0.8),
            _eval(theory_id="B", overall_score=0.5),
        ]
        with pytest.raises(ValueError, match="counts differ"):
            StrategySelector().select(theories, evals, _plan())


# ---------------------------------------------------------------------------
# StrategySelection metadata
# ---------------------------------------------------------------------------

class TestStrategySelectionMetadata:
    def test_last_selection_populated_after_select(self):
        theories = [_theory(option_id="A")]
        evals = [_eval(theory_id="A", overall_score=0.9)]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection is not None
        assert isinstance(sel._last_selection, StrategySelection)

    def test_winner_theory_id_correct(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.9),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.winner_theory_id == "A"

    def test_winner_score_correct(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.9),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert abs(sel._last_selection.winner_score - 0.9) < 1e-9

    def test_runner_up_theory_id_correct(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.9),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.runner_up_theory_id == "B"

    def test_runner_up_score_correct(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.9),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert abs(sel._last_selection.runner_up_score - 0.5) < 1e-9

    def test_score_margin_correct(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.9),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert abs(sel._last_selection.score_margin - 0.4) < 1e-6

    def test_no_tie_breaker_when_scores_differ(self):
        theories = [_theory(option_id="A"), _theory(option_id="B")]
        evals = [
            _eval(theory_id="A", overall_score=0.8),
            _eval(theory_id="B", overall_score=0.5),
        ]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.tie_breaker_used is None

    def test_single_theory_no_runner_up(self):
        theories = [_theory(option_id="ONLY")]
        evals = [_eval(theory_id="ONLY", overall_score=0.7)]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        assert sel._last_selection.runner_up_theory_id is None
        assert sel._last_selection.runner_up_score is None
        assert sel._last_selection.score_margin is None

    def test_selection_is_frozen(self):
        theories = [_theory(option_id="A")]
        evals = [_eval(theory_id="A", overall_score=0.7)]
        sel = StrategySelector()
        sel.select(theories, evals, _plan())
        with pytest.raises(Exception):
            sel._last_selection.winner_score = 0.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# StrategyCoordinator integration
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorSelectedTheory:
    def test_selected_theory_none_before_build(self):
        coord = StrategyCoordinator()
        assert coord._selected_theory is None

    def test_selected_theory_is_theory_of_winning_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert isinstance(coord._selected_theory, TheoryOfWinning)

    def test_selected_theory_populated_on_each_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        first = coord._selected_theory
        coord.build(_full_ctx())
        # Both calls produce a TheoryOfWinning (same ctx → same result)
        assert isinstance(coord._selected_theory, TheoryOfWinning)
        assert first is not coord._selected_theory  # new object each call

    def test_selection_metadata_stored(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._selection is not None
        assert isinstance(coord._selection, StrategySelection)

    def test_selection_winner_id_matches_selected_theory(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        # winner_theory_id in StrategySelection matches the selected theory's theory_id
        assert coord._selection.winner_theory_id == coord._selected_theory.theory_id


# ---------------------------------------------------------------------------
# StrategicPosition construction using selected theory
# ---------------------------------------------------------------------------

class TestStrategicPositionFromSelectedTheory:
    def test_position_theory_of_winning_is_selected_theory(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        # The position's theory_of_winning must be the selected theory object
        assert pos.theory_of_winning is coord._selected_theory

    def test_position_theory_of_winning_is_theory_of_winning_type(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert isinstance(pos.theory_of_winning, TheoryOfWinning)

    def test_recommendation_aligned_with_selected_theory_option_id(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert (
            pos.recommendation.recommended_option_id
            == coord._selected_theory.recommended_option_id
        )

    def test_recommendation_aligned_with_selected_theory_title(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert (
            pos.recommendation.recommended_option_title
            == coord._selected_theory.recommended_option_title
        )

    def test_position_run_id_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"

    def test_position_question_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.question == "What should we do?"

    def test_position_does_not_carry_evaluations(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert not hasattr(pos, "evaluations")
        assert not hasattr(pos, "_evaluations")

    def test_position_does_not_carry_selection(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert not hasattr(pos, "selection")
        assert not hasattr(pos, "_selection")

    def test_position_justification_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.justification is not None

    def test_position_execution_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.execution is not None

    def test_position_strategic_options_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert len(pos.strategic_options) == 3
