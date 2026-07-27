"""PH10.6b — Identity Contract Completion tests.

Covers:
- TheoryOfWinning.theory_id is required (no default)
- Construction without theory_id raises ValidationError
- Blank theory_id raises ValidationError
- Whitespace-only theory_id raises ValidationError
- Valid theory_id is accepted unchanged
- Serialization round-trip intact with required field
- StrategySelector: duplicate evaluation theory_ids raise ValueError
- StrategySelector: blank evaluation theory_id raises ValueError
- StrategySelector: all eight rejection cases enforced
- Strict one-to-one matching still enforced (no positional fallback)
- Selection ranking unchanged with required theory_ids
- StrategicPosition still uses selected theory object
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    StrategyCoordinator,
    StrategySelector,
)
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _plan() -> StrategyPlan:
    return StrategyPlan(plan_id="P-TEST", framework="executive", active_dimensions=[])


def _theory(tid: str, oid: str = "OPT-A", scid: str = "SCS-X") -> TheoryOfWinning:
    return TheoryOfWinning(theory_id=tid, recommended_option_id=oid, source_choice_set_id=scid)


def _eval(tid: str, score: float = 0.8, confidence: str = "High") -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={"x": CriterionScore(score=score, rationale="r", weight=1.0)},
        strengths=[], weaknesses=[], residual_risks=[],
        overall_score=score, confidence=confidence, metadata={},
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
                "option_id": "OPT-A", "title": "Option A", "description": "First.",
                "strategic_objective": "Grow.", "expected_outcomes": ["O1"],
                "supporting_assumption_ids": [], "associated_risk_ids": [],
                "associated_opportunity_ids": [], "supporting_recommendation_ids": [],
                "advantages": ["Fast"], "disadvantages": ["Risky"],
                "implementation_complexity": "Low", "estimated_time_horizon": "Near-term",
                "capital_intensity": "Low", "confidence": "High",
                "recommended": True, "rationale": "Best.",
            },
        ],
        assumptions=[], risks=[], opportunities=[], recommendations=[],
        decision_model={"strategic_question": "What should we do?"},
        decision_analysis={
            "recommended_option_id": "OPT-A", "rationale": "Best.",
            "key_tradeoffs": [], "decision_matrix": [],
        },
        executive_confidence={
            "overall_confidence": "High", "board_recommendation": "Proceed.",
            "decision_readiness": "Ready", "confidence_drivers": [],
            "confidence_limiters": [], "critical_unknowns": [], "validation_priorities": [],
        },
        preferred_option={"option_id": "OPT-A", "title": "Option A"},
        research_strategy={},
    )


# ---------------------------------------------------------------------------
# TheoryOfWinning — required field
# ---------------------------------------------------------------------------

class TestTheoryOfWinningRequiredField:
    def test_construction_without_theory_id_raises(self):
        with pytest.raises(ValidationError, match="theory_id"):
            TheoryOfWinning(recommended_option_id="OPT-A")

    def test_construction_with_empty_theory_id_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            TheoryOfWinning(theory_id="", recommended_option_id="OPT-A")

    def test_construction_with_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            TheoryOfWinning(theory_id="   ", recommended_option_id="OPT-A")

    def test_construction_with_tab_only_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            TheoryOfWinning(theory_id="\t", recommended_option_id="OPT-A")

    def test_construction_with_newline_only_raises(self):
        with pytest.raises(ValidationError, match="non-empty"):
            TheoryOfWinning(theory_id="\n", recommended_option_id="OPT-A")

    def test_valid_theory_id_accepted(self):
        t = TheoryOfWinning(theory_id="TH-SCS-0", recommended_option_id="OPT-A", source_choice_set_id="SCS-0")
        assert t.theory_id == "TH-SCS-0"

    def test_single_char_theory_id_accepted(self):
        t = TheoryOfWinning(theory_id="X", source_choice_set_id="SCS-X")
        assert t.theory_id == "X"

    def test_theory_id_value_preserved_unchanged(self):
        t = TheoryOfWinning(theory_id="TH-SCS-0-20260101-120000", source_choice_set_id="SCS-0")
        assert t.theory_id == "TH-SCS-0-20260101-120000"

    def test_recommended_option_id_remains_optional(self):
        # recommended_option_id keeps its default; theory_id and source_choice_set_id are required
        t = TheoryOfWinning(theory_id="TH-TEST", source_choice_set_id="SCS-X")
        assert t.recommended_option_id == ""

    def test_serialization_round_trip_intact(self):
        t = TheoryOfWinning(theory_id="TH-SCS-7", recommended_option_id="OPT-B", confidence="High", source_choice_set_id="SCS-7")
        d = t.model_dump(mode="json")
        t2 = TheoryOfWinning.model_validate(d)
        assert t2.theory_id == "TH-SCS-7"
        assert t2.recommended_option_id == "OPT-B"


# ---------------------------------------------------------------------------
# StrategySelector — duplicate evaluation IDs
# ---------------------------------------------------------------------------

class TestStrategySelectorDuplicateEvalIds:
    def test_duplicate_eval_theory_ids_raises(self):
        theories = [_theory("TH-A"), _theory("TH-B")]
        evals = [_eval("TH-A"), _eval("TH-A")]  # TH-A duplicated in evaluations
        with pytest.raises(ValueError, match="duplicate theory_id='TH-A' in evaluations"):
            StrategySelector().select(theories, evals, _plan())

    def test_all_duplicate_eval_ids_raises(self):
        theories = [_theory("TH-A"), _theory("TH-B"), _theory("TH-C")]
        evals = [_eval("TH-A"), _eval("TH-B"), _eval("TH-B")]
        with pytest.raises(ValueError, match="duplicate theory_id='TH-B' in evaluations"):
            StrategySelector().select(theories, evals, _plan())

    def test_unique_eval_ids_does_not_raise(self):
        theories = [_theory("TH-A"), _theory("TH-B")]
        evals = [_eval("TH-A"), _eval("TH-B")]
        result = StrategySelector().select(theories, evals, _plan())
        assert isinstance(result, TheoryOfWinning)


# ---------------------------------------------------------------------------
# StrategySelector — blank evaluation theory_ids
# ---------------------------------------------------------------------------

class TestStrategySelectorBlankEvalIds:
    def test_blank_eval_theory_id_raises(self):
        theories = [_theory("TH-A")]
        evals = [TheoryEvaluation(
            theory_id="",
            criteria_scores={"x": CriterionScore(score=0.8, rationale="r", weight=1.0)},
            strengths=[], weaknesses=[], residual_risks=[],
            overall_score=0.8, confidence="High", metadata={},
        )]
        with pytest.raises(ValueError, match="blank theory_id"):
            StrategySelector().select(theories, evals, _plan())

    def test_whitespace_eval_theory_id_raises(self):
        theories = [_theory("TH-A")]
        evals = [TheoryEvaluation(
            theory_id="   ",
            criteria_scores={"x": CriterionScore(score=0.8, rationale="r", weight=1.0)},
            strengths=[], weaknesses=[], residual_risks=[],
            overall_score=0.8, confidence="High", metadata={},
        )]
        with pytest.raises(ValueError, match="blank theory_id"):
            StrategySelector().select(theories, evals, _plan())


# ---------------------------------------------------------------------------
# StrategySelector — all eight rejection cases
# ---------------------------------------------------------------------------

class TestStrategySelectorRejectionCases:
    def test_empty_theories_raises(self):
        with pytest.raises(ValueError, match="theories list is empty"):
            StrategySelector().select([], [_eval("TH-A")], _plan())

    def test_empty_evaluations_raises(self):
        with pytest.raises(ValueError, match="evaluations list is empty"):
            StrategySelector().select([_theory("TH-A")], [], _plan())

    def test_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="counts differ"):
            StrategySelector().select([_theory("TH-A")], [_eval("TH-A"), _eval("TH-B")], _plan())

    def test_duplicate_theory_ids_raises(self):
        t1 = TheoryOfWinning(theory_id="TH-SAME", recommended_option_id="OPT-A", source_choice_set_id="SCS-X")
        t2 = TheoryOfWinning(theory_id="TH-SAME", recommended_option_id="OPT-B", source_choice_set_id="SCS-X")
        with pytest.raises(ValueError, match="duplicate theory_id='TH-SAME' in theories"):
            StrategySelector().select([t1, t2], [_eval("TH-SAME"), _eval("TH-X")], _plan())

    def test_duplicate_eval_ids_raises(self):
        with pytest.raises(ValueError, match="duplicate theory_id.*in evaluations"):
            StrategySelector().select(
                [_theory("TH-A"), _theory("TH-B")],
                [_eval("TH-A"), _eval("TH-A")],
                _plan(),
            )

    def test_unmatched_eval_id_raises(self):
        with pytest.raises(ValueError, match="TH-GHOST"):
            StrategySelector().select(
                [_theory("TH-A"), _theory("TH-B")],
                [_eval("TH-A"), _eval("TH-GHOST")],
                _plan(),
            )

    def test_theory_without_eval_raises(self):
        # theory count > eval count → caught by _validate_counts
        with pytest.raises(ValueError, match="counts differ"):
            StrategySelector().select(
                [_theory("TH-A"), _theory("TH-B")],
                [_eval("TH-A")],
                _plan(),
            )

    def test_eval_without_theory_raises(self):
        # eval count > theory count → caught by _validate_counts
        with pytest.raises(ValueError, match="counts differ"):
            StrategySelector().select(
                [_theory("TH-A")],
                [_eval("TH-A"), _eval("TH-B")],
                _plan(),
            )


# ---------------------------------------------------------------------------
# Ranking unchanged
# ---------------------------------------------------------------------------

class TestSelectionRankingUnchanged:
    def test_highest_score_still_wins(self):
        theories = [_theory("TH-A"), _theory("TH-B"), _theory("TH-C")]
        evals = [
            _eval("TH-A", score=0.5),
            _eval("TH-B", score=0.9),
            _eval("TH-C", score=0.7),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.theory_id == "TH-B"

    def test_confidence_tiebreaker_unchanged(self):
        theories = [_theory("TH-A"), _theory("TH-B")]
        evals = [
            _eval("TH-A", score=0.7, confidence="Low"),
            _eval("TH-B", score=0.7, confidence="High"),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.theory_id == "TH-B"

    def test_order_tiebreaker_unchanged(self):
        theories = [_theory("TH-FIRST"), _theory("TH-SECOND")]
        evals = [
            _eval("TH-FIRST", score=0.8, confidence="High"),
            _eval("TH-SECOND", score=0.8, confidence="High"),
        ]
        result = StrategySelector().select(theories, evals, _plan())
        assert result.theory_id == "TH-FIRST"


# ---------------------------------------------------------------------------
# StrategyCoordinator — end-to-end with required theory_id
# ---------------------------------------------------------------------------

class TestCoordinatorWithRequiredTheoryId:
    def test_coordinator_build_succeeds(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos is not None

    def test_all_theories_have_valid_theory_ids(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for t in coord._theories:
            assert t.theory_id and t.theory_id.strip()

    def test_position_theory_of_winning_is_selected_theory(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.theory_of_winning is coord._selected_theory

    def test_recommendation_aligned_with_selected_theory(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert (
            pos.recommendation.recommended_option_id
            == coord._selected_theory.recommended_option_id
        )

    def test_selection_winner_id_matches_selected_theory_theory_id(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert coord._selection.winner_theory_id == coord._selected_theory.theory_id
