"""PH10.4 — TheoryEvaluation unit tests.

Covers:
- CriterionScore: field defaults, validators, immutability, to_dict/from_dict
- TheoryEvaluation: field defaults, validators, immutability, to_dict/from_dict
- criteria_scores: generic keys — no hard-coded framework concepts
- overall_score validator: [0.0, 1.0]
- CriterionScore.score validator: [0.0, 1.0]
- CriterionScore.weight validator: >= 0.0
- weighted_score(): empty set, uniform weights, varied weights, all-zero weights
- criterion_names() and score_for() accessors
- Round-trip: nested CriterionScore survives to_dict → from_dict
- Extra fields allowed on both models
- Pipeline unchanged: existing StrategyCoordinator / StrategicPosition unaffected
"""

from __future__ import annotations

import pytest

from functional_agents.strategy import (
    CriterionScore,
    StrategyCoordinator,
    TheoryEvaluation,
)
from functional_agents.context import AgentContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cs(score: float = 0.8, rationale: str = "Strong", weight: float = 1.0) -> CriterionScore:
    return CriterionScore(score=score, rationale=rationale, weight=weight)


def _eval(
    theory_id: str = "TOW-001",
    criteria: dict[str, CriterionScore] | None = None,
    overall_score: float = 0.75,
    confidence: str = "High",
) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=theory_id,
        criteria_scores=criteria or {},
        overall_score=overall_score,
        confidence=confidence,
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
            }
        ],
        assumptions=[],
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
# CriterionScore — defaults
# ---------------------------------------------------------------------------

class TestCriterionScoreDefaults:
    def test_default_score_is_zero(self):
        cs = CriterionScore()
        assert cs.score == 0.0

    def test_default_rationale_is_empty(self):
        cs = CriterionScore()
        assert cs.rationale == ""

    def test_default_weight_is_one(self):
        cs = CriterionScore()
        assert cs.weight == 1.0

    def test_default_metadata_is_empty_dict(self):
        cs = CriterionScore()
        assert cs.metadata == {}

    def test_instantiate_with_values(self):
        cs = CriterionScore(score=0.9, rationale="Excellent", weight=2.0)
        assert cs.score == 0.9
        assert cs.rationale == "Excellent"
        assert cs.weight == 2.0


# ---------------------------------------------------------------------------
# CriterionScore — score validator
# ---------------------------------------------------------------------------

class TestCriterionScoreScoreValidator:
    def test_zero_is_valid(self):
        assert CriterionScore(score=0.0).score == 0.0

    def test_one_is_valid(self):
        assert CriterionScore(score=1.0).score == 1.0

    def test_midpoint_is_valid(self):
        assert CriterionScore(score=0.5).score == 0.5

    def test_negative_score_raises(self):
        with pytest.raises(Exception):
            CriterionScore(score=-0.1)

    def test_above_one_raises(self):
        with pytest.raises(Exception):
            CriterionScore(score=1.01)


# ---------------------------------------------------------------------------
# CriterionScore — weight validator
# ---------------------------------------------------------------------------

class TestCriterionScoreWeightValidator:
    def test_zero_weight_is_valid(self):
        assert CriterionScore(score=0.5, weight=0.0).weight == 0.0

    def test_large_weight_is_valid(self):
        assert CriterionScore(score=0.5, weight=100.0).weight == 100.0

    def test_negative_weight_raises(self):
        with pytest.raises(Exception):
            CriterionScore(score=0.5, weight=-0.1)


# ---------------------------------------------------------------------------
# CriterionScore — immutability
# ---------------------------------------------------------------------------

class TestCriterionScoreImmutability:
    def test_score_assignment_raises(self):
        cs = _cs()
        with pytest.raises(Exception):
            cs.score = 0.5

    def test_rationale_assignment_raises(self):
        cs = _cs()
        with pytest.raises(Exception):
            cs.rationale = "changed"

    def test_weight_assignment_raises(self):
        cs = _cs()
        with pytest.raises(Exception):
            cs.weight = 2.0


# ---------------------------------------------------------------------------
# CriterionScore — serialization
# ---------------------------------------------------------------------------

class TestCriterionScoreSerialization:
    def test_to_dict_returns_dict(self):
        assert isinstance(_cs().to_dict(), dict)

    def test_to_dict_contains_score(self):
        d = _cs(score=0.7).to_dict()
        assert d["score"] == 0.7

    def test_to_dict_contains_rationale(self):
        d = _cs(rationale="Good fit").to_dict()
        assert d["rationale"] == "Good fit"

    def test_to_dict_contains_weight(self):
        d = _cs(weight=3.0).to_dict()
        assert d["weight"] == 3.0

    def test_round_trip_score(self):
        cs = _cs(score=0.6)
        assert CriterionScore.from_dict(cs.to_dict()).score == 0.6

    def test_round_trip_rationale(self):
        cs = _cs(rationale="Compelling")
        assert CriterionScore.from_dict(cs.to_dict()).rationale == "Compelling"

    def test_round_trip_weight(self):
        cs = _cs(weight=2.5)
        assert CriterionScore.from_dict(cs.to_dict()).weight == 2.5

    def test_from_dict_empty_uses_defaults(self):
        cs = CriterionScore.from_dict({})
        assert cs.score == 0.0
        assert cs.weight == 1.0


# ---------------------------------------------------------------------------
# CriterionScore — extra fields allowed
# ---------------------------------------------------------------------------

class TestCriterionScoreExtraFields:
    def test_extra_field_accepted(self):
        cs = CriterionScore(score=0.5, custom_tag="framework-v2")
        assert cs.model_extra.get("custom_tag") == "framework-v2"

    def test_extra_field_survives_round_trip(self):
        cs = CriterionScore(score=0.5, custom_tag="v2")
        restored = CriterionScore.from_dict(cs.to_dict())
        assert restored.model_extra.get("custom_tag") == "v2"


# ---------------------------------------------------------------------------
# TheoryEvaluation — defaults
# ---------------------------------------------------------------------------

class TestTheoryEvaluationDefaults:
    def test_default_theory_id_is_empty(self):
        assert TheoryEvaluation().theory_id == ""

    def test_default_criteria_scores_is_empty_dict(self):
        assert TheoryEvaluation().criteria_scores == {}

    def test_default_strengths_is_empty_list(self):
        assert TheoryEvaluation().strengths == []

    def test_default_weaknesses_is_empty_list(self):
        assert TheoryEvaluation().weaknesses == []

    def test_default_residual_risks_is_empty_list(self):
        assert TheoryEvaluation().residual_risks == []

    def test_default_overall_score_is_zero(self):
        assert TheoryEvaluation().overall_score == 0.0

    def test_default_confidence_is_empty(self):
        assert TheoryEvaluation().confidence == ""

    def test_default_metadata_is_empty_dict(self):
        assert TheoryEvaluation().metadata == {}


# ---------------------------------------------------------------------------
# TheoryEvaluation — overall_score validator
# ---------------------------------------------------------------------------

class TestTheoryEvaluationOverallScoreValidator:
    def test_zero_is_valid(self):
        assert TheoryEvaluation(overall_score=0.0).overall_score == 0.0

    def test_one_is_valid(self):
        assert TheoryEvaluation(overall_score=1.0).overall_score == 1.0

    def test_midpoint_is_valid(self):
        assert TheoryEvaluation(overall_score=0.5).overall_score == 0.5

    def test_negative_raises(self):
        with pytest.raises(Exception):
            TheoryEvaluation(overall_score=-0.01)

    def test_above_one_raises(self):
        with pytest.raises(Exception):
            TheoryEvaluation(overall_score=1.001)


# ---------------------------------------------------------------------------
# TheoryEvaluation — immutability
# ---------------------------------------------------------------------------

class TestTheoryEvaluationImmutability:
    def test_theory_id_assignment_raises(self):
        ev = _eval()
        with pytest.raises(Exception):
            ev.theory_id = "other"

    def test_overall_score_assignment_raises(self):
        ev = _eval()
        with pytest.raises(Exception):
            ev.overall_score = 0.1

    def test_confidence_assignment_raises(self):
        ev = _eval()
        with pytest.raises(Exception):
            ev.confidence = "Low"

    def test_criteria_scores_assignment_raises(self):
        ev = _eval()
        with pytest.raises(Exception):
            ev.criteria_scores = {}


# ---------------------------------------------------------------------------
# TheoryEvaluation — criteria_scores genericity
# ---------------------------------------------------------------------------

class TestCriteriaScoresGenericity:
    def test_arbitrary_criterion_key_accepted(self):
        ev = TheoryEvaluation(
            theory_id="T-001",
            criteria_scores={"risk_adjusted_return": _cs(0.9)},
        )
        assert "risk_adjusted_return" in ev.criteria_scores

    def test_multiple_arbitrary_keys(self):
        ev = TheoryEvaluation(
            theory_id="T-001",
            criteria_scores={
                "alignment":    _cs(0.8),
                "feasibility":  _cs(0.7),
                "market_fit":   _cs(0.6),
                "sustainability": _cs(0.9),
            },
        )
        assert len(ev.criteria_scores) == 4

    def test_no_executive_keys_hard_coded(self):
        # Model accepts any key — none are required
        ev = TheoryEvaluation(theory_id="T-001", criteria_scores={
            "completely_custom_criterion_xyz": _cs(0.5),
        })
        assert ev.criteria_scores["completely_custom_criterion_xyz"].score == 0.5

    def test_criteria_scores_values_are_criterion_score_instances(self):
        ev = TheoryEvaluation(
            theory_id="T-001",
            criteria_scores={"alpha": _cs(0.75)},
        )
        assert isinstance(ev.criteria_scores["alpha"], CriterionScore)


# ---------------------------------------------------------------------------
# TheoryEvaluation — qualitative fields
# ---------------------------------------------------------------------------

class TestQualitativeFields:
    def test_strengths_preserved(self):
        ev = TheoryEvaluation(strengths=["Strong market position", "Low capex"])
        assert ev.strengths == ["Strong market position", "Low capex"]

    def test_weaknesses_preserved(self):
        ev = TheoryEvaluation(weaknesses=["Regulatory uncertainty"])
        assert ev.weaknesses == ["Regulatory uncertainty"]

    def test_residual_risks_preserved(self):
        risks = [{"description": "Grid bottleneck", "severity": "High"}]
        ev = TheoryEvaluation(residual_risks=risks)
        assert ev.residual_risks == risks

    def test_residual_risks_are_dicts(self):
        risks = [{"description": "R1"}, {"description": "R2"}]
        ev = TheoryEvaluation(residual_risks=risks)
        assert all(isinstance(r, dict) for r in ev.residual_risks)


# ---------------------------------------------------------------------------
# TheoryEvaluation — weighted_score accessor
# ---------------------------------------------------------------------------

class TestWeightedScore:
    def test_empty_criteria_returns_overall_score(self):
        ev = TheoryEvaluation(overall_score=0.6)
        assert ev.weighted_score() == 0.6

    def test_uniform_weights_equal_mean(self):
        ev = TheoryEvaluation(
            overall_score=0.0,
            criteria_scores={
                "a": _cs(0.8, weight=1.0),
                "b": _cs(0.4, weight=1.0),
            },
        )
        assert abs(ev.weighted_score() - 0.6) < 1e-9

    def test_varied_weights(self):
        ev = TheoryEvaluation(
            overall_score=0.0,
            criteria_scores={
                "a": _cs(1.0, weight=3.0),
                "b": _cs(0.0, weight=1.0),
            },
        )
        # (1.0*3 + 0.0*1) / 4 = 0.75
        assert abs(ev.weighted_score() - 0.75) < 1e-9

    def test_all_zero_weights_returns_zero(self):
        ev = TheoryEvaluation(
            overall_score=0.9,
            criteria_scores={
                "a": _cs(0.8, weight=0.0),
                "b": _cs(0.6, weight=0.0),
            },
        )
        assert ev.weighted_score() == 0.0

    def test_single_criterion_equals_its_score(self):
        ev = TheoryEvaluation(
            overall_score=0.0,
            criteria_scores={"only": _cs(0.55)},
        )
        assert abs(ev.weighted_score() - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# TheoryEvaluation — criterion_names and score_for accessors
# ---------------------------------------------------------------------------

class TestCriterionAccessors:
    def test_criterion_names_empty_when_no_scores(self):
        assert _eval(criteria={}).criterion_names() == []

    def test_criterion_names_returns_all_keys(self):
        ev = TheoryEvaluation(criteria_scores={
            "alpha": _cs(0.8),
            "beta":  _cs(0.6),
        })
        assert set(ev.criterion_names()) == {"alpha", "beta"}

    def test_score_for_returns_criterion_score(self):
        ev = TheoryEvaluation(criteria_scores={"alpha": _cs(0.8)})
        result = ev.score_for("alpha")
        assert result is not None
        assert result.score == 0.8

    def test_score_for_missing_key_returns_none(self):
        ev = TheoryEvaluation(criteria_scores={"alpha": _cs(0.8)})
        assert ev.score_for("nonexistent") is None

    def test_score_for_empty_criteria_returns_none(self):
        assert _eval(criteria={}).score_for("anything") is None


# ---------------------------------------------------------------------------
# TheoryEvaluation — serialization and round-trip
# ---------------------------------------------------------------------------

class TestTheoryEvaluationSerialization:
    def test_to_dict_returns_dict(self):
        assert isinstance(_eval().to_dict(), dict)

    def test_to_dict_contains_theory_id(self):
        d = _eval(theory_id="TOW-XYZ").to_dict()
        assert d["theory_id"] == "TOW-XYZ"

    def test_to_dict_contains_overall_score(self):
        d = _eval(overall_score=0.82).to_dict()
        assert d["overall_score"] == 0.82

    def test_to_dict_contains_confidence(self):
        d = _eval(confidence="Low").to_dict()
        assert d["confidence"] == "Low"

    def test_round_trip_theory_id(self):
        ev = _eval(theory_id="TOW-999")
        assert TheoryEvaluation.from_dict(ev.to_dict()).theory_id == "TOW-999"

    def test_round_trip_overall_score(self):
        ev = _eval(overall_score=0.33)
        assert TheoryEvaluation.from_dict(ev.to_dict()).overall_score == 0.33

    def test_round_trip_nested_criterion_score(self):
        ev = TheoryEvaluation(
            theory_id="T-1",
            criteria_scores={"feasibility": CriterionScore(score=0.72, rationale="Proven tech", weight=2.0)},
            overall_score=0.72,
        )
        restored = TheoryEvaluation.from_dict(ev.to_dict())
        assert "feasibility" in restored.criteria_scores
        cs = restored.criteria_scores["feasibility"]
        assert cs.score == 0.72
        assert cs.rationale == "Proven tech"
        assert cs.weight == 2.0

    def test_round_trip_strengths(self):
        ev = TheoryEvaluation(strengths=["S1", "S2"])
        assert TheoryEvaluation.from_dict(ev.to_dict()).strengths == ["S1", "S2"]

    def test_round_trip_weaknesses(self):
        ev = TheoryEvaluation(weaknesses=["W1"])
        assert TheoryEvaluation.from_dict(ev.to_dict()).weaknesses == ["W1"]

    def test_round_trip_residual_risks(self):
        risks = [{"description": "Grid risk", "severity": "High"}]
        ev = TheoryEvaluation(residual_risks=risks)
        restored = TheoryEvaluation.from_dict(ev.to_dict())
        assert restored.residual_risks == risks

    def test_round_trip_multiple_criteria(self):
        ev = TheoryEvaluation(
            criteria_scores={
                "alpha": CriterionScore(score=0.9),
                "beta":  CriterionScore(score=0.4),
            },
            overall_score=0.65,
        )
        restored = TheoryEvaluation.from_dict(ev.to_dict())
        assert set(restored.criterion_names()) == {"alpha", "beta"}

    def test_from_dict_empty_uses_defaults(self):
        ev = TheoryEvaluation.from_dict({})
        assert ev.theory_id == ""
        assert ev.criteria_scores == {}
        assert ev.overall_score == 0.0

    def test_round_trip_preserves_weighted_score(self):
        ev = TheoryEvaluation(
            criteria_scores={
                "a": CriterionScore(score=0.8, weight=2.0),
                "b": CriterionScore(score=0.4, weight=1.0),
            },
            overall_score=0.67,
        )
        restored = TheoryEvaluation.from_dict(ev.to_dict())
        assert abs(restored.weighted_score() - ev.weighted_score()) < 1e-9


# ---------------------------------------------------------------------------
# TheoryEvaluation — extra fields allowed
# ---------------------------------------------------------------------------

class TestTheoryEvaluationExtraFields:
    def test_extra_field_accepted(self):
        ev = TheoryEvaluation(theory_id="T-1", framework_version="2.0")
        assert ev.model_extra.get("framework_version") == "2.0"

    def test_extra_field_survives_round_trip(self):
        ev = TheoryEvaluation(theory_id="T-1", framework_version="2.0")
        restored = TheoryEvaluation.from_dict(ev.to_dict())
        assert restored.model_extra.get("framework_version") == "2.0"


# ---------------------------------------------------------------------------
# Pipeline unchanged — StrategyCoordinator / StrategicPosition unaffected
# ---------------------------------------------------------------------------

class TestPipelineUnchanged:
    def test_coordinator_build_still_returns_strategic_position(self):
        from functional_agents.strategy import StrategicPosition
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert isinstance(pos, StrategicPosition)

    def test_strategic_position_has_no_theory_evaluation(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert not hasattr(pos, "theory_evaluation")
        assert not hasattr(pos, "evaluations")

    def test_coordinator_has_no_evaluations_attribute(self):
        coord = StrategyCoordinator()
        assert not hasattr(coord, "_evaluations")

    def test_existing_position_fields_intact(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"
        assert pos.question == "What should we do?"
        assert pos.recommendation.recommended_option_id == "OPT-A"
        assert pos.theory_of_winning is not None

    def test_theories_still_produced(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._theories) == 3

    def test_choice_sets_still_produced(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._choice_sets) == 3
