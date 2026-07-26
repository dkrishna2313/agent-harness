"""PH10.5 / PH10.5a — TheoryEvaluator unit tests.

Covers:
- TheoryEvaluator.build(): return type is TheoryEvaluation
- theory_id: derived from theory.recommended_option_id
- criteria_scores: seven generic criteria present, all CriterionScore instances
- overall_score: in [0.0, 1.0], weighted mean of criteria
- confidence: carried from theory.confidence; falls back to score-based derivation
- strengths: criteria with score >= 0.75 and success_conditions
- weaknesses: criteria with score < 0.50
- residual_risks: carries theory.failure_modes
- metadata: plan_id, dimension counts, evidence/assumption counts
- PH10.5a criterion-selection algorithm:
    - DEFAULT mode (empty weights): all 7 built-in criteria evaluated
    - CONFIGURED mode (non-empty weights): ONLY those named criteria evaluated
    - Unrecognised configured criteria receive deterministic neutral fallback
- Plan interaction: evaluation_model.weights defines criterion set in configured mode
- Plan interaction: validation_policy.require_evidence forces evidence_quality to 0.0
- Plan interaction: validation_policy.require_assumptions forces assumption_coverage to 0.0
- PH10.5a three-tier rationale:
    - score >= 0.75 → high_rationale [+ (detail)]
    - 0 < score < 0.75 → detail string (accurate partial description)
    - score == 0.0 → low_rationale
- choice_completeness: 1.0 when no active dims; proportional when dims present
- evidence_quality scoring: 0 → 0.0, 1 → 0.33, 3+ → 1.0
- assumption_coverage scoring: 0 → 0.0, 1 → 0.5, 2+ → 1.0
- risk_awareness: 1.0 when failure_modes non-empty; 0.5 when empty
- Independent evaluation: two theories produce two evaluations, no cross-reference
- StrategyCoordinator: _evaluations empty before build; list[TheoryEvaluation] after
- StrategyCoordinator: exactly three evaluations produced by default
- StrategyCoordinator: StrategicPosition unchanged
"""

from __future__ import annotations

import types

import pytest

from functional_agents.context import AgentContext
from functional_agents.strategy import (
    CriterionScore,
    StrategyCoordinator,
    TheoryEvaluation,
    TheoryEvaluator,
)
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_plan import (
    EvaluationModel,
    StrategyPlan,
    ValidationPolicy,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _plan(
    *,
    active_dimensions: list[str] | None = None,
    weights: dict[str, float] | None = None,
    require_evidence: bool = False,
    require_assumptions: bool = False,
) -> StrategyPlan:
    return StrategyPlan(
        plan_id="P-TEST",
        framework="executive",
        active_dimensions=active_dimensions or [],
        evaluation_model=EvaluationModel(weights=weights or {}),
        validation_policy=ValidationPolicy(
            require_evidence=require_evidence,
            require_assumptions=require_assumptions,
        ),
    )


def _theory(
    *,
    recommended_option_id: str = "OPT-A",
    recommended_option_title: str = "Alpha",
    winning_position: str = "Posture 0 (recommended): 2 dimension(s) covered.",
    winning_mechanism: str = "Deploy at full scale.",
    strategic_choices: list[dict] | None = None,
    success_conditions: list[str] | None = None,
    failure_modes: list[dict] | None = None,
    assumptions: list[dict] | None = None,
    evidence: list[str] | None = None,
    confidence: str = "High",
) -> TheoryOfWinning:
    return TheoryOfWinning(
        recommended_option_id=recommended_option_id,
        recommended_option_title=recommended_option_title,
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        strategic_choices=strategic_choices or [],
        success_conditions=success_conditions or [],
        failure_modes=failure_modes or [],
        assumptions=assumptions or [],
        evidence=evidence or [],
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

class TestTheoryEvaluatorReturnType:
    def test_returns_theory_evaluation(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert isinstance(ev, TheoryEvaluation)

    def test_does_not_mutate_theory(self):
        t = _theory()
        original_id = t.recommended_option_id
        TheoryEvaluator().build(t, _plan(), None)
        assert t.recommended_option_id == original_id

    def test_does_not_mutate_plan(self):
        p = _plan(active_dimensions=["market"])
        original_dims = list(p.active_dimensions)
        TheoryEvaluator().build(_theory(), p, None)
        assert p.active_dimensions == original_dims


# ---------------------------------------------------------------------------
# theory_id
# ---------------------------------------------------------------------------

class TestTheoryId:
    def test_theory_id_from_recommended_option_id(self):
        ev = TheoryEvaluator().build(_theory(recommended_option_id="OPT-B"), _plan(), None)
        assert ev.theory_id == "OPT-B"

    def test_theory_id_empty_when_no_recommended_id(self):
        ev = TheoryEvaluator().build(_theory(recommended_option_id=""), _plan(), None)
        assert ev.theory_id == ""


# ---------------------------------------------------------------------------
# criteria_scores — structure
# ---------------------------------------------------------------------------

class TestCriteriaScoresStructure:
    _EXPECTED = {
        "option_identified",
        "position_articulated",
        "mechanism_defined",
        "choice_completeness",
        "evidence_quality",
        "assumption_coverage",
        "risk_awareness",
    }

    def test_all_seven_criteria_present(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert set(ev.criteria_scores.keys()) == self._EXPECTED

    def test_all_values_are_criterion_score_instances(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert all(isinstance(cs, CriterionScore) for cs in ev.criteria_scores.values())

    def test_all_scores_in_valid_range(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        for cs in ev.criteria_scores.values():
            assert 0.0 <= cs.score <= 1.0

    def test_all_weights_non_negative(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        for cs in ev.criteria_scores.values():
            assert cs.weight >= 0.0

    def test_criterion_names_match_accessor(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert set(ev.criterion_names()) == self._EXPECTED


# ---------------------------------------------------------------------------
# criteria_scores — individual criterion scoring
# ---------------------------------------------------------------------------

class TestOptionIdentifiedCriterion:
    def test_score_one_when_option_present(self):
        ev = TheoryEvaluator().build(_theory(recommended_option_id="OPT-A"), _plan(), None)
        assert ev.criteria_scores["option_identified"].score == 1.0

    def test_score_zero_when_option_absent(self):
        ev = TheoryEvaluator().build(_theory(recommended_option_id=""), _plan(), None)
        assert ev.criteria_scores["option_identified"].score == 0.0


class TestPositionArticulatedCriterion:
    def test_score_one_when_position_present(self):
        ev = TheoryEvaluator().build(_theory(winning_position="Clear position"), _plan(), None)
        assert ev.criteria_scores["position_articulated"].score == 1.0

    def test_score_zero_when_position_absent(self):
        ev = TheoryEvaluator().build(_theory(winning_position=""), _plan(), None)
        assert ev.criteria_scores["position_articulated"].score == 0.0


class TestMechanismDefinedCriterion:
    def test_score_one_when_mechanism_present(self):
        ev = TheoryEvaluator().build(_theory(winning_mechanism="Deploy at scale."), _plan(), None)
        assert ev.criteria_scores["mechanism_defined"].score == 1.0

    def test_score_zero_when_mechanism_absent(self):
        ev = TheoryEvaluator().build(_theory(winning_mechanism=""), _plan(), None)
        assert ev.criteria_scores["mechanism_defined"].score == 0.0


class TestChoiceCompletenessCriterion:
    def test_score_one_when_no_active_dimensions(self):
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=[]),
            _plan(active_dimensions=[]),
            None,
        )
        assert ev.criteria_scores["choice_completeness"].score == 1.0

    def test_score_one_when_fully_covered(self):
        choices = [{"dimension": "market"}, {"dimension": "technology"}]
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=choices),
            _plan(active_dimensions=["market", "technology"]),
            None,
        )
        assert ev.criteria_scores["choice_completeness"].score == 1.0

    def test_score_half_when_half_covered(self):
        choices = [{"dimension": "market"}]
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=choices),
            _plan(active_dimensions=["market", "technology"]),
            None,
        )
        assert abs(ev.criteria_scores["choice_completeness"].score - 0.5) < 1e-9

    def test_score_zero_when_no_choices_with_dims(self):
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=[]),
            _plan(active_dimensions=["market"]),
            None,
        )
        assert ev.criteria_scores["choice_completeness"].score == 0.0

    def test_score_capped_at_one(self):
        # More choices than dimensions — should not exceed 1.0
        choices = [{"dimension": "a"}, {"dimension": "b"}, {"dimension": "c"}]
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=choices),
            _plan(active_dimensions=["market"]),
            None,
        )
        assert ev.criteria_scores["choice_completeness"].score <= 1.0


class TestEvidenceQualityCriterion:
    def test_score_zero_when_no_evidence(self):
        ev = TheoryEvaluator().build(_theory(evidence=[]), _plan(), None)
        assert ev.criteria_scores["evidence_quality"].score == 0.0

    def test_score_one_third_for_one_evidence(self):
        ev = TheoryEvaluator().build(_theory(evidence=["E1"]), _plan(), None)
        assert abs(ev.criteria_scores["evidence_quality"].score - 1.0 / 3.0) < 1e-9

    def test_score_two_thirds_for_two_evidence(self):
        ev = TheoryEvaluator().build(_theory(evidence=["E1", "E2"]), _plan(), None)
        assert abs(ev.criteria_scores["evidence_quality"].score - 2.0 / 3.0) < 1e-9

    def test_score_one_for_three_or_more_evidence(self):
        ev = TheoryEvaluator().build(_theory(evidence=["E1", "E2", "E3"]), _plan(), None)
        assert ev.criteria_scores["evidence_quality"].score == 1.0

    def test_score_capped_at_one_with_many_evidence(self):
        ev = TheoryEvaluator().build(
            _theory(evidence=["E"] * 10), _plan(), None
        )
        assert ev.criteria_scores["evidence_quality"].score == 1.0

    def test_require_evidence_forces_zero_when_empty(self):
        ev = TheoryEvaluator().build(
            _theory(evidence=[]),
            _plan(require_evidence=True),
            None,
        )
        assert ev.criteria_scores["evidence_quality"].score == 0.0

    def test_require_evidence_does_not_penalise_when_present(self):
        ev = TheoryEvaluator().build(
            _theory(evidence=["E1", "E2", "E3"]),
            _plan(require_evidence=True),
            None,
        )
        assert ev.criteria_scores["evidence_quality"].score == 1.0


class TestAssumptionCoverageCriterion:
    def test_score_zero_when_no_assumptions(self):
        ev = TheoryEvaluator().build(_theory(assumptions=[]), _plan(), None)
        assert ev.criteria_scores["assumption_coverage"].score == 0.0

    def test_score_half_for_one_assumption(self):
        ev = TheoryEvaluator().build(
            _theory(assumptions=[{"statement": "A1"}]), _plan(), None
        )
        assert abs(ev.criteria_scores["assumption_coverage"].score - 0.5) < 1e-9

    def test_score_one_for_two_or_more_assumptions(self):
        ev = TheoryEvaluator().build(
            _theory(assumptions=[{"statement": "A1"}, {"statement": "A2"}]),
            _plan(),
            None,
        )
        assert ev.criteria_scores["assumption_coverage"].score == 1.0

    def test_require_assumptions_forces_zero_when_empty(self):
        ev = TheoryEvaluator().build(
            _theory(assumptions=[]),
            _plan(require_assumptions=True),
            None,
        )
        assert ev.criteria_scores["assumption_coverage"].score == 0.0


class TestRiskAwarenessCriterion:
    def test_score_one_when_failure_modes_present(self):
        ev = TheoryEvaluator().build(
            _theory(failure_modes=[{"description": "Delay", "severity": "High"}]),
            _plan(),
            None,
        )
        assert ev.criteria_scores["risk_awareness"].score == 1.0

    def test_score_half_when_no_failure_modes(self):
        ev = TheoryEvaluator().build(_theory(failure_modes=[]), _plan(), None)
        assert ev.criteria_scores["risk_awareness"].score == 0.5


# ---------------------------------------------------------------------------
# Plan weight override
# ---------------------------------------------------------------------------

class TestPlanWeightOverride:
    def test_plan_weight_overrides_default(self):
        # CONFIGURED mode: exactly these two criteria scored with these weights
        custom_weights = {"option_identified": 10.0, "evidence_quality": 0.1}
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights=custom_weights), None
        )
        assert ev.criteria_scores["option_identified"].weight == 10.0
        assert ev.criteria_scores["evidence_quality"].weight == 0.1

    def test_configured_mode_excludes_unconfigured_built_ins(self):
        # PH10.5a: non-empty weights = CONFIGURED mode.
        # Only "option_identified" is configured → position_articulated is absent.
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"option_identified": 5.0}), None
        )
        assert "option_identified" in ev.criteria_scores
        assert "position_articulated" not in ev.criteria_scores

    def test_empty_plan_weights_uses_all_defaults(self):
        ev = TheoryEvaluator().build(_theory(), _plan(weights={}), None)
        # risk_awareness default weight is 1.0
        assert ev.criteria_scores["risk_awareness"].weight == 1.0


# ---------------------------------------------------------------------------
# overall_score
# ---------------------------------------------------------------------------

class TestOverallScore:
    def test_overall_score_in_range(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert 0.0 <= ev.overall_score <= 1.0

    def test_well_populated_theory_scores_high(self):
        ev = TheoryEvaluator().build(
            _theory(
                evidence=["E1", "E2", "E3"],
                assumptions=[{"statement": "A1"}, {"statement": "A2"}],
                failure_modes=[{"description": "Risk", "severity": "High"}],
                strategic_choices=[{"dimension": "market"}],
            ),
            _plan(active_dimensions=["market"]),
            None,
        )
        assert ev.overall_score >= 0.75

    def test_sparse_theory_scores_lower(self):
        ev = TheoryEvaluator().build(
            _theory(
                recommended_option_id="",
                winning_position="",
                winning_mechanism="",
                evidence=[],
                assumptions=[],
                failure_modes=[],
                strategic_choices=[],
            ),
            _plan(active_dimensions=["market"]),
            None,
        )
        assert ev.overall_score < 0.75

    def test_overall_score_is_weighted_mean(self):
        # Give each criterion equal weight=1.0 and known scores → verify mean
        ev = TheoryEvaluator().build(
            _theory(
                recommended_option_id="OPT-A",   # score 1.0
                winning_position="P",             # score 1.0
                winning_mechanism="M",            # score 1.0
                strategic_choices=[],             # score 1.0 (no dims)
                evidence=[],                      # score 0.0
                assumptions=[],                   # score 0.0
                failure_modes=[],                 # score 0.5
                confidence="High",
            ),
            _plan(
                active_dimensions=[],
                weights={k: 1.0 for k in [
                    "option_identified", "position_articulated", "mechanism_defined",
                    "choice_completeness", "evidence_quality", "assumption_coverage",
                    "risk_awareness",
                ]},
            ),
            None,
        )
        # (1+1+1+1+0+0+0.5) / 7 ≈ 0.6429
        assert abs(ev.overall_score - (4.5 / 7.0)) < 1e-6


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def test_confidence_high_carried_from_theory(self):
        ev = TheoryEvaluator().build(_theory(confidence="High"), _plan(), None)
        assert ev.confidence == "High"

    def test_confidence_medium_carried_from_theory(self):
        ev = TheoryEvaluator().build(_theory(confidence="Medium"), _plan(), None)
        assert ev.confidence == "Medium"

    def test_confidence_low_carried_from_theory(self):
        ev = TheoryEvaluator().build(_theory(confidence="Low"), _plan(), None)
        assert ev.confidence == "Low"

    def test_confidence_case_insensitive(self):
        ev = TheoryEvaluator().build(_theory(confidence="high"), _plan(), None)
        assert ev.confidence == "High"

    def test_confidence_score_based_when_theory_confidence_empty(self):
        sparse = _theory(
            recommended_option_id="",
            winning_position="",
            winning_mechanism="",
            evidence=[],
            assumptions=[],
            failure_modes=[],
            strategic_choices=[],
            confidence="",
        )
        ev = TheoryEvaluator().build(sparse, _plan(active_dimensions=["market"]), None)
        assert ev.confidence in {"High", "Medium", "Low"}

    def test_confidence_high_when_score_above_threshold(self):
        rich = _theory(
            evidence=["E1", "E2", "E3"],
            assumptions=[{"statement": "A"}, {"statement": "B"}],
            failure_modes=[{"description": "R"}],
            strategic_choices=[],
            confidence="",
        )
        ev = TheoryEvaluator().build(rich, _plan(), None)
        if ev.overall_score >= 0.75:
            assert ev.confidence == "High"


# ---------------------------------------------------------------------------
# strengths and weaknesses
# ---------------------------------------------------------------------------

class TestStrengths:
    def test_strengths_is_list(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert isinstance(ev.strengths, list)

    def test_fully_populated_theory_has_strengths(self):
        ev = TheoryEvaluator().build(
            _theory(evidence=["E1", "E2", "E3"]),
            _plan(),
            None,
        )
        assert len(ev.strengths) > 0

    def test_success_conditions_appended_to_strengths(self):
        ev = TheoryEvaluator().build(
            _theory(success_conditions=["Policy tailwinds", "Technology maturity"]),
            _plan(),
            None,
        )
        assert "Policy tailwinds" in ev.strengths
        assert "Technology maturity" in ev.strengths

    def test_strengths_strings(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert all(isinstance(s, str) for s in ev.strengths)


class TestWeaknesses:
    def test_weaknesses_is_list(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert isinstance(ev.weaknesses, list)

    def test_sparse_theory_has_weaknesses(self):
        ev = TheoryEvaluator().build(
            _theory(evidence=[], assumptions=[], recommended_option_id=""),
            _plan(active_dimensions=["market"]),
            None,
        )
        assert len(ev.weaknesses) > 0

    def test_weaknesses_strings(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert all(isinstance(w, str) for w in ev.weaknesses)


# ---------------------------------------------------------------------------
# residual_risks
# ---------------------------------------------------------------------------

class TestResidualRisks:
    def test_residual_risks_carries_failure_modes(self):
        fm = [{"description": "Grid delay", "severity": "High"}]
        ev = TheoryEvaluator().build(_theory(failure_modes=fm), _plan(), None)
        assert ev.residual_risks == fm

    def test_residual_risks_empty_when_no_failure_modes(self):
        ev = TheoryEvaluator().build(_theory(failure_modes=[]), _plan(), None)
        assert ev.residual_risks == []

    def test_multiple_failure_modes_all_carried(self):
        fm = [
            {"description": "Risk A", "severity": "High"},
            {"description": "Risk B", "severity": "High"},
        ]
        ev = TheoryEvaluator().build(_theory(failure_modes=fm), _plan(), None)
        assert len(ev.residual_risks) == 2


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_metadata_contains_plan_id(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        assert "plan_id" in ev.metadata

    def test_metadata_plan_id_matches_plan(self):
        p = StrategyPlan(plan_id="PLAN-XYZ", framework="executive", active_dimensions=[])
        ev = TheoryEvaluator().build(_theory(), p, None)
        assert ev.metadata["plan_id"] == "PLAN-XYZ"

    def test_metadata_n_active_dimensions(self):
        ev = TheoryEvaluator().build(
            _theory(), _plan(active_dimensions=["market", "technology"]), None
        )
        assert ev.metadata["n_active_dimensions"] == 2

    def test_metadata_n_evidence(self):
        ev = TheoryEvaluator().build(_theory(evidence=["E1", "E2"]), _plan(), None)
        assert ev.metadata["n_evidence"] == 2

    def test_metadata_n_assumptions(self):
        ev = TheoryEvaluator().build(
            _theory(assumptions=[{"statement": "A"}]), _plan(), None
        )
        assert ev.metadata["n_assumptions"] == 1

    def test_metadata_n_failure_modes(self):
        fm = [{"description": "R1"}, {"description": "R2"}]
        ev = TheoryEvaluator().build(_theory(failure_modes=fm), _plan(), None)
        assert ev.metadata["n_failure_modes"] == 2


# ---------------------------------------------------------------------------
# Independent evaluation — theories do not influence each other
# ---------------------------------------------------------------------------

class TestIndependentEvaluation:
    def test_two_theories_produce_two_evaluations(self):
        gen = TheoryEvaluator()
        p = _plan()
        t1 = _theory(recommended_option_id="OPT-A", winning_mechanism="Deploy.")
        t2 = _theory(recommended_option_id="OPT-B", winning_mechanism="")
        ev1 = gen.build(t1, p, None)
        ev2 = gen.build(t2, p, None)
        assert ev1.theory_id == "OPT-A"
        assert ev2.theory_id == "OPT-B"

    def test_evaluations_differ_when_theories_differ(self):
        gen = TheoryEvaluator()
        p = _plan()
        t_rich = _theory(evidence=["E1", "E2", "E3"])
        t_sparse = _theory(evidence=[])
        ev_rich = gen.build(t_rich, p, None)
        ev_sparse = gen.build(t_sparse, p, None)
        assert ev_rich.overall_score != ev_sparse.overall_score

    def test_evaluation_is_frozen(self):
        ev = TheoryEvaluator().build(_theory(), _plan(), None)
        with pytest.raises(Exception):
            ev.overall_score = 0.0


# ---------------------------------------------------------------------------
# StrategyCoordinator — _evaluations attribute
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorEvaluations:
    def test_evaluations_empty_before_build(self):
        coord = StrategyCoordinator()
        assert coord._evaluations == []

    def test_evaluations_is_list_after_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert isinstance(coord._evaluations, list)

    def test_evaluations_has_three_elements(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._evaluations) == 3

    def test_evaluations_all_theory_evaluation(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert all(isinstance(ev, TheoryEvaluation) for ev in coord._evaluations)

    def test_evaluations_count_equals_theories_count(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        assert len(coord._evaluations) == len(coord._theories)

    def test_evaluations_updated_on_second_build(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        first_count = len(coord._evaluations)
        coord.build(_full_ctx())
        assert len(coord._evaluations) == first_count

    def test_all_evaluation_scores_in_range(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for ev in coord._evaluations:
            assert 0.0 <= ev.overall_score <= 1.0

    def test_all_evaluations_have_seven_criteria(self):
        coord = StrategyCoordinator()
        coord.build(_full_ctx())
        for ev in coord._evaluations:
            assert len(ev.criteria_scores) == 7


# ---------------------------------------------------------------------------
# StrategyCoordinator — StrategicPosition unchanged
# ---------------------------------------------------------------------------

class TestStrategyPositionUnchangedByEvaluations:
    def test_position_run_id_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.run_id == "run001"

    def test_position_question_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.question == "What should we do?"

    def test_position_theory_of_winning_still_populated(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.theory_of_winning is not None
        assert pos.theory_of_winning.recommended_option_id == "OPT-A"

    def test_position_recommendation_unchanged(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert pos.recommendation.recommended_option_id == "OPT-A"

    def test_position_does_not_carry_evaluations(self):
        coord = StrategyCoordinator()
        pos = coord.build(_full_ctx())
        assert not hasattr(pos, "evaluations")
        assert not hasattr(pos, "_evaluations")


# ---------------------------------------------------------------------------
# PH10.5a — Configured mode criterion selection (Issue 1)
# ---------------------------------------------------------------------------

class TestConfiguredModeSelection:
    """Criterion-selection algorithm: non-empty weights define the complete set."""

    def test_configured_mode_only_shows_configured_criteria(self):
        # Two custom criteria → exactly those two in result
        weights = {"option_identified": 1.0, "evidence_quality": 1.0}
        ev = TheoryEvaluator().build(_theory(), _plan(weights=weights), None)
        assert set(ev.criteria_scores.keys()) == {"option_identified", "evidence_quality"}

    def test_configured_mode_excludes_all_other_built_ins(self):
        # Configuring one criterion must not auto-add the other six
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"mechanism_defined": 2.0}), None
        )
        built_ins = {
            "option_identified", "position_articulated", "choice_completeness",
            "evidence_quality", "assumption_coverage", "risk_awareness",
        }
        for name in built_ins:
            assert name not in ev.criteria_scores

    def test_default_mode_still_produces_seven_criteria(self):
        # Empty weights → DEFAULT mode → all 7
        ev = TheoryEvaluator().build(_theory(), _plan(weights={}), None)
        expected = {
            "option_identified", "position_articulated", "mechanism_defined",
            "choice_completeness", "evidence_quality", "assumption_coverage",
            "risk_awareness",
        }
        assert set(ev.criteria_scores.keys()) == expected

    def test_unknown_configured_criterion_gets_fallback_score(self):
        # Unrecognised name → _FALLBACK_SCORE = 0.5
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"custom_alignment_score": 1.0}), None
        )
        assert ev.criteria_scores["custom_alignment_score"].score == 0.5

    def test_unknown_configured_criterion_gets_fallback_rationale(self):
        # Unrecognised name → rationale = "Criterion not recognized..."
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"custom_alignment_score": 1.0}), None
        )
        rationale = ev.criteria_scores["custom_alignment_score"].rationale
        assert "not recognized" in rationale.lower()

    def test_unknown_configured_criterion_uses_configured_weight(self):
        # The weight from the plan must be honoured even for unknown criteria
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"custom_alignment_score": 3.5}), None
        )
        assert ev.criteria_scores["custom_alignment_score"].weight == 3.5

    def test_mixed_known_and_unknown_criteria(self):
        # Known criterion scores normally; unknown gets fallback
        weights = {"option_identified": 2.0, "bespoke_dimension": 1.0}
        ev = TheoryEvaluator().build(_theory(), _plan(weights=weights), None)
        assert set(ev.criteria_scores.keys()) == {"option_identified", "bespoke_dimension"}
        assert ev.criteria_scores["option_identified"].score == 1.0  # OPT-A present
        assert ev.criteria_scores["bespoke_dimension"].score == 0.5  # fallback

    def test_configured_mode_weight_is_honored_for_known_criterion(self):
        # Custom weight for a recognised criterion is stored correctly
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"risk_awareness": 9.9}), None
        )
        assert ev.criteria_scores["risk_awareness"].weight == 9.9

    def test_configured_mode_unknown_criterion_is_neutral_in_overall_score(self):
        # Unknown criterion at 0.5 contributes proportionally; overall between 0 and 1
        ev = TheoryEvaluator().build(
            _theory(), _plan(weights={"mystery": 1.0}), None
        )
        assert ev.overall_score == 0.5


# ---------------------------------------------------------------------------
# PH10.5a — Three-tier rationale (Issue 2)
# ---------------------------------------------------------------------------

class TestRationaleTiers:
    """Rationale distinguishes high (>=0.75), partial (0<score<0.75), zero (0.0)."""

    # --- evidence_quality ---

    def test_evidence_rationale_high_tier(self):
        # 3 evidence items → score=1.0 → high tier: "Supporting evidence is cited. (…)"
        ev = TheoryEvaluator().build(_theory(evidence=["E1", "E2", "E3"]), _plan(), None)
        rationale = ev.criteria_scores["evidence_quality"].rationale
        assert rationale.startswith("Supporting evidence is cited.")
        assert "3 evidence item(s) cited" in rationale

    def test_evidence_rationale_partial_tier(self):
        # 1 evidence item → score≈0.33 → partial tier: detail string directly
        ev = TheoryEvaluator().build(_theory(evidence=["E1"]), _plan(), None)
        rationale = ev.criteria_scores["evidence_quality"].rationale
        # Must NOT start with the low_rationale "No supporting evidence is cited."
        assert not rationale.startswith("No supporting evidence is cited.")
        # Must be the detail string
        assert "1 evidence item(s) cited" in rationale

    def test_evidence_rationale_low_tier(self):
        # 0 evidence items → score=0.0 → low tier: low_rationale, no detail
        ev = TheoryEvaluator().build(_theory(evidence=[]), _plan(), None)
        rationale = ev.criteria_scores["evidence_quality"].rationale
        assert rationale == "No supporting evidence is cited."

    def test_evidence_partial_does_not_include_low_rationale_prefix(self):
        # The bug was: low_rationale + " (1 evidence item(s) cited)"
        # Fixed: only the detail string
        ev = TheoryEvaluator().build(_theory(evidence=["E1"]), _plan(), None)
        rationale = ev.criteria_scores["evidence_quality"].rationale
        assert "No supporting evidence" not in rationale

    def test_evidence_two_items_partial_tier(self):
        # 2 evidence items → score≈0.67 → partial tier
        ev = TheoryEvaluator().build(_theory(evidence=["E1", "E2"]), _plan(), None)
        rationale = ev.criteria_scores["evidence_quality"].rationale
        assert "2 evidence item(s) cited" in rationale
        assert "No supporting evidence" not in rationale

    # --- assumption_coverage ---

    def test_assumption_rationale_high_tier(self):
        # 2 assumptions → score=1.0 → high tier
        ev = TheoryEvaluator().build(
            _theory(assumptions=[{"statement": "A1"}, {"statement": "A2"}]),
            _plan(), None,
        )
        rationale = ev.criteria_scores["assumption_coverage"].rationale
        assert "Key assumptions are documented." in rationale
        assert "2 assumption(s) documented" in rationale

    def test_assumption_rationale_partial_tier(self):
        # 1 assumption → score=0.5 → partial tier: detail string
        ev = TheoryEvaluator().build(
            _theory(assumptions=[{"statement": "A1"}]), _plan(), None
        )
        rationale = ev.criteria_scores["assumption_coverage"].rationale
        assert "1 assumption(s) documented" in rationale
        assert "No assumptions are documented." not in rationale

    def test_assumption_rationale_low_tier(self):
        # 0 assumptions → score=0.0 → low tier
        ev = TheoryEvaluator().build(_theory(assumptions=[]), _plan(), None)
        rationale = ev.criteria_scores["assumption_coverage"].rationale
        assert rationale == "No assumptions are documented."

    # --- risk_awareness ---

    def test_risk_awareness_rationale_partial_when_no_failure_modes(self):
        # 0 failure modes → score=0.5 → partial tier: detail = "no failure modes identified"
        ev = TheoryEvaluator().build(_theory(failure_modes=[]), _plan(), None)
        rationale = ev.criteria_scores["risk_awareness"].rationale
        assert "no failure modes identified" in rationale
        # Must NOT be the low_rationale (which would only appear at score=0.0)
        assert rationale != "No failure modes have been identified."

    def test_risk_awareness_rationale_high_when_failure_modes_present(self):
        # 1+ failure modes → score=1.0 → high tier
        fm = [{"description": "Delay", "severity": "High"}]
        ev = TheoryEvaluator().build(_theory(failure_modes=fm), _plan(), None)
        rationale = ev.criteria_scores["risk_awareness"].rationale
        assert "Failure modes are identified." in rationale
        assert "1 failure mode(s) identified" in rationale

    # --- choice_completeness ---

    def test_choice_completeness_rationale_partial_tier(self):
        # 1 choice / 2 dims → score=0.5 → partial tier
        choices = [{"dimension": "market"}]
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=choices),
            _plan(active_dimensions=["market", "technology"]),
            None,
        )
        rationale = ev.criteria_scores["choice_completeness"].rationale
        assert "1 of 2 dimension(s) covered" in rationale
        assert "Active plan dimensions are incompletely covered." not in rationale

    def test_choice_completeness_rationale_high_tier(self):
        # 2 choices / 2 dims → score=1.0 → high tier
        choices = [{"dimension": "market"}, {"dimension": "technology"}]
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=choices),
            _plan(active_dimensions=["market", "technology"]),
            None,
        )
        rationale = ev.criteria_scores["choice_completeness"].rationale
        assert "All active plan dimensions are covered by choices." in rationale

    def test_choice_completeness_rationale_low_tier(self):
        # 0 choices / 1 dim → score=0.0 → low tier
        ev = TheoryEvaluator().build(
            _theory(strategic_choices=[]),
            _plan(active_dimensions=["market"]),
            None,
        )
        rationale = ev.criteria_scores["choice_completeness"].rationale
        assert rationale == "Active plan dimensions are incompletely covered."
