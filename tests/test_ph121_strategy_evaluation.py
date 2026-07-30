"""PH12.1 — Theory-Specific Evaluation and Recommendation Alignment tests.

Covers:
  - ConstraintEvaluator: avoidance and preservation constraints
  - SaturationDetector: detection and message
  - OptionMapper: keyword matching and confidence tiers
  - AlignmentEvaluator: confirmed / refined / challenged / unresolved
  - TheoryEvaluator: constraint penalty in strategic_fit / risk_resilience
  - TheoryEvaluator: execution_complexity-based execution_feasibility
  - TheoryEvaluator: wait_and_monitor penalty in opportunity_capture
  - TheoryEvaluator: assumption_robustness with /3 denominator
  - StrategySelection: new optional PH12.1 fields present and defaulted
  - StrategicChoiceGenerator: execution_complexity in metadata
  - ConfigurationResolver: parses execution_complexity into ChoiceConfig
  - Full end-to-end: score differentiation across 3 configured theories
"""
from __future__ import annotations

import pytest

from functional_agents.strategy.alignment import (
    AlignmentResult,
    ConstraintResult,
    OptionMapping,
)
from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
from functional_agents.strategy.constraint_evaluator import ConstraintEvaluator
from functional_agents.strategy.option_mapper import OptionMapper
from functional_agents.strategy.saturation_detector import SaturationDetector
from functional_agents.strategy.strategy_selector import StrategySelection
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation
from functional_agents.strategy.theory_evaluator import TheoryEvaluator
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_plan import (
    EvaluationModel,
    GenerationPolicy,
    StrategyPlan,
    ValidationPolicy,
)
from functional_agents.strategy.strategy_config import ChoiceConfig, DimensionConfig
from functional_agents.strategy.configuration_resolver import ConfigurationResolver
from functional_agents.strategy.strategy_config import StrategyConfig, StrategyConstraints


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_plan(
    weights: dict | None = None,
    constraints: list[str] | None = None,
) -> StrategyPlan:
    return StrategyPlan(
        plan_id="TEST-PLAN",
        framework="executive",
        active_dimensions=["d1", "d2", "d3"],
        constraints=constraints or [],
        evaluation_model=EvaluationModel(weights=weights or {}),
        generation_policy=GenerationPolicy(),
        validation_policy=ValidationPolicy(),
    )


def _make_theory(
    theory_id: str = "TH-1",
    choices: list[dict] | None = None,
    failure_modes: list[dict] | None = None,
    success_conditions: list[str] | None = None,
    assumptions: list | None = None,
    winning_position: str = "Position A",
    winning_mechanism: str = "Mechanism A",
) -> TheoryOfWinning:
    return TheoryOfWinning(
        theory_id=theory_id,
        source_choice_set_id=f"SCS-{theory_id}",
        recommended_option_id="opt1",
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        strategic_choices=choices or [],
        failure_modes=failure_modes or [],
        success_conditions=success_conditions or [],
        assumptions=assumptions or [],
        evidence=["ev1", "ev2", "ev3"],
    )


def _make_constraint_result(status: str) -> ConstraintResult:
    return ConstraintResult(
        constraint="test constraint",
        status=status,
        score={"satisfied": 1.0, "partially_satisfied": 0.5,
               "violated": 0.0, "not_assessable": 0.75}[status],
        rationale=f"status={status}",
    )


def _make_evaluation(score: float, theory_id: str = "TH-1") -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=theory_id,
        criteria_scores={"c1": CriterionScore(score=score, rationale="r", weight=1.0)},
        overall_score=score,
        confidence="Medium",
        strengths=[],
        weaknesses=[],
        residual_risks=[],
    )


# ---------------------------------------------------------------------------
# ConstraintEvaluator
# ---------------------------------------------------------------------------

class TestConstraintEvaluator:

    def test_avoidance_violated_by_concentrated_choice(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "concentrated",
            "metadata": {"choice_title": "Concentrated"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Avoid strategies dependent on single-state concentration"
        ])
        results = ce.evaluate(theory, plan)
        assert len(results) == 1
        assert results[0].status == "violated"
        assert results[0].score == 0.0

    def test_avoidance_satisfied_by_diversified_choice(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "diversified",
            "metadata": {"choice_title": "Diversified"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Avoid strategies dependent on single-state concentration"
        ])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "satisfied"
        assert results[0].score == 1.0

    def test_preservation_satisfied_by_diversified(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "diversified",
            "metadata": {"choice_title": "Diversified"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Preserve at least one alternative state for contingency development"
        ])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "satisfied"
        assert results[0].score == 1.0

    def test_preservation_partially_satisfied_by_staged(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "staged",
            "metadata": {"choice_title": "Staged Portfolio"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Preserve at least one alternative state for contingency development"
        ])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "partially_satisfied"
        assert results[0].score == 0.5

    def test_preservation_violated_by_concentrated(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "concentrated",
            "metadata": {"choice_title": "Concentrated"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Preserve at least one alternative state for contingency development"
        ])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "violated"
        assert results[0].score == 0.0

    def test_excluded_option_violated(self):
        ce = ConstraintEvaluator()
        theory = _make_theory()
        # theory.recommended_option_id = "opt1"
        plan = _make_plan(constraints=["excluded_option:opt1"])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "violated"

    def test_excluded_option_satisfied(self):
        ce = ConstraintEvaluator()
        theory = _make_theory()
        plan = _make_plan(constraints=["excluded_option:opt99"])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "satisfied"

    def test_not_assessable_unknown_prefix(self):
        ce = ConstraintEvaluator()
        theory = _make_theory()
        plan = _make_plan(constraints=["unknown_prefix:some constraint"])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "not_assessable"
        assert results[0].score == 0.75

    def test_not_assessable_unrecognised_verb(self):
        ce = ConstraintEvaluator()
        theory = _make_theory()
        plan = _make_plan(constraints=[
            "required_condition:Consider alternative approaches where possible"
        ])
        results = ce.evaluate(theory, plan)
        assert results[0].status == "not_assessable"

    def test_no_constraints_returns_empty(self):
        ce = ConstraintEvaluator()
        theory = _make_theory()
        plan = _make_plan(constraints=[])
        results = ce.evaluate(theory, plan)
        assert results == []

    def test_multiple_constraints_independent(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "concentrated",
            "metadata": {"choice_title": "Concentrated"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Avoid strategies dependent on single-state concentration",
            "required_condition:Preserve at least one alternative state for contingency development",
        ])
        results = ce.evaluate(theory, plan)
        assert len(results) == 2
        assert all(r.status == "violated" for r in results)

    def test_diversified_satisfies_both_constraints(self):
        ce = ConstraintEvaluator()
        theory = _make_theory(choices=[{
            "selected_value": "diversified",
            "metadata": {"choice_title": "Diversified"},
        }])
        plan = _make_plan(constraints=[
            "required_condition:Avoid strategies dependent on single-state concentration",
            "required_condition:Preserve at least one alternative state for contingency development",
        ])
        results = ce.evaluate(theory, plan)
        assert all(r.status == "satisfied" for r in results)


# ---------------------------------------------------------------------------
# SaturationDetector
# ---------------------------------------------------------------------------

class TestSaturationDetector:

    def test_all_same_score_1_0_detected(self):
        evals = [_make_evaluation(1.0, f"T{i}") for i in range(3)]
        detected, msg = SaturationDetector().check(evals)
        assert detected is True
        assert "1.0" in msg
        assert "presence-only" in msg

    def test_all_same_score_non_1_0_detected(self):
        evals = [_make_evaluation(0.75, f"T{i}") for i in range(3)]
        detected, msg = SaturationDetector().check(evals)
        assert detected is True
        assert "saturation detected" in msg.lower()

    def test_different_scores_not_saturated(self):
        evals = [
            _make_evaluation(0.9, "T1"),
            _make_evaluation(0.7, "T2"),
            _make_evaluation(0.5, "T3"),
        ]
        detected, _ = SaturationDetector().check(evals)
        assert detected is False

    def test_single_evaluation_not_saturated(self):
        evals = [_make_evaluation(1.0, "T1")]
        detected, msg = SaturationDetector().check(evals)
        assert detected is False
        assert "not applicable" in msg.lower()

    def test_empty_list_not_saturated(self):
        detected, _ = SaturationDetector().check([])
        assert detected is False


# ---------------------------------------------------------------------------
# OptionMapper
# ---------------------------------------------------------------------------

class TestOptionMapper:

    def _make_research_with_options(self, options: list[dict]):
        class FakeResearch:
            strategic_options = options
        return FakeResearch()

    def test_high_confidence_match(self):
        research = self._make_research_with_options([
            {"option_id": "opt_diversified", "title": "Diversified Portfolio",
             "description": "Spread investment across diversified states"},
        ])
        theory = _make_theory(choices=[{
            "selected_value": "diversified",
            "metadata": {"choice_title": "Diversified", "choice_description": "spread investment"},
        }])
        result = OptionMapper().map(theory, research)
        assert result.mapped_option_id == "opt_diversified"
        assert result.mapping_confidence in ("High", "Medium", "Low")

    def test_no_options_returns_none_confidence(self):
        class FakeResearch:
            strategic_options = []
        theory = _make_theory()
        result = OptionMapper().map(theory, FakeResearch())
        assert result.mapped_option_id is None
        assert result.mapping_confidence == "None"

    def test_confidence_tiers(self):
        # Instance method since PH12.2a (thresholds now come from mapping_config)
        mapper = OptionMapper()
        assert mapper._confidence(0.80, 0.30, False) == "High"
        assert mapper._confidence(0.30, 0.10, False) == "Medium"
        assert mapper._confidence(0.10, 0.05, False) == "Low"
        assert mapper._confidence(0.00, 0.00, False) == "None"

    def test_no_keywords_returns_none_confidence(self):
        research = self._make_research_with_options([
            {"option_id": "opt1", "title": "Some Option", "description": "desc"},
        ])
        # Theory with no metadata to extract keywords
        theory = _make_theory(choices=[])
        result = OptionMapper().map(theory, research)
        assert result.mapped_option_id is None


# ---------------------------------------------------------------------------
# AlignmentEvaluator
# ---------------------------------------------------------------------------

class TestAlignmentEvaluator:

    def _make_selection(
        self,
        score_margin: float = 0.1,
        tie_breaker: str | None = None,
    ) -> StrategySelection:
        return StrategySelection(
            winner_theory_id="TH-1",
            winner_score=0.85,
            runner_up_theory_id="TH-2",
            runner_up_score=0.85 - score_margin,
            score_margin=score_margin,
            tie_breaker_used=tie_breaker,
        )

    def _make_mapping(
        self,
        mapped_id: str | None = "opt1",
        confidence: str = "High",
    ) -> OptionMapping:
        return OptionMapping(
            mapped_option_id=mapped_id,
            mapping_score=0.8,
            mapping_confidence=confidence,
        )

    def _make_research(self, preferred_id: str = "") -> object:
        class Res:
            preferred_option = {"option_id": preferred_id} if preferred_id else {}
        return Res()

    def test_confirmed_when_same_option_clear_margin(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.15)
        mapping = self._make_mapping("opt1", "High")
        research = self._make_research("opt1")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "confirmed"

    def test_refined_when_same_option_narrow_margin(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.02)  # < default threshold 0.05
        mapping = self._make_mapping("opt1", "High")
        research = self._make_research("opt1")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "refined"

    def test_challenged_when_different_option_clear_margin(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.15)
        mapping = self._make_mapping("opt_other", "High")
        research = self._make_research("opt1")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "challenged"

    def test_unresolved_when_no_preferred_option(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.15)
        mapping = self._make_mapping("opt1", "High")
        research = self._make_research("")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "unresolved"

    def test_unresolved_when_low_confidence(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.15)
        mapping = self._make_mapping("opt1", "Low")
        research = self._make_research("opt1")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "unresolved"

    def test_unresolved_when_tie_breaker_used(self):
        theory = _make_theory("TH-1")
        sel = self._make_selection(0.0, "order")
        mapping = self._make_mapping("opt1", "High")
        research = self._make_research("opt1")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "unresolved"

    def test_result_is_frozen(self):
        result = AlignmentResult(status="confirmed", rationale="ok")
        with pytest.raises(Exception):
            result.status = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TheoryEvaluator — constraint penalties
# ---------------------------------------------------------------------------

class TestTheoryEvaluatorPH121:

    def _plan_with_weights(self, weights: dict, constraints: list[str] | None = None) -> StrategyPlan:
        return StrategyPlan(
            plan_id="TP",
            framework="executive",
            active_dimensions=["d1", "d2"],
            constraints=constraints or [],
            evaluation_model=EvaluationModel(weights=weights),
            generation_policy=GenerationPolicy(),
            validation_policy=ValidationPolicy(),
        )

    def test_strategic_fit_no_penalty_when_no_constraints(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=[])
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(1.0, abs=1e-3)

    def test_strategic_fit_penalty_for_violated_constraint(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        cr = [_make_constraint_result("violated")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        # base=1.0, penalty=0.25*1=0.25 → 0.75
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(0.75, abs=1e-3)

    def test_strategic_fit_penalty_for_two_violated_constraints(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        cr = [_make_constraint_result("violated"), _make_constraint_result("violated")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        # base=1.0, penalty=0.50 → 0.50
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(0.50, abs=1e-3)

    def test_strategic_fit_penalty_for_partial_constraint(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        cr = [_make_constraint_result("partially_satisfied")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        # base=1.0, penalty=0.10*1=0.10 → 0.90
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(0.90, abs=1e-3)

    def test_strategic_fit_clamped_to_zero(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        cr = [_make_constraint_result("violated")] * 6  # penalty 1.5 > 1.0 base
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        assert ev.criteria_scores["strategic_fit"].score == 0.0

    def test_risk_resilience_penalty_for_violated_constraint(self):
        plan = self._plan_with_weights({"risk_resilience": 1.0})
        theory = _make_theory(failure_modes=[{"severity": "high", "description": "risk1"}])
        cr = [_make_constraint_result("violated")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        # n_failure=1 → base=0.6, penalty=0.25 → 0.35
        assert ev.criteria_scores["risk_resilience"].score == pytest.approx(0.35, abs=1e-3)

    def test_risk_resilience_no_failure_modes_with_penalty(self):
        plan = self._plan_with_weights({"risk_resilience": 1.0})
        theory = _make_theory(failure_modes=[])
        cr = [_make_constraint_result("violated"), _make_constraint_result("violated")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        # base=0.3, penalty=0.5 → clamped to 0.0
        assert ev.criteria_scores["risk_resilience"].score == 0.0

    def test_execution_feasibility_from_complexity_metadata(self):
        plan = self._plan_with_weights({"execution_feasibility": 1.0})
        theory = _make_theory(choices=[
            {"selected_value": "opt1", "metadata": {"execution_complexity": "high"}},
            {"selected_value": "opt2", "metadata": {"execution_complexity": "medium"}},
            {"selected_value": "opt3", "metadata": {"execution_complexity": "low"}},
        ])
        ev = TheoryEvaluator().build(theory, plan, None)
        # (0.5 + 0.75 + 1.0) / 3 = 0.75
        assert ev.criteria_scores["execution_feasibility"].score == pytest.approx(0.75, abs=1e-3)

    def test_execution_feasibility_high_complexity_scores_0_5(self):
        plan = self._plan_with_weights({"execution_feasibility": 1.0})
        theory = _make_theory(choices=[
            {"selected_value": "c1", "metadata": {"execution_complexity": "high"}},
            {"selected_value": "c2", "metadata": {"execution_complexity": "high"}},
        ])
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["execution_feasibility"].score == pytest.approx(0.5, abs=1e-3)

    def test_execution_feasibility_low_complexity_scores_1_0(self):
        plan = self._plan_with_weights({"execution_feasibility": 1.0})
        theory = _make_theory(choices=[
            {"selected_value": "c1", "metadata": {"execution_complexity": "low"}},
        ])
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["execution_feasibility"].score == pytest.approx(1.0, abs=1e-3)

    def test_execution_feasibility_unknown_complexity_defaults_075(self):
        plan = self._plan_with_weights({"execution_feasibility": 1.0})
        theory = _make_theory(choices=[
            {"selected_value": "c1", "metadata": {"execution_complexity": "unknown"}},
        ])
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["execution_feasibility"].score == pytest.approx(0.75, abs=1e-3)

    def test_opportunity_capture_wait_and_monitor_penalty(self):
        plan = self._plan_with_weights({"opportunity_capture": 1.0})
        theory = _make_theory(
            choices=[{"selected_value": "wait_and_monitor", "metadata": {}}],
            success_conditions=["c1", "c2", "c3"],  # base=1.0
        )
        ev = TheoryEvaluator().build(theory, plan, None)
        # base=1.0, penalty=0.15 → 0.85
        assert ev.criteria_scores["opportunity_capture"].score == pytest.approx(0.85, abs=1e-3)

    def test_opportunity_capture_no_penalty_without_wait(self):
        plan = self._plan_with_weights({"opportunity_capture": 1.0})
        theory = _make_theory(
            choices=[{"selected_value": "accelerate", "metadata": {}}],
            success_conditions=["c1", "c2", "c3"],
        )
        ev = TheoryEvaluator().build(theory, plan, None)
        assert ev.criteria_scores["opportunity_capture"].score == pytest.approx(1.0, abs=1e-3)

    def test_assumption_robustness_denominator_3(self):
        plan = self._plan_with_weights({"assumption_robustness": 1.0})
        theory = _make_theory(assumptions=[{"text": "a1"}, {"text": "a2"}, {"text": "a3"}])
        ev = TheoryEvaluator().build(theory, plan, None)
        # min(3/3, 1.0) = 1.0
        assert ev.criteria_scores["assumption_robustness"].score == pytest.approx(1.0, abs=1e-3)

    def test_assumption_robustness_one_assumption(self):
        plan = self._plan_with_weights({"assumption_robustness": 1.0})
        theory = _make_theory(assumptions=[{"text": "a1"}])
        ev = TheoryEvaluator().build(theory, plan, None)
        # min(1/3, 1.0) ≈ 0.333
        assert ev.criteria_scores["assumption_robustness"].score == pytest.approx(1 / 3, abs=1e-3)

    def test_no_constraint_results_produces_no_penalty(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0, "risk_resilience": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=None)
        # No constraint_results → no penalty
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(1.0, abs=1e-3)

    def test_satisfied_constraints_produce_no_penalty(self):
        plan = self._plan_with_weights({"strategic_fit": 1.0})
        theory = _make_theory(winning_position="P", winning_mechanism="M")
        cr = [_make_constraint_result("satisfied"), _make_constraint_result("satisfied")]
        ev = TheoryEvaluator().build(theory, plan, None, constraint_results=cr)
        assert ev.criteria_scores["strategic_fit"].score == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# StrategySelection PH12.1 fields
# ---------------------------------------------------------------------------

class TestStrategySelectionPH121Fields:

    def test_default_selection_status(self):
        sel = StrategySelection(winner_theory_id="T1", winner_score=0.8)
        assert sel.selection_status == "selected"

    def test_default_alignment_status_empty(self):
        sel = StrategySelection(winner_theory_id="T1", winner_score=0.8)
        assert sel.alignment_status == ""

    def test_default_saturation_false(self):
        sel = StrategySelection(winner_theory_id="T1", winner_score=0.8)
        assert sel.saturation_detected is False

    def test_default_mapped_option_id_none(self):
        sel = StrategySelection(winner_theory_id="T1", winner_score=0.8)
        assert sel.mapped_option_id is None

    def test_selection_is_frozen(self):
        sel = StrategySelection(winner_theory_id="T1", winner_score=0.8)
        with pytest.raises(Exception):
            sel.saturation_detected = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConfigurationResolver: execution_complexity in ChoiceConfig
# ---------------------------------------------------------------------------

class TestConfigurationResolverExecutionComplexity:

    def _config_with_choices(self, choices_raw: list[dict]) -> StrategyConfig:
        return StrategyConfig(
            framework="executive",
            version="1.0",
        )

    def test_execution_complexity_parsed_into_choice_config(self):
        resolver = ConfigurationResolver()
        engagement = {
            "dimensions": [{
                "id": "d1",
                "title": "Dimension 1",
                "required": True,
                "choices": [
                    {"id": "c1", "title": "Choice 1", "execution_complexity": "high"},
                    {"id": "c2", "title": "Choice 2", "execution_complexity": "low"},
                ],
            }]
        }
        resolved = resolver.resolve_from_engagement(StrategyConfig(), engagement)
        assert len(resolved.dimension_configs) == 1
        dim = resolved.dimension_configs[0]
        c1 = next(c for c in dim.choices if c.id == "c1")
        c2 = next(c for c in dim.choices if c.id == "c2")
        assert getattr(c1, "execution_complexity", None) == "high"
        assert getattr(c2, "execution_complexity", None) == "low"

    def test_missing_execution_complexity_does_not_fail(self):
        resolver = ConfigurationResolver()
        engagement = {
            "dimensions": [{
                "id": "d1",
                "title": "Dimension 1",
                "required": True,
                "choices": [{"id": "c1", "title": "Choice 1"}],
            }]
        }
        resolved = resolver.resolve_from_engagement(StrategyConfig(), engagement)
        dim = resolved.dimension_configs[0]
        choice = dim.choices[0]
        # Should not raise; execution_complexity may or may not be present
        _ = getattr(choice, "execution_complexity", None)


# ---------------------------------------------------------------------------
# Score differentiation integration test
# ---------------------------------------------------------------------------

class TestScoreDifferentiation:
    """Verify that the three theories produce meaningfully different scores."""

    def _make_configured_plan(self) -> StrategyPlan:
        return StrategyPlan(
            plan_id="DIFF-PLAN",
            framework="executive",
            active_dimensions=["geographic_portfolio", "power_pathway", "market_timing"],
            constraints=[
                "required_condition:Avoid strategies dependent on unvalidated single-state concentration",
                "required_condition:Preserve at least one alternative state for contingency development",
            ],
            evaluation_model=EvaluationModel(weights={
                "strategic_fit": 2.0,
                "execution_feasibility": 1.5,
                "risk_resilience": 1.5,
                "opportunity_capture": 1.0,
            }),
            generation_policy=GenerationPolicy(),
            validation_policy=ValidationPolicy(),
        )

    def _make_concentrated_theory(self) -> TheoryOfWinning:
        return TheoryOfWinning(
            theory_id="TH-concentrated",
            source_choice_set_id="SCS-0",
            recommended_option_id="concentrated",
            winning_position="Concentrated | Grid First | Accelerate",
            winning_mechanism="Accelerate: ...",
            strategic_choices=[
                {"selected_value": "concentrated", "metadata": {
                    "choice_title": "Concentrated", "execution_complexity": "high"
                }},
                {"selected_value": "grid_first", "metadata": {
                    "choice_title": "Grid First", "execution_complexity": "high"
                }},
                {"selected_value": "accelerate", "metadata": {
                    "choice_title": "Accelerate", "execution_complexity": "high"
                }},
            ],
            success_conditions=["sc1", "sc2", "sc3"],
            failure_modes=[{"severity": "high", "description": "r1"},
                           {"severity": "high", "description": "r2"}],
            assumptions=[{"text": "a1"}, {"text": "a2"}],
            evidence=["e1", "e2"],
        )

    def _make_diversified_theory(self) -> TheoryOfWinning:
        return TheoryOfWinning(
            theory_id="TH-diversified",
            source_choice_set_id="SCS-1",
            recommended_option_id="diversified",
            winning_position="Diversified | BTM First | Milestone Gated",
            winning_mechanism="Milestone Gated: ...",
            strategic_choices=[
                {"selected_value": "diversified", "metadata": {
                    "choice_title": "Diversified", "execution_complexity": "medium"
                }},
                {"selected_value": "btm_first", "metadata": {
                    "choice_title": "BTM First", "execution_complexity": "medium"
                }},
                {"selected_value": "milestone_gated", "metadata": {
                    "choice_title": "Milestone Gated", "execution_complexity": "medium"
                }},
            ],
            success_conditions=["sc1", "sc2", "sc3"],
            failure_modes=[{"severity": "high", "description": "r1"}],
            assumptions=[{"text": "a1"}, {"text": "a2"}, {"text": "a3"}],
            evidence=["e1", "e2"],
        )

    def _make_staged_theory(self) -> TheoryOfWinning:
        return TheoryOfWinning(
            theory_id="TH-staged",
            source_choice_set_id="SCS-2",
            recommended_option_id="staged",
            winning_position="Staged | Hybrid | Wait and Monitor",
            winning_mechanism="Wait and Monitor: ...",
            strategic_choices=[
                {"selected_value": "staged", "metadata": {
                    "choice_title": "Staged Portfolio", "execution_complexity": "medium"
                }},
                {"selected_value": "hybrid", "metadata": {
                    "choice_title": "Hybrid", "execution_complexity": "high"
                }},
                {"selected_value": "wait_and_monitor", "metadata": {
                    "choice_title": "Wait and Monitor", "execution_complexity": "low"
                }},
            ],
            success_conditions=["sc1", "sc2"],
            failure_modes=[{"severity": "high", "description": "r1"}],
            assumptions=[{"text": "a1"}],
            evidence=["e1"],
        )

    def test_concentrated_scores_lower_than_diversified(self):
        plan = self._make_configured_plan()
        evaluator = TheoryEvaluator()
        ce = ConstraintEvaluator()

        t0 = self._make_concentrated_theory()
        t1 = self._make_diversified_theory()

        cr0 = ce.evaluate(t0, plan)
        cr1 = ce.evaluate(t1, plan)

        ev0 = evaluator.build(t0, plan, None, cr0)
        ev1 = evaluator.build(t1, plan, None, cr1)

        assert ev0.overall_score < ev1.overall_score, (
            f"Expected concentrated ({ev0.overall_score:.3f}) < "
            f"diversified ({ev1.overall_score:.3f})"
        )

    def test_concentrated_violated_two_constraints(self):
        plan = self._make_configured_plan()
        ce = ConstraintEvaluator()
        t0 = self._make_concentrated_theory()
        results = ce.evaluate(t0, plan)
        violated = [r for r in results if r.status == "violated"]
        assert len(violated) == 2

    def test_diversified_satisfies_all_constraints(self):
        plan = self._make_configured_plan()
        ce = ConstraintEvaluator()
        t1 = self._make_diversified_theory()
        results = ce.evaluate(t1, plan)
        assert all(r.status == "satisfied" for r in results)

    def test_staged_partially_satisfies_preservation_constraint(self):
        plan = self._make_configured_plan()
        ce = ConstraintEvaluator()
        t2 = self._make_staged_theory()
        results = ce.evaluate(t2, plan)
        # First: avoidance (satisfied — not concentrated)
        # Second: preservation (partially_satisfied — staged)
        statuses = {r.status for r in results}
        assert "partially_satisfied" in statuses

    def test_wait_and_monitor_applies_opportunity_penalty(self):
        plan = self._make_configured_plan()
        evaluator = TheoryEvaluator()
        ce = ConstraintEvaluator()
        t2 = self._make_staged_theory()
        cr2 = ce.evaluate(t2, plan)
        ev2 = evaluator.build(t2, plan, None, cr2)
        # Without penalty: n=2 → base=0.8; with penalty: 0.65
        assert ev2.criteria_scores["opportunity_capture"].score == pytest.approx(0.65, abs=1e-3)

    def test_three_theories_produce_distinct_scores(self):
        plan = self._make_configured_plan()
        evaluator = TheoryEvaluator()
        ce = ConstraintEvaluator()
        theories = [
            self._make_concentrated_theory(),
            self._make_diversified_theory(),
            self._make_staged_theory(),
        ]
        scores = []
        for t in theories:
            cr = ce.evaluate(t, plan)
            ev = evaluator.build(t, plan, None, cr)
            scores.append(ev.overall_score)
        assert len(set(round(s, 4) for s in scores)) == 3, (
            f"Expected 3 distinct scores; got {scores}"
        )

    def test_saturation_not_detected_with_constraint_scoring(self):
        plan = self._make_configured_plan()
        evaluator = TheoryEvaluator()
        ce = ConstraintEvaluator()
        theories = [
            self._make_concentrated_theory(),
            self._make_diversified_theory(),
            self._make_staged_theory(),
        ]
        evals = [evaluator.build(t, plan, None, ce.evaluate(t, plan)) for t in theories]
        detected, _ = SaturationDetector().check(evals)
        assert detected is False
