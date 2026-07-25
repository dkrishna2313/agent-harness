"""PH9.3 — StrategyPlan and StrategyPlanner unit tests.

Covers:
- StrategyPlan model: fields, defaults, serialization
- StrategyPlanner: objectives merged, evaluation mapped, generation mapped,
  validation mapped, search_budget derived, dimensions extracted,
  constraints translated
- StrategyCoordinator: holds _plan, plan reflects config
- Determinism: same config always produces same plan structure
- Immutability: config not mutated during planning
"""

import pytest

from functional_agents.strategy import (
    ConfigurationResolver,
    EvaluationModel,
    FrameworkDefaults,
    GenerationPolicy,
    SearchBudget,
    StrategyConfig,
    StrategyCoordinator,
    StrategyPlan,
    StrategyPlanner,
    StrategyConstraints,
    StrategyDimensions,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyObjectives,
    StrategyValidation,
    ValidationPolicy,
)


# ---------------------------------------------------------------------------
# StrategyPlan model
# ---------------------------------------------------------------------------

class TestStrategyPlanModel:
    def test_default_instantiation(self):
        plan = StrategyPlan()
        assert plan.plan_id == ""
        assert plan.framework == ""
        assert plan.active_dimensions == []
        assert plan.objectives == []
        assert plan.constraints == []

    def test_typed_section_defaults(self):
        plan = StrategyPlan()
        assert isinstance(plan.evaluation_model, EvaluationModel)
        assert isinstance(plan.generation_policy, GenerationPolicy)
        assert isinstance(plan.validation_policy, ValidationPolicy)
        assert isinstance(plan.search_budget, SearchBudget)

    def test_evaluation_model_defaults(self):
        em = EvaluationModel()
        assert em.method == "multi_criteria"
        assert em.weights == {}
        assert em.min_score_threshold == 0.0

    def test_generation_policy_defaults(self):
        gp = GenerationPolicy()
        assert gp.max_candidates == 3
        assert gp.diversity_required is True

    def test_validation_policy_defaults(self):
        vp = ValidationPolicy()
        assert vp.require_evidence is False
        assert vp.min_confidence == ""
        assert vp.require_assumptions is False

    def test_search_budget_defaults(self):
        sb = SearchBudget()
        assert sb.max_iterations == 1
        assert sb.max_candidates == 3

    def test_serialization_round_trip(self):
        plan = StrategyPlan(
            plan_id="SPLAN-20260725",
            framework="executive",
            objectives=["win"],
            active_dimensions=["financial"],
        )
        d = plan.to_dict()
        restored = StrategyPlan.from_dict(d)
        assert restored.plan_id == plan.plan_id
        assert restored.framework == plan.framework
        assert restored.objectives == plan.objectives
        assert restored.active_dimensions == plan.active_dimensions

    def test_to_dict_returns_dict(self):
        plan = StrategyPlan()
        assert isinstance(plan.to_dict(), dict)
        assert "plan_id" in plan.to_dict()
        assert "evaluation_model" in plan.to_dict()


# ---------------------------------------------------------------------------
# StrategyPlanner — objectives
# ---------------------------------------------------------------------------

class TestStrategyPlannerObjectives:
    def test_primary_objectives_included(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(primary=["primary one", "primary two"])
        )
        plan = StrategyPlanner().build(cfg)
        assert "primary one" in plan.objectives
        assert "primary two" in plan.objectives

    def test_secondary_objectives_included(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(secondary=["secondary one"])
        )
        plan = StrategyPlanner().build(cfg)
        assert "secondary one" in plan.objectives

    def test_primary_before_secondary(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(
                primary=["first"],
                secondary=["second"],
            )
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.objectives == ["first", "second"]

    def test_empty_objectives_produces_empty_list(self):
        cfg = StrategyConfig(objectives=StrategyObjectives())
        plan = StrategyPlanner().build(cfg)
        assert plan.objectives == []

    def test_executive_defaults_objectives_propagate(self):
        cfg = ConfigurationResolver().resolve(StrategyConfig())
        plan = StrategyPlanner().build(cfg)
        exec_defaults = FrameworkDefaults.get("executive")
        expected = exec_defaults.objectives.primary + exec_defaults.objectives.secondary
        assert plan.objectives == expected


# ---------------------------------------------------------------------------
# StrategyPlanner — evaluation model
# ---------------------------------------------------------------------------

class TestStrategyPlannerEvaluationModel:
    def test_method_mapped(self):
        cfg = StrategyConfig(evaluation=StrategyEvaluation(method="scoring"))
        plan = StrategyPlanner().build(cfg)
        assert plan.evaluation_model.method == "scoring"

    def test_weights_mapped(self):
        cfg = StrategyConfig(
            evaluation=StrategyEvaluation(weights={"financial": 0.6, "risk": 0.4})
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.evaluation_model.weights == {"financial": 0.6, "risk": 0.4}

    def test_min_score_threshold_mapped(self):
        cfg = StrategyConfig(
            evaluation=StrategyEvaluation(min_score_threshold=0.5)
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.evaluation_model.min_score_threshold == 0.5


# ---------------------------------------------------------------------------
# StrategyPlanner — generation policy
# ---------------------------------------------------------------------------

class TestStrategyPlannerGenerationPolicy:
    def test_max_candidates_mapped(self):
        cfg = StrategyConfig(generation=StrategyGeneration(max_candidates=5))
        plan = StrategyPlanner().build(cfg)
        assert plan.generation_policy.max_candidates == 5

    def test_diversity_required_mapped(self):
        cfg = StrategyConfig(
            generation=StrategyGeneration(diversity_required=False)
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.generation_policy.diversity_required is False


# ---------------------------------------------------------------------------
# StrategyPlanner — validation policy
# ---------------------------------------------------------------------------

class TestStrategyPlannerValidationPolicy:
    def test_require_evidence_mapped(self):
        cfg = StrategyConfig(
            validation=StrategyValidation(require_evidence=True)
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.validation_policy.require_evidence is True

    def test_min_confidence_mapped(self):
        cfg = StrategyConfig(
            validation=StrategyValidation(min_confidence="High")
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.validation_policy.min_confidence == "High"

    def test_require_assumptions_mapped(self):
        cfg = StrategyConfig(
            validation=StrategyValidation(require_assumptions=True)
        )
        plan = StrategyPlanner().build(cfg)
        assert plan.validation_policy.require_assumptions is True


# ---------------------------------------------------------------------------
# StrategyPlanner — search budget
# ---------------------------------------------------------------------------

class TestStrategyPlannerSearchBudget:
    def test_max_iterations_is_one(self):
        cfg = StrategyConfig()
        plan = StrategyPlanner().build(cfg)
        assert plan.search_budget.max_iterations == 1

    def test_max_candidates_derived_from_generation(self):
        cfg = StrategyConfig(generation=StrategyGeneration(max_candidates=7))
        plan = StrategyPlanner().build(cfg)
        assert plan.search_budget.max_candidates == 7


# ---------------------------------------------------------------------------
# StrategyPlanner — constraints
# ---------------------------------------------------------------------------

class TestStrategyPlannerConstraints:
    def test_excluded_options_translated(self):
        cfg = StrategyConfig(
            constraints=StrategyConstraints(excluded_options=["OPT-Z"])
        )
        plan = StrategyPlanner().build(cfg)
        assert any("OPT-Z" in c for c in plan.constraints)

    def test_required_conditions_translated(self):
        cfg = StrategyConfig(
            constraints=StrategyConstraints(required_conditions=["market_ready"])
        )
        plan = StrategyPlanner().build(cfg)
        assert any("market_ready" in c for c in plan.constraints)

    def test_empty_constraints_produce_empty_list(self):
        cfg = StrategyConfig(constraints=StrategyConstraints())
        plan = StrategyPlanner().build(cfg)
        assert plan.constraints == []


# ---------------------------------------------------------------------------
# StrategyPlanner — dimensions
# ---------------------------------------------------------------------------

class TestStrategyPlannerDimensions:
    def test_empty_dimensions_produce_empty_list(self):
        cfg = StrategyConfig()
        plan = StrategyPlanner().build(cfg)
        assert plan.active_dimensions == []

    def test_framework_field_mapped(self):
        cfg = StrategyConfig(framework="executive")
        plan = StrategyPlanner().build(cfg)
        assert plan.framework == "executive"


# ---------------------------------------------------------------------------
# StrategyPlanner — immutability and determinism
# ---------------------------------------------------------------------------

class TestStrategyPlannerImmutabilityAndDeterminism:
    def test_config_not_mutated(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(primary=["original"])
        )
        original_primary = list(cfg.objectives.primary)
        StrategyPlanner().build(cfg)
        assert cfg.objectives.primary == original_primary

    def test_plan_is_new_object(self):
        cfg = StrategyConfig()
        p1 = StrategyPlanner().build(cfg)
        p2 = StrategyPlanner().build(cfg)
        assert p1 is not p2

    def test_same_config_produces_same_structure(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(primary=["win"]),
            generation=StrategyGeneration(max_candidates=4),
        )
        p1 = StrategyPlanner().build(cfg)
        p2 = StrategyPlanner().build(cfg)
        assert p1.objectives == p2.objectives
        assert p1.generation_policy.max_candidates == p2.generation_policy.max_candidates
        assert p1.framework == p2.framework


# ---------------------------------------------------------------------------
# StrategyCoordinator — holds _plan
# ---------------------------------------------------------------------------

class TestStrategyCoordinatorHasPlan:
    def test_coordinator_has_plan_attribute(self):
        coord = StrategyCoordinator()
        assert hasattr(coord, "_plan")

    def test_plan_is_strategy_plan_instance(self):
        coord = StrategyCoordinator()
        assert isinstance(coord._plan, StrategyPlan)

    def test_plan_framework_matches_config(self):
        coord = StrategyCoordinator()
        assert coord._plan.framework == coord._config.framework

    def test_plan_objectives_derived_from_resolved_config(self):
        coord = StrategyCoordinator()
        exec_defaults = FrameworkDefaults.get("executive")
        expected = exec_defaults.objectives.primary + exec_defaults.objectives.secondary
        assert coord._plan.objectives == expected

    def test_custom_config_flows_through_to_plan(self):
        cfg = StrategyConfig(
            objectives=StrategyObjectives(primary=["custom goal"]),
            generation=StrategyGeneration(max_candidates=5),
        )
        coord = StrategyCoordinator(config=cfg)
        assert "custom goal" in coord._plan.objectives
        assert coord._plan.generation_policy.max_candidates == 5

    def test_coordinator_config_still_accessible(self):
        coord = StrategyCoordinator()
        assert isinstance(coord._config, StrategyConfig)
