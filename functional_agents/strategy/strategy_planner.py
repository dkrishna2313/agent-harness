"""StrategyPlanner — transforms a resolved StrategyConfig into a StrategyPlan (PH9.3).

Sits between ConfigurationResolver and StrategyCoordinator in the pipeline:

    Resolved StrategyConfig
            │
            ▼
    StrategyPlanner.build()
            │
            ▼
    StrategyPlan
            │
            ▼
    StrategyCoordinator

Responsibilities:
  - Map resolved configuration fields to concrete plan values
  - Derive active dimensions from configured dimension names
  - Merge primary and secondary objectives into an ordered objectives list
  - Translate constraint fields into plan-level constraint strings
  - Set search budget from generation policy

Rules:
  - No LLM calls
  - No reasoning
  - No framework plugins
  - No engagement overrides
  - Deterministic: same config always produces the same plan
  - Does not mutate the input StrategyConfig

Not in scope for PH9.3:
  - StrategicChoice generation
  - Theory generation or evaluation
  - Engagement overrides
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .strategy_config import StrategyConfig
from .strategy_plan import (
    EvaluationModel,
    GenerationPolicy,
    SearchBudget,
    StrategyPlan,
    ValidationPolicy,
)

LOGGER = logging.getLogger(__name__)


class StrategyPlanner:
    """Maps a resolved StrategyConfig to an executable StrategyPlan.

    Deterministic and stateless — no LLM calls, no side effects.
    """

    def build(self, config: StrategyConfig) -> StrategyPlan:
        """Produce a StrategyPlan from a resolved StrategyConfig.

        Does not mutate config. Always returns a new StrategyPlan instance.
        """
        created_at = datetime.now(timezone.utc).isoformat()
        plan_id = f"SPLAN-{created_at[:10].replace('-', '')}"

        active_dimensions = self._extract_dimensions(config)
        objectives = self._build_objectives(config)
        constraints = self._build_constraints(config)
        evaluation_model = self._build_evaluation_model(config)
        generation_policy = self._build_generation_policy(config)
        validation_policy = self._build_validation_policy(config)
        search_budget = self._build_search_budget(config)

        plan = StrategyPlan(
            plan_id=plan_id,
            created_at=created_at,
            framework=config.framework,
            active_dimensions=active_dimensions,
            objectives=objectives,
            constraints=constraints,
            evaluation_model=evaluation_model,
            generation_policy=generation_policy,
            validation_policy=validation_policy,
            search_budget=search_budget,
            dimension_configs=list(config.dimension_configs),
            alignment_policy=config.alignment_policy,
            scoring_policy=config.scoring_policy,
            content_config=config.content,
        )

        LOGGER.debug(
            "[StrategyPlanner] plan built: framework=%r dimensions=%d objectives=%d",
            plan.framework,
            len(plan.active_dimensions),
            len(plan.objectives),
        )

        return plan

    # ------------------------------------------------------------------
    # Internal mappers — each reads from config, returns a plan value
    # ------------------------------------------------------------------

    def _extract_dimensions(self, config: StrategyConfig) -> list[str]:
        """Derive active dimension names from configured dimensions.

        PH12.0: if dimension_configs is non-empty, use their IDs (preserving order).
        Legacy: StrategyDimensions uses extra fields to store named dimensions;
        an empty StrategyDimensions means dimensions will be derived at
        theory-generation time from the decision model.
        """
        if config.dimension_configs:
            return [d.id for d in config.dimension_configs if d.id]
        extra = config.dimensions.model_extra or {}
        return sorted(extra.keys())

    def _build_objectives(self, config: StrategyConfig) -> list[str]:
        """Merge primary and secondary objectives into an ordered list.

        Primary objectives come first; secondary objectives follow.
        """
        return list(config.objectives.primary) + list(config.objectives.secondary)

    def _build_constraints(self, config: StrategyConfig) -> list[str]:
        """Translate constraint fields into plan-level constraint strings."""
        result: list[str] = []
        for opt in config.constraints.excluded_options:
            result.append(f"excluded_option:{opt}")
        for cond in config.constraints.required_conditions:
            result.append(f"required_condition:{cond}")
        return result

    def _build_evaluation_model(self, config: StrategyConfig) -> EvaluationModel:
        return EvaluationModel(
            method=config.evaluation.method,
            weights=dict(config.evaluation.weights),
            min_score_threshold=config.evaluation.min_score_threshold,
        )

    def _build_generation_policy(self, config: StrategyConfig) -> GenerationPolicy:
        return GenerationPolicy(
            max_candidates=config.generation.max_candidates,
            diversity_required=config.generation.diversity_required,
        )

    def _build_validation_policy(self, config: StrategyConfig) -> ValidationPolicy:
        return ValidationPolicy(
            require_evidence=config.validation.require_evidence,
            min_confidence=config.validation.min_confidence,
            require_assumptions=config.validation.require_assumptions,
        )

    def _build_search_budget(self, config: StrategyConfig) -> SearchBudget:
        """Derive search budget from generation policy.

        PH9.3: single iteration; max_candidates taken from generation policy.
        """
        return SearchBudget(
            max_iterations=1,
            max_candidates=config.generation.max_candidates,
        )
