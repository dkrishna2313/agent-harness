"""StrategyPlan — executable plan produced by StrategyPlanner (PH9.3).

Canonical object that represents the resolved, executable strategy search plan.
Consumed by the theory generation and evaluation steps in future phases.

Produced by: StrategyPlanner.build(resolved_config)
Consumed by: StrategyCoordinator (PH9.3), TheoryGenerator (future)

Relationship to StrategyConfig:
  StrategyConfig  →  declarative intent (what the user configured)
  StrategyPlan    →  executable plan   (what the planner decided to do)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .strategy_config import DimensionConfig


# ---------------------------------------------------------------------------
# Plan section models
# ---------------------------------------------------------------------------

class EvaluationModel(BaseModel):
    """How competing theories are evaluated and ranked."""

    method: str = "multi_criteria"
    weights: dict[str, float] = Field(default_factory=dict)
    min_score_threshold: float = 0.0

    model_config = {"extra": "allow"}


class GenerationPolicy(BaseModel):
    """How candidate theories of winning are generated."""

    max_candidates: int = 3
    diversity_required: bool = True

    model_config = {"extra": "allow"}


class ValidationPolicy(BaseModel):
    """Validation rules a theory must satisfy before acceptance."""

    require_evidence: bool = False
    min_confidence: str = ""
    require_assumptions: bool = False

    model_config = {"extra": "allow"}


class SearchBudget(BaseModel):
    """Resource envelope for the strategy search process."""

    max_iterations: int = 1
    max_candidates: int = 3

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# StrategyPlan — root object
# ---------------------------------------------------------------------------

class StrategyPlan(BaseModel):
    """Executable plan for the strategy generation process.

    Produced by StrategyPlanner from a resolved StrategyConfig. All values
    are concrete — no further resolution is required before running the
    theory generation and evaluation pipeline.
    """

    plan_id: str = ""
    created_at: str = ""
    framework: str = ""

    # Dimensions that theories will be evaluated against
    active_dimensions: list[str] = Field(default_factory=list)

    # Objectives the strategy should optimise for (merged primary + secondary)
    objectives: list[str] = Field(default_factory=list)

    # Hard constraints expressed as plain strings for plan consumers
    constraints: list[str] = Field(default_factory=list)

    # Typed plan sections
    evaluation_model: EvaluationModel = Field(default_factory=EvaluationModel)
    generation_policy: GenerationPolicy = Field(default_factory=GenerationPolicy)
    validation_policy: ValidationPolicy = Field(default_factory=ValidationPolicy)
    search_budget: SearchBudget = Field(default_factory=SearchBudget)

    # PH12.0 — structured dimension configs forwarded from StrategyConfig
    dimension_configs: list[DimensionConfig] = Field(default_factory=list)

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyPlan":
        return cls.model_validate(data)
