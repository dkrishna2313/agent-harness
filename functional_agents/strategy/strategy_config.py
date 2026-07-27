"""StrategyConfig — canonical configuration object for the Strategy Layer (PH9.0/PH12.0/PH12.1a).

Defines strategic intent without changing platform code.
Frameworks provide defaults; engagements can override them in later phases.

PH9.0 scope: canonical Pydantic model with sensible defaults.
PH12.0 scope: ChoiceConfig, DimensionConfig, dimension_configs field on StrategyConfig.
PH12.1a scope: AlignmentPolicy, ScoringPolicy — policy blocks for configured evaluation.
Not in scope: YAML loading, framework plugins.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# Supported mapping confidence levels for AlignmentPolicy validation
_SUPPORTED_MAPPING_CONFIDENCES: frozenset[str] = frozenset({"High", "Medium", "Low", "None"})


# ---------------------------------------------------------------------------
# PH12.1a — Policy models
# ---------------------------------------------------------------------------

class AlignmentPolicy(BaseModel):
    """Policy governing how upstream recommendation and selected theory are aligned."""

    preferred_option_authority: bool = True
    minimum_challenge_margin: float = 0.05
    unresolved_on_tie: bool = True
    # Minimum OptionMapping confidence to proceed to confirmed/refined/challenged
    # "High" | "Medium" | "Low" | "None"
    minimum_mapping_confidence: str = "Medium"

    model_config = {"frozen": True, "extra": "allow"}


class ScoringPolicy(BaseModel):
    """Policy governing configured-mode theory scoring penalties and detection."""

    constraint_violation_penalty: float = 0.25
    partial_constraint_penalty: float = 0.10
    wait_and_monitor_penalty: float = 0.15
    saturation_detection: bool = True

    model_config = {"frozen": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# PH12.0 — Engagement dimension models
# ---------------------------------------------------------------------------

class ChoiceConfig(BaseModel):
    """A single selectable choice within a strategic dimension."""

    id: str = ""
    title: str = ""
    description: str = ""

    model_config = {"frozen": True, "extra": "allow"}


class DimensionConfig(BaseModel):
    """A configured strategic decision dimension with its available choices.

    Each required dimension must be covered exactly once per StrategicChoiceSet.
    """

    id: str = ""
    title: str = ""
    description: str = ""
    required: bool = True
    choices: list[ChoiceConfig] = Field(default_factory=list)

    model_config = {"frozen": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------

class StrategyObjectives(BaseModel):
    """What the strategy aims to achieve."""

    primary: list[str] = Field(default_factory=list)
    secondary: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class StrategyDimensions(BaseModel):
    """Evaluation axes along which strategic options are scored.

    Keys are dimension names; values are dimension descriptors.
    An empty instance means "derive dimensions from the decision model."
    """

    model_config = {"extra": "allow"}

    def add(self, name: str, descriptor: Any = None) -> None:
        """Add a named dimension, stored in Pydantic's extra field dict.

        Uses __pydantic_extra__ directly so the dimension survives
        model_dump() / model_validate() round-trips.
        """
        if self.__pydantic_extra__ is None:
            object.__setattr__(self, "__pydantic_extra__", {name: descriptor})
        else:
            self.__pydantic_extra__[name] = descriptor


class StrategyEvaluation(BaseModel):
    """How competing theories are evaluated and ranked."""

    method: str = "multi_criteria"
    weights: dict[str, float] = Field(default_factory=dict)
    min_score_threshold: float = 0.0

    model_config = {"extra": "allow"}


class StrategyGeneration(BaseModel):
    """How candidate theories of winning are generated."""

    max_candidates: int = 3
    diversity_required: bool = True

    model_config = {"extra": "allow"}


class StrategyConstraints(BaseModel):
    """Hard constraints that any selected theory must satisfy."""

    excluded_options: list[str] = Field(default_factory=list)
    required_conditions: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class StrategyValidation(BaseModel):
    """Validation policies applied before a theory is accepted."""

    require_evidence: bool = False
    min_confidence: str = ""
    require_assumptions: bool = False

    model_config = {"extra": "allow"}


class StrategyMetadata(BaseModel):
    """Metadata about this strategy configuration."""

    author: str = ""
    engagement_id: str = ""
    notes: str = ""

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# StrategyConfig — root object
# ---------------------------------------------------------------------------

class StrategyConfig(BaseModel):
    """Canonical configuration for the Strategy Layer.

    An empty StrategyConfig represents the default behavior — no constraints,
    no custom objectives, standard multi-criteria evaluation. All fields have
    sensible defaults so the Strategy Layer runs correctly with no config.

    Version field enables schema negotiation in future phases when YAML
    loading and framework plugins are introduced.
    """

    version: str = "1.0"
    framework: str = "executive"

    objectives: StrategyObjectives = Field(default_factory=StrategyObjectives)
    dimensions: StrategyDimensions = Field(default_factory=StrategyDimensions)
    evaluation: StrategyEvaluation = Field(default_factory=StrategyEvaluation)
    generation: StrategyGeneration = Field(default_factory=StrategyGeneration)
    constraints: StrategyConstraints = Field(default_factory=StrategyConstraints)
    validation: StrategyValidation = Field(default_factory=StrategyValidation)
    metadata: StrategyMetadata = Field(default_factory=StrategyMetadata)

    # PH12.0 — structured dimension definitions with choices
    # When non-empty, supersedes StrategyDimensions.extra for active_dimensions
    dimension_configs: list[DimensionConfig] = Field(default_factory=list)

    # PH12.1a — policy blocks (optional; defaults preserve PH12.1 behavior)
    alignment_policy: AlignmentPolicy = Field(default_factory=AlignmentPolicy)
    scoring_policy: ScoringPolicy = Field(default_factory=ScoringPolicy)

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls.model_validate(data)
