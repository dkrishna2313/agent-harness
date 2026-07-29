"""StrategyConfig — canonical configuration object for the Strategy Layer.

Defines strategic intent without changing platform code.
Frameworks provide defaults; engagements can override them in later phases.

PH9.0 scope: canonical Pydantic model with sensible defaults.
PH12.0 scope: ChoiceConfig, DimensionConfig, dimension_configs field on StrategyConfig.
PH12.1a scope: AlignmentPolicy, ScoringPolicy — policy blocks for configured evaluation.
PH12.2 scope: ContentConfig — theory content assignment configuration.
PH12.2b scope: ContentConfig extended with discrimination controls.
PH12.2a scope: Full policy surface — evaluation, constraint, mapping, alignment, reporting,
               diagnostics, relationship priority, discrimination, coverage, confidence,
               homogenization, and fallback config blocks. All defaults reproduce PH12.2b
               behavior exactly so no existing pipeline code is affected by the addition.
Not in scope: YAML loading, framework plugins.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Supported mapping confidence levels for AlignmentPolicy validation
_SUPPORTED_MAPPING_CONFIDENCES: frozenset[str] = frozenset({"High", "Medium", "Low", "None"})

# Allowed content type names for homogenization and fallback configs
_ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset({
    "assumptions",
    "risks",
    "opportunities",
    "recommendations",
    "evidence",
    "success_conditions",
})

# Diagnostic severity levels (ascending order: ignore < info < warning < error)
_VALID_SEVERITIES: frozenset[str] = frozenset({"ignore", "info", "warning", "error"})


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
# PH12.2a — Extended strategy configuration models
# ---------------------------------------------------------------------------

class StrategyCriterionConfig(BaseModel):
    """Weight and enabled flag for a single evaluation criterion."""

    weight: float = 0.0
    enabled: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Criterion weight must be in [0.0, 1.0], got {v}")
        return v


class StrategyEvaluationConfig(BaseModel):
    """Evaluation policy: criterion weights, scoring precision, and saturation detection.

    When weight_policy='strict', all enabled criteria weights must sum to 1.0 ± 0.001.
    When weight_policy='normalize', weights are normalized at runtime.
    """

    criteria: dict[str, StrategyCriterionConfig] = Field(default_factory=dict)
    weight_policy: str = "strict"
    score_precision: int = 6
    minimum_winner_margin: float = 0.05
    saturation_threshold: float = 0.95
    saturation_spread_threshold: float = 0.02
    tie_tolerance: float = 1e-6

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("weight_policy")
    @classmethod
    def _validate_weight_policy(cls, v: str) -> str:
        if v not in {"strict", "normalize"}:
            raise ValueError(f"weight_policy must be 'strict' or 'normalize', got {v!r}")
        return v

    @field_validator("score_precision")
    @classmethod
    def _validate_score_precision(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"score_precision must be a positive integer, got {v}")
        return v

    @field_validator(
        "minimum_winner_margin",
        "saturation_threshold",
        "saturation_spread_threshold",
        "tie_tolerance",
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_strict_weights(self) -> "StrategyEvaluationConfig":
        if self.weight_policy == "strict" and self.criteria:
            total = sum(c.weight for c in self.criteria.values() if c.enabled)
            if abs(total - 1.0) > 0.001:
                raise ValueError(
                    f"When weight_policy='strict', enabled criteria weights must sum to "
                    f"1.0 ± 0.001, got {total:.6f}"
                )
        elif self.weight_policy == "normalize" and self.criteria:
            total = sum(c.weight for c in self.criteria.values() if c.enabled)
            if total <= 0.0:
                raise ValueError(
                    "When weight_policy='normalize', at least one enabled criterion must "
                    "have a weight > 0.0 to allow normalization"
                )
        return self


class StrategyConstraintConfig(BaseModel):
    """Policy governing how hard constraint violations are handled."""

    required_violation_policy: str = "penalize"
    default_required_penalty: float = 0.15
    default_preferred_penalty: float = 0.05
    maximum_total_penalty: float = 0.50
    fail_on_unknown_constraint_reference: bool = True
    include_constraint_rationale_in_report: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("required_violation_policy")
    @classmethod
    def _validate_policy(cls, v: str) -> str:
        if v not in {"penalize", "disqualify", "warn"}:
            raise ValueError(
                f"required_violation_policy must be 'penalize', 'disqualify', or 'warn', got {v!r}"
            )
        return v

    @field_validator(
        "default_required_penalty",
        "default_preferred_penalty",
        "maximum_total_penalty",
    )
    @classmethod
    def _validate_penalty(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Penalty must be in [0.0, 1.0], got {v}")
        return v


class StrategyMappingConfidenceConfig(BaseModel):
    """Thresholds governing option-mapping confidence tier assignment."""

    minimum_authoritative_score: float = 0.20
    minimum_authoritative_margin: float = 0.05
    high_score_threshold: float = 0.40
    high_margin_threshold: float = 0.15
    disallow_high_with_contradiction: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "minimum_authoritative_score",
        "minimum_authoritative_margin",
        "high_score_threshold",
        "high_margin_threshold",
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_threshold_ordering(self) -> "StrategyMappingConfidenceConfig":
        if self.high_score_threshold < self.minimum_authoritative_score:
            raise ValueError(
                f"high_score_threshold ({self.high_score_threshold}) must be >= "
                f"minimum_authoritative_score ({self.minimum_authoritative_score})"
            )
        if self.high_margin_threshold < self.minimum_authoritative_margin:
            raise ValueError(
                f"high_margin_threshold ({self.high_margin_threshold}) must be >= "
                f"minimum_authoritative_margin ({self.minimum_authoritative_margin})"
            )
        return self


class StrategyMappingAuthorityConfig(BaseModel):
    """Authority guarantees for the option-mapping pass."""

    single_pass: bool = True
    fail_on_mapping_mismatch: bool = True
    allow_content_resolution_remap: bool = False

    model_config = {"frozen": True}


class StrategyMappingConfig(BaseModel):
    """Full option-mapping policy: authority, confidence, posture weights, and penalties."""

    enabled: bool = True
    unresolved_policy: str = "preserve_upstream"
    authority: StrategyMappingAuthorityConfig = Field(
        default_factory=StrategyMappingAuthorityConfig
    )
    confidence: StrategyMappingConfidenceConfig = Field(
        default_factory=StrategyMappingConfidenceConfig
    )
    posture_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "geographic": 0.35,
            "power": 0.30,
            "timing": 0.35,
        }
    )
    contradiction_penalties: dict[str, float] = Field(
        default_factory=lambda: {
            "geographic_hard": 0.35,
            "geographic_soft": 0.12,
            "power_hard": 0.30,
            "power_soft": 0.10,
            "timing_hard": 0.20,
            "timing_soft": 0.10,
        }
    )
    generic_overlap: dict[str, float] = Field(
        default_factory=lambda: {
            "per_match_weight": 0.01,
            "maximum_bonus": 0.10,
        }
    )

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("unresolved_policy")
    @classmethod
    def _validate_unresolved_policy(cls, v: str) -> str:
        if v not in {"preserve_upstream", "report_unresolved", "fail"}:
            raise ValueError(
                f"unresolved_policy must be 'preserve_upstream', 'report_unresolved', "
                f"or 'fail', got {v!r}"
            )
        return v

    @model_validator(mode="after")
    def _validate_weights_and_penalties(self) -> "StrategyMappingConfig":
        for k, v in self.posture_weights.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"posture_weights[{k!r}] must be in [0.0, 1.0], got {v}")
        p = self.contradiction_penalties
        for k, v in p.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"contradiction_penalties[{k!r}] must be in [0.0, 1.0], got {v}")
        for prefix in ("geographic", "power", "timing"):
            hard_key = f"{prefix}_hard"
            soft_key = f"{prefix}_soft"
            if hard_key in p and soft_key in p:
                if p[hard_key] < p[soft_key]:
                    raise ValueError(
                        f"contradiction_penalties[{hard_key!r}] ({p[hard_key]}) must be >= "
                        f"[{soft_key!r}] ({p[soft_key]})"
                    )
        for k, v in self.generic_overlap.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"generic_overlap[{k!r}] must be in [0.0, 1.0], got {v}")
        return self


class StrategyAlignmentConfig(BaseModel):
    """Alignment policy: margin thresholds, confidence gates, and status assignments."""

    enabled: bool = True
    minimum_challenge_margin: float = 0.10
    minimum_challenge_confidence: str = "Medium"
    same_option_high_confidence_status: str = "confirmed"
    same_option_medium_confidence_status: str = "refined"
    different_option_below_margin_status: str = "unresolved"
    low_mapping_confidence_status: str = "unresolved"
    unresolved_authority: str = "upstream_preferred"

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("minimum_challenge_margin")
    @classmethod
    def _validate_margin(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"minimum_challenge_margin must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("minimum_challenge_confidence")
    @classmethod
    def _validate_confidence(cls, v: str) -> str:
        if v not in {"Low", "Medium", "High"}:
            raise ValueError(
                f"minimum_challenge_confidence must be 'Low', 'Medium', or 'High', got {v!r}"
            )
        return v

    @field_validator(
        "same_option_high_confidence_status",
        "same_option_medium_confidence_status",
        "different_option_below_margin_status",
        "low_mapping_confidence_status",
    )
    @classmethod
    def _validate_status(cls, v: str) -> str:
        if v not in {"confirmed", "challenged", "refined", "unresolved"}:
            raise ValueError(
                f"Status must be one of 'confirmed', 'challenged', 'refined', 'unresolved', "
                f"got {v!r}"
            )
        return v

    @field_validator("unresolved_authority")
    @classmethod
    def _validate_authority(cls, v: str) -> str:
        if v not in {"upstream_preferred", "strategy_selected", "none"}:
            raise ValueError(
                f"unresolved_authority must be 'upstream_preferred', 'strategy_selected', "
                f"or 'none', got {v!r}"
            )
        return v


class StrategyRelationshipPriorityConfig(BaseModel):
    """Scoring priorities for content-to-theory relationship assignment.

    All values must be in [-1.0, 1.0]. posture_contradiction must be <= 0.
    explicit_option_link must be strictly greater than semantic_overlap.
    """

    explicit_option_link: float = 1.00
    explicit_recommendation_link: float = 0.80
    explicit_assumption_risk_link: float = 0.70
    direct_posture_match: float = 0.50
    compatible_posture_match: float = 0.25
    category_match: float = 0.15
    semantic_overlap: float = 0.05
    posture_contradiction: float = -0.50

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "explicit_option_link",
        "explicit_recommendation_link",
        "explicit_assumption_risk_link",
        "direct_posture_match",
        "compatible_posture_match",
        "category_match",
        "semantic_overlap",
        "posture_contradiction",
    )
    @classmethod
    def _validate_range(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"Priority must be in [-1.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_ordering(self) -> "StrategyRelationshipPriorityConfig":
        if self.posture_contradiction > 0:
            raise ValueError(
                f"posture_contradiction must be <= 0 (contradiction), "
                f"got {self.posture_contradiction}"
            )
        if self.explicit_option_link <= self.semantic_overlap:
            raise ValueError(
                f"explicit_option_link ({self.explicit_option_link}) must be > "
                f"semantic_overlap ({self.semantic_overlap})"
            )
        return self


class StrategyDiscriminationConfig(BaseModel):
    """Scores and multipliers for discrimination lineage classification."""

    unique_theory_score: float = 1.00
    subset_shared_score: float = 0.50
    global_shared_score: float = 0.00
    shared_context_multiplier: float = 0.35
    distinctive_content_multiplier: float = 1.00
    evidence_inherits_lineage: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "unique_theory_score",
        "subset_shared_score",
        "global_shared_score",
        "shared_context_multiplier",
        "distinctive_content_multiplier",
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Score/multiplier must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_score_ordering(self) -> "StrategyDiscriminationConfig":
        if not (
            self.unique_theory_score >= self.subset_shared_score >= self.global_shared_score
        ):
            raise ValueError(
                f"Discrimination scores must satisfy: "
                f"unique ({self.unique_theory_score}) >= "
                f"subset ({self.subset_shared_score}) >= "
                f"global ({self.global_shared_score})"
            )
        if self.shared_context_multiplier > self.distinctive_content_multiplier:
            raise ValueError(
                f"shared_context_multiplier ({self.shared_context_multiplier}) must be <= "
                f"distinctive_content_multiplier ({self.distinctive_content_multiplier})"
            )
        return self


class StrategyCoverageConfig(BaseModel):
    """Coverage threshold tiers: sufficient, partial, and fallback-heavy."""

    sufficient_threshold: float = 0.75
    partial_threshold: float = 0.50
    fallback_heavy_threshold: float = 0.25

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("sufficient_threshold", "partial_threshold", "fallback_heavy_threshold")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_ordering(self) -> "StrategyCoverageConfig":
        if not (
            self.sufficient_threshold >= self.partial_threshold >= self.fallback_heavy_threshold
        ):
            raise ValueError(
                f"Coverage thresholds must satisfy: "
                f"sufficient ({self.sufficient_threshold}) >= "
                f"partial ({self.partial_threshold}) >= "
                f"fallback_heavy ({self.fallback_heavy_threshold})"
            )
        return self


class StrategyContentConfidenceConfig(BaseModel):
    """Per-tier share thresholds for content confidence classification (High/Medium/Low).

    High thresholds must be at least as strict as Medium thresholds:
      - High minimums >= Medium minimums (harder to achieve)
      - high_maximum_fallback_share <= medium_maximum_fallback_share (stricter upper bound)
    """

    high_minimum_distinctive_share: float = 0.40
    high_minimum_evidence_share: float = 0.30
    high_maximum_shared_share: float = 0.60
    high_maximum_fallback_share: float = 0.10
    medium_minimum_distinctive_share: float = 0.20
    medium_minimum_evidence_share: float = 0.20
    medium_maximum_fallback_share: float = 0.35
    cap_at_medium_on_partial_homogenization: bool = True
    cap_at_low_on_substantial_homogenization: bool = True
    require_authoritative_mapping_for_high: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "high_minimum_distinctive_share",
        "high_minimum_evidence_share",
        "high_maximum_shared_share",
        "high_maximum_fallback_share",
        "medium_minimum_distinctive_share",
        "medium_minimum_evidence_share",
        "medium_maximum_fallback_share",
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Share threshold must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_strictness_ordering(self) -> "StrategyContentConfidenceConfig":
        if self.high_minimum_distinctive_share < self.medium_minimum_distinctive_share:
            raise ValueError(
                f"high_minimum_distinctive_share ({self.high_minimum_distinctive_share}) "
                f"must be >= medium_minimum_distinctive_share "
                f"({self.medium_minimum_distinctive_share})"
            )
        if self.high_minimum_evidence_share < self.medium_minimum_evidence_share:
            raise ValueError(
                f"high_minimum_evidence_share ({self.high_minimum_evidence_share}) "
                f"must be >= medium_minimum_evidence_share ({self.medium_minimum_evidence_share})"
            )
        if self.high_maximum_fallback_share > self.medium_maximum_fallback_share:
            raise ValueError(
                f"high_maximum_fallback_share ({self.high_maximum_fallback_share}) "
                f"must be <= medium_maximum_fallback_share ({self.medium_maximum_fallback_share})"
            )
        return self


class StrategyHomogenizationConfig(BaseModel):
    """Homogenization detection: thresholds, material dimensions, and reporting policy."""

    enabled: bool = True
    partial_threshold: float = 0.75
    substantial_threshold: float = 0.90
    full_threshold: float = 0.99
    maximum_identical_dimensions_before_partial: int = 2
    material_dimensions: list[str] = Field(
        default_factory=lambda: [
            "assumptions",
            "risks",
            "opportunities",
            "recommendations",
            "evidence",
            "success_conditions",
        ]
    )
    allow_relationship_based_justification: bool = True
    emit_report_warning: bool = True

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("partial_threshold", "substantial_threshold", "full_threshold")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {v}")
        return v

    @field_validator("maximum_identical_dimensions_before_partial")
    @classmethod
    def _validate_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(
                f"maximum_identical_dimensions_before_partial must be >= 0, got {v}"
            )
        return v

    @model_validator(mode="after")
    def _validate_homogenization(self) -> "StrategyHomogenizationConfig":
        if not (self.partial_threshold <= self.substantial_threshold <= self.full_threshold):
            raise ValueError(
                f"Thresholds must satisfy: "
                f"partial ({self.partial_threshold}) <= "
                f"substantial ({self.substantial_threshold}) <= "
                f"full ({self.full_threshold})"
            )
        invalid = set(self.material_dimensions) - _ALLOWED_CONTENT_TYPES
        if invalid:
            raise ValueError(
                f"Invalid material_dimensions: {sorted(invalid)}. "
                f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}"
            )
        return self


class StrategyFallbackConfig(BaseModel):
    """Fallback policy: which content types may fall back and what share is permitted."""

    enabled: bool = True
    allowed_content_types: list[str] = Field(
        default_factory=lambda: [
            "assumptions",
            "risks",
            "opportunities",
            "recommendations",
            "evidence",
            "success_conditions",
        ]
    )
    maximum_fallback_share: float = 0.40
    confidence_penalty: float = 0.20
    report_fallback_usage: bool = True
    fail_above_maximum_share: bool = False

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("maximum_fallback_share", "confidence_penalty")
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value must be in [0.0, 1.0], got {v}")
        return v

    @model_validator(mode="after")
    def _validate_content_types(self) -> "StrategyFallbackConfig":
        invalid = set(self.allowed_content_types) - _ALLOWED_CONTENT_TYPES
        if invalid:
            raise ValueError(
                f"Invalid allowed_content_types: {sorted(invalid)}. "
                f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}"
            )
        return self


class StrategyReportingAuthorityConfig(BaseModel):
    """Toggle flags for the authority section of the strategic direction report."""

    show_upstream_preferred_option: bool = True
    show_mapped_option: bool = True
    show_alignment_status: bool = True
    show_selection_status: bool = True
    show_scores: bool = True
    show_mapping_confidence: bool = True
    show_saturation_status: bool = True

    model_config = {"frozen": True, "extra": "allow"}


class StrategyReportingContentConfig(BaseModel):
    """Toggle flags for the content section of the strategic direction report."""

    show_why_strategy_won: bool = True
    show_linked_recommendations: bool = True
    show_key_assumptions: bool = True
    show_primary_risks: bool = True
    show_opportunities: bool = True
    show_success_conditions: bool = True
    show_theory_specific_evidence: bool = True
    show_shared_context: bool = True
    show_content_coverage: bool = True
    show_distinctive_coverage: bool = True
    show_content_confidence: bool = True
    show_homogenization_status: bool = True
    show_alternative_differentiation: bool = True

    model_config = {"frozen": True, "extra": "allow"}


class StrategyReportingMandatoryConfig(BaseModel):
    """Mandatory disclosure flags — when True the report must surface these conditions."""

    challenged_alignment: bool = True
    unresolved_alignment: bool = True
    low_mapping_confidence: bool = True
    saturation: bool = True
    substantial_homogenization: bool = True
    excessive_fallback: bool = True
    insufficient_coverage: bool = True

    model_config = {"frozen": True, "extra": "allow"}


class StrategyReportingLimitsConfig(BaseModel):
    """Display limits for per-section item counts and text truncation."""

    maximum_assumptions: int = 5
    maximum_risks: int = 5
    maximum_opportunities: int = 5
    maximum_evidence: int = 8
    maximum_success_conditions: int = 8
    maximum_alternatives: int = 5
    sentence_limit: int = 300

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "maximum_assumptions",
        "maximum_risks",
        "maximum_opportunities",
        "maximum_evidence",
        "maximum_success_conditions",
        "maximum_alternatives",
        "sentence_limit",
    )
    @classmethod
    def _validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Reporting limit must be a positive integer, got {v}")
        return v


class StrategyReportingConfig(BaseModel):
    """Unified reporting policy: section toggles, mandatory disclosures, and display limits."""

    enabled: bool = True
    authority: StrategyReportingAuthorityConfig = Field(
        default_factory=StrategyReportingAuthorityConfig
    )
    content: StrategyReportingContentConfig = Field(
        default_factory=StrategyReportingContentConfig
    )
    mandatory: StrategyReportingMandatoryConfig = Field(
        default_factory=StrategyReportingMandatoryConfig
    )
    limits: StrategyReportingLimitsConfig = Field(
        default_factory=StrategyReportingLimitsConfig
    )

    model_config = {"frozen": True, "extra": "allow"}


class StrategyDiagnosticsConfig(BaseModel):
    """Diagnostic severity policy for strategy pipeline conditions.

    mapping_mismatch and invalid_weight_sum are locked to 'error' and cannot be lowered.
    All other fields accept: 'ignore' | 'info' | 'warning' | 'error'.
    """

    unknown_reference: str = "error"
    mapping_mismatch: str = "error"
    invalid_weight_sum: str = "error"
    low_mapping_confidence: str = "warning"
    partial_homogenization: str = "warning"
    substantial_homogenization: str = "warning"
    fallback_heavy: str = "warning"
    insufficient_coverage: str = "warning"
    missing_report_section: str = "warning"

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "unknown_reference",
        "mapping_mismatch",
        "invalid_weight_sum",
        "low_mapping_confidence",
        "partial_homogenization",
        "substantial_homogenization",
        "fallback_heavy",
        "insufficient_coverage",
        "missing_report_section",
    )
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        if v not in _VALID_SEVERITIES:
            raise ValueError(f"Severity must be one of {sorted(_VALID_SEVERITIES)}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _validate_minimum_severities(self) -> "StrategyDiagnosticsConfig":
        if self.mapping_mismatch != "error":
            raise ValueError(
                f"mapping_mismatch severity cannot be below 'error', got {self.mapping_mismatch!r}"
            )
        if self.invalid_weight_sum != "error":
            raise ValueError(
                f"invalid_weight_sum severity cannot be below 'error', "
                f"got {self.invalid_weight_sum!r}"
            )
        return self


# ---------------------------------------------------------------------------
# PH12.2 — Theory content assignment configuration (extended in PH12.2a)
# ---------------------------------------------------------------------------

class ContentConfig(BaseModel):
    """Configuration for theory-specific content assignment (PH12.2).

    All fields have sensible defaults so PH12.2 runs without explicit configuration.
    Validation raises ValueError on out-of-range values — no silent clamping.
    PH12.2a adds: minimum_distinctive_coverage, maximum_shared_risks/opportunities,
    and six sub-model policy blocks for relationship priorities, discrimination,
    coverage, confidence, homogenization, and fallback.
    """

    minimum_relevance_score: float = 0.20
    maximum_assumptions_per_theory: int = 5
    maximum_risks_per_theory: int = 5
    maximum_opportunities_per_theory: int = 5
    maximum_recommendations_per_theory: int = 5
    maximum_evidence_per_theory: int = 12
    allow_symmetric_fallback: bool = True
    minimum_content_coverage: float = 0.50

    # PH12.2a — distinctive coverage threshold
    minimum_distinctive_coverage: float = 0.20

    # PH12.2b — discrimination controls
    minimum_discrimination_score: float = 0.20
    maximum_shared_assumptions: int = 5
    maximum_shared_risks: int = 5
    maximum_shared_opportunities: int = 5
    maximum_shared_recommendations: int = 5
    maximum_shared_evidence: int = 12
    partial_homogenization_threshold: float = 0.75
    full_homogenization_threshold: float = 0.95
    maximum_identical_dimensions: int = 2

    # PH12.2a — sub-model policy blocks
    relationship_priorities: StrategyRelationshipPriorityConfig = Field(
        default_factory=StrategyRelationshipPriorityConfig
    )
    discrimination: StrategyDiscriminationConfig = Field(
        default_factory=StrategyDiscriminationConfig
    )
    coverage_policy: StrategyCoverageConfig = Field(
        default_factory=StrategyCoverageConfig
    )
    confidence_policy: StrategyContentConfidenceConfig = Field(
        default_factory=StrategyContentConfidenceConfig
    )
    homogenization_policy: StrategyHomogenizationConfig = Field(
        default_factory=StrategyHomogenizationConfig
    )
    fallback_policy: StrategyFallbackConfig = Field(
        default_factory=StrategyFallbackConfig
    )

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator(
        "minimum_relevance_score",
        "minimum_content_coverage",
        "minimum_distinctive_coverage",
        "minimum_discrimination_score",
        "partial_homogenization_threshold",
        "full_homogenization_threshold",
    )
    @classmethod
    def _validate_fraction(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"Value must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator(
        "maximum_assumptions_per_theory",
        "maximum_risks_per_theory",
        "maximum_opportunities_per_theory",
        "maximum_recommendations_per_theory",
        "maximum_evidence_per_theory",
        "maximum_shared_assumptions",
        "maximum_shared_risks",
        "maximum_shared_opportunities",
        "maximum_shared_recommendations",
        "maximum_shared_evidence",
    )
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Maximum count must be a positive integer, got {v}")
        return v

    @field_validator("maximum_identical_dimensions")
    @classmethod
    def _validate_non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Value must be a non-negative integer, got {v}")
        return v

    @model_validator(mode="after")
    def _validate_homogenization_thresholds(self) -> "ContentConfig":
        if self.full_homogenization_threshold < self.partial_homogenization_threshold:
            raise ValueError(
                f"full_homogenization_threshold ({self.full_homogenization_threshold}) "
                f"must be >= partial_homogenization_threshold "
                f"({self.partial_homogenization_threshold})"
            )
        return self


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

    PH12.2a adds: enabled, config_version, and six new top-level policy blocks
    (evaluation_config, constraint_config, mapping_config, alignment_config,
    reporting_config, diagnostics_config). All existing fields are preserved
    for backward compatibility.
    """

    enabled: bool = True
    config_version: str = "ph12.2a-v1"
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

    # PH12.2 — theory content assignment configuration
    content: ContentConfig = Field(default_factory=ContentConfig)

    # PH12.2a — extended policy blocks
    evaluation_config: StrategyEvaluationConfig = Field(
        default_factory=StrategyEvaluationConfig
    )
    constraint_config: StrategyConstraintConfig = Field(
        default_factory=StrategyConstraintConfig
    )
    mapping_config: StrategyMappingConfig = Field(
        default_factory=StrategyMappingConfig
    )
    alignment_config: StrategyAlignmentConfig = Field(
        default_factory=StrategyAlignmentConfig
    )
    reporting_config: StrategyReportingConfig = Field(
        default_factory=StrategyReportingConfig
    )
    diagnostics_config: StrategyDiagnosticsConfig = Field(
        default_factory=StrategyDiagnosticsConfig
    )

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def compute_fingerprint(self) -> str:
        """Return the first 16 hex characters of the SHA-256 hash of the canonical JSON.

        The canonical form sorts all keys and uses compact separators so the result
        is deterministic regardless of insertion order or Python version.
        """
        canonical = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        return cls.model_validate(data)

    @classmethod
    def from_yaml_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
        """Construct a StrategyConfig from a YAML-loaded dict, handling legacy field names.

        This is the canonical entry point for YAML-sourced configuration. Add entries
        to _FIELD_RENAMES when a field is renamed in a later phase so older engagement
        files continue to work without modification.
        """
        _FIELD_RENAMES: dict[str, str] = {
            # Example: "old_field_name": "new_field_name"
            # No renames defined as of PH12.2a; extend here as needed.
        }
        normalized: dict[str, Any] = {
            _FIELD_RENAMES.get(k, k): v for k, v in data.items()
        }

        # Engagement YAMLs store objectives/dimensions/constraints as lists.
        # StrategyObjectives/StrategyDimensions/StrategyConstraints expect dicts.
        # Coerce list shapes here so old engagement files continue to validate.
        if isinstance(normalized.get("objectives"), list):
            normalized["objectives"] = {"primary": normalized["objectives"]}

        if isinstance(normalized.get("dimensions"), list):
            # Convert [{id, title, ...}, ...] → {id: {title, ...}, ...}
            raw_dims: list[dict] = normalized["dimensions"]
            normalized["dimensions"] = {
                d["id"]: {k: v for k, v in d.items() if k != "id"}
                for d in raw_dims
                if isinstance(d, dict) and "id" in d
            }

        if isinstance(normalized.get("constraints"), list):
            normalized["constraints"] = {"required_conditions": normalized["constraints"]}

        return cls.model_validate(normalized)
