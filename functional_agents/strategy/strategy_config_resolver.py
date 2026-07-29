"""strategy_config_resolver — load, validate, and resolve StrategyConfig from YAML.

PH12.2a scope:
- Resolve strategy config from engagement YAML (under the ``strategy:`` key)
- Apply defaults for absent fields; track which fields used defaults
- Translate deprecated field names to their canonical equivalents with warnings
- Return a frozen ResolvedStrategyConfig that includes fingerprint, source, and
  full provenance (defaults_applied, deprecations, warnings)
- Export a JSON schema catalogue for UI / validation tooling
"""

from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field

from .strategy_config import (
    AlignmentPolicy,
    ContentConfig,
    ScoringPolicy,
    StrategyConfig,
    StrategyAlignmentConfig,
    StrategyConstraintConfig,
    StrategyDiagnosticsConfig,
    StrategyEvaluationConfig,
    StrategyMappingConfig,
    StrategyReportingConfig,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIG_VERSION = "ph12.2a-v1"
_KNOWN_CONFIG_VERSIONS: frozenset[str] = frozenset({"ph12.2a-v1"})

# Top-level fields that exist on StrategyConfig — used for default tracking.
# We only track the fields added in PH12.2a and the major policy blocks; the
# older fine-grained sub-fields (objectives, dimensions, etc.) are omitted to
# keep the defaults_applied list useful rather than noisy.
_TOP_LEVEL_POLICY_FIELDS: tuple[str, ...] = (
    "enabled",
    "config_version",
    "version",
    "framework",
    "objectives",
    "dimensions",
    "evaluation",
    "generation",
    "constraints",
    "validation",
    "metadata",
    "dimension_configs",
    "alignment_policy",
    "scoring_policy",
    "content",
    "evaluation_config",
    "constraint_config",
    "mapping_config",
    "alignment_config",
    "reporting_config",
    "diagnostics_config",
)


# ---------------------------------------------------------------------------
# ResolvedStrategyConfig
# ---------------------------------------------------------------------------

class ResolvedStrategyConfig(BaseModel):
    """Fully resolved, validated strategy configuration with full provenance.

    ``raw`` is the caller-supplied YAML dict (unmodified).
    ``resolved`` is the final StrategyConfig with all defaults applied.
    ``defaults_applied`` lists the top-level field paths whose values came
    entirely from defaults (i.e., the caller did not supply them).
    ``deprecations`` captures old-field → new-field translation notices.
    ``warnings`` captures non-fatal validation notices.
    ``fingerprint`` is the first 16 hex chars of the SHA-256 of the canonical
    JSON of ``resolved``.
    """

    raw: dict[str, Any]
    resolved: StrategyConfig
    defaults_applied: list[str]
    deprecations: list[str]
    warnings: list[str]
    fingerprint: str
    source: str = "engagement_yaml"
    config_version: str = _CONFIG_VERSION

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _track_defaults(normalized_input: dict[str, Any]) -> list[str]:
    """Return field paths whose values were not supplied in the YAML input."""
    return [
        field
        for field in _TOP_LEVEL_POLICY_FIELDS
        if field not in normalized_input
    ]


def _migrate_alignment_policy(
    data: dict[str, Any],
    deprecations: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Copy legacy ``alignment_policy`` → ``alignment_config`` when the new key is absent.

    The legacy AlignmentPolicy has a subset of the fields that StrategyAlignmentConfig
    exposes, so we map the overlapping scalars and let StrategyAlignmentConfig defaults
    fill the rest.
    """
    if "alignment_policy" in data and "alignment_config" not in data:
        old: dict[str, Any] = data["alignment_policy"]
        if isinstance(old, dict):
            migrated: dict[str, Any] = {}
            # minimum_challenge_margin exists under both names
            if "minimum_challenge_margin" in old:
                migrated["minimum_challenge_margin"] = old["minimum_challenge_margin"]
            # preferred_option_authority → kept under alignment_policy only; no direct mapping
            # minimum_mapping_confidence → minimum_challenge_confidence (closest analogue)
            if "minimum_mapping_confidence" in old:
                migrated["minimum_challenge_confidence"] = old["minimum_mapping_confidence"]
            data = {**data, "alignment_config": migrated}
            deprecations.append(
                "alignment_policy → alignment_config: "
                "minimum_challenge_margin and minimum_mapping_confidence migrated"
            )
        else:
            warnings.append(
                "alignment_policy present but is not a dict; skipping migration to alignment_config"
            )
    return data


def _migrate_scoring_policy(
    data: dict[str, Any],
    deprecations: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Copy legacy ``scoring_policy`` penalty fields → ``constraint_config`` when absent."""
    if "scoring_policy" in data and "constraint_config" not in data:
        old: dict[str, Any] = data["scoring_policy"]
        if isinstance(old, dict):
            migrated: dict[str, Any] = {}
            # constraint_violation_penalty → default_required_penalty (closest semantic match)
            if "constraint_violation_penalty" in old:
                migrated["default_required_penalty"] = old["constraint_violation_penalty"]
            # partial_constraint_penalty → default_preferred_penalty
            if "partial_constraint_penalty" in old:
                migrated["default_preferred_penalty"] = old["partial_constraint_penalty"]
            data = {**data, "constraint_config": migrated}
            deprecations.append(
                "scoring_policy → constraint_config: "
                "constraint_violation_penalty and partial_constraint_penalty migrated"
            )
        else:
            warnings.append(
                "scoring_policy present but is not a dict; skipping migration to constraint_config"
            )
    return data


def _migrate_evaluation_weights(
    data: dict[str, Any],
    deprecations: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    """Migrate legacy evaluation.method / evaluation.weights → evaluation_config.criteria.

    Legacy shape::

        evaluation:
          method: multi_criteria
          weights:
            geographic: 0.35
            power: 0.30
            timing: 0.35

    PH12.2a shape::

        evaluation_config:
          criteria:
            geographic: {weight: 0.35, enabled: true}
            ...
    """
    eval_block = data.get("evaluation")
    if not isinstance(eval_block, dict):
        return data
    weights_dict = eval_block.get("weights")
    if not isinstance(weights_dict, dict):
        return data
    if "evaluation_config" in data:
        # evaluation_config already supplied — do not override
        return data
    criteria: dict[str, Any] = {
        name: {"weight": float(w), "enabled": True}
        for name, w in weights_dict.items()
    }
    eval_config: dict[str, Any] = {"criteria": criteria}
    data = {**data, "evaluation_config": eval_config}
    deprecations.append(
        "evaluation.weights → evaluation_config.criteria: "
        "weight dict converted to StrategyCriterionConfig entries"
    )
    return data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_strategy_config(raw_yaml_dict: dict[str, Any] | None) -> ResolvedStrategyConfig:
    """Resolve and validate a StrategyConfig from a YAML-sourced dict.

    Parameters
    ----------
    raw_yaml_dict:
        The dict parsed from the ``strategy:`` block of an engagement YAML, or
        ``None`` / empty dict if no strategy block was supplied.

    Returns
    -------
    ResolvedStrategyConfig
        Frozen container with the fully validated config, provenance lists,
        fingerprint, and source tag.
    """
    raw: dict[str, Any] = raw_yaml_dict if raw_yaml_dict else {}
    source: str = "defaults_only" if not raw else "engagement_yaml"

    deprecations: list[str] = []
    warnings: list[str] = []

    # Deep-copy so we never mutate the caller's dict
    data: dict[str, Any] = deepcopy(raw)

    # 0. Version gate — warn on unrecognised config_version values
    raw_version = raw.get("config_version")
    if raw_version and raw_version not in _KNOWN_CONFIG_VERSIONS:
        warnings.append(
            f"config_version {raw_version!r} is not recognised; "
            f"known versions: {sorted(_KNOWN_CONFIG_VERSIONS)}. "
            "Config will be processed as-is but some fields may be unsupported."
        )

    # 1. Apply deprecated-field migrations (order matters: alignment before scoring)
    data = _migrate_alignment_policy(data, deprecations, warnings)
    data = _migrate_scoring_policy(data, deprecations, warnings)
    data = _migrate_evaluation_weights(data, deprecations, warnings)

    # 2. Track which top-level fields were absent in the (pre-migration) raw dict —
    #    these all took their values from defaults.  We use `raw` (not `data`) so
    #    that migrated synthetic keys are not listed as "user supplied".
    defaults_applied: list[str] = _track_defaults(raw)

    # 3. Build and validate the resolved config
    resolved: StrategyConfig = StrategyConfig.from_yaml_dict(data)

    # 4. Compute fingerprint from the resolved (fully-defaulted) model
    fingerprint: str = resolved.compute_fingerprint()

    return ResolvedStrategyConfig(
        raw=raw,
        resolved=resolved,
        defaults_applied=defaults_applied,
        deprecations=deprecations,
        warnings=warnings,
        fingerprint=fingerprint,
        source=source,
        config_version=_CONFIG_VERSION,
    )


# ---------------------------------------------------------------------------
# JSON schema catalogue export
# ---------------------------------------------------------------------------

def _schema_entry(
    *,
    path: str,
    title: str,
    description: str,
    type: str,
    default: Any,
    minimum: float | int | None = None,
    maximum: float | int | None = None,
    allowed_values: list[str] | None = None,
    required: bool = False,
    category: str,
    unsafe_to_hide: bool = False,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "path": path,
        "title": title,
        "description": description,
        "type": type,
        "default": default,
        "required": required,
        "category": category,
        "unsafe_to_hide": unsafe_to_hide,
    }
    if minimum is not None:
        entry["minimum"] = minimum
    if maximum is not None:
        entry["maximum"] = maximum
    if allowed_values is not None:
        entry["allowed_values"] = allowed_values
    return entry


def export_config_schema() -> dict[str, Any]:
    """Return a structured schema catalogue for all major StrategyConfig fields.

    Each entry describes a field path for UI generation, documentation, or
    runtime validation tooling.  The catalogue is also written to
    ``functional_agents/strategy/strategy_config_schema.json`` alongside this
    module.
    """
    fields: list[dict[str, Any]] = [
        # ----- Root -------------------------------------------------------
        _schema_entry(
            path="enabled",
            title="Strategy Layer Enabled",
            description="When False the entire strategy pipeline is bypassed.",
            type="boolean",
            default=True,
            category="root",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="config_version",
            title="Config Version",
            description="Schema version tag for this configuration object.",
            type="string",
            default=_CONFIG_VERSION,
            category="root",
        ),
        _schema_entry(
            path="framework",
            title="Strategy Framework",
            description="Named framework variant to apply (e.g. 'executive').",
            type="string",
            default="executive",
            allowed_values=["executive"],
            category="root",
        ),

        # ----- Evaluation config -----------------------------------------
        _schema_entry(
            path="evaluation_config.weight_policy",
            title="Weight Policy",
            description=(
                "How criterion weights are enforced. "
                "'strict' requires enabled weights to sum to 1.0 ± 0.001; "
                "'normalize' normalises at runtime."
            ),
            type="string",
            default="normalize",
            allowed_values=["strict", "normalize"],
            category="evaluation",
        ),
        _schema_entry(
            path="evaluation_config.score_precision",
            title="Score Precision",
            description="Decimal places used when rounding theory scores.",
            type="integer",
            default=6,
            minimum=1,
            category="evaluation",
        ),
        _schema_entry(
            path="evaluation_config.minimum_winner_margin",
            title="Minimum Winner Margin",
            description="Score gap required to declare a clear winner over the runner-up.",
            type="number",
            default=0.05,
            minimum=0.0,
            maximum=1.0,
            category="evaluation",
        ),
        _schema_entry(
            path="evaluation_config.saturation_threshold",
            title="Saturation Threshold",
            description="Score above which a theory is considered saturated.",
            type="number",
            default=0.95,
            minimum=0.0,
            maximum=1.0,
            category="evaluation",
        ),
        _schema_entry(
            path="evaluation_config.tie_tolerance",
            title="Tie Tolerance",
            description="Maximum absolute score difference that is treated as a tie.",
            type="number",
            default=1e-6,
            minimum=0.0,
            maximum=1.0,
            category="evaluation",
        ),

        # ----- Constraint config -----------------------------------------
        _schema_entry(
            path="constraint_config.required_violation_policy",
            title="Required Violation Policy",
            description="How hard constraint violations are handled.",
            type="string",
            default="penalize",
            allowed_values=["penalize", "disqualify", "warn"],
            category="constraint",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="constraint_config.default_required_penalty",
            title="Required Constraint Penalty",
            description="Score deduction applied when a required constraint is violated.",
            type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            category="constraint",
        ),
        _schema_entry(
            path="constraint_config.default_preferred_penalty",
            title="Preferred Constraint Penalty",
            description="Score deduction applied when a preferred constraint is violated.",
            type="number",
            default=0.05,
            minimum=0.0,
            maximum=1.0,
            category="constraint",
        ),
        _schema_entry(
            path="constraint_config.maximum_total_penalty",
            title="Maximum Total Penalty",
            description="Cap on cumulative constraint penalties per theory.",
            type="number",
            default=0.50,
            minimum=0.0,
            maximum=1.0,
            category="constraint",
        ),
        _schema_entry(
            path="constraint_config.fail_on_unknown_constraint_reference",
            title="Fail on Unknown Constraint Reference",
            description="Raise an error if a constraint references an unknown theory or option.",
            type="boolean",
            default=True,
            category="constraint",
            unsafe_to_hide=True,
        ),

        # ----- Mapping config --------------------------------------------
        _schema_entry(
            path="mapping_config.enabled",
            title="Option Mapping Enabled",
            description="When False the option-mapping pass is skipped entirely.",
            type="boolean",
            default=True,
            category="mapping",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="mapping_config.unresolved_policy",
            title="Unresolved Mapping Policy",
            description="Behaviour when a theory cannot be mapped to any upstream option.",
            type="string",
            default="preserve_upstream",
            allowed_values=["preserve_upstream", "report_unresolved", "fail"],
            category="mapping",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="mapping_config.authority.fail_on_mapping_mismatch",
            title="Fail on Mapping Mismatch",
            description="Raise an error when the mapped option does not match the upstream option.",
            type="boolean",
            default=True,
            category="mapping",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="mapping_config.confidence.high_score_threshold",
            title="High Confidence Score Threshold",
            description="Minimum score to qualify for High mapping confidence.",
            type="number",
            default=0.40,
            minimum=0.0,
            maximum=1.0,
            category="mapping",
        ),
        _schema_entry(
            path="mapping_config.confidence.high_margin_threshold",
            title="High Confidence Margin Threshold",
            description="Minimum margin over runner-up to qualify for High mapping confidence.",
            type="number",
            default=0.15,
            minimum=0.0,
            maximum=1.0,
            category="mapping",
        ),

        # ----- Alignment config ------------------------------------------
        _schema_entry(
            path="alignment_config.enabled",
            title="Alignment Enabled",
            description="When False the alignment pass is skipped entirely.",
            type="boolean",
            default=True,
            category="alignment",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="alignment_config.minimum_challenge_margin",
            title="Minimum Challenge Margin",
            description=(
                "Score advantage the strategy selection must have over the upstream option "
                "to register as a 'challenged' alignment."
            ),
            type="number",
            default=0.10,
            minimum=0.0,
            maximum=1.0,
            category="alignment",
        ),
        _schema_entry(
            path="alignment_config.minimum_challenge_confidence",
            title="Minimum Challenge Confidence",
            description="Minimum mapping confidence required to declare a challenge.",
            type="string",
            default="Medium",
            allowed_values=["Low", "Medium", "High"],
            category="alignment",
        ),
        _schema_entry(
            path="alignment_config.unresolved_authority",
            title="Unresolved Authority",
            description="Which recommendation is authoritative when alignment is unresolved.",
            type="string",
            default="upstream_preferred",
            allowed_values=["upstream_preferred", "strategy_selected", "none"],
            category="alignment",
            unsafe_to_hide=True,
        ),

        # ----- Content config --------------------------------------------
        _schema_entry(
            path="content.minimum_relevance_score",
            title="Minimum Relevance Score",
            description="Minimum score for a content item to be assigned to a theory.",
            type="number",
            default=0.20,
            minimum=0.0,
            maximum=1.0,
            category="content",
        ),
        _schema_entry(
            path="content.minimum_content_coverage",
            title="Minimum Content Coverage",
            description="Fraction of available content items that must be assigned.",
            type="number",
            default=0.50,
            minimum=0.0,
            maximum=1.0,
            category="content",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="content.minimum_distinctive_coverage",
            title="Minimum Distinctive Coverage",
            description="Fraction of content that must be distinctive (not shared) per theory.",
            type="number",
            default=0.20,
            minimum=0.0,
            maximum=1.0,
            category="content",
        ),
        _schema_entry(
            path="content.allow_symmetric_fallback",
            title="Allow Symmetric Fallback",
            description="When True, shared content may be symmetric across theories.",
            type="boolean",
            default=True,
            category="content",
        ),
        _schema_entry(
            path="content.partial_homogenization_threshold",
            title="Partial Homogenization Threshold",
            description="Shared-content fraction above which partial homogenization is flagged.",
            type="number",
            default=0.75,
            minimum=0.0,
            maximum=1.0,
            category="content",
            unsafe_to_hide=True,
        ),

        # ----- Reporting config ------------------------------------------
        _schema_entry(
            path="reporting_config.enabled",
            title="Reporting Enabled",
            description="When False the report assembly pass is skipped.",
            type="boolean",
            default=True,
            category="reporting",
        ),
        _schema_entry(
            path="reporting_config.mandatory.challenged_alignment",
            title="Mandate Challenged Alignment Disclosure",
            description="When True, challenged alignment must always appear in the report.",
            type="boolean",
            default=True,
            category="reporting",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="reporting_config.mandatory.substantial_homogenization",
            title="Mandate Homogenization Disclosure",
            description="When True, substantial homogenization must always appear in the report.",
            type="boolean",
            default=True,
            category="reporting",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="reporting_config.limits.maximum_assumptions",
            title="Maximum Assumptions per Report",
            description="Maximum number of assumption items shown per theory in the report.",
            type="integer",
            default=5,
            minimum=1,
            category="reporting",
        ),
        _schema_entry(
            path="reporting_config.limits.maximum_risks",
            title="Maximum Risks per Report",
            description="Maximum number of risk items shown per theory in the report.",
            type="integer",
            default=5,
            minimum=1,
            category="reporting",
        ),

        # ----- Diagnostics config ----------------------------------------
        _schema_entry(
            path="diagnostics_config.unknown_reference",
            title="Unknown Reference Severity",
            description="Diagnostic severity when an unknown theory/option reference is encountered.",
            type="string",
            default="error",
            allowed_values=["ignore", "info", "warning", "error"],
            category="diagnostics",
            unsafe_to_hide=True,
        ),
        _schema_entry(
            path="diagnostics_config.low_mapping_confidence",
            title="Low Mapping Confidence Severity",
            description="Diagnostic severity when mapping confidence falls below the minimum.",
            type="string",
            default="warning",
            allowed_values=["ignore", "info", "warning", "error"],
            category="diagnostics",
        ),
        _schema_entry(
            path="diagnostics_config.partial_homogenization",
            title="Partial Homogenization Severity",
            description="Diagnostic severity when partial content homogenization is detected.",
            type="string",
            default="warning",
            allowed_values=["ignore", "info", "warning", "error"],
            category="diagnostics",
        ),
        _schema_entry(
            path="diagnostics_config.fallback_heavy",
            title="Fallback-Heavy Severity",
            description="Diagnostic severity when a theory relies heavily on fallback content.",
            type="string",
            default="warning",
            allowed_values=["ignore", "info", "warning", "error"],
            category="diagnostics",
        ),
        _schema_entry(
            path="diagnostics_config.insufficient_coverage",
            title="Insufficient Coverage Severity",
            description="Diagnostic severity when coverage falls below the configured minimum.",
            type="string",
            default="warning",
            allowed_values=["ignore", "info", "warning", "error"],
            category="diagnostics",
            unsafe_to_hide=True,
        ),
    ]

    schema: dict[str, Any] = {
        "schema_version": _CONFIG_VERSION,
        "description": "StrategyConfig field catalogue for UI generation and validation tooling.",
        "field_count": len(fields),
        "fields": fields,
    }

    # Write alongside this module
    _dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(_dir, "strategy_config_schema.json")
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2, ensure_ascii=False)

    return schema
