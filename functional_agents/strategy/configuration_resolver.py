"""ConfigurationResolver — canonical entry point into the Strategy Layer (PH9.1/PH9.2).

Sits between an incoming StrategyConfig and the StrategyCoordinator:

    Framework Defaults
            +
    Input StrategyConfig
            │
            ▼
    ConfigurationResolver.resolve()
            │
            ▼
    Resolved StrategyConfig
            │
            ▼
    StrategyCoordinator

PH9.1: defensive validation and immutable copy.
PH9.2: merge with framework defaults before returning. Merge rule: if the caller's
field value differs from the zero-opinion baseline (StrategyConfig()), keep the
caller's value; otherwise use the framework default. Neither input is mutated.

Not in scope:
  - YAML loading
  - Engagement overrides
  - StrategyPlan production
  - Theory generation or evaluation
"""

from __future__ import annotations

import logging
from typing import Any

from .framework_defaults import FrameworkDefaults
from .strategy_config import (
    ChoiceConfig,
    DimensionConfig,
    StrategyConfig,
    StrategyConstraints,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyObjectives,
    StrategyValidation,
)

LOGGER = logging.getLogger(__name__)


class ConfigurationResolver:
    """Produces a validated, resolved StrategyConfig.

    PH9.2: merges framework defaults with the caller's config.
    For each top-level field, the caller's value wins if it differs from the
    zero-opinion baseline (``StrategyConfig()``); otherwise the framework
    default is used. The caller's object and the defaults object are never
    mutated.

    If the caller specifies an unknown framework, a warning is logged and the
    caller's config is used as-is (no defaults applied).
    """

    def resolve(self, config: StrategyConfig) -> StrategyConfig:
        """Validate and merge config with framework defaults.

        The returned instance is always a new object — callers that hold a
        reference to the original will not observe any change.

        Raises:
            ValueError: if any field violates a hard invariant.
        """
        self._validate(config)

        if FrameworkDefaults.is_known(config.framework):
            defaults = FrameworkDefaults.get(config.framework)
            merged_dict = self._merge(defaults, config)
        else:
            LOGGER.warning(
                "[ConfigurationResolver] unknown framework %r — no defaults applied",
                config.framework,
            )
            merged_dict = config.to_dict()

        resolved = StrategyConfig.from_dict(merged_dict)

        LOGGER.debug(
            "[ConfigurationResolver] resolved: framework=%r version=%r",
            resolved.framework,
            resolved.version,
        )

        return resolved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge(self, defaults: StrategyConfig, caller: StrategyConfig) -> dict[str, Any]:
        """Merge caller config over framework defaults.

        For each top-level field: if the caller's serialized value differs
        from the zero-opinion baseline (``StrategyConfig()``), keep the
        caller's value. Otherwise use the framework default value.

        Does not mutate either input.
        """
        baseline_d = StrategyConfig().to_dict()
        defaults_d = defaults.to_dict()
        caller_d = caller.to_dict()

        merged: dict[str, Any] = {}
        for key in defaults_d:
            caller_val = caller_d.get(key)
            baseline_val = baseline_d.get(key)
            if caller_val != baseline_val:
                # caller explicitly set this field — preserve it
                merged[key] = caller_val
            else:
                # caller left it at baseline — use framework default
                merged[key] = defaults_d[key]

        return merged

    def resolve_from_engagement(
        self,
        config: StrategyConfig,
        engagement_strategy: dict[str, Any],
    ) -> StrategyConfig:
        """Merge an engagement-level strategy block into the base config and resolve.

        Merge order: FrameworkDefaults → base config → engagement overrides → resolved.

        The engagement dict may contain:
          framework, objectives (list[str]), dimensions (list[dict]),
          evaluation (dict with criteria sub-dict), generation (dict),
          validation (dict), constraints (list[str]).

        Raises ValueError for:
          - duplicate dimension IDs
          - duplicate choice IDs within a dimension
          - required dimension with no choices
          - blank dimension or choice IDs
        """
        overrides = self._parse_engagement_strategy(engagement_strategy)
        merged_dict = {**config.to_dict(), **overrides}
        merged = StrategyConfig.from_dict(merged_dict)
        return self.resolve(merged)

    # ------------------------------------------------------------------
    # Engagement parsing helpers
    # ------------------------------------------------------------------

    def _parse_engagement_strategy(
        self, raw: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse an engagement strategy dict into StrategyConfig field overrides."""
        result: dict[str, Any] = {}

        if "framework" in raw:
            result["framework"] = raw["framework"]

        if "objectives" in raw:
            obj_list = raw["objectives"] or []
            result["objectives"] = StrategyObjectives(primary=list(obj_list)).model_dump()

        if "dimensions" in raw:
            dims = self._parse_dimensions(raw["dimensions"] or [])
            result["dimension_configs"] = [d.model_dump() for d in dims]

        if "evaluation" in raw:
            eval_raw = raw["evaluation"] or {}
            result["evaluation"] = self._parse_evaluation(eval_raw).model_dump()

        if "generation" in raw:
            gen_raw = raw["generation"] or {}
            result["generation"] = StrategyGeneration(
                max_candidates=int(gen_raw.get("max_candidates", 3)),
                diversity_required=bool(gen_raw.get("diversity_required", True)),
            ).model_dump()

        if "validation" in raw:
            val_raw = raw["validation"] or {}
            min_conf_raw = val_raw.get("min_confidence", "")
            result["validation"] = StrategyValidation(
                require_evidence=bool(val_raw.get("require_evidence", False)),
                require_assumptions=bool(val_raw.get("require_assumptions", False)),
                min_confidence=str(min_conf_raw) if min_conf_raw else "",
            ).model_dump()

        if "constraints" in raw:
            c_list = raw["constraints"] or []
            result["constraints"] = StrategyConstraints(
                required_conditions=list(c_list),
            ).model_dump()

        return result

    def _parse_dimensions(self, raw_dims: list[Any]) -> list[DimensionConfig]:
        """Parse engagement dimension list into DimensionConfig objects.

        Validates:
        - No blank dimension IDs.
        - No duplicate dimension IDs.
        - No duplicate choice IDs within a dimension.
        - Required dimensions must have at least one choice.
        """
        seen_dim_ids: set[str] = set()
        parsed: list[DimensionConfig] = []

        for dim_raw in raw_dims:
            if not isinstance(dim_raw, dict):
                continue
            dim_id = str(dim_raw.get("id", "")).strip()
            if not dim_id:
                raise ValueError(
                    "Engagement strategy: dimension has a blank or missing 'id'."
                )
            if dim_id in seen_dim_ids:
                raise ValueError(
                    f"Engagement strategy: duplicate dimension id={dim_id!r}."
                )
            seen_dim_ids.add(dim_id)

            required = bool(dim_raw.get("required", True))
            choices = self._parse_choices(dim_id, dim_raw.get("choices") or [])

            if required and not choices:
                raise ValueError(
                    f"Engagement strategy: required dimension {dim_id!r} has no choices."
                )

            parsed.append(DimensionConfig(
                id=dim_id,
                title=str(dim_raw.get("title", dim_id)),
                description=str(dim_raw.get("description", "")),
                required=required,
                choices=choices,
            ))

        return parsed

    def _parse_choices(
        self, dimension_id: str, raw_choices: list[Any]
    ) -> list[ChoiceConfig]:
        """Parse choice list for one dimension, validating uniqueness."""
        seen_ids: set[str] = set()
        choices: list[ChoiceConfig] = []

        for c_raw in raw_choices:
            if not isinstance(c_raw, dict):
                continue
            c_id = str(c_raw.get("id", "")).strip()
            if not c_id:
                raise ValueError(
                    f"Engagement strategy: choice in dimension {dimension_id!r} "
                    f"has a blank or missing 'id'."
                )
            if c_id in seen_ids:
                raise ValueError(
                    f"Engagement strategy: duplicate choice id={c_id!r} "
                    f"in dimension {dimension_id!r}."
                )
            seen_ids.add(c_id)
            choices.append(ChoiceConfig(
                id=c_id,
                title=str(c_raw.get("title", c_id)),
                description=str(c_raw.get("description", "")),
            ))

        return choices

    def _parse_evaluation(self, eval_raw: dict[str, Any]) -> StrategyEvaluation:
        """Parse evaluation block, extracting weights from 'criteria' sub-dict."""
        weights: dict[str, float] = {}
        criteria_raw = eval_raw.get("criteria") or {}
        for crit_name, crit_cfg in criteria_raw.items():
            if isinstance(crit_cfg, dict):
                weights[str(crit_name)] = float(crit_cfg.get("weight", 1.0))
            elif isinstance(crit_cfg, (int, float)):
                weights[str(crit_name)] = float(crit_cfg)

        return StrategyEvaluation(
            method=str(eval_raw.get("method", "multi_criteria")),
            weights=weights,
            min_score_threshold=float(eval_raw.get("min_score_threshold", 0.0)),
        )

    def _validate(self, config: StrategyConfig) -> None:
        """Enforce hard invariants that Pydantic field types do not cover."""
        if not config.version:
            raise ValueError("StrategyConfig.version must not be empty")

        if not config.framework:
            raise ValueError("StrategyConfig.framework must not be empty")

        if config.generation.max_candidates < 1:
            raise ValueError(
                f"StrategyConfig.generation.max_candidates must be >= 1, "
                f"got {config.generation.max_candidates}"
            )

        if config.evaluation.min_score_threshold < 0.0:
            raise ValueError(
                f"StrategyConfig.evaluation.min_score_threshold must be >= 0.0, "
                f"got {config.evaluation.min_score_threshold}"
            )
