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
from .strategy_config import StrategyConfig

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
