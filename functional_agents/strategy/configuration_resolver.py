"""ConfigurationResolver — canonical entry point into the Strategy Layer (PH9.1).

Sits between an incoming StrategyConfig and the StrategyCoordinator:

    StrategyConfig
            │
            ▼
    ConfigurationResolver.resolve()
            │
            ▼
    Resolved StrategyConfig
            │
            ▼
    StrategyCoordinator

PH9.1 scope: pass-through with defensive validation.
The resolver establishes the architectural seam for future milestones that
will add framework defaults, engagement overrides, and dimension population
between the raw config and the resolved config.

Not in scope for PH9.1:
  - Framework defaults
  - YAML loading
  - Engagement overrides
  - StrategyPlan production
  - Theory generation or evaluation
"""

from __future__ import annotations

import logging

from .strategy_config import StrategyConfig

LOGGER = logging.getLogger(__name__)


class ConfigurationResolver:
    """Produces a validated, resolved StrategyConfig.

    PH9.1: pass-through. The resolver validates the input, then returns a
    fresh immutable copy — the caller's object is never mutated. Future
    milestones will add framework defaults and engagement overrides inside
    this class without changing its interface.
    """

    def resolve(self, config: StrategyConfig) -> StrategyConfig:
        """Validate config and return a resolved copy.

        The returned instance is always a new object — callers that hold a
        reference to the original will not observe any change.

        Raises:
            ValueError: if any field violates a hard invariant.
        """
        self._validate(config)

        # Round-trip through dict to produce a fully independent copy.
        # model_validate re-runs Pydantic validation, so the result is
        # guaranteed to satisfy all field-level constraints.
        resolved = StrategyConfig.from_dict(config.to_dict())

        LOGGER.debug(
            "[ConfigurationResolver] resolved: framework=%r version=%r",
            resolved.framework,
            resolved.version,
        )

        return resolved

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

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
