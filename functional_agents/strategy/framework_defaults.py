"""FrameworkDefaults — built-in defaults for named strategy frameworks (PH9.2).

Registry of StrategyConfig defaults keyed by framework name. Each framework
provides sensible starting points; callers override only the fields they care
about — everything else falls through to the framework default.

PH9.2 scope: built-in 'executive' framework only.
Not in scope: YAML loading, plugin discovery, engagement overrides.
Future phases will add additional built-in frameworks and a plugin registration
path.
"""

from __future__ import annotations

from .strategy_config import (
    StrategyConfig,
    StrategyConstraints,
    StrategyDimensions,
    StrategyEvaluation,
    StrategyGeneration,
    StrategyMetadata,
    StrategyObjectives,
    StrategyValidation,
)

# ---------------------------------------------------------------------------
# Built-in: executive framework
# ---------------------------------------------------------------------------

_EXECUTIVE = StrategyConfig(
    version="1.0",
    framework="executive",
    objectives=StrategyObjectives(
        primary=[
            "Identify the option that maximises risk-adjusted value",
            "Preserve strategic optionality where possible",
        ],
        secondary=[
            "Build decision confidence through validated assumptions",
        ],
    ),
    dimensions=StrategyDimensions(),  # derived from decision model at runtime
    evaluation=StrategyEvaluation(
        method="multi_criteria",
        weights={},          # equal weighting across all dimensions
        min_score_threshold=0.0,
    ),
    generation=StrategyGeneration(
        max_candidates=3,
        diversity_required=True,
    ),
    constraints=StrategyConstraints(
        excluded_options=[],
        required_conditions=[],
    ),
    validation=StrategyValidation(
        require_evidence=False,
        min_confidence="",
        require_assumptions=False,
    ),
    metadata=StrategyMetadata(
        notes="Built-in executive framework defaults (PH9.2)",
    ),
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, StrategyConfig] = {
    "executive": _EXECUTIVE,
}


class FrameworkDefaults:
    """Registry of built-in strategy framework defaults.

    ``get()`` is the primary interface — it returns the default StrategyConfig
    for a named framework. Future phases will add a ``register()`` method for
    framework plugins.
    """

    @staticmethod
    def get(framework: str) -> StrategyConfig:
        """Return the default StrategyConfig for a named framework.

        Raises:
            ValueError: if the framework is not registered.
        """
        try:
            return _REGISTRY[framework]
        except KeyError:
            available = ", ".join(f"{k!r}" for k in sorted(_REGISTRY))
            raise ValueError(
                f"Unknown strategy framework: {framework!r}. "
                f"Available: {available}"
            )

    @staticmethod
    def known() -> list[str]:
        """Return the names of all registered frameworks, sorted."""
        return sorted(_REGISTRY)

    @staticmethod
    def is_known(framework: str) -> bool:
        """Return True if the framework has registered defaults."""
        return framework in _REGISTRY
