"""Strategy Layer (PH8/PH9).

Sits between Research and Communication in the canonical pipeline:

    ResearchObject  (Research Layer output)
        ↓
    StrategyCoordinator.build(ctx, config)
        ↓
    StrategicPosition  (Strategy Layer output / Communication Layer input)
        ↓
    EditorialCoordinator.build(position)

The Strategy Layer answers: Given what is strategically true, how do we win?

PH9.0 addition: StrategyConfig is the canonical configuration object for the
Strategy Layer. StrategyCoordinator accepts it optionally; if omitted, a default
instance is constructed. Behavior is unchanged.
"""

from .strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
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
from .strategy_coordinator import StrategyCoordinator

__all__ = [
    "StrategicExecution",
    "StrategicJustification",
    "StrategicPosition",
    "StrategicRecommendation",
    "StrategyConfig",
    "StrategyConstraints",
    "StrategyCoordinator",
    "StrategyDimensions",
    "StrategyEvaluation",
    "StrategyGeneration",
    "StrategyMetadata",
    "StrategyObjectives",
    "StrategyValidation",
    "TheoryOfWinning",
]
