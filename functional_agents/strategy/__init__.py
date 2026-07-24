"""Strategy Layer (PH8).

Sits between Research and Communication in the canonical pipeline:

    ResearchObject  (Research Layer output)
        ↓
    StrategyCoordinator.build(ctx)
        ↓
    StrategicPosition  (Strategy Layer output / Communication Layer input)
        ↓
    EditorialCoordinator.build(position)

The Strategy Layer answers: Given what is strategically true, how do we win?
"""

from .strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from .strategy_coordinator import StrategyCoordinator

__all__ = [
    "StrategicExecution",
    "StrategicJustification",
    "StrategicPosition",
    "StrategicRecommendation",
    "StrategyCoordinator",
    "TheoryOfWinning",
]
