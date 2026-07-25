"""StrategicChoiceSet — canonical data contract for a coherent set of strategic choices (PH10.0).

A StrategicChoiceSet groups all choices made across every strategic dimension
into a single object. The set is coherent when it covers all required dimensions
and contains no internal conflicts.

Produced by: StrategicChoiceGenerator (PH10.x, not yet implemented)
Consumed by: TheoryGenerator, TheoryEvaluator (future phases)

Immutability: frozen=True prevents attribute assignment after construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .strategic_choice import StrategicChoice


class StrategicChoiceSet(BaseModel):
    """A coherent set of strategic choices across all active dimensions.

    The set is the unit of theory generation: one StrategicChoiceSet maps
    to one candidate TheoryOfWinning. Its ``completeness`` and
    ``internal_conflicts`` fields allow evaluators to score the set before
    committing to theory construction.
    """

    # Identity
    id: str = ""

    # The individual choices — one per strategic dimension
    choices: list[StrategicChoice] = Field(default_factory=list)

    # Aggregate confidence across all choices
    overall_confidence: str = ""  # "High" | "Medium" | "Low" | ""

    # Detected conflicts between pairs (or groups) of choices.
    # Each entry: {"choice_ids": [str, ...], "description": str}
    internal_conflicts: list[dict[str, Any]] = Field(default_factory=list)

    # Fraction of required dimensions covered: 0.0 (none) to 1.0 (all)
    completeness: float = 0.0

    rationale: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("completeness")
    @classmethod
    def _validate_completeness(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"completeness must be in [0.0, 1.0], got {v}"
            )
        return v

    # ------------------------------------------------------------------
    # Convenience accessors (read-only — do not mutate returned objects)
    # ------------------------------------------------------------------

    def choice_by_dimension(self, dimension: str) -> StrategicChoice | None:
        """Return the choice for a given dimension, or None if not present."""
        for choice in self.choices:
            if choice.dimension == dimension:
                return choice
        return None

    def dimensions_covered(self) -> list[str]:
        """Return the list of dimensions that have a choice in this set."""
        return [c.dimension for c in self.choices]

    def has_conflicts(self) -> bool:
        """Return True if any internal conflicts were detected."""
        return bool(self.internal_conflicts)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicChoiceSet":
        return cls.model_validate(data)
