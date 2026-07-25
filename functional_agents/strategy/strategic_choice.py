"""StrategicChoice — canonical data contract for a single strategic choice (PH10.0).

A StrategicChoice represents a concrete decision on one strategic dimension:
which option was selected, why, and with what confidence.

Produced by: StrategicChoiceGenerator (PH10.x, not yet implemented)
Consumed by: TheoryOfWinning construction, StrategicChoiceSet

Immutability: frozen=True prevents attribute assignment after construction.
List/dict fields are not deeply frozen — callers should treat them as
read-only by convention.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class StrategicChoice(BaseModel):
    """A concrete decision on one strategic dimension.

    Represents a resolved choice: which option was selected for a specific
    strategic dimension, why it was selected, and what alternatives were
    considered but set aside.
    """

    # Identity
    id: str = ""
    dimension: str = ""

    # The decision
    selected_value: str = ""
    rationale: str = ""

    # Evidence and assumptions supporting this choice
    supporting_evidence: list[str] = Field(default_factory=list)
    supporting_assumptions: list[str] = Field(default_factory=list)

    # Confidence in this choice
    confidence: str = ""  # "High" | "Medium" | "Low" | ""

    # Alternatives that were considered but not selected.
    # Each entry: {"value": str, "reason_not_selected": str}
    alternatives_considered: list[dict[str, Any]] = Field(default_factory=list)

    # Whether this choice must be present in any valid StrategicChoiceSet.
    # "required" | "optional" | "conditional"
    requiredness: str = "optional"

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("requiredness")
    @classmethod
    def _validate_requiredness(cls, v: str) -> str:
        allowed = {"required", "optional", "conditional", ""}
        if v not in allowed:
            raise ValueError(
                f"requiredness must be one of {sorted(allowed)!r}, got {v!r}"
            )
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicChoice":
        return cls.model_validate(data)
