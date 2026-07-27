"""PH12.1 — alignment models for theory-specific evaluation.

ConstraintResult: outcome of checking one plan constraint against a theory.
OptionMapping:    best upstream strategic option the theory maps to.
AlignmentResult:  relationship between upstream recommendation and selected theory.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConstraintResult(BaseModel):
    """Result of evaluating one plan constraint against a theory's choices."""

    constraint: str
    # "satisfied" | "partially_satisfied" | "violated" | "not_assessable"
    status: str
    score: float
    rationale: str

    model_config = {"frozen": True, "extra": "allow"}


class OptionMapping(BaseModel):
    """Maps a theory's recommended choices to an upstream strategic option."""

    mapped_option_id: str | None = None
    mapping_score: float = 0.0
    mapping_rationale: str = ""
    # "High" | "Medium" | "Low" | "None"
    mapping_confidence: str = "Low"

    model_config = {"frozen": True, "extra": "allow"}


class AlignmentResult(BaseModel):
    """Relationship between the upstream preferred option and the selected theory."""

    # "confirmed" | "refined" | "challenged" | "unresolved"
    status: str = "unresolved"
    preferred_option_id: str = ""
    selected_theory_id: str = ""
    mapped_option_id: str | None = None
    score_margin: float = 0.0
    rationale: str = ""
    criterion_advantages: list[str] = Field(default_factory=list)

    model_config = {"frozen": True, "extra": "allow"}
