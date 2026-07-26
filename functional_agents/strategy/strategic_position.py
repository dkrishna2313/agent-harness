"""StrategicPosition — canonical output of the Strategy Layer (PH8).

Produced by StrategyCoordinator.build().
Consumed by EditorialCoordinator.build().

The Strategy Layer answers: Given what is strategically true, how do we win?

This object is the boundary between Strategy and Communication.
The Communication Layer must not read from AgentContext directly.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Theory of Winning
# ---------------------------------------------------------------------------

class TheoryOfWinning(BaseModel):
    """How the organisation wins given what is strategically true."""

    theory_id: str = ""
    recommended_option_id: str = ""
    recommended_option_title: str = ""
    winning_position: str = ""
    winning_mechanism: str = ""
    strategic_choices: list[dict[str, Any]] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    failure_modes: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    confidence: str = ""

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Strategic Recommendation
# ---------------------------------------------------------------------------

class StrategicRecommendation(BaseModel):
    """The selected strategic direction and its decision context."""

    recommended_option_id: str = ""
    recommended_option_title: str = ""
    board_recommendation: str = ""
    decision_readiness: str = ""
    overall_confidence: str = ""
    key_conditions: list[str] = Field(default_factory=list)
    critical_unknowns: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Strategic Justification
# ---------------------------------------------------------------------------

class StrategicJustification(BaseModel):
    """The evidence base and analytical foundation for the recommendation."""

    decision_analysis: dict[str, Any] = Field(default_factory=dict)
    strategic_options: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Strategic Execution
# ---------------------------------------------------------------------------

class StrategicExecution(BaseModel):
    """Immediate actions and implementation priorities."""

    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    validation_priorities: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# StrategicPosition — root object
# ---------------------------------------------------------------------------

class StrategicPosition(BaseModel):
    """Canonical output of the Strategy Layer.

    Represents the selected strategy to be communicated. Every field the
    Communication Layer needs is carried here; it never reads AgentContext.

    Raw reasoning outputs (decision_analysis, executive_confidence, etc.) are
    stored at top level for direct access by EditorialCoordinator._build_*
    methods, which mirror the AgentContext field names to minimise diff.
    The typed sub-objects (theory_of_winning, recommendation, justification,
    execution) represent the canonical spec structure.
    """

    position_id: str = ""
    created_at: str = ""

    # Provenance
    run_id: str = ""
    question: str = ""
    profiles: list[str] = Field(default_factory=list)
    execution_profile: str = ""
    decision_model: dict[str, Any] = Field(default_factory=dict)
    engagement: dict[str, Any] = Field(default_factory=dict)
    preferred_option: dict[str, Any] = Field(default_factory=dict)
    research_object: dict[str, Any] = Field(default_factory=dict)

    # Raw reasoning outputs — consumed by EditorialCoordinator._build_* methods
    decision_analysis: dict[str, Any] = Field(default_factory=dict)
    executive_confidence: dict[str, Any] = Field(default_factory=dict)
    strategic_options: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    opportunities: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)

    # Canonical spec structure
    theory_of_winning: TheoryOfWinning = Field(default_factory=TheoryOfWinning)
    recommendation: StrategicRecommendation = Field(default_factory=StrategicRecommendation)
    justification: StrategicJustification = Field(default_factory=StrategicJustification)
    execution: StrategicExecution = Field(default_factory=StrategicExecution)

    model_config = {"extra": "allow"}

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategicPosition":
        return cls.model_validate(data)
