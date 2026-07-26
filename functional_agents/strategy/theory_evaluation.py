"""TheoryEvaluation — canonical evaluation artifact for a single TheoryOfWinning (PH10.4).

A TheoryEvaluation records the result of evaluating one TheoryOfWinning against
a set of criteria. It does not compare theories, does not select a winner, and
does not reference any specific evaluation framework.

The ``criteria_scores`` field is deliberately generic: its keys are arbitrary
criterion names defined at evaluation time by the caller. No Executive, MECE,
or framework-specific concepts are hard-coded here.

Produced by: TheoryEvaluator (future phase)
Consumed by: StrategySelector (future phase)

Immutability: frozen=True prevents attribute assignment after construction.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class CriterionScore(BaseModel):
    """Score for a single evaluation criterion.

    The criterion name lives in the parent ``criteria_scores`` dict key.
    This model records only the numeric score, an explanatory rationale,
    and an optional contribution weight used by aggregators.

    The model is open (``extra="allow"``) so evaluators may attach
    framework-specific supplementary data without breaking the contract.
    """

    # Normalised score for this criterion: 0.0 (worst) to 1.0 (best)
    score: float = 0.0

    # Human-readable explanation of why this score was assigned
    rationale: str = ""

    # Contribution weight when computing a weighted overall_score.
    # Defaults to 1.0 (equal weight). Must be >= 0.0.
    weight: float = 1.0

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("score")
    @classmethod
    def _validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"score must be in [0.0, 1.0], got {v!r}")
        return v

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"weight must be >= 0.0, got {v!r}")
        return v

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriterionScore":
        return cls.model_validate(data)


class TheoryEvaluation(BaseModel):
    """Evaluation of a single TheoryOfWinning.

    Records scores for each evaluation criterion, qualitative observations
    (strengths, weaknesses, residual risks), an aggregated overall_score,
    and the evaluator's confidence in its own assessment.

    The model makes no assumptions about which framework produced it or
    which criteria were used — those are carried as keys in ``criteria_scores``.
    """

    # ID of the TheoryOfWinning being evaluated
    theory_id: str = ""

    # Per-criterion scores: {criterion_name: CriterionScore}
    # Keys are defined by the calling evaluator; no names are hard-coded.
    criteria_scores: dict[str, CriterionScore] = Field(default_factory=dict)

    # Qualitative observations
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    # Risks that remain after the theory is accepted.
    # Each entry is a free-form dict: {"description": str, "severity": str, ...}
    residual_risks: list[dict[str, Any]] = Field(default_factory=list)

    # Aggregated score across all criteria: 0.0 (worst) to 1.0 (best)
    overall_score: float = 0.0

    # Confidence in this evaluation: "High" | "Medium" | "Low" | ""
    confidence: str = ""

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "allow"}

    @field_validator("overall_score")
    @classmethod
    def _validate_overall_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"overall_score must be in [0.0, 1.0], got {v!r}")
        return v

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def criterion_names(self) -> list[str]:
        """Return the names of all criteria that have a score."""
        return list(self.criteria_scores.keys())

    def score_for(self, criterion: str) -> CriterionScore | None:
        """Return the CriterionScore for a given criterion name, or None."""
        return self.criteria_scores.get(criterion)

    def weighted_score(self) -> float:
        """Return the weighted mean of all criterion scores.

        Falls back to ``overall_score`` when ``criteria_scores`` is empty.
        Returns 0.0 when all weights are zero.
        """
        if not self.criteria_scores:
            return self.overall_score
        total_weight = sum(cs.weight for cs in self.criteria_scores.values())
        if total_weight == 0.0:
            return 0.0
        return sum(
            cs.score * cs.weight for cs in self.criteria_scores.values()
        ) / total_weight

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TheoryEvaluation":
        return cls.model_validate(data)
