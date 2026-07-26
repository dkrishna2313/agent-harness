"""StrategyTrace — canonical trace artifact for the Strategy Layer (PH11.0).

Records the full execution chain of the Strategy sub-pipeline:

    StrategyPlan
        → list[StrategicChoiceSet]
        → list[TheoryOfWinning]
        → list[TheoryEvaluation]
        → StrategySelection
        → StrategicPosition

Produced by: StrategyCoordinator.build() (stored as _trace)
Consumed by: pipeline_trace.build_canonical_trace() (exposed as "strategy_trace")
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from .strategic_choice_set import StrategicChoiceSet
from .strategic_position import StrategicPosition, TheoryOfWinning
from .strategy_plan import StrategyPlan
from .strategy_selector import StrategySelection
from .theory_evaluation import TheoryEvaluation


class StrategyTrace(BaseModel):
    """Canonical trace of one complete Strategy Layer execution.

    Captures the full derivation chain from StrategyPlan to StrategicPosition.
    Immutable after construction.

    Validation rules enforced on construction:
      1.  choice_sets is non-empty
      2.  theories is non-empty
      3.  evaluations is non-empty
      4.  theories and evaluations have the same count (one eval per theory)
      5.  theories and choice_sets have the same count (one theory per set)
      6.  no duplicate theory_ids in theories
      7.  no duplicate theory_ids in evaluations
      8.  every evaluation.theory_id resolves to a theory (full bijection)
      9.  selection.winner_theory_id references a known theory
      10. strategic_position.theory_of_winning.theory_id == winner_theory_id
    """

    trace_id: str
    created_at: str
    plan: StrategyPlan
    choice_sets: list[StrategicChoiceSet] = Field(default_factory=list)
    theories: list[TheoryOfWinning] = Field(default_factory=list)
    evaluations: list[TheoryEvaluation] = Field(default_factory=list)
    selection: StrategySelection
    strategic_position: StrategicPosition
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "allow"}

    @model_validator(mode="after")
    def _validate_chain_consistency(self) -> "StrategyTrace":
        theories = self.theories
        evaluations = self.evaluations
        choice_sets = self.choice_sets
        selection = self.selection
        position = self.strategic_position

        # Rules 1–3: no empty collections
        if not choice_sets:
            raise ValueError("StrategyTrace: choice_sets must not be empty.")
        if not theories:
            raise ValueError("StrategyTrace: theories must not be empty.")
        if not evaluations:
            raise ValueError("StrategyTrace: evaluations must not be empty.")

        # Rule 4: one evaluation per theory
        if len(theories) != len(evaluations):
            raise ValueError(
                f"StrategyTrace: theories ({len(theories)}) and "
                f"evaluations ({len(evaluations)}) must have the same count."
            )

        # Rule 5: one theory per choice_set (traceability chain)
        if len(theories) != len(choice_sets):
            raise ValueError(
                f"StrategyTrace: theories ({len(theories)}) and "
                f"choice_sets ({len(choice_sets)}) must have the same count."
            )

        # Rule 6: no duplicate theory_ids in theories
        seen_t: set[str] = set()
        for t in theories:
            if t.theory_id in seen_t:
                raise ValueError(
                    f"StrategyTrace: duplicate theory_id={t.theory_id!r} in theories."
                )
            seen_t.add(t.theory_id)

        # Rule 7: no duplicate theory_ids in evaluations
        seen_ev: set[str] = set()
        for ev in evaluations:
            if ev.theory_id in seen_ev:
                raise ValueError(
                    f"StrategyTrace: duplicate theory_id={ev.theory_id!r} in evaluations."
                )
            seen_ev.add(ev.theory_id)

        # Rule 8: every evaluation theory_id resolves to a theory
        unresolved = seen_ev - seen_t
        if unresolved:
            raise ValueError(
                f"StrategyTrace: evaluation theory_id(s) {sorted(unresolved)!r} "
                f"have no matching theory."
            )

        # Rule 9: selection.winner_theory_id references a known theory
        if selection.winner_theory_id not in seen_t:
            raise ValueError(
                f"StrategyTrace: selection.winner_theory_id={selection.winner_theory_id!r} "
                f"not found in theories. Available: {sorted(seen_t)}"
            )

        # Rule 10: StrategicPosition.theory_of_winning.theory_id == winner
        tow = position.theory_of_winning
        if hasattr(tow, "theory_id") and tow.theory_id != selection.winner_theory_id:
            raise ValueError(
                f"StrategyTrace: strategic_position.theory_of_winning.theory_id="
                f"{tow.theory_id!r} does not match "
                f"selection.winner_theory_id={selection.winner_theory_id!r}."
            )

        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyTrace":
        return cls.model_validate(data)
