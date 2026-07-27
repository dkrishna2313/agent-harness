"""StrategyLineageLink and lineage builder (PH11.2).

StrategyLineageLink records one directed provenance edge in the Strategy
Layer derivation chain. build_strategy_lineage() produces the complete set
of edges from ResearchObject through StrategyTrace.

Links are immutable (frozen=True, extra=forbid) and validate that all
ID and relationship fields are non-empty, non-whitespace strings.

PH11.2 lineage categories (8 types):
  1. ResearchObject → StrategyPlan           (informs      — 1 link)
  2. StrategyPlan → StrategicChoiceSet[]    (generates    — one per set)
  3. StrategicChoiceSet → TheoryOfWinning[] (produces     — one per pair)
  4. TheoryOfWinning → TheoryEvaluation[]   (evaluated_by — one per pair)
  5. TheoryEvaluation → StrategySelection   (contributes_to — all evals)
  6. StrategySelection → selected Theory    (selects      — 1 link)
  7. selected Theory → StrategicPosition    (grounds      — 1 link)
  8. StrategicPosition → StrategyTrace      (captured_in  — 1 link)

Total link count for N choice_sets: 4N + 4.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from .strategic_choice_set import StrategicChoiceSet
from .strategic_position import StrategicPosition, TheoryOfWinning
from .strategy_plan import StrategyPlan
from .strategy_selector import StrategySelection
from .theory_evaluation import TheoryEvaluation


class StrategyLineageLink(BaseModel):
    """One directed provenance edge in the Strategy Layer derivation chain."""

    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relationship: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True, "extra": "forbid"}

    @field_validator("source_type", "source_id", "target_type", "target_id", "relationship")
    @classmethod
    def _nonempty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(
                f"{info.field_name} must be a non-empty, non-whitespace string."
            )
        return v


def build_strategy_lineage(
    *,
    research_id: str,
    plan: StrategyPlan,
    choice_sets: list[StrategicChoiceSet],
    theories: list[TheoryOfWinning],
    evaluations: list[TheoryEvaluation],
    selection: StrategySelection,
    strategic_position: StrategicPosition,
    trace_id: str,
) -> list[StrategyLineageLink]:
    """Build the canonical 8-category lineage chain for a Strategy Layer run.

    Returns one StrategyLineageLink per relationship edge, ordered to follow
    the execution chain:

        ResearchObject → StrategyPlan → StrategicChoiceSet[]
        → TheoryOfWinning[] → TheoryEvaluation[] → StrategySelection
        → selected TheoryOfWinning → StrategicPosition → StrategyTrace

    Each theory must have a non-empty ``source_choice_set_id`` field.
    If a theory lacks one, the StrategicChoiceSet→TheoryOfWinning link
    for that theory will have an empty ``source_id`` and
    ``StrategyLineageLink`` validation will raise.

    Parameters
    ----------
    research_id:
        Canonical ID of the source ResearchObject
        (``research_object.get("id") or run_id``).
    plan:
        StrategyPlan produced by StrategyPlanner.
    choice_sets:
        Ordered list of StrategicChoiceSets.
    theories:
        Ordered list of TheoryOfWinning objects (same order as choice_sets).
    evaluations:
        Ordered list of TheoryEvaluation objects (same order as theories).
    selection:
        StrategySelection produced by StrategySelector.
    strategic_position:
        StrategicPosition produced by StrategyCoordinator.
    trace_id:
        trace_id of the StrategyTrace being assembled
        (``f"STRAT-{plan.plan_id}"``).
    """
    links: list[StrategyLineageLink] = []
    sel_id = f"SEL-{trace_id}"  # stable canonical ID for the selection node

    # 1. ResearchObject → StrategyPlan
    links.append(StrategyLineageLink(
        source_type="research_object",
        source_id=research_id,
        target_type="strategy_plan",
        target_id=plan.plan_id,
        relationship="informs",
    ))

    # 2. StrategyPlan → StrategicChoiceSet (one per set)
    for cs in choice_sets:
        links.append(StrategyLineageLink(
            source_type="strategy_plan",
            source_id=plan.plan_id,
            target_type="strategic_choice_set",
            target_id=cs.id,
            relationship="generates",
        ))

    # 3. StrategicChoiceSet → TheoryOfWinning (one per theory, via source_choice_set_id)
    for theory in theories:
        links.append(StrategyLineageLink(
            source_type="strategic_choice_set",
            source_id=theory.source_choice_set_id,
            target_type="theory_of_winning",
            target_id=theory.theory_id,
            relationship="produces",
        ))

    # 4. TheoryOfWinning → TheoryEvaluation (one per pair; evaluation keyed by theory_id)
    eval_ids: set[str] = {ev.theory_id for ev in evaluations}
    for theory in theories:
        if theory.theory_id in eval_ids:
            links.append(StrategyLineageLink(
                source_type="theory_of_winning",
                source_id=theory.theory_id,
                target_type="theory_evaluation",
                target_id=theory.theory_id,
                relationship="evaluated_by",
            ))

    # 5. TheoryEvaluation → StrategySelection (all evaluations contribute to selection)
    for ev in evaluations:
        links.append(StrategyLineageLink(
            source_type="theory_evaluation",
            source_id=ev.theory_id,
            target_type="strategy_selection",
            target_id=sel_id,
            relationship="contributes_to",
        ))

    # 6. StrategySelection → selected TheoryOfWinning
    links.append(StrategyLineageLink(
        source_type="strategy_selection",
        source_id=sel_id,
        target_type="theory_of_winning",
        target_id=selection.winner_theory_id,
        relationship="selects",
    ))

    # 7. Selected TheoryOfWinning → StrategicPosition
    links.append(StrategyLineageLink(
        source_type="theory_of_winning",
        source_id=selection.winner_theory_id,
        target_type="strategic_position",
        target_id=strategic_position.position_id,
        relationship="grounds",
    ))

    # 8. StrategicPosition → StrategyTrace
    links.append(StrategyLineageLink(
        source_type="strategic_position",
        source_id=strategic_position.position_id,
        target_type="strategy_trace",
        target_id=trace_id,
        relationship="captured_in",
    ))

    return links
