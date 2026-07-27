"""StrategyNarrative — presentation-focused editorial model for Strategy content (PH11.4).

Carries selected Strategy reasoning into the editorial layer.
Produced by build_strategy_narrative() from a StrategyTrace.
Consumed by StrategyWriter and MarkdownRenderer.

No LLM calls. No file I/O. No modifications to the StrategyTrace.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StrategyAlternativeSummary(BaseModel):
    """Compact summary of a non-winning theory for editorial presentation."""

    theory_id: str
    recommended_option_title: str = ""
    score: float = 0.0
    confidence: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    residual_risks: list[str] = Field(default_factory=list)

    model_config = {"frozen": True}


class StrategyNarrative(BaseModel):
    """Presentation-focused model carrying Strategy Layer reasoning into the editorial layer.

    Derived from a StrategyTrace by build_strategy_narrative().
    Contains no raw reasoning fields — only presentation-ready content for
    StrategyWriter and MarkdownRenderer to consume.

    Immutable after construction (frozen=True).
    """

    # Identity
    trace_id: str
    framework: str = ""
    strategic_position_id: str = ""

    # Winner
    winner_theory_id: str
    winner_option_title: str = ""
    winning_position: str = ""
    winning_mechanism: str = ""
    winner_score: float = 0.0
    overall_confidence: str = ""

    # Runner-up / score comparison (None when only one theory existed)
    runner_up_theory_id: str | None = None
    runner_up_score: float | None = None
    score_margin: float | None = None
    tie_breaker_used: str | None = None

    # Evaluation criteria and per-criterion scores
    evaluation_criteria: list[str] = Field(default_factory=list)
    criterion_scores: dict[str, float] = Field(default_factory=dict)
    winner_evaluation_strengths: list[str] = Field(default_factory=list)

    # Winner theory content
    assumptions: list[str] = Field(default_factory=list)
    success_conditions: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    winner_strategic_choices: list[str] = Field(default_factory=list)

    # Alternatives (non-winner theories, sorted by score descending)
    alternatives: list[StrategyAlternativeSummary] = Field(default_factory=list)

    model_config = {"frozen": True}


def build_strategy_narrative(trace: Any) -> StrategyNarrative:
    """Build a StrategyNarrative from a StrategyTrace.

    Pure function: no LLM calls, no file I/O, no mutations to the trace.

    Parameters
    ----------
    trace:
        A StrategyTrace instance. Typed as Any to avoid circular imports;
        callers should pass a validated StrategyTrace.

    Returns
    -------
    StrategyNarrative
        Presentation-ready editorial model derived from the trace.
    """
    sel = trace.selection

    winner_theory = next(t for t in trace.theories if t.theory_id == sel.winner_theory_id)
    winner_eval = next(ev for ev in trace.evaluations if ev.theory_id == sel.winner_theory_id)

    # Evaluation criteria and scores from the winner's evaluation
    criteria_names = list(winner_eval.criteria_scores.keys())
    crit_scores = {k: cs.score for k, cs in winner_eval.criteria_scores.items()}

    # Extract assumption statements (list[dict] on TheoryOfWinning)
    assumptions: list[str] = []
    for a in winner_theory.assumptions:
        if isinstance(a, dict):
            assumptions.append(a.get("statement", str(a)))
        else:
            assumptions.append(str(a))

    # Extract failure mode descriptions (list[dict] on TheoryOfWinning)
    failure_modes: list[str] = []
    for fm in winner_theory.failure_modes:
        if isinstance(fm, dict):
            failure_modes.append(fm.get("description", fm.get("mode", str(fm))))
        else:
            failure_modes.append(str(fm))

    # Winner evaluation strengths
    winner_evaluation_strengths = list(winner_eval.strengths)

    # Extract strategic choices as readable strings (list[dict] on TheoryOfWinning)
    winner_strategic_choices: list[str] = []
    for sc in winner_theory.strategic_choices:
        if isinstance(sc, dict):
            dim = sc.get("dimension", sc.get("id", ""))
            val = sc.get("selected_value", "")
            conf = sc.get("confidence", "")
            label = f"{dim}: {val}" + (f" ({conf} confidence)" if conf else "")
            winner_strategic_choices.append(label)
        else:
            winner_strategic_choices.append(str(sc))

    # Build alternatives (non-winner theories), sorted by score descending
    eval_by_id = {ev.theory_id: ev for ev in trace.evaluations}
    alternatives: list[StrategyAlternativeSummary] = []
    for theory in trace.theories:
        if theory.theory_id == sel.winner_theory_id:
            continue
        ev = eval_by_id.get(theory.theory_id)
        # Extract residual risk descriptions
        residual_risk_descs: list[str] = []
        if ev:
            for rr in ev.residual_risks:
                if isinstance(rr, dict):
                    residual_risk_descs.append(rr.get("description", str(rr)))
                else:
                    residual_risk_descs.append(str(rr))
        alternatives.append(
            StrategyAlternativeSummary(
                theory_id=theory.theory_id,
                recommended_option_title=theory.recommended_option_title,
                score=ev.overall_score if ev else 0.0,
                confidence=ev.confidence if ev else "",
                strengths=list(ev.strengths) if ev else [],
                weaknesses=list(ev.weaknesses) if ev else [],
                residual_risks=residual_risk_descs,
            )
        )
    alternatives.sort(key=lambda a: -a.score)

    framework = trace.metadata.get("framework", "") or getattr(trace.plan, "framework", "")

    return StrategyNarrative(
        trace_id=trace.trace_id,
        framework=framework,
        strategic_position_id=trace.strategic_position.position_id,
        winner_theory_id=sel.winner_theory_id,
        winner_option_title=winner_theory.recommended_option_title,
        winning_position=winner_theory.winning_position,
        winning_mechanism=winner_theory.winning_mechanism,
        winner_score=sel.winner_score,
        overall_confidence=winner_eval.confidence,
        runner_up_theory_id=sel.runner_up_theory_id,
        runner_up_score=sel.runner_up_score,
        score_margin=sel.score_margin,
        tie_breaker_used=sel.tie_breaker_used,
        evaluation_criteria=criteria_names,
        criterion_scores=crit_scores,
        winner_evaluation_strengths=winner_evaluation_strengths,
        assumptions=assumptions,
        success_conditions=list(winner_theory.success_conditions),
        failure_modes=failure_modes,
        winner_strategic_choices=winner_strategic_choices,
        alternatives=alternatives,
    )
