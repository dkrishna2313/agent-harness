"""StrategyNarrative — presentation-focused editorial model for Strategy content (PH11.4).
PH12.1b — Added authority fields: alignment_status, alignment_narrative, mapped_option_id,
preferred_option_id, mapping_confidence, saturation_detected, selection_status,
constraint_outcomes, winner_theory_label.

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

    # PH12.1b — authority fields (backward-compatible optional)
    winner_theory_label: str = ""          # "Diversified / BTM First / Milestone Gated"
    alignment_status: str = ""
    alignment_narrative: str = ""
    mapped_option_id: str | None = None
    mapped_option_title: str = ""
    preferred_option_id: str = ""
    preferred_option_title: str = ""
    mapping_confidence: str = ""
    saturation_detected: bool = False
    selection_status: str = "selected"
    constraint_outcomes: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Alignment narrative builder (deterministic, status-driven)
# ---------------------------------------------------------------------------

def _build_alignment_narrative(status: str, winner_theory_label: str) -> str:
    """Build a short deterministic authority narrative based on alignment status."""
    if status == "confirmed":
        return "The configured Strategy evaluation confirms the upstream preferred option."
    if status == "refined":
        posture = f" through a {winner_theory_label.lower()} execution posture" if winner_theory_label else ""
        return (
            f"The configured Strategy evaluation reinforces the upstream preferred option"
            f"{posture}."
        )
    if status == "challenged":
        return (
            "The configured Strategy evaluation selects a different option and exceeds "
            "the upstream preferred option by the required challenge margin."
        )
    # unresolved or empty
    return (
        "The configured Strategy evaluation did not establish a sufficiently reliable "
        "option mapping. The upstream preferred option remains authoritative."
    )


def build_strategy_narrative(trace: Any) -> "StrategyNarrative":
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
            text = (
                fm.get("statement")
                or fm.get("description")
                or fm.get("mode")
                or str(fm)
            )
            sev = fm.get("severity", "")
            lik = fm.get("likelihood", "")
            mit = fm.get("mitigation_notes") or fm.get("mitigation", "")
            suffix = ""
            if sev:
                suffix += f" [Severity: {sev}]"
            if lik:
                suffix += f" [Likelihood: {lik}]"
            if mit:
                suffix += f" — Mitigation: {mit}"
            failure_modes.append(f"{text}{suffix}")
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

    # PH12.1b — winner theory label (human-readable choice summary)
    winner_theory_label_parts: list[str] = []
    for sc in winner_theory.strategic_choices:
        if isinstance(sc, dict):
            meta = sc.get("metadata") or {}
            title = meta.get("choice_title") or sc.get("selected_value", "")
            if title:
                winner_theory_label_parts.append(str(title).title())
    winner_theory_label = " / ".join(winner_theory_label_parts)

    # Build alternatives (non-winner theories), sorted by score descending
    eval_by_id = {ev.theory_id: ev for ev in trace.evaluations}
    alternatives: list[StrategyAlternativeSummary] = []
    for theory in trace.theories:
        if theory.theory_id == sel.winner_theory_id:
            continue
        ev = eval_by_id.get(theory.theory_id)
        residual_risk_descs: list[str] = []
        if ev:
            for rr in ev.residual_risks:
                if isinstance(rr, dict):
                    residual_risk_descs.append(
                        rr.get("statement")
                        or rr.get("description")
                        or rr.get("mode")
                        or str(rr)
                    )
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

    # PH12.1b — extract authority fields from trace
    alignment_block = getattr(trace, "alignment", {}) or {}
    saturation_block = getattr(trace, "saturation", {}) or {}

    alignment_status = alignment_block.get("status", "")
    mapped_option_id = alignment_block.get("mapped_option_id")
    preferred_option_id = alignment_block.get("preferred_option_id", "")

    # mapping_confidence: look up from theory_option_mappings for the winner
    mapping_confidence = ""
    for tom in (getattr(trace, "theory_option_mappings", None) or []):
        if isinstance(tom, dict) and tom.get("theory_id") == sel.winner_theory_id:
            mapping_confidence = tom.get("mapping_confidence", "")
            break

    saturation_detected = bool(saturation_block.get("detected", False))
    selection_status = getattr(sel, "selection_status", "selected") or "selected"

    # constraint_outcomes for winner theory
    constraint_results = getattr(trace, "constraint_results", {}) or {}
    raw_constraints = constraint_results.get(sel.winner_theory_id, [])
    constraint_outcomes: list[dict[str, Any]] = []
    for cr in raw_constraints:
        if isinstance(cr, dict):
            constraint_outcomes.append(cr)

    alignment_narrative = _build_alignment_narrative(alignment_status, winner_theory_label)

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
        # PH12.1b authority fields
        winner_theory_label=winner_theory_label,
        alignment_status=alignment_status,
        alignment_narrative=alignment_narrative,
        mapped_option_id=mapped_option_id,
        mapped_option_title="",       # title lookup deferred to renderer via brief
        preferred_option_id=preferred_option_id,
        preferred_option_title="",    # title lookup deferred to renderer via brief
        mapping_confidence=mapping_confidence,
        saturation_detected=saturation_detected,
        selection_status=selection_status,
        constraint_outcomes=constraint_outcomes,
    )
