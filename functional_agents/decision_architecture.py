"""Decision Architecture — J9.2.

A Decision Architecture sits between the Strategic Engagement and the research
process. It reframes the engagement as an executive decision (statement, scope,
success definition, strategic themes, decision streams, executive unknowns, and
the board decisions required) so that research becomes a supporting workstream
rather than the primary product.

Research questions become children of *decision streams* rather than top-level
objects. The existing framing outputs (decision_areas, research_questions,
critical_uncertainties, evidence_requirements) are preserved unchanged — the
Decision Architecture is derived alongside them.

Derivation is deterministic (no LLM call): it maps the ProblemFramingAgent
payload plus the optional structured engagement onto the executive structure.
This keeps J9.2 free of new truncation/verbosity risk. Fields the client never
supplied are left explicitly empty rather than invented (the J9.1 principle).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

# Bounds — keep the architecture compact and executive (mirrors J9.1b discipline).
_MAX_STREAMS = 8
_MIN_STREAMS_TARGET = 4
_MAX_THEMES = 8
_MAX_SUCCESS = 8
_MAX_EXEC_UNKNOWNS = 6
_MAX_BOARD_DECISIONS = 6
_MAX_IN_SCOPE = 12

_STOPWORDS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with",
    "what", "which", "how", "is", "are", "be", "by", "at", "that", "this",
    "will", "can", "should", "do", "does", "within", "into", "from", "as",
}


class DecisionStream(BaseModel):
    """An executive workstream. Research questions are its children (J9.2)."""

    title: str
    executive_objective: str = ""
    related_strategic_themes: list[str] = Field(default_factory=list)
    research_questions: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)


class DecisionScope(BaseModel):
    """Explicit in/out scope to keep downstream reasoning bounded (J9.2)."""

    in_scope: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)


class DecisionArchitecture(BaseModel):
    """Executive decision framing derived from engagement + problem framing (J9.2)."""

    decision_statement: str = ""
    decision_scope: DecisionScope = Field(default_factory=DecisionScope)
    success_definition: list[str] = Field(default_factory=list)
    strategic_themes: list[str] = Field(default_factory=list)
    decision_streams: list[DecisionStream] = Field(default_factory=list)
    executive_unknowns: list[str] = Field(default_factory=list)
    board_decisions_required: list[str] = Field(default_factory=list)
    out_of_scope_items: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        s = (it or "").strip()
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _assign_questions_to_streams(
    questions: list[str], themes: list[str]
) -> dict[int, list[str]]:
    """Map each research question to the best-matching theme by keyword overlap.

    Every question is assigned exactly once; unmatched questions fall back to
    round-robin so none are dropped (research questions become stream children).
    """
    assignment: dict[int, list[str]] = {i: [] for i in range(len(themes))}
    if not themes:
        return assignment
    theme_kw = [_keywords(t) for t in themes]
    rr = 0
    for q in questions:
        qkw = _keywords(q)
        best_idx, best_score = -1, 0
        for i, tkw in enumerate(theme_kw):
            score = len(qkw & tkw)
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx < 0:
            best_idx = rr % len(themes)
            rr += 1
        assignment[best_idx].append(q)
    return assignment


def _board_decision_for(theme: str) -> str:
    """Turn a strategic theme into an executive approval item."""
    t = theme.strip().rstrip(".")
    low = t.lower()
    if low.startswith(("approve", "decide", "select", "authorize")):
        return t
    return f"Approve {t} approach" if t else "Approve strategy"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_decision_architecture(
    payload: Any,
    engagement: dict | None = None,
) -> DecisionArchitecture:
    """Derive a Decision Architecture from a framing payload + optional engagement.

    ``payload`` is a DecisionModelPayload (objective, decision_areas,
    critical_uncertainties, research_questions, evidence_requirements).
    ``engagement`` is the structured EngagementSpec dict (J9.1) when the run was
    engagement-driven, else None (goal/question mode).
    """
    eng = engagement or {}

    objective = (getattr(payload, "objective", "") or "").strip()
    decision_areas = list(getattr(payload, "decision_areas", None) or [])
    research_questions = list(getattr(payload, "research_questions", None) or [])
    critical_uncertainties = list(getattr(payload, "critical_uncertainties", None) or [])

    eng_objectives = list(eng.get("objectives") or [])
    eng_priorities = list(eng.get("priorities") or [])
    eng_success = list(eng.get("success_criteria") or [])
    eng_horizon = (eng.get("decision_horizon") or "").strip()
    eng_known_unknowns = list(eng.get("known_unknowns") or [])
    eng_constraints = list(eng.get("constraints") or [])

    # --- Decision statement: the condensed objective is already an executive
    #     restatement (J9.1a bounds it). Fall back to the engagement title.
    decision_statement = objective or (eng.get("title") or "").strip()

    # --- Strategic themes: framing decision areas are the workstream dimensions;
    #     augment with client priorities. Deduped and capped.
    strategic_themes = _dedupe_keep_order(decision_areas + eng_priorities)[:_MAX_THEMES]

    # --- Scope: in-scope = themes + client objectives. Out-of-scope is not
    #     supplied by the engagement contract, so record it explicitly empty.
    in_scope = _dedupe_keep_order(strategic_themes + eng_objectives)[:_MAX_IN_SCOPE]
    out_of_scope: list[str] = []

    # --- Success definition: client success criteria + decision horizon.
    success_definition = _dedupe_keep_order(eng_success)[:_MAX_SUCCESS]
    if eng_horizon:
        success_definition = (success_definition + [f"Decision horizon: {eng_horizon}"])[:_MAX_SUCCESS]

    # --- Decision streams: one per theme; research questions become children.
    streams_themes = strategic_themes[:_MAX_STREAMS]
    q_assignment = _assign_questions_to_streams(research_questions, streams_themes)
    decision_streams: list[DecisionStream] = []
    for i, theme in enumerate(streams_themes):
        decision_streams.append(
            DecisionStream(
                title=theme,
                executive_objective=f"Establish the recommended {theme} approach and its trade-offs.",
                related_strategic_themes=[theme],
                research_questions=q_assignment.get(i, []),
                expected_outputs=[
                    f"Assessment of {theme}",
                    "Recommended approach with risks and trade-offs",
                ],
            )
        )
    # If framing produced no themes at all, create a single catch-all stream so
    # research questions are still parented (never dropped).
    if not decision_streams and research_questions:
        decision_streams.append(
            DecisionStream(
                title="Primary Decision Workstream",
                executive_objective=decision_statement or "Resolve the core decision.",
                related_strategic_themes=[],
                research_questions=research_questions,
                expected_outputs=["Recommended approach with risks and trade-offs"],
            )
        )

    # --- Executive unknowns: client known-unknowns are executive-level; fall
    #     back to framing critical uncertainties. These differ from research gaps.
    executive_unknowns = _dedupe_keep_order(
        eng_known_unknowns + critical_uncertainties
    )[:_MAX_EXEC_UNKNOWNS]

    # --- Board decisions required: derived from the strategic themes.
    board_decisions_required = _dedupe_keep_order(
        [_board_decision_for(t) for t in strategic_themes]
    )[:_MAX_BOARD_DECISIONS]

    return DecisionArchitecture(
        decision_statement=decision_statement,
        decision_scope=DecisionScope(in_scope=in_scope, out_of_scope=out_of_scope),
        success_definition=success_definition,
        strategic_themes=strategic_themes,
        decision_streams=decision_streams,
        executive_unknowns=executive_unknowns,
        board_decisions_required=board_decisions_required,
        # The engagement contract does not carry explicit exclusions, so record
        # out-of-scope items explicitly empty rather than inventing them.
        out_of_scope_items=out_of_scope,
    )


def architecture_trace_metadata(arch: DecisionArchitecture) -> dict[str, Any]:
    """Compact counts for the execution trace (J9.2)."""
    return {
        "decision_stream_count": len(arch.decision_streams),
        "strategic_theme_count": len(arch.strategic_themes),
        "board_decision_count": len(arch.board_decisions_required),
        "executive_unknown_count": len(arch.executive_unknowns),
        "success_criteria_count": len(arch.success_definition),
        "research_questions_parented": sum(
            len(s.research_questions) for s in arch.decision_streams
        ),
    }
