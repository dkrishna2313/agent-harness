"""Shared Context Compactor (PH3.2).

The functional pipeline accumulates a growing shared context (decision model,
research strategy, evidence, hypotheses, recommendations, ...) as it runs.
Each downstream agent only actually reads a subset of that context — PH3.1
found the pipeline is heavily LLM-bound with large, overlapping context
re-sent across many sequential calls.

This module provides a reusable, deterministic compactor:

    build_context_sections(context)         → full named-section snapshot
    compact_context(sections, profile)      → drop unused + duplicate sections
    compact_context_for_agent(ctx, name)    → the two composed for one agent
    measure_and_record(ctx, name)           → compute + record onto the
                                               active PerformanceTracker

The compactor never summarizes, rewrites, or truncates a kept section — it
only decides, at whole-section granularity, which sections an agent's
*documented* profile needs and drops the rest. Given the same input it always
produces the same output (no timestamps, no randomness).

PH3.2 integration is measurement-only (see base.py `FunctionalAgent.run()`):
it computes what compaction *would* achieve for each agent's real context and
records it as diagnostics. It does not change what any agent actually reads,
what any prompt says, or what any LLM call sends — so reasoning, prompt
wording, and outputs are unchanged. Acting on the measured savings (actually
scoping what is sent per call) is a follow-up milestone, gated on reviewing
this data — the same "measure before optimizing" pattern established in PH3.1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Matches the existing instrumentation-only heuristic in
# functional_agents/research_strategy_agent.py (_CHARS_PER_TOKEN = 4).
CHARS_PER_TOKEN = 4


def _stringify(value: Any) -> str:
    """Canonical, deterministic string form of a section's content.

    Dicts/lists are serialized with sorted keys so that key-insertion-order
    differences don't cause two logically-identical sections to be treated as
    different (and vice versa) — required for stable duplicate detection.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except TypeError:
        return str(value)


def estimate_tokens(value: Any) -> int:
    """Best-effort token count for instrumentation only — not a live tokenizer."""
    if value is None:
        return 0
    if isinstance(value, (str, list, dict, tuple, set)) and len(value) == 0:
        return 0
    text = _stringify(value)
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Compaction result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DuplicateSection:
    name: str
    duplicate_of: str


@dataclass(frozen=True)
class CompactionResult:
    """Outcome of compacting one context bundle down to one agent's profile."""

    compacted: dict[str, Any]
    original_sections: tuple[str, ...]
    kept_sections: tuple[str, ...]
    removed_unused: tuple[str, ...]
    removed_duplicate: tuple[DuplicateSection, ...]
    original_tokens: int
    compacted_tokens: int
    section_tokens: dict[str, int]

    @property
    def tokens_saved(self) -> int:
        return max(0, self.original_tokens - self.compacted_tokens)

    @property
    def reduction_pct(self) -> float:
        if self.original_tokens <= 0:
            return 0.0
        return round(100.0 * self.tokens_saved / self.original_tokens, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_sections": list(self.original_sections),
            "kept_sections": list(self.kept_sections),
            "removed_unused": list(self.removed_unused),
            "removed_duplicate": [
                {"name": d.name, "duplicate_of": d.duplicate_of} for d in self.removed_duplicate
            ],
            "original_tokens": self.original_tokens,
            "compacted_tokens": self.compacted_tokens,
            "tokens_saved": self.tokens_saved,
            "reduction_pct": self.reduction_pct,
        }


def compact_context(sections: dict[str, Any], profile: Sequence[str]) -> CompactionResult:
    """Compact a named-section context bundle down to a target profile.

    Deterministic and non-destructive to kept content:
      1. Sections not named in ``profile`` are dropped (``removed_unused``).
      2. Among the remaining sections, any section whose content is
         byte-identical (after canonical stringification) to an earlier-kept
         section is dropped as a duplicate (``removed_duplicate``) — its
         information is still present under the first section's name, so no
         semantics are lost.

    This function never rewrites, truncates, or summarizes a kept section's
    content; it only chooses which whole sections to keep. Pure function of
    its inputs — same input always yields the same output.
    """
    if not isinstance(sections, dict):
        raise TypeError("sections must be a dict of {name: value}")

    profile_set = set(profile)
    original_names = tuple(sections.keys())
    section_tokens = {name: estimate_tokens(value) for name, value in sections.items()}
    original_tokens = sum(section_tokens.values())

    removed_unused = tuple(name for name in original_names if name not in profile_set)

    kept: dict[str, Any] = {}
    seen_content: dict[str, str] = {}
    removed_duplicate: list[DuplicateSection] = []
    for name in original_names:
        if name not in profile_set:
            continue
        content_key = _stringify(sections[name])
        if content_key and content_key in seen_content:
            removed_duplicate.append(DuplicateSection(name=name, duplicate_of=seen_content[content_key]))
            continue
        seen_content[content_key] = name
        kept[name] = sections[name]

    compacted_tokens = sum(section_tokens[name] for name in kept)

    return CompactionResult(
        compacted=kept,
        original_sections=original_names,
        kept_sections=tuple(kept.keys()),
        removed_unused=removed_unused,
        removed_duplicate=tuple(removed_duplicate),
        original_tokens=original_tokens,
        compacted_tokens=compacted_tokens,
        section_tokens=section_tokens,
    )


# ---------------------------------------------------------------------------
# Canonical section catalog + AgentContext extraction
# ---------------------------------------------------------------------------

# Fixed canonical order — independent of any one profile's ordering, so
# "preserve ordering" always yields the same section sequence.
SECTION_ORDER: tuple[str, ...] = (
    "question",
    "profiles",
    "decision_model",
    "research_strategy",
    "plan",
    "evidence_notes",
    "hypotheses",
    "surviving_hypotheses",
    "hypothesis_challenges",
    "strategic_synthesis",
    "contradictions",
    "recommendations",
    "recommendation_portfolio",
    "assumptions",
    "risks",
    "opportunities",
    "strategic_options",
    "decision_analysis",
    "executive_confidence",
    "artifacts",
    "documents",
    "memo",
)


def _get_contradictions(ctx: Any) -> list:
    ro = getattr(ctx, "research_object", None)
    if ro:
        return ro.get("contradictions", [])
    return getattr(ctx, "validated_contradictions", None) or []


def _get_evidence_note(ctx: Any) -> dict:
    notes = getattr(ctx, "evidence_notes", None)
    return notes[0] if notes else {}


def _get_trace_value(ctx: Any, key: str) -> Any:
    trace = getattr(ctx, "trace", None)
    return trace.get(key) if isinstance(trace, dict) else None


_SECTION_ACCESSORS: dict[str, Callable[[Any], Any]] = {
    "question": lambda ctx: getattr(ctx, "question", None),
    "profiles": lambda ctx: sorted(getattr(ctx, "profiles", None) or []),
    "decision_model": lambda ctx: getattr(ctx, "decision_model", None) or {},
    "research_strategy": lambda ctx: getattr(ctx, "research_strategy", None) or {},
    "plan": lambda ctx: getattr(ctx, "plan", None) or {},
    "evidence_notes": _get_evidence_note,
    "hypotheses": lambda ctx: getattr(ctx, "hypotheses", None) or [],
    "surviving_hypotheses": lambda ctx: getattr(ctx, "surviving_hypotheses", None) or [],
    "hypothesis_challenges": lambda ctx: getattr(ctx, "hypothesis_challenges", None) or [],
    "strategic_synthesis": lambda ctx: getattr(ctx, "strategic_synthesis", None) or {},
    "contradictions": _get_contradictions,
    "recommendations": lambda ctx: getattr(ctx, "recommendations", None) or [],
    "recommendation_portfolio": lambda ctx: getattr(ctx, "recommendation_portfolio", None) or {},
    "assumptions": lambda ctx: getattr(ctx, "assumptions", None) or [],
    "risks": lambda ctx: getattr(ctx, "risks", None) or [],
    "opportunities": lambda ctx: getattr(ctx, "opportunities", None) or [],
    "strategic_options": lambda ctx: getattr(ctx, "strategic_options", None) or [],
    "decision_analysis": lambda ctx: getattr(ctx, "decision_analysis", None) or {},
    "executive_confidence": lambda ctx: getattr(ctx, "executive_confidence", None) or {},
    "artifacts": lambda ctx: getattr(ctx, "artifacts", None) or {},
    "documents": lambda ctx: _get_trace_value(ctx, "_documents") or [],
    "memo": lambda ctx: _get_trace_value(ctx, "_memo"),
}


def build_context_sections(context: Any, section_names: Sequence[str] | None = None) -> dict[str, Any]:
    """Read-only, deterministic {section_name: value} snapshot of an AgentContext.

    Never mutates ``context``. ``section_names`` defaults to the full
    :data:`SECTION_ORDER` catalog; pass a subset to snapshot fewer sections.
    An accessor error yields ``None`` for that section rather than raising,
    since this is diagnostic/measurement tooling, not business logic.
    """
    names = section_names if section_names is not None else SECTION_ORDER
    sections: dict[str, Any] = {}
    for name in names:
        accessor = _SECTION_ACCESSORS.get(name)
        if accessor is None:
            continue
        try:
            sections[name] = accessor(context)
        except Exception:
            sections[name] = None
    return sections


# ---------------------------------------------------------------------------
# Agent Context Profiles — derived from current prompt/business-logic usage
# (verified against source at PH3.2 time; NOT guessed). Each comment cites the
# reading that established it. An agent's own outputs are never listed as
# inputs to itself.
# ---------------------------------------------------------------------------

AGENT_CONTEXT_PROFILES: dict[str, tuple[str, ...]] = {
    # planner_agent.py: context.question / target.question (~L59),
    # context.decision_model, context.research_strategy (~L63-64) passed into
    # plan_research_question_raw(); context.profiles (~L170) for profile listing.
    "PlannerAgent": ("question", "profiles", "decision_model", "research_strategy"),

    # evidence_agent.py: KB path uses context.plan (subquestions/investigation_areas,
    # ~L430-431) and context.question (~L434) to drive retrieval, context.profiles
    # (~L638) for profile attribution; legacy path calls
    # agent.analyze(context.question, documents) (~L753). decision_model and
    # research_strategy are NOT read by EvidenceAgent.
    "EvidenceAgent": ("question", "plan", "profiles"),

    # hypothesis_agent.py _execute_single (~L93-117): evidence_notes (evidence_items
    # + profile_coverage_by_profile), context.profiles (fallback coverage),
    # contradictions (research_object), context.decision_model, context.research_strategy
    # — all passed into _generate_hypotheses().
    "HypothesisAgent": (
        "evidence_notes", "profiles", "contradictions", "decision_model", "research_strategy",
    ),

    # recommendation_agent.py _execute (~L59-92): hypotheses, evidence_notes
    # (evidence_items), surviving_hypotheses, hypothesis_challenges, decision_model,
    # research_strategy, validated_contradictions, strategic_synthesis — all
    # passed into _generate_recommendations().
    "RecommendationAgent": (
        "hypotheses", "surviving_hypotheses", "hypothesis_challenges",
        "evidence_notes", "decision_model", "research_strategy",
        "contradictions", "strategic_synthesis",
    ),

    # report_agent.py _build_j7_executive_report(): reads nearly the entire
    # decision graph (question, plan, evidence_notes, hypotheses,
    # surviving_hypotheses, hypothesis_challenges, assumptions, risks,
    # opportunities, strategic_options, decision_analysis, executive_confidence,
    # recommendations, recommendation_portfolio, profiles, artifacts, plus
    # trace-held documents/memo). Report legitimately needs almost everything —
    # there is little compaction opportunity here, which is itself a useful
    # finding (see docs/architecture/PH3.2).
    "ReportAgent": (
        "question", "plan", "evidence_notes", "hypotheses", "surviving_hypotheses",
        "hypothesis_challenges", "assumptions", "risks", "opportunities",
        "strategic_options", "decision_analysis", "executive_confidence",
        "recommendations", "recommendation_portfolio", "profiles",
        "documents", "memo", "artifacts",
    ),
}


def compact_context_for_agent(context: Any, agent_name: str) -> CompactionResult | None:
    """Compose build_context_sections + compact_context for one named agent.

    Returns None when the agent has no documented profile (e.g. agents not
    yet covered by PH3.2, or non-LLM agents with no meaningful context need).
    """
    profile = AGENT_CONTEXT_PROFILES.get(agent_name)
    if profile is None:
        return None
    sections = build_context_sections(context)
    return compact_context(sections, profile)


def measure_and_record(context: Any, agent_name: str) -> CompactionResult | None:
    """Measure this agent's context compaction and record it on the active
    PerformanceTracker (no-op without one). Read-only: never mutates context,
    never changes what the agent actually reads or sends downstream.
    """
    result = compact_context_for_agent(context, agent_name)
    if result is None:
        return None
    trace = getattr(context, "trace", None)
    tracker = trace.get("_perf_tracker") if isinstance(trace, dict) else None
    if tracker is not None:
        tracker.record_context_compaction(result.to_dict())
    return result
