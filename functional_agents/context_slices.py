"""Per-agent LLM input slicing (PH3.3).

PH3.2 built a measurement-only Shared Context Compactor
(`functional_agents/context_compactor.py`) that computed, but never applied,
each agent's context-compaction opportunity. PH3.3 performs the actual
cutover for the LLM-backed agents whose real prompt inputs were verified by
reading the source of both prompt-building paths:

  - the LIVE prompt builder (`research_agent/claude_client.py`, e.g.
    `_planning_prompt`, `_hypothesis_prompt`, `_recommendation_prompt`,
    `_strategic_synthesis_prompt`)
  - the corresponding `MockClaudeClient` deterministic method

Slicing here is restricted to TOP-LEVEL DICT SUB-KEYS of metadata objects
(`decision_model`, `research_strategy`, `decision_architecture`) plus one
confirmed-fully-unused LIST field (`domain_evidence`, for Strategic
Synthesis). Evidence-bearing / reference-bearing lists — `evidence_items`,
`hypotheses`, `surviving_hypotheses`, `hypothesis_challenges`,
`contradictions` / `validated_contradictions` — are NEVER trimmed here; they
pass through in full. `strategic_synthesis` is passed through in full too:
its 7 fields were verified to be exactly the 7 fields
`_strategic_synthesis_section` reads — there is no unused sub-key to remove.

Every excluded field was verified UNREAD in BOTH the live prompt-building
function and MockClaudeClient's corresponding method, so removing it cannot
change the prompt text, cannot change the mock's deterministic output, and
cannot change a live model's response (an argument a prompt never
interpolates cannot influence what the model sees).

Verified field usage (as of PH3.3; re-verify against source if the prompt
builders change):

  PlannerAgent      (_planning_prompt / MockClaudeClient.plan_research_question):
    decision_model:      objective, decision_areas, critical_uncertainties,
                          research_questions, evidence_requirements
    research_strategy:   research_question_priorities, required_evidence,
                          source_priorities, coverage_targets

  HypothesisAgent   (_hypothesis_prompt / MockClaudeClient.generate_hypotheses):
    decision_model:      objective, decision_areas, critical_uncertainties
    research_strategy:   research_question_priorities
    (profile_coverage, contradictions are read by the live prompt but not
     trimmed here — they are already small, capped-at-render lists/dicts)

  RecommendationAgent (_recommendation_prompt / MockClaudeClient.generate_recommendations):
    decision_model:      objective, decision_areas
    research_strategy:   NONE — confirmed dead in both the live prompt
                          (parameter never referenced in the f-string) and
                          the mock (parameter never referenced in the body)

  StrategicSynthesisAgent (_strategic_synthesis_prompt / MockClaudeClient.generate_strategic_synthesis):
    decision_architecture: strategic_themes, decision_statement,
                            executive_unknowns
    domain_evidence:        NONE — confirmed dead in both the live prompt and
                             the mock generator
    domain_plans, domain_hypotheses: passed through in full (domain_plans'
      content is read as a mock fallback when domain_hypotheses is empty)
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Verified-used sub-keys per (agent, field) — see module docstring.
# ---------------------------------------------------------------------------

_PLANNER_DECISION_MODEL_KEYS = (
    "objective", "decision_areas", "critical_uncertainties",
    "research_questions", "evidence_requirements",
)
_PLANNER_RESEARCH_STRATEGY_KEYS = (
    "research_question_priorities", "required_evidence",
    "source_priorities", "coverage_targets",
)

_HYPOTHESIS_DECISION_MODEL_KEYS = ("objective", "decision_areas", "critical_uncertainties")
_HYPOTHESIS_RESEARCH_STRATEGY_KEYS = ("research_question_priorities",)

_RECOMMENDATION_DECISION_MODEL_KEYS = ("objective", "decision_areas")
_RECOMMENDATION_RESEARCH_STRATEGY_KEYS: tuple[str, ...] = ()  # confirmed unused

_STRATEGIC_SYNTHESIS_DECISION_ARCHITECTURE_KEYS = (
    "strategic_themes", "decision_statement", "executive_unknowns",
)


def _slice_dict(d: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    """Keep only ``keys`` present in ``d`` — deterministic, order-stable."""
    d = d or {}
    return {k: d[k] for k in keys if k in d}


def _byte_size(value: Any) -> int:
    try:
        return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
    except TypeError:
        return len(str(value).encode("utf-8"))


# ---------------------------------------------------------------------------
# Per-agent input slices
# ---------------------------------------------------------------------------

def planner_input_slice(context: Any) -> dict[str, Any]:
    """PlannerAgent's real prompt inputs: question + trimmed decision_model/research_strategy.

    ``profiles_context`` is intentionally NOT rebuilt here — PlannerAgent
    already constructs it minimally via ``_build_profiles_context()`` from
    ``DomainProfile`` objects that aren't reachable from ``AgentContext`` alone.
    """
    decision_model = getattr(context, "decision_model", None) or {}
    research_strategy = getattr(context, "research_strategy", None) or {}
    return {
        "question": getattr(context, "question", None),
        "decision_model": _slice_dict(decision_model, _PLANNER_DECISION_MODEL_KEYS),
        "research_strategy": _slice_dict(research_strategy, _PLANNER_RESEARCH_STRATEGY_KEYS),
    }


def hypothesis_input_slice(context: Any) -> dict[str, Any]:
    """HypothesisAgent's real prompt inputs."""
    decision_model = getattr(context, "decision_model", None) or {}
    research_strategy = getattr(context, "research_strategy", None) or {}
    evidence_notes = getattr(context, "evidence_notes", None) or []
    evidence_note = evidence_notes[0] if evidence_notes else {}
    evidence_items = evidence_note.get("evidence_items", [])
    raw_coverage = evidence_note.get("profile_coverage_by_profile", {})
    profile_coverage = {
        name: (entry.get("coverage_level", "NONE") or "NONE").lower()
        for name, entry in raw_coverage.items()
    }
    if not profile_coverage:
        profiles = getattr(context, "profiles", None) or []
        profile_coverage = {p: "unknown" for p in profiles}
    research_object = getattr(context, "research_object", None) or {}
    contradictions = research_object.get("contradictions", [])
    return {
        "decision_model": _slice_dict(decision_model, _HYPOTHESIS_DECISION_MODEL_KEYS),
        "research_strategy": _slice_dict(research_strategy, _HYPOTHESIS_RESEARCH_STRATEGY_KEYS),
        "evidence_items": evidence_items,
        "profile_coverage": profile_coverage,
        "contradictions": contradictions,
    }


def recommendation_input_slice(context: Any) -> dict[str, Any]:
    """RecommendationAgent's real prompt inputs."""
    decision_model = getattr(context, "decision_model", None) or {}
    research_strategy = getattr(context, "research_strategy", None) or {}
    evidence_notes = getattr(context, "evidence_notes", None) or []
    evidence_note = evidence_notes[0] if evidence_notes else {}
    research_object = getattr(context, "research_object", None) or {}
    validated_contradictions = (
        getattr(context, "validated_contradictions", None)
        or research_object.get("validated_contradictions", [])
    )
    return {
        "hypotheses": getattr(context, "hypotheses", None) or [],
        "surviving_hypotheses": getattr(context, "surviving_hypotheses", None) or [],
        "hypothesis_challenges": getattr(context, "hypothesis_challenges", None) or [],
        "evidence_items": evidence_note.get("evidence_items", []),
        "decision_model": _slice_dict(decision_model, _RECOMMENDATION_DECISION_MODEL_KEYS),
        "research_strategy": _slice_dict(research_strategy, _RECOMMENDATION_RESEARCH_STRATEGY_KEYS),
        "validated_contradictions": validated_contradictions,
        "strategic_synthesis": getattr(context, "strategic_synthesis", None) or {},
    }


def strategic_synthesis_input_slice(context: Any) -> dict[str, Any]:
    """StrategicSynthesisAgent's real prompt inputs.

    ``domain_evidence`` is excluded entirely: verified unread by both the
    live prompt (``_strategic_synthesis_prompt``) and the mock generator.
    """
    decision_architecture = getattr(context, "decision_architecture", None) or {}
    return {
        "domain_plans": getattr(context, "domain_plans", None) or [],
        "domain_evidence": [],
        "domain_hypotheses": getattr(context, "domain_hypotheses", None) or [],
        "decision_architecture": _slice_dict(
            decision_architecture, _STRATEGIC_SYNTHESIS_DECISION_ARCHITECTURE_KEYS
        ),
    }


# ---------------------------------------------------------------------------
# Observability — byte-size diagnostics (PH3.3 point 4)
# ---------------------------------------------------------------------------

def slice_diagnostics(original: dict[str, Any], sliced: dict[str, Any]) -> dict[str, Any]:
    """Compute before/after byte-size diagnostics for one agent's slice call.

    ``original`` is the payload that would have been sent pre-PH3.3 (the
    unsliced context fields); ``sliced`` is what is actually sent now.
    Reports overall byte reduction plus, for each dict-shaped field, the
    included/excluded sub-keys, and for list-shaped fields that were fully
    zeroed out, an explicit exclusion note.
    """
    original_bytes = _byte_size(original)
    sliced_bytes = _byte_size(sliced)
    reduction_pct = (
        round(100.0 * (original_bytes - sliced_bytes) / original_bytes, 1)
        if original_bytes else 0.0
    )

    fields_included: list[str] = []
    fields_excluded: list[str] = []
    for key, orig_val in original.items():
        sliced_val = sliced.get(key)
        if isinstance(orig_val, dict):
            kept = set((sliced_val or {}).keys())
            dropped = set(orig_val.keys()) - kept
            fields_included.extend(f"{key}.{k}" for k in sorted(kept))
            fields_excluded.extend(f"{key}.{k}" for k in sorted(dropped))
        elif isinstance(orig_val, list):
            if len(orig_val) > 0 and not sliced_val:
                fields_excluded.append(f"{key} (all {len(orig_val)} items)")
            else:
                fields_included.append(key)
        else:
            fields_included.append(key)

    return {
        "original_bytes": original_bytes,
        "sliced_bytes": sliced_bytes,
        "bytes_saved": max(0, original_bytes - sliced_bytes),
        "reduction_pct": reduction_pct,
        "fields_included": fields_included,
        "fields_excluded_count": len(fields_excluded),
        "fields_excluded": fields_excluded,
    }


def record_slice_diagnostics(
    context: Any, agent_label: str, original: dict[str, Any], sliced: dict[str, Any],
) -> dict[str, Any]:
    """Compute and record PH3.3 slice diagnostics for one agent's LLM call.

    Records into:
      - ``context.trace["_<agent_label>_prompt_slice"]`` (always, JSON-safe)
      - the active ``PerformanceTracker`` as the current agent's pending
        prompt-slice record (flushed onto its ``AgentPerfRecord`` in
        ``FunctionalAgent.run()``, mirroring the sub-phase/compaction pattern)

    Read-only with respect to business state — the only context mutation is
    the additive trace scratch key above.
    """
    diag = slice_diagnostics(original, sliced)
    trace = getattr(context, "trace", None)
    if isinstance(trace, dict):
        trace[f"_{agent_label}_prompt_slice"] = diag
        tracker = trace.get("_perf_tracker")
        if tracker is not None and hasattr(tracker, "record_prompt_slice"):
            tracker.record_prompt_slice(diag)
    return diag
