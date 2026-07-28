"""PH12.2 — PostureRelevance: normalized concept dictionaries for posture-based content scoring.

Maps canonical posture values (geographic/power/timing) to content-domain keywords.
Used by content assigners to score assumption, risk, opportunity relevance to a theory's
strategic posture — without LLM calls, embeddings, or engagement-specific rules.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Posture → content concept dictionaries
# ---------------------------------------------------------------------------
# Keys are canonical posture values from PostureNormalizer.
# Values are frozensets of lowercase substring triggers that indicate relevance.

_GEO_CONCENTRATED_CONCEPTS: frozenset[str] = frozenset([
    "concentrat", "single-state", "single state", "single market", "anchor",
    "dominan", "focus", "undiversif", "all-in", "committed", "concentration",
    "single-rto", "one market", "geographic risk", "geographic concentrat",
    "execution risk", "capital-at-risk", "single jurisdiction",
])

_GEO_DIVERSIFIED_CONCEPTS: frozenset[str] = frozenset([
    "diversif", "multi-state", "multi-rto", "multi rto", "portfolio", "hedge",
    "resilience", "distribut", "spread", "multiple market", "three-state",
    "three state", "geographic diversif", "correlated risk", "multi-market",
    "coordination", "reallocation", "cross-market",
])

_GEO_STAGED_CONCEPTS: frozenset[str] = frozenset([
    "staged", "phased", "optionality", "conditional", "deferred", "sequential",
    "commit later", "expansion", "scale-up", "initial market", "first market",
    "information gain", "learning", "option value", "contingent",
])

_PWR_GRID_FIRST_CONCEPTS: frozenset[str] = frozenset([
    "grid", "interconnect", "transmission", "queue", "utility", "congestion",
    "curtailment", "grid capacity", "power purchase", "ppa", "grid-depend",
    "network upgrade", "substation", "access charge", "transmission cost",
    "grid reliability", "frequency response",
])

_PWR_BTM_FIRST_CONCEPTS: frozenset[str] = frozenset([
    "behind-the-meter", "behind the meter", "btm", "on-site", "onsite",
    "generation", "fuel", "permitting", "offtake", "generation economics",
    "self-generat", "energy independence", "grid-independ", "gas supply",
    "hydrogen", "fuel cell", "solar", "storage", "battery", "diesel",
    "carbon footprint", "power purchase", "ppa", "capex",
])

_PWR_HYBRID_CONCEPTS: frozenset[str] = frozenset([
    "hybrid", "dual", "parallel", "both", "combination", "grid and btm",
    "grid-and-generation", "mixed pathway", "dual-track", "flexibility",
    "operational complexity", "two-track",
])

_TMG_ACCELERATE_CONCEPTS: frozenset[str] = frozenset([
    "speed", "first-mover", "first mover", "queue position", "early",
    "aggressive", "accelerat", "ahead of", "rapid", "fast", "commit now",
    "pre-empt", "preempt", "window closing", "market timing", "immediate",
    "lock-in", "competitive advantage", "urgency",
])

_TMG_MILESTONE_CONCEPTS: frozenset[str] = frozenset([
    "milestone", "gated", "conditional", "gate", "validation", "trigger",
    "contingent capital", "stage-gate", "approval", "permit", "regulatory",
    "utility confirm", "milestone-based", "condition precedent",
    "decision point", "phased capital", "staged capital",
])

_TMG_WAIT_CONCEPTS: frozenset[str] = frozenset([
    "wait", "monitor", "delay", "defer", "optionality", "opportunity cost",
    "preserve", "hold", "watch", "patience", "uncertainty", "volatility",
    "late entry", "second mover", "observe", "signal", "information",
    "market signal", "option value", "flexibility", "downside protection",
])

# Posture → concept set lookup
POSTURE_CONCEPTS: dict[str, frozenset[str]] = {
    ("geographic", "concentrated"): _GEO_CONCENTRATED_CONCEPTS,
    ("geographic", "diversified"):  _GEO_DIVERSIFIED_CONCEPTS,
    ("geographic", "staged"):       _GEO_STAGED_CONCEPTS,
    ("power", "grid_first"):        _PWR_GRID_FIRST_CONCEPTS,
    ("power", "btm_first"):         _PWR_BTM_FIRST_CONCEPTS,
    ("power", "hybrid"):            _PWR_HYBRID_CONCEPTS,
    ("timing", "accelerate"):       _TMG_ACCELERATE_CONCEPTS,
    ("timing", "milestone_gated"):  _TMG_MILESTONE_CONCEPTS,
    ("timing", "wait_and_monitor"): _TMG_WAIT_CONCEPTS,
}

# Posture pairs that are directly contradictory (for content contradiction scoring)
POSTURE_CONTRADICTIONS: frozenset[tuple[str, str]] = frozenset([
    ("geographic:concentrated", "geographic:diversified"),
    ("geographic:diversified",  "geographic:concentrated"),
    ("timing:accelerate",       "timing:wait_and_monitor"),
    ("timing:wait_and_monitor", "timing:accelerate"),
    ("timing:accelerate",       "timing:milestone_gated"),
    ("power:btm_first",         "power:grid_first"),
    ("power:grid_first",        "power:btm_first"),
])


# ---------------------------------------------------------------------------
# Scoring API
# ---------------------------------------------------------------------------

def posture_relevance_score(
    obj_text: str,
    theory_postures: dict[str, str],
) -> tuple[float, list[str]]:
    """Score an object's relevance to the theory's strategic posture.

    Returns (score, matched_concepts) where score is in [0.0, 1.0].
    Each posture category contributes independently up to a per-category cap.

    Weights:
        direct posture match (title/primary text): +0.50 per category
        compatible concept match:                  +0.25 per category
        generic posture category match:            +0.15 per category
    """
    if not theory_postures or not obj_text:
        return 0.0, []

    text = obj_text.lower()
    matched: list[str] = []
    total_score = 0.0

    for cat, val in theory_postures.items():
        concepts = POSTURE_CONCEPTS.get((cat, val), frozenset())
        if not concepts:
            continue
        hits = [c for c in concepts if c in text]
        if hits:
            # Weight by count capped at per-category contribution
            contrib = min(0.50, len(hits) * 0.15)
            total_score += contrib
            matched.extend(hits[:3])

    return min(round(total_score, 4), 1.0), list(dict.fromkeys(matched))


def contradiction_score(
    obj_text: str,
    theory_postures: dict[str, str],
) -> float:
    """Return a penalty (0.0–0.50) if the object strongly contradicts theory posture.

    Contradiction is detected when the object text contains concepts from the
    opposite posture — i.e. the posture that directly conflicts with the theory.
    """
    if not theory_postures or not obj_text:
        return 0.0

    text = obj_text.lower()
    penalty = 0.0

    for cat, val in theory_postures.items():
        # Find the opposite posture for this category
        if cat == "geographic":
            if val == "concentrated":
                opposite = "diversified"
            elif val == "diversified":
                opposite = "concentrated"
            else:
                opposite = ""
        elif cat == "power":
            if val == "btm_first":
                opposite = "grid_first"
            elif val == "grid_first":
                opposite = "btm_first"
            else:
                opposite = ""
        elif cat == "timing":
            if val == "accelerate":
                opposite = "wait_and_monitor"
            elif val == "wait_and_monitor":
                opposite = "accelerate"
            else:
                opposite = ""
        else:
            opposite = ""

        if not opposite:
            continue
        opp_concepts = POSTURE_CONCEPTS.get((cat, opposite), frozenset())
        if any(c in text for c in opp_concepts):
            penalty += 0.25

    return min(round(penalty, 4), 0.50)


def _obj_text(obj: dict[str, Any]) -> str:
    """Extract all searchable text from a canonical object dict."""
    parts = []
    for field in (
        "statement", "title", "description", "rationale", "category",
        "impact", "mitigation", "mitigation_notes", "expected_benefit",
        "opportunity", "text", "content",
    ):
        val = obj.get(field)
        if isinstance(val, str) and val:
            parts.append(val)
    return " ".join(parts).lower()


def score_item(
    obj: dict[str, Any],
    theory_postures: dict[str, str],
    assignment_type_bonus: float = 0.0,
) -> float:
    """Score a single content object for relevance to the theory.

    assignment_type_bonus: pre-existing bonus from relationship type
        (e.g. +1.0 for explicit option link, +0.80 for rec link).

    Returns score in [0.0, 2.0] (can exceed 1.0 when explicit links add bonus).
    """
    text = _obj_text(obj)
    posture_score, _ = posture_relevance_score(text, theory_postures)
    contra = contradiction_score(text, theory_postures)
    return max(0.0, round(assignment_type_bonus + posture_score - contra, 4))
