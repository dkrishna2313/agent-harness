"""PH12.1b — PostureNormalizer: canonical posture synonym groups and contradiction table.

Maps theory choice values and option text to canonical posture values.
Used by OptionMapper to enable contradiction-aware, posture-weighted mapping.

No LLM calls. No external services. Fully deterministic.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Geographic posture synonym triggers (substring matching, lowercased)
# ---------------------------------------------------------------------------
_GEO_CONCENTRATED: frozenset[str] = frozenset([
    "concentrat", "single-state", "single_state", "one-state", "anchor-market",
    "dominan", "texas-first", "texas first", "regional-focus", "regional focus",
    "focused market", "undiversif", "single market",
])
_GEO_DIVERSIFIED: frozenset[str] = frozenset([
    "diversif", "multi-state", "multi-rto", "multi rto", "portfolio-hedge",
    "portfolio hedge", "three-state", "three state", "four-state", "distribut",
    "geographic-resilience", "geographic resilience", "multiple state",
    "spread investment", "multiple market",
])
_GEO_STAGED: frozenset[str] = frozenset([
    "staged", "phased", "optionality", "conditional-commit", "conditional commit",
    "deferred-commit", "deferred commit", "begin concentrat", "start concentrat",
    "then diversif",
])

# ---------------------------------------------------------------------------
# Power pathway synonym triggers
# ---------------------------------------------------------------------------
_PWR_GRID_FIRST: frozenset[str] = frozenset([
    "grid-first", "grid first", "utility-interconnect", "utility interconnect",
    "grid-depend", "grid depend", "utility-scale", "utility scale",
    "interconnect-led", "transmission-led", "grid-only", "grid only",
    "utility-led",
])
_PWR_BTM_FIRST: frozenset[str] = frozenset([
    "behind-the-meter", "behind the meter", "btm-first", "btm first",
    "on-site-generat", "on-site generat", "onsite generat", "generation-led",
    "generation led", "self-generat", "btm-led", "btm led",
    "behind-meter", "behind meter",
])
_PWR_HYBRID: frozenset[str] = frozenset([
    "hybrid-grid", "hybrid grid", "parallel-grid", "parallel grid",
    "dual-pathway", "dual pathway", "grid-and-btm", "grid and btm",
    "both grid", "mixed pathway", "combination grid",
])

# ---------------------------------------------------------------------------
# Market timing synonym triggers
# ---------------------------------------------------------------------------
_TMG_ACCELERATE: frozenset[str] = frozenset([
    "accelerat", "aggressive", "immediate-deploy", "immediate deploy",
    "move-quickly", "move quickly", "fast-mover", "fast mover",
    "first-mover", "first mover", "rapid deploy", "ahead of", "ahead-of",
    "commit now", "commit immediately",
])
_TMG_MILESTONE_GATED: frozenset[str] = frozenset([
    "milestone-gated", "milestone gated", "conditional-commit", "conditional commit",
    "gated-deploy", "gated deploy", "proceed-with-condition", "proceed with condition",
    "milestone-based", "milestone based", "gated-capital", "gated capital",
    "capital-on-milestone",
])
_TMG_WAIT_AND_MONITOR: frozenset[str] = frozenset([
    "wait-and-monitor", "wait and monitor", "preserve-optionality",
    "preserve optionality", "defer", "conservative-commit", "conservative commit",
    "watch-and-wait", "watch and wait", "delay-commit", "delay commit",
    "monitor before", "hold capital",
])

# ---------------------------------------------------------------------------
# Canonical posture categories: category -> {canonical_value -> trigger_set}
# ---------------------------------------------------------------------------
POSTURE_CATEGORIES: dict[str, dict[str, frozenset[str]]] = {
    "geographic": {
        "concentrated": _GEO_CONCENTRATED,
        "diversified":  _GEO_DIVERSIFIED,
        "staged":       _GEO_STAGED,
    },
    "power": {
        "grid_first":  _PWR_GRID_FIRST,
        "btm_first":   _PWR_BTM_FIRST,
        "hybrid":      _PWR_HYBRID,
    },
    "timing": {
        "accelerate":       _TMG_ACCELERATE,
        "milestone_gated":  _TMG_MILESTONE_GATED,
        "wait_and_monitor": _TMG_WAIT_AND_MONITOR,
    },
}

# ---------------------------------------------------------------------------
# Choice-value -> canonical posture (for direct theory choice extraction)
# These complement substring matching for exact engagement choice IDs.
# ---------------------------------------------------------------------------
_CHOICE_CANONICAL: dict[str, tuple[str, str]] = {
    # geographic choices
    "concentrated":       ("geographic", "concentrated"),
    "diversified":        ("geographic", "diversified"),
    "staged":             ("geographic", "staged"),
    "staged_portfolio":   ("geographic", "staged"),
    "staged-portfolio":   ("geographic", "staged"),
    # power choices
    "grid_first":   ("power", "grid_first"),
    "grid-first":   ("power", "grid_first"),
    "btm_first":    ("power", "btm_first"),
    "btm-first":    ("power", "btm_first"),
    "hybrid":       ("power", "hybrid"),
    # timing choices
    "accelerate":         ("timing", "accelerate"),
    "milestone_gated":    ("timing", "milestone_gated"),
    "milestone-gated":    ("timing", "milestone_gated"),
    "wait_and_monitor":   ("timing", "wait_and_monitor"),
    "wait-and-monitor":   ("timing", "wait_and_monitor"),
}

# ---------------------------------------------------------------------------
# Contradiction table: (theory_cat, theory_val, option_cat, option_val) -> penalty
# ---------------------------------------------------------------------------
CONTRADICTIONS: dict[tuple[str, str, str, str], float] = {
    ("geographic", "diversified",  "geographic", "concentrated"):     0.35,
    ("geographic", "concentrated", "geographic", "diversified"):      0.35,
    ("timing",     "accelerate",   "timing",     "wait_and_monitor"): 0.30,
    ("timing",     "wait_and_monitor", "timing", "accelerate"):       0.30,
    ("timing",     "milestone_gated", "timing",  "accelerate"):       0.20,
    ("timing",     "accelerate",   "timing",     "milestone_gated"):  0.20,
    ("power",      "btm_first",    "power",      "grid_first"):       0.25,
    ("power",      "grid_first",   "power",      "btm_first"):        0.25,
}


class PostureNormalizer:
    """Extracts canonical posture values from theory choices and option text."""

    def theory_postures(self, theory_choices: list[dict[str, Any]]) -> dict[str, str]:
        """Extract canonical posture dict from a theory's strategic_choices.

        Tries direct choice-ID lookup first, then substring matching.
        Returns dict like {"geographic": "diversified", "power": "btm_first"}.
        """
        result: dict[str, str] = {}
        for choice in theory_choices:
            if not isinstance(choice, dict):
                continue
            val = str(choice.get("selected_value", "")).strip().lower().replace(" ", "_")
            title = str((choice.get("metadata") or {}).get("choice_title", "")).strip().lower()

            # Direct lookup on normalised choice value
            if val in _CHOICE_CANONICAL:
                cat, canonical = _CHOICE_CANONICAL[val]
                if cat not in result:
                    result[cat] = canonical
                continue

            # Substring matching on value and title candidates
            for candidate in [val.replace("_", "-"), val, title.replace("_", "-"), title]:
                if not candidate:
                    continue
                matched = False
                for cat, postures in POSTURE_CATEGORIES.items():
                    if cat in result:
                        continue
                    for posture_name, triggers in postures.items():
                        if any(trigger in candidate for trigger in triggers):
                            result[cat] = posture_name
                            matched = True
                            break
                    if matched:
                        break

        return result

    def option_postures(self, opt: dict[str, Any]) -> dict[str, str]:
        """Extract canonical posture dict from an option's text fields.

        Priority order: title -> description -> strategic_objective -> other fields.
        Returns dict like {"geographic": "concentrated", "timing": "accelerate"}.
        """
        result: dict[str, str] = {}

        # Tier 1: title (most authoritative)
        title = str(opt.get("title", "")).lower().replace("_", "-")
        self._match_postures_from_text(title, result)

        # Tier 2: description
        desc = str(opt.get("description", "")).lower().replace("_", "-")
        self._match_postures_from_text(desc, result)

        # Tier 3: strategic_objective
        obj = str(opt.get("strategic_objective", "")).lower().replace("_", "-")
        self._match_postures_from_text(obj, result)

        # Tier 4: advantages, disadvantages, expected_outcomes
        for field in ("advantages", "disadvantages", "expected_outcomes",
                      "implementation_complexity", "time_horizon"):
            val = opt.get(field)
            if val:
                text = str(val).lower().replace("_", "-")
                self._match_postures_from_text(text, result)

        return result

    @staticmethod
    def _match_postures_from_text(text: str, result: dict[str, str]) -> None:
        """Update result dict with posture values found in text (first-match per category)."""
        for cat, postures in POSTURE_CATEGORIES.items():
            if cat in result:
                continue
            for posture_name, triggers in postures.items():
                if any(trigger in text for trigger in triggers):
                    result[cat] = posture_name
                    break

    def normalize_text(self, text: str) -> dict[str, str]:
        """Return canonical posture values detected in arbitrary text."""
        return self.option_postures({"title": text, "description": text})
