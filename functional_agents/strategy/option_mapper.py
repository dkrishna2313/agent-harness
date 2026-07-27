"""PH12.1a — OptionMapper (rewritten).

Maps a theory's recommended choices to the closest upstream strategic option
using posture-category weighted scoring. No LLM calls.

Algorithm:
  1. Extract posture categories from theory choices (geographic, power, timing).
  2. For each upstream option, count how many of the theory's posture tokens
     appear in the combined option text (option_id + title + description +
     advantages + disadvantages + other rich fields).
  3. The option with the highest posture-token overlap wins.
  4. Score = matched / total posture tokens; confidence thresholds apply.

This prevents generic word overlap from cross-mapping theories to wrong options.
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import OptionMapping
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)

# Posture-category token sets. Each frozenset contains sub-string prefixes
# (or full tokens) that characterise one posture value. Matching uses `in`
# so "concentrat" matches "concentrated", "concentration", etc.
_GEOGRAPHIC_TOKENS: dict[str, frozenset[str]] = {
    "concentrated": frozenset(["concentrat", "single-state", "single_state", "focused", "undiversif"]),
    "diversified":  frozenset(["diversif", "multi-state", "multiple", "portfolio", "spread"]),
    "staged":       frozenset(["staged", "phased", "optionality", "contingenc", "defer"]),
}
_POWER_TOKENS: dict[str, frozenset[str]] = {
    "grid_first":   frozenset(["grid", "interconnect", "utility", "transmission"]),
    "btm_first":    frozenset(["btm", "behind-the-meter", "generation", "onsite", "self-generat"]),
    "hybrid":       frozenset(["hybrid", "mixed", "combination"]),
}
_TIMING_TOKENS: dict[str, frozenset[str]] = {
    "accelerate":       frozenset(["accelerat", "fast", "rapid", "first-mover", "immediate"]),
    "milestone_gated":  frozenset(["milestone", "gated", "conditional", "milestone-gate"]),
    "wait_and_monitor": frozenset(["wait", "monitor", "defer", "optionality", "delay"]),
}

# Merged lookup: posture key → frozenset of discriminating tokens
_ALL_POSTURE_TOKENS: dict[str, frozenset[str]] = {
    **_GEOGRAPHIC_TOKENS,
    **_POWER_TOKENS,
    **_TIMING_TOKENS,
}

# Confidence thresholds (fraction of posture tokens matched)
_HIGH_THRESHOLD   = 0.50
_MEDIUM_THRESHOLD = 0.20


class OptionMapper:
    """Matches each theory to the most relevant upstream strategic option.

    Uses posture-category token matching to prevent generic word overlap
    from cross-mapping theories (e.g. a diversified theory should not map
    to a concentrated option merely because both mention "state").
    """

    def map(self, theory: TheoryOfWinning, research: Any) -> OptionMapping:
        """Return an OptionMapping for the given theory and research context."""
        options: list[dict[str, Any]] = list(
            getattr(research, "strategic_options", None) or []
        )

        if not options:
            return OptionMapping(
                mapped_option_id=None,
                mapping_score=0.0,
                mapping_rationale="No upstream strategic options available for mapping.",
                mapping_confidence="None",
            )

        posture_tokens = self._extract_posture_tokens(theory)

        if not posture_tokens:
            return OptionMapping(
                mapped_option_id=None,
                mapping_score=0.0,
                mapping_rationale="No posture keywords available for mapping.",
                mapping_confidence="None",
            )

        best_id: str | None = None
        best_score = 0.0
        best_matched: list[str] = []

        for opt in options:
            if not isinstance(opt, dict):
                continue
            matched = self._matched_tokens(opt, posture_tokens)
            score = len(matched) / len(posture_tokens)
            if score > best_score or (score == best_score and best_id is None):
                best_score = score
                best_matched = list(matched)
                best_id = opt.get("option_id") or opt.get("id") or None

        confidence = self._confidence(best_score)
        if confidence == "None":
            best_id = None

        rationale = (
            f"Posture tokens {best_matched} matched option {best_id!r} "
            f"(score={best_score:.2f}, {len(best_matched)}/{len(posture_tokens)} tokens)."
            if best_id
            else "No option matched the theory's posture tokens at sufficient confidence."
        )

        LOGGER.debug(
            "[OptionMapper] theory=%s mapped_option=%s score=%.2f confidence=%s",
            theory.theory_id, best_id, best_score, confidence,
        )

        return OptionMapping(
            mapped_option_id=best_id,
            mapping_score=round(best_score, 4),
            mapping_rationale=rationale,
            mapping_confidence=confidence,
        )

    # ------------------------------------------------------------------

    @classmethod
    def _extract_posture_tokens(cls, theory: TheoryOfWinning) -> list[str]:
        """Return the union of posture tokens for all choices in the theory."""
        tokens: list[str] = []
        for c in theory.strategic_choices:
            if not isinstance(c, dict):
                continue
            candidates = [
                str(c.get("selected_value", "")).lower().replace("_", "-"),
                str((c.get("metadata") or {}).get("choice_title", "")).lower(),
            ]
            for candidate in candidates:
                for cat_key, cat_tokens in _ALL_POSTURE_TOKENS.items():
                    # Check if any posture trigger appears in the candidate
                    if any(trigger in candidate.replace("_", "-") for trigger in cat_tokens):
                        tokens.extend(cat_tokens)
                        break
                    # Also match on the category key itself
                    if cat_key.replace("_", "-") in candidate.replace("_", "-"):
                        tokens.extend(cat_tokens)
                        break
        return list(dict.fromkeys(tokens))  # deduplicated, ordered

    @staticmethod
    def _matched_tokens(opt: dict[str, Any], posture_tokens: list[str]) -> list[str]:
        """Return the posture tokens that appear in the option's text."""
        text = " ".join(str(opt.get(k, "")) for k in (
            "option_id", "title", "description",
            "advantages", "disadvantages",
            "strategic_objective", "implementation_complexity",
            "time_horizon", "capital_intensity",
        )).lower()
        return [tok for tok in posture_tokens if tok in text]

    @staticmethod
    def _confidence(score: float) -> str:
        if score > _HIGH_THRESHOLD:
            return "High"
        if score > _MEDIUM_THRESHOLD:
            return "Medium"
        if score > 0.0:
            return "Low"
        return "None"
