"""PH12.1 — OptionMapper.

Maps a theory's recommended choices to the closest upstream strategic option
from research using keyword overlap scoring. No LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import OptionMapping
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)


class OptionMapper:
    """Matches a theory to the most relevant upstream strategic option.

    Scoring:
      For each upstream option, count how many of the theory's choice keywords
      appear in the option's text fields (option_id, title, description).
      The option with the most matches wins.

    Confidence thresholds:
      score >= 0.70 → High
      score >= 0.40 → Medium
      score >  0.00 → Low
      score == 0.00 → None
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

        keywords = self._extract_keywords(theory)

        if not keywords:
            return OptionMapping(
                mapped_option_id=None,
                mapping_score=0.0,
                mapping_rationale="No choice keywords available for matching.",
                mapping_confidence="None",
            )

        best_id: str | None = None
        best_score = 0.0

        for opt in options:
            if not isinstance(opt, dict):
                continue
            score = self._score_option(opt, keywords)
            if score > best_score:
                best_score = score
                best_id = opt.get("option_id") or opt.get("id") or None

        normalized = min(best_score / max(len(keywords), 1), 1.0)
        confidence = self._confidence(normalized)

        rationale = (
            f"Matched option {best_id!r} with score {normalized:.2f} "
            f"({int(best_score)} keyword(s) matched out of {len(keywords)})."
            if best_id
            else "No option matched the theory's choice keywords."
        )

        LOGGER.debug(
            "[OptionMapper] theory=%s mapped_option=%s score=%.2f confidence=%s",
            theory.theory_id, best_id, normalized, confidence,
        )

        return OptionMapping(
            mapped_option_id=best_id,
            mapping_score=round(normalized, 4),
            mapping_rationale=rationale,
            mapping_confidence=confidence,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(theory: TheoryOfWinning) -> list[str]:
        keywords: list[str] = []
        for c in theory.strategic_choices:
            if not isinstance(c, dict):
                continue
            for field in ("selected_value",):
                v = str(c.get(field, "")).lower()
                keywords.extend(w for w in v.replace("_", " ").split() if len(w) > 3)
            meta: dict = c.get("metadata", {}) or {}
            for key in ("choice_title", "choice_description"):
                v = str(meta.get(key, "")).lower()
                keywords.extend(w for w in v.split() if len(w) > 3)
        return list(dict.fromkeys(keywords))

    @staticmethod
    def _score_option(opt: dict[str, Any], keywords: list[str]) -> float:
        text = " ".join([
            str(opt.get("option_id", "")),
            str(opt.get("title", "")),
            str(opt.get("description", "")),
        ]).lower()
        return float(sum(1 for kw in keywords if kw in text))

    @staticmethod
    def _confidence(score: float) -> str:
        if score >= 0.70:
            return "High"
        if score >= 0.40:
            return "Medium"
        if score > 0.0:
            return "Low"
        return "None"
