"""PH12.1b — OptionMapper (rewritten): contradiction-aware posture-weighted mapper.

Maps each theory's strategic posture to the closest upstream strategic option.
Uses three-tier scoring:
  Tier 1 (dominant): posture match/contradiction between theory and option postures
  Tier 2 (secondary): posture match from lower-priority option fields
  Tier 3 (low weight): generic keyword overlap (cannot override posture signals)

Algorithm:
  1. Extract theory canonical postures from strategic_choices.
  2. Extract option canonical postures from title/description/objective.
  3. For each theory-option pair:
     a. Score posture matches: +_MATCH_TITLE if match detected in title/objective,
        +_MATCH_DESC if detected in description/other fields.
     b. Score contradictions: subtract penalty from CONTRADICTIONS table.
     c. Score generic overlap: bounded at _GENERIC_CAP.
  4. Option with highest total score wins.
  5. Compute confidence from winner score and score separation.
  6. Return OptionMapping with full per-option diagnostics.

No LLM calls. No engagement-specific rules. Fully deterministic.
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import OptionMapping
from .posture_normalizer import CONTRADICTIONS, POSTURE_CATEGORIES, PostureNormalizer
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)

# Scoring weights
_MATCH_TITLE = 0.35   # posture match detected in title/strategic_objective
_MATCH_DESC  = 0.25   # posture match detected in description/advantages/other
_GENERIC_CAP = 0.10   # maximum generic keyword overlap contribution

# Confidence thresholds
_HIGH_THRESHOLD  = 0.40   # >= this -> High (also needs separation)
_MEDIUM_THRESHOLD = 0.20  # >= this -> Medium
_SEPARATION_HIGH = 0.20   # score separation needed for High
_SEPARATION_MED  = 0.05   # score separation needed for Medium

# Generic posture tokens — strictly bounded, never override posture signals
_GENERIC_TOKENS: frozenset[str] = frozenset([
    "portfolio", "diversif", "concentrat", "grid", "btm", "behind-the-meter",
    "milestone", "gated", "accelerat", "staged", "phased", "hybrid",
    "multi-state", "multi-rto", "single-state", "generation", "interconnect",
    "optionality", "wait", "monitor", "aggressive",
])


class OptionMapper:
    """Maps each theory to the most semantically consistent upstream strategic option.

    Uses posture-category weighted scoring with explicit contradiction penalties.
    Generic keyword overlap is bounded and cannot overcome posture signals.
    """

    def __init__(self) -> None:
        self._normalizer = PostureNormalizer()

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
                option_scores=[],
                theory_postures={},
            )

        # Extract canonical postures from theory choices
        theory_postures = self._normalizer.theory_postures(
            [c for c in (theory.strategic_choices or []) if isinstance(c, dict)]
        )

        if not theory_postures:
            # Fallback: scan theory prose for posture signals
            theory_text = " ".join(filter(None, [
                theory.winning_position or "",
                theory.winning_mechanism or "",
                theory.recommended_option_title or "",
            ]))
            theory_postures = self._normalizer.normalize_text(theory_text)

        if not theory_postures:
            return OptionMapping(
                mapped_option_id=None,
                mapping_score=0.0,
                mapping_rationale="No posture keywords available for mapping.",
                mapping_confidence="None",
                option_scores=[],
                theory_postures={},
            )

        # Score every option
        option_scores: list[dict[str, Any]] = []
        for opt in options:
            if not isinstance(opt, dict):
                continue
            option_scores.append(self._score_option(theory_postures, opt))

        if not option_scores:
            return OptionMapping(
                mapped_option_id=None,
                mapping_score=0.0,
                mapping_rationale="No scorable options found.",
                mapping_confidence="None",
                option_scores=[],
                theory_postures=theory_postures,
            )

        # Sort descending by score; stable sort preserves original order on ties
        option_scores.sort(key=lambda e: -e["score"])

        winner = option_scores[0]
        runner_up_score = option_scores[1]["score"] if len(option_scores) > 1 else winner["score"] - 1.0
        separation = winner["score"] - runner_up_score

        confidence = self._confidence(winner["score"], separation, bool(winner["contradictions"]))
        mapped_id = winner["option_id"] if confidence != "None" else None

        rationale = self._build_rationale(winner, confidence, mapped_id, theory_postures)

        LOGGER.debug(
            "[OptionMapper] theory=%s mapped=%s score=%.3f conf=%s contradictions=%d",
            theory.theory_id, mapped_id, winner["score"], confidence,
            len(winner["contradictions"]),
        )

        return OptionMapping(
            mapped_option_id=mapped_id,
            mapping_score=round(max(-1.0, winner["score"]), 4),
            mapping_confidence=confidence,
            mapping_rationale=rationale,
            option_scores=option_scores,
            theory_postures=theory_postures,
        )

    # ------------------------------------------------------------------
    # Per-option scoring
    # ------------------------------------------------------------------

    def _score_option(
        self,
        theory_postures: dict[str, str],
        opt: dict[str, Any],
    ) -> dict[str, Any]:
        """Score one option against the theory postures.

        Returns diagnostic dict: score, posture_matches, contradictions,
        generic_matches, penalties, option_postures, rationale.
        """
        opt_id = opt.get("option_id") or opt.get("id") or ""
        opt_postures = self._normalizer.option_postures(opt)

        # Title + strategic_objective text — higher weight tier
        title_text = " ".join([
            str(opt.get("title", "")).lower().replace("_", "-"),
            str(opt.get("strategic_objective", "")).lower().replace("_", "-"),
        ])

        posture_score = 0.0
        posture_matches: list[dict[str, Any]] = []
        contradictions: list[dict[str, Any]] = []

        for t_cat, t_val in theory_postures.items():
            o_val = opt_postures.get(t_cat)
            if o_val is None:
                continue  # option doesn't express this posture dimension — neutral

            if o_val == t_val:
                triggers = POSTURE_CATEGORIES.get(t_cat, {}).get(t_val, frozenset())
                in_title = any(tok in title_text for tok in triggers)
                weight = _MATCH_TITLE if in_title else _MATCH_DESC
                posture_score += weight
                posture_matches.append({
                    "category": t_cat,
                    "value": t_val,
                    "weight": round(weight, 3),
                    "source": "title" if in_title else "description",
                })
            else:
                penalty = CONTRADICTIONS.get((t_cat, t_val, t_cat, o_val), 0.0)
                if penalty > 0.0:
                    posture_score -= penalty
                    contradictions.append({
                        "category": t_cat,
                        "theory_value": t_val,
                        "option_value": o_val,
                        "penalty": penalty,
                        "rationale": (
                            f"{t_cat.title()} posture conflict: "
                            f"theory={t_val!r} vs option={o_val!r}."
                        ),
                    })

        # Generic overlap — very low weight, strictly bounded
        all_opt_text = " ".join(
            str(opt.get(k, "")) for k in (
                "option_id", "title", "description", "strategic_objective",
                "advantages", "disadvantages", "expected_outcomes",
                "implementation_complexity", "time_horizon", "capital_intensity",
            )
        ).lower().replace("_", "-")
        generic_matches = [tok for tok in _GENERIC_TOKENS if tok in all_opt_text]
        generic_score = min(_GENERIC_CAP, len(generic_matches) * 0.01)

        total = max(-1.0, min(1.5, posture_score + generic_score))

        return {
            "option_id": opt_id,
            "option_postures": opt_postures,
            "score": round(total, 4),
            "posture_matches": posture_matches,
            "contradictions": contradictions,
            "generic_matches": generic_matches[:10],
            "penalties": [{"category": c["category"], "penalty": c["penalty"]} for c in contradictions],
            "rationale": self._entry_rationale(posture_matches, contradictions, generic_score),
        }

    # ------------------------------------------------------------------
    # Confidence and rationale helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence(score: float, separation: float, has_contradictions: bool) -> str:
        if score <= 0.0:
            return "None"
        if score >= _HIGH_THRESHOLD and separation >= _SEPARATION_HIGH and not has_contradictions:
            return "High"
        if score >= _MEDIUM_THRESHOLD and separation >= _SEPARATION_MED:
            return "Medium"
        if score > 0.0:
            return "Low"
        return "None"

    @staticmethod
    def _build_rationale(
        winner: dict[str, Any],
        confidence: str,
        mapped_id: str | None,
        theory_postures: dict[str, str],
    ) -> str:
        if not mapped_id:
            contradictions = winner.get("contradictions", [])
            if contradictions:
                c = contradictions[0]
                return (
                    f"No authoritative mapping: strongest option has direct contradiction "
                    f"({c['category']}: theory={c['theory_value']!r}, "
                    f"option={c['option_value']!r}, penalty={c['penalty']})."
                )
            return (
                f"No authoritative mapping: winner score {winner['score']:.2f} "
                "below minimum confidence threshold."
            )

        matches = winner.get("posture_matches", [])
        contradictions = winner.get("contradictions", [])
        match_summary = (
            ", ".join(f"{m['category']}:{m['value']}" for m in matches)
            or "generic overlap"
        )
        theory_summary = ", ".join(f"{k}:{v}" for k, v in theory_postures.items())
        parts = [
            f"Theory postures [{theory_summary}] mapped to {mapped_id!r}",
            f"(score={winner['score']:.2f}, confidence={confidence},",
            f"posture_matches=[{match_summary}]",
        ]
        if contradictions:
            parts.append(f"contradictions={len(contradictions)}")
        parts[-1] += ")."
        return " ".join(parts)

    @staticmethod
    def _entry_rationale(
        posture_matches: list[dict[str, Any]],
        contradictions: list[dict[str, Any]],
        generic_score: float,
    ) -> str:
        parts = []
        if posture_matches:
            parts.append(f"{len(posture_matches)} posture match(es)")
        if contradictions:
            parts.append(f"{len(contradictions)} contradiction(s)")
        if generic_score > 0:
            parts.append(f"generic overlap={generic_score:.2f}")
        return "; ".join(parts) or "no posture signal"
