"""PH12.1 — AlignmentEvaluator.

Determines the relationship between the upstream preferred option (from research)
and the theory selected by the Strategy Layer.

Status values:
  confirmed  — selected theory maps to the same option, strong margin
  refined    — selected theory maps to same option but margin is narrow or
               confidence is low
  challenged — selected theory maps to a different option with significant margin
  unresolved — no preferred option, or tie, or mapping confidence too low
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import AlignmentResult, OptionMapping
from .strategic_position import TheoryOfWinning
from .strategy_selector import StrategySelection

LOGGER = logging.getLogger(__name__)

_MINIMUM_CHALLENGE_MARGIN = 0.05


class AlignmentEvaluator:
    """Evaluates alignment between upstream recommendation and selected theory."""

    def evaluate(
        self,
        selected_theory: TheoryOfWinning,
        option_mapping: OptionMapping,
        selection: StrategySelection,
        research: Any,
        minimum_challenge_margin: float = _MINIMUM_CHALLENGE_MARGIN,
    ) -> AlignmentResult:
        """Return an AlignmentResult describing the recommendation relationship.

        Parameters
        ----------
        selected_theory:
            The theory chosen by StrategySelector.
        option_mapping:
            Result of OptionMapper.map() for the selected theory.
        selection:
            The StrategySelection record from the selector.
        research:
            AgentContext; provides the upstream preferred option.
        minimum_challenge_margin:
            Score margin above which a disagreement is classified as "challenged"
            rather than "unresolved".
        """
        preferred = self._extract_preferred_option(research)
        preferred_id = preferred.get("option_id", "") or preferred.get("id", "") or ""

        selected_id = selected_theory.theory_id
        mapped_id = option_mapping.mapped_option_id
        margin = selection.score_margin or 0.0
        conf = option_mapping.mapping_confidence

        # No preferred option → unresolved
        if not preferred_id:
            return AlignmentResult(
                status="unresolved",
                preferred_option_id="",
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale="No upstream preferred option available for alignment comparison.",
            )

        # Low mapping confidence → unresolved
        if conf in ("None", "Low") or mapped_id is None:
            return AlignmentResult(
                status="unresolved",
                preferred_option_id=preferred_id,
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale=(
                    f"Mapping confidence is {conf!r}; "
                    "alignment cannot be reliably determined."
                ),
            )

        # Tie → unresolved
        if margin == 0.0 or (selection.tie_breaker_used is not None):
            return AlignmentResult(
                status="unresolved",
                preferred_option_id=preferred_id,
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale="Selection was decided by tie-breaker; alignment is not clear-cut.",
            )

        # Agreement vs disagreement
        if mapped_id == preferred_id:
            status = "confirmed" if margin >= minimum_challenge_margin else "refined"
            rationale = (
                f"Selected theory maps to upstream preferred option {preferred_id!r} "
                f"(margin={margin:.3f})."
            )
        else:
            if margin >= minimum_challenge_margin:
                status = "challenged"
                rationale = (
                    f"Selected theory maps to {mapped_id!r}, not the upstream preferred "
                    f"option {preferred_id!r} (margin={margin:.3f} ≥ threshold)."
                )
            else:
                status = "unresolved"
                rationale = (
                    f"Selected theory maps to {mapped_id!r} vs upstream preferred "
                    f"{preferred_id!r}, but margin {margin:.3f} < threshold."
                )

        LOGGER.debug(
            "[AlignmentEvaluator] status=%s preferred=%s mapped=%s margin=%.3f",
            status, preferred_id, mapped_id, margin,
        )

        return AlignmentResult(
            status=status,
            preferred_option_id=preferred_id,
            selected_theory_id=selected_id,
            mapped_option_id=mapped_id,
            score_margin=margin,
            rationale=rationale,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _extract_preferred_option(research: Any) -> dict[str, Any]:
        val = getattr(research, "preferred_option", None)
        return val if isinstance(val, dict) else {}
