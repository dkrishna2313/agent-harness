"""PH12.1a — AlignmentEvaluator (updated to consume AlignmentPolicy).
PH12.1b — Adds confirmed/refined distinction.

Status values:
  confirmed  — same option, High-confidence mapping, clear margin
  refined    — same option, Medium-confidence mapping (theory adds specificity)
  challenged — different option with significant margin and sufficient confidence
  unresolved — no preferred option, tie, low confidence, or below margin
"""

from __future__ import annotations

import logging
from typing import Any

from .alignment import AlignmentResult, OptionMapping
from .strategic_position import TheoryOfWinning
from .strategy_config import AlignmentPolicy
from .strategy_selector import StrategySelection

LOGGER = logging.getLogger(__name__)

_DEFAULT_POLICY = AlignmentPolicy()

# Numeric rank for mapping confidence levels
_CONF_RANK: dict[str, int] = {"High": 3, "Medium": 2, "Low": 1, "None": 0}


class AlignmentEvaluator:
    """Evaluates alignment between upstream recommendation and selected theory."""

    def evaluate(
        self,
        selected_theory: TheoryOfWinning,
        option_mapping: OptionMapping,
        selection: StrategySelection,
        research: Any,
        policy: AlignmentPolicy | None = None,
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
        policy:
            AlignmentPolicy from resolved config; falls back to default policy.
        """
        ap = policy if policy is not None else _DEFAULT_POLICY
        preferred = self._extract_preferred_option(research)
        preferred_id = preferred.get("option_id", "") or preferred.get("id", "") or ""

        selected_id = selected_theory.theory_id
        mapped_id = option_mapping.mapped_option_id
        margin = selection.score_margin or 0.0
        conf = option_mapping.mapping_confidence

        # No preferred option -> unresolved
        if not preferred_id:
            return AlignmentResult(
                status="unresolved",
                preferred_option_id="",
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale="No upstream preferred option available for alignment comparison.",
            )

        # Check mapping confidence against configured minimum
        min_conf_rank = _CONF_RANK.get(ap.minimum_mapping_confidence, 2)
        actual_conf_rank = _CONF_RANK.get(conf or "None", 0)

        if actual_conf_rank < min_conf_rank or mapped_id is None:
            return AlignmentResult(
                status="unresolved",
                preferred_option_id=preferred_id,
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale=(
                    f"Mapping confidence {conf!r} is below minimum "
                    f"{ap.minimum_mapping_confidence!r}; "
                    "alignment cannot be reliably determined."
                ),
            )

        # Tie -> unresolved (when configured)
        if ap.unresolved_on_tie and (margin == 0.0 or (selection.tie_breaker_used is not None)):
            return AlignmentResult(
                status="unresolved",
                preferred_option_id=preferred_id,
                selected_theory_id=selected_id,
                mapped_option_id=mapped_id,
                score_margin=margin,
                rationale="Selection was decided by tie-breaker; alignment is not clear-cut.",
            )

        # Agreement vs disagreement
        min_margin = ap.minimum_challenge_margin
        if mapped_id == preferred_id:
            # PH12.1b: confirmed vs refined distinction.
            # confirmed: High confidence (option covers all theory dimensions strongly)
            # refined:   Medium confidence (theory adds specificity beyond option coverage)
            if conf == "High" and margin >= min_margin:
                status = "confirmed"
            else:
                status = "refined"
            rationale = (
                f"Selected theory maps to upstream preferred option {preferred_id!r} "
                f"(margin={margin:.3f}, mapping_confidence={conf!r})."
            )
        else:
            if margin >= min_margin:
                status = "challenged"
                rationale = (
                    f"Selected theory maps to {mapped_id!r}, not the upstream preferred "
                    f"option {preferred_id!r} (margin={margin:.3f} >= threshold={min_margin})."
                )
            else:
                status = "unresolved"
                rationale = (
                    f"Selected theory maps to {mapped_id!r} vs upstream preferred "
                    f"{preferred_id!r}, but margin {margin:.3f} < threshold {min_margin}."
                )

        LOGGER.debug(
            "[AlignmentEvaluator] status=%s preferred=%s mapped=%s margin=%.3f conf=%s",
            status, preferred_id, mapped_id, margin, conf,
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
