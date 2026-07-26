"""StrategySelector — selects the winning TheoryOfWinning (PH10.6).

Public interface:
    select(theories, evaluations, plan) -> TheoryOfWinning

Selection is fully deterministic — no LLM calls, no randomness.

--- Matching (PH10.6a) ---

Each TheoryEvaluation is matched to its TheoryOfWinning by strict ID:
    evaluation.theory_id == theory.theory_id

Duplicate theory_ids in the theories list raise ValueError.
An evaluation whose theory_id cannot be resolved raises ValueError.
No positional fallback is used.

--- Selection algorithm ---

Candidates (theory, evaluation) pairs are ranked by, in order:

  1. overall_score (descending) — primary criterion
  2. confidence rank: High > Medium > Low > unknown (descending)
  3. residual_risks count (ascending)
  4. original theory order (ascending) — final stable tie-breaker

--- StrategySelection ---

After each call, the selector stores self._last_selection: StrategySelection.
StrategySelection records the winning and runner-up theory IDs, scores,
score margin, and which tie-breaker (if any) was decisive.

Produced by: StrategyCoordinator.build()
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from .strategic_position import TheoryOfWinning
from .strategy_plan import StrategyPlan
from .theory_evaluation import TheoryEvaluation

LOGGER = logging.getLogger(__name__)

# Numeric rank for confidence strings (higher = better)
_CONFIDENCE_RANK: dict[str, int] = {"high": 2, "medium": 1, "low": 0}


def _conf_rank(confidence: str) -> int:
    return _CONFIDENCE_RANK.get(confidence.strip().lower(), -1)


class StrategySelection(BaseModel):
    """Immutable record of a completed selection decision.

    Accessible via ``StrategySelector._last_selection`` after ``select()``
    returns. Stored on ``StrategyCoordinator._selection`` for diagnostics.
    """

    model_config = {"frozen": True, "extra": "allow"}

    winner_theory_id: str
    winner_score: float
    runner_up_theory_id: str | None = None
    runner_up_score: float | None = None
    score_margin: float | None = None
    # "confidence", "residual_risks", "order", or None (scores differed)
    tie_breaker_used: str | None = None


class StrategySelector:
    """Selects the highest-scoring TheoryOfWinning deterministically.

    Selection algorithm (all deterministic):
      1. Primary:   highest TheoryEvaluation.overall_score
      2. Secondary: higher evaluation confidence (High > Medium > Low)
      3. Tertiary:  fewer residual_risks
      4. Final:     earlier original theory order (stable)

    After each call:
      self._last_selection: StrategySelection  — records the decision.
    """

    def __init__(self) -> None:
        self._last_selection: StrategySelection | None = None

    def select(
        self,
        theories: list[TheoryOfWinning],
        evaluations: list[TheoryEvaluation],
        plan: StrategyPlan,
    ) -> TheoryOfWinning:
        """Select the winning TheoryOfWinning.

        Parameters
        ----------
        theories:
            List of TheoryOfWinning from TheoryGenerator. Must be non-empty
            and the same length as evaluations.
        evaluations:
            List of TheoryEvaluation from TheoryEvaluator. Must be non-empty
            and the same length as theories.
        plan:
            The resolved StrategyPlan (reserved for future policy use).

        Returns
        -------
        TheoryOfWinning
            The theory with the best evaluation under deterministic
            tie-breaking rules.

        Raises
        ------
        ValueError
            If theories is empty, evaluations is empty, counts differ, or
            an evaluation's theory_id cannot be matched to any theory when
            IDs are unique.
        """
        self._validate_counts(theories, evaluations)

        # Pair theories with evaluations
        pairs = self._pair(theories, evaluations)

        # Sort: score desc → confidence rank desc → residual_risks asc → index asc
        def _sort_key(entry: tuple[TheoryOfWinning, TheoryEvaluation, int]) -> tuple:
            _, ev, idx = entry
            return (
                -ev.overall_score,
                -_conf_rank(ev.confidence),
                len(ev.residual_risks),
                idx,
            )

        pairs.sort(key=_sort_key)

        winner_theory, winner_ev, _ = pairs[0]

        # Determine tie-breaker used (vs runner-up only)
        tie_breaker_used: str | None = None
        runner_up_theory_id: str | None = None
        runner_up_score: float | None = None
        score_margin: float | None = None

        if len(pairs) > 1:
            _, runner_ev, _ = pairs[1]
            runner_up_theory_id = runner_ev.theory_id
            runner_up_score = runner_ev.overall_score
            score_margin = round(winner_ev.overall_score - runner_ev.overall_score, 6)

            if winner_ev.overall_score == runner_ev.overall_score:
                if _conf_rank(winner_ev.confidence) != _conf_rank(runner_ev.confidence):
                    tie_breaker_used = "confidence"
                elif len(winner_ev.residual_risks) != len(runner_ev.residual_risks):
                    tie_breaker_used = "residual_risks"
                else:
                    tie_breaker_used = "order"

        self._last_selection = StrategySelection(
            winner_theory_id=winner_ev.theory_id,
            winner_score=winner_ev.overall_score,
            runner_up_theory_id=runner_up_theory_id,
            runner_up_score=runner_up_score,
            score_margin=score_margin,
            tie_breaker_used=tie_breaker_used,
        )

        LOGGER.debug(
            "[StrategySelector] winner=%s score=%.3f margin=%s tie_breaker=%s",
            self._last_selection.winner_theory_id,
            self._last_selection.winner_score,
            self._last_selection.score_margin,
            self._last_selection.tie_breaker_used,
        )

        return winner_theory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_counts(
        theories: list[TheoryOfWinning],
        evaluations: list[TheoryEvaluation],
    ) -> None:
        if not theories:
            raise ValueError("StrategySelector: theories list is empty.")
        if not evaluations:
            raise ValueError("StrategySelector: evaluations list is empty.")
        if len(theories) != len(evaluations):
            raise ValueError(
                f"StrategySelector: theories ({len(theories)}) and "
                f"evaluations ({len(evaluations)}) counts differ."
            )

    @staticmethod
    def _pair(
        theories: list[TheoryOfWinning],
        evaluations: list[TheoryEvaluation],
    ) -> list[tuple[TheoryOfWinning, TheoryEvaluation, int]]:
        """Pair each evaluation with its theory by strict theory_id match.

        Raises ValueError for:
        - blank theory_ids in theories (defensive; normally prevented by model)
        - duplicate theory_ids in theories
        - blank theory_ids in evaluations
        - duplicate theory_ids in evaluations
        - any evaluation whose theory_id has no matching theory
        """
        # Theories: blank and duplicate checks
        seen_theory: set[str] = set()
        for t in theories:
            if not t.theory_id or not t.theory_id.strip():
                raise ValueError("StrategySelector: theory with blank theory_id found.")
            if t.theory_id in seen_theory:
                raise ValueError(
                    f"StrategySelector: duplicate theory_id={t.theory_id!r} in theories."
                )
            seen_theory.add(t.theory_id)

        # Evaluations: blank and duplicate checks
        seen_eval: set[str] = set()
        for ev in evaluations:
            if not ev.theory_id or not ev.theory_id.strip():
                raise ValueError(
                    "StrategySelector: evaluation with blank theory_id found."
                )
            if ev.theory_id in seen_eval:
                raise ValueError(
                    f"StrategySelector: duplicate theory_id={ev.theory_id!r} in evaluations."
                )
            seen_eval.add(ev.theory_id)

        # Strict ID-based matching
        theory_by_id: dict[str, TheoryOfWinning] = {t.theory_id: t for t in theories}
        pairs: list[tuple[TheoryOfWinning, TheoryEvaluation, int]] = []
        for i, ev in enumerate(evaluations):
            theory = theory_by_id.get(ev.theory_id)
            if theory is None:
                raise ValueError(
                    f"StrategySelector: evaluation.theory_id={ev.theory_id!r} "
                    f"has no matching theory. "
                    f"Available theory IDs: {sorted(theory_by_id)}"
                )
            pairs.append((theory, ev, i))
        return pairs
