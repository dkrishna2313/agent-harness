"""PH12.1 — SaturationDetector.

Detects when all theory evaluations have collapsed to the same overall score,
which indicates that scoring logic is presence-only rather than theory-specific.
"""

from __future__ import annotations

import logging

from .theory_evaluation import TheoryEvaluation

LOGGER = logging.getLogger(__name__)


class SaturationDetector:
    """Detects evaluation saturation (all theories score identically).

    Saturation is a signal that the scoring model is not differentiating
    between theories — typically caused by presence-only scoring where every
    theory that has the minimum required fields scores the same.
    """

    def check(
        self, evaluations: list[TheoryEvaluation]
    ) -> tuple[bool, str]:
        """Return (is_saturated, message).

        is_saturated is True when all overall_scores are equal.
        message explains what was detected.
        """
        if len(evaluations) <= 1:
            return False, "Single theory; saturation not applicable."

        scores = [round(ev.overall_score, 6) for ev in evaluations]
        unique_scores = set(scores)

        if len(unique_scores) == 1:
            score_val = scores[0]
            if score_val == 1.0:
                msg = (
                    f"Score saturation detected: all {len(evaluations)} theories "
                    f"score 1.0 — evaluation criteria are presence-only."
                )
            else:
                msg = (
                    f"Score saturation detected: all {len(evaluations)} theories "
                    f"share overall_score={score_val:.4f}."
                )
            LOGGER.warning("[SaturationDetector] %s", msg)
            return True, msg

        spread = max(scores) - min(scores)
        msg = (
            f"No saturation: {len(unique_scores)} distinct score(s) "
            f"across {len(evaluations)} theories (spread={spread:.4f})."
        )
        LOGGER.debug("[SaturationDetector] %s", msg)
        return False, msg
