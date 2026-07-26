"""TheoryEvaluator — evaluates a single TheoryOfWinning (PH10.5).

Public interface:
    build(theory, plan, research) -> TheoryEvaluation

One TheoryEvaluation is produced per TheoryOfWinning. Theories are
evaluated independently; no comparison or winner selection is performed.

Criteria are generic (no Executive or framework-specific names hard-coded).
Default criterion weights can be overridden via plan.evaluation_model.weights.
plan.validation_policy drives whether certain criteria are scored strictly (0.0)
when their evidence is absent.

No LLM calls, no search, no optimisation.

Produced by: StrategyCoordinator.build() (one per TheoryOfWinning)
Consumed by: StrategyCoordinator (stored as _evaluations; not yet wired
             into StrategicPosition)
"""

from __future__ import annotations

import logging
from typing import Any

from .strategic_position import TheoryOfWinning
from .strategy_plan import StrategyPlan
from .theory_evaluation import CriterionScore, TheoryEvaluation

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criterion registry
# ---------------------------------------------------------------------------
# Each entry: criterion_name -> (high_rationale, low_rationale, default_weight)
# high_rationale is used when score >= 0.75 (strength candidate)
# low_rationale  is used when score <  0.50 (weakness candidate)
# The criterion names are deliberately generic — callers may configure weights
# for them via plan.evaluation_model.weights without needing to know their
# internal semantics.

_CRITERION_META: dict[str, tuple[str, str, float]] = {
    "option_identified": (
        "A recommended option is identified.",
        "No recommended option is identified.",
        2.0,
    ),
    "position_articulated": (
        "The winning position is clearly articulated.",
        "The winning position is not articulated.",
        1.5,
    ),
    "mechanism_defined": (
        "The winning mechanism is defined.",
        "No winning mechanism is defined.",
        1.5,
    ),
    "choice_completeness": (
        "All active plan dimensions are covered by choices.",
        "Active plan dimensions are incompletely covered.",
        2.0,
    ),
    "evidence_quality": (
        "Supporting evidence is cited.",
        "No supporting evidence is cited.",
        1.0,
    ),
    "assumption_coverage": (
        "Key assumptions are documented.",
        "No assumptions are documented.",
        1.0,
    ),
    "risk_awareness": (
        "Failure modes are identified.",
        "No failure modes have been identified.",
        1.0,
    ),
}

# Score at or above this threshold → strength candidate
_STRENGTH_THRESHOLD = 0.75
# Score below this threshold → weakness candidate
_WEAKNESS_THRESHOLD = 0.50


class TheoryEvaluator:
    """Evaluates a single TheoryOfWinning against generic plan-driven criteria.

    Evaluation is independent per theory. No comparison, no ranking,
    no winner selection.
    """

    def build(
        self,
        theory: TheoryOfWinning,
        plan: StrategyPlan,
        research: Any,
    ) -> TheoryEvaluation:
        """Produce one TheoryEvaluation for a single TheoryOfWinning.

        Parameters
        ----------
        theory:
            The TheoryOfWinning to evaluate.
        plan:
            The resolved StrategyPlan; provides evaluation weights and
            validation policy.
        research:
            Available research output (AgentContext in PH10.x). Reserved
            for future use; currently accessed for structural completeness.

        Returns
        -------
        TheoryEvaluation
            Scores for each criterion, qualitative observations, an
            overall_score, and confidence in the evaluation.
        """
        criteria_scores = self._score_criteria(theory, plan)
        overall_score = self._compute_overall_score(criteria_scores)
        strengths = self._extract_strengths(criteria_scores, theory)
        weaknesses = self._extract_weaknesses(criteria_scores, theory)
        residual_risks = list(theory.failure_modes)
        confidence = self._derive_confidence(theory, overall_score)
        theory_id = theory.recommended_option_id or ""

        evaluation = TheoryEvaluation(
            theory_id=theory_id,
            criteria_scores=criteria_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            residual_risks=residual_risks,
            overall_score=overall_score,
            confidence=confidence,
            metadata={
                "plan_id": plan.plan_id,
                "n_active_dimensions": len(plan.active_dimensions),
                "n_choices": len(theory.strategic_choices),
                "n_evidence": len(theory.evidence),
                "n_assumptions": len(theory.assumptions),
                "n_failure_modes": len(theory.failure_modes),
            },
        )
        LOGGER.debug(
            "[TheoryEvaluator] theory_id=%s overall_score=%.3f confidence=%s",
            theory_id, overall_score, confidence,
        )
        return evaluation

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_criteria(
        self,
        theory: TheoryOfWinning,
        plan: StrategyPlan,
    ) -> dict[str, CriterionScore]:
        """Compute a CriterionScore for each entry in _CRITERION_META."""
        plan_weights = plan.evaluation_model.weights
        vp = plan.validation_policy
        n_dims = len(plan.active_dimensions)

        scores: dict[str, CriterionScore] = {}

        # ---- option_identified ----
        score_oi = 1.0 if theory.recommended_option_id else 0.0
        scores["option_identified"] = self._make_score(
            "option_identified", score_oi, plan_weights,
            theory.recommended_option_id or None,
        )

        # ---- position_articulated ----
        score_pa = 1.0 if theory.winning_position else 0.0
        scores["position_articulated"] = self._make_score(
            "position_articulated", score_pa, plan_weights,
            theory.winning_position or None,
        )

        # ---- mechanism_defined ----
        score_md = 1.0 if theory.winning_mechanism else 0.0
        scores["mechanism_defined"] = self._make_score(
            "mechanism_defined", score_md, plan_weights,
            theory.winning_mechanism or None,
        )

        # ---- choice_completeness ----
        n_choices = len(theory.strategic_choices)
        if n_dims == 0:
            score_cc = 1.0  # vacuously complete when no dimensions required
        else:
            score_cc = min(float(n_choices) / float(n_dims), 1.0)
        scores["choice_completeness"] = self._make_score(
            "choice_completeness", score_cc, plan_weights,
            f"{n_choices} of {n_dims} dimension(s) covered" if n_dims > 0 else "no dimensions required",
        )

        # ---- evidence_quality ----
        n_evidence = len(theory.evidence)
        if vp.require_evidence and n_evidence == 0:
            score_eq = 0.0
        else:
            score_eq = min(float(n_evidence) / 3.0, 1.0)
        scores["evidence_quality"] = self._make_score(
            "evidence_quality", score_eq, plan_weights,
            f"{n_evidence} evidence item(s) cited" if n_evidence > 0 else None,
        )

        # ---- assumption_coverage ----
        n_assumptions = len(theory.assumptions)
        if vp.require_assumptions and n_assumptions == 0:
            score_ac = 0.0
        else:
            score_ac = min(float(n_assumptions) / 2.0, 1.0)
        scores["assumption_coverage"] = self._make_score(
            "assumption_coverage", score_ac, plan_weights,
            f"{n_assumptions} assumption(s) documented" if n_assumptions > 0 else None,
        )

        # ---- risk_awareness ----
        n_failure_modes = len(theory.failure_modes)
        score_ra = 1.0 if n_failure_modes > 0 else 0.5
        scores["risk_awareness"] = self._make_score(
            "risk_awareness", score_ra, plan_weights,
            f"{n_failure_modes} failure mode(s) identified" if n_failure_modes > 0 else None,
        )

        return scores

    @staticmethod
    def _make_score(
        criterion: str,
        score: float,
        plan_weights: dict[str, float],
        detail: str | None,
    ) -> CriterionScore:
        """Build a CriterionScore, substituting the plan-level weight if present."""
        high_rationale, low_rationale, default_weight = _CRITERION_META[criterion]
        weight = plan_weights.get(criterion, default_weight)

        if score >= _STRENGTH_THRESHOLD:
            rationale = high_rationale
            if detail:
                rationale = f"{high_rationale} ({detail})"
        else:
            rationale = low_rationale
            if detail:
                rationale = f"{low_rationale} ({detail})"

        return CriterionScore(score=score, rationale=rationale, weight=weight)

    @staticmethod
    def _compute_overall_score(criteria_scores: dict[str, CriterionScore]) -> float:
        """Weighted mean of all criterion scores, clamped to [0.0, 1.0]."""
        if not criteria_scores:
            return 0.0
        total_weight = sum(cs.weight for cs in criteria_scores.values())
        if total_weight == 0.0:
            return 0.0
        raw = sum(cs.score * cs.weight for cs in criteria_scores.values()) / total_weight
        return min(max(raw, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Qualitative derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_strengths(
        criteria_scores: dict[str, CriterionScore],
        theory: TheoryOfWinning,
    ) -> list[str]:
        """Return rationales for criteria that scored at or above the strength threshold."""
        strengths = [
            cs.rationale
            for cs in criteria_scores.values()
            if cs.score >= _STRENGTH_THRESHOLD
        ]
        if theory.success_conditions:
            strengths.extend(theory.success_conditions)
        return strengths

    @staticmethod
    def _extract_weaknesses(
        criteria_scores: dict[str, CriterionScore],
        theory: TheoryOfWinning,
    ) -> list[str]:
        """Return rationales for criteria that scored below the weakness threshold."""
        return [
            cs.rationale
            for cs in criteria_scores.values()
            if cs.score < _WEAKNESS_THRESHOLD
        ]

    @staticmethod
    def _derive_confidence(theory: TheoryOfWinning, overall_score: float) -> str:
        """Derive evaluation confidence from theory confidence and overall_score.

        Uses theory.confidence as the primary signal when available.
        Falls back to score-based derivation.
        """
        theory_conf = (theory.confidence or "").strip().lower()
        if theory_conf in {"high", "medium", "low"}:
            return theory_conf.capitalize()
        # Score-based fallback
        if overall_score >= 0.75:
            return "High"
        if overall_score >= 0.50:
            return "Medium"
        return "Low"
