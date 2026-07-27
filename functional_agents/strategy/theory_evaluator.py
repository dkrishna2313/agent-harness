"""TheoryEvaluator — evaluates a single TheoryOfWinning (PH10.5 / PH10.5a).

Public interface:
    build(theory, plan, research) -> TheoryEvaluation

One TheoryEvaluation is produced per TheoryOfWinning. Theories are
evaluated independently; no comparison or winner selection is performed.

--- Criterion-selection algorithm ---

The set of criteria that appear in the resulting TheoryEvaluation is
determined by plan.evaluation_model.weights:

  CONFIGURED mode (plan.evaluation_model.weights is non-empty):
    The key set of plan.evaluation_model.weights defines the COMPLETE
    criterion set. No built-in criteria are added automatically.
    For each key:
      - Recognised names (present in _CRITERION_META): scored by the
        registered deterministic scorer with the configured weight.
      - Unrecognised names (not in _CRITERION_META): scored by the
        documented deterministic fallback (_FALLBACK_SCORE = 0.5,
        rationale = _FALLBACK_RATIONALE) with the configured weight.

  DEFAULT mode (plan.evaluation_model.weights is empty):
    All seven built-in criteria from _CRITERION_META are evaluated with
    their registered default weights.

plan.validation_policy drives strict-zero scoring for recognised criteria
when evidence or assumptions are absent and required.

--- Evidence rationale tiers ---

For criteria with partial-score states (evidence_quality, assumption_coverage,
choice_completeness, risk_awareness) the rationale distinguishes:

  score == 1.0  (sufficient)  → high_rationale + detail
  0 < score < 1  (partial)   → detail string (accurate description of state)
  score == 0.0  (none / fail) → low_rationale

No LLM calls, no search, no optimisation.

Produced by: StrategyCoordinator.build() (one per TheoryOfWinning)
Consumed by: StrategyCoordinator (stored as _evaluations; not yet wired
             into StrategicPosition)
"""

from __future__ import annotations

import logging
from typing import Any

from .strategic_position import TheoryOfWinning
from .strategy_plan import StrategyPlan, ValidationPolicy
from .theory_evaluation import CriterionScore, TheoryEvaluation

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Criterion registry
# ---------------------------------------------------------------------------
# (high_rationale, low_rationale, default_weight)
# high_rationale: score >= _STRENGTH_THRESHOLD
# low_rationale:  score == 0.0
# partial score:  the detail string is used directly (see _make_score)

_CRITERION_META: dict[str, tuple[str, str, float]] = {
    # --- PH10.5 built-in criteria ---
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

# PH12.0 criteria — available in CONFIGURED mode only; not included in
# DEFAULT mode so the "seven built-in criteria" contract stays intact.
_CONFIGURED_CRITERION_META: dict[str, tuple[str, str, float]] = {
    "strategic_fit": (
        "Strategic position, mechanism, and choice coverage confirmed.",
        "Strategic fit cannot be assessed — position and mechanism are absent.",
        2.0,
    ),
    "assumption_robustness": (
        "Key assumptions are documented and support the strategy.",
        "No assumptions documented; strategic robustness cannot be assessed.",
        1.5,
    ),
    "execution_feasibility": (
        "Choice set is complete and execution-ready.",
        "Insufficient choice coverage for execution feasibility.",
        1.5,
    ),
    "risk_resilience": (
        "Multiple failure modes identified; risk plan is comprehensive.",
        "Failure modes have not been assessed.",
        1.5,
    ),
    "opportunity_capture": (
        "Multiple success conditions identified; opportunity capture is structured.",
        "No success conditions identified; opportunity capture cannot be assessed.",
        1.0,
    ),
}

# Combined lookup used by _score_one / _make_score — covers both default and
# configured criteria without polluting the 7-entry DEFAULT mode set.
_ALL_CRITERION_META: dict[str, tuple[str, str, float]] = {
    **_CRITERION_META,
    **_CONFIGURED_CRITERION_META,
}

# Score at or above this threshold → strength candidate
_STRENGTH_THRESHOLD = 0.75
# Score strictly below this threshold → weakness candidate
_WEAKNESS_THRESHOLD = 0.50

# Fallback for unrecognised configured criteria
_FALLBACK_SCORE = 0.5
_FALLBACK_RATIONALE = (
    "Criterion not recognized by this evaluator; neutral score assigned."
)


class TheoryEvaluator:
    """Evaluates a single TheoryOfWinning against plan-driven criteria.

    Criterion-selection algorithm
    ─────────────────────────────
    • CONFIGURED mode  (plan.evaluation_model.weights non-empty):
        Only the criteria named in weights are evaluated.
        Unrecognised names receive the deterministic neutral fallback.
    • DEFAULT mode  (plan.evaluation_model.weights empty):
        All seven built-in criteria are evaluated at their default weights.

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
            The resolved StrategyPlan; provides the criterion weight map
            (evaluation_model.weights) and validation policy.
        research:
            Available research output (AgentContext in PH10.x). Reserved
            for future use; currently a structural parameter.

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
        theory_id = theory.theory_id or ""

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
            "[TheoryEvaluator] theory_id=%s overall_score=%.3f confidence=%s "
            "mode=%s n_criteria=%d",
            theory_id, overall_score, confidence,
            "configured" if plan.evaluation_model.weights else "default",
            len(criteria_scores),
        )
        return evaluation

    # ------------------------------------------------------------------
    # Criterion selection and scoring
    # ------------------------------------------------------------------

    def _score_criteria(
        self,
        theory: TheoryOfWinning,
        plan: StrategyPlan,
    ) -> dict[str, CriterionScore]:
        """Return one CriterionScore per criterion in the active criterion set.

        CONFIGURED mode: key set of plan.evaluation_model.weights (non-empty).
        DEFAULT mode:    all entries in _CRITERION_META (empty weights).
        """
        plan_weights = plan.evaluation_model.weights
        vp = plan.validation_policy
        n_dims = len(plan.active_dimensions)

        if plan_weights:
            # CONFIGURED mode — criterion set is exactly the plan's weight keys
            return {
                name: self._score_one(name, weight, theory, vp, n_dims)
                for name, weight in plan_weights.items()
            }
        else:
            # DEFAULT mode — all built-in criteria at registered default weights
            return {
                name: self._score_one(name, meta[2], theory, vp, n_dims)
                for name, meta in _CRITERION_META.items()
            }

    @staticmethod
    def _score_one(
        name: str,
        weight: float,
        theory: TheoryOfWinning,
        vp: ValidationPolicy,
        n_dims: int,
    ) -> CriterionScore:
        """Score a single criterion by name.

        Unrecognised names receive the deterministic neutral fallback
        (_FALLBACK_SCORE, _FALLBACK_RATIONALE) with the configured weight.
        """
        if name not in _ALL_CRITERION_META:
            return CriterionScore(
                score=_FALLBACK_SCORE,
                rationale=_FALLBACK_RATIONALE,
                weight=weight,
            )
        score, detail = TheoryEvaluator._raw_score(name, theory, vp, n_dims)
        return TheoryEvaluator._make_score(name, score, weight, detail)

    @staticmethod
    def _raw_score(
        name: str,
        theory: TheoryOfWinning,
        vp: ValidationPolicy,
        n_dims: int,
    ) -> tuple[float, str | None]:
        """Return (score, detail) for a recognised criterion.

        detail is a short descriptive string used as the rationale when the
        score is in the partial range (0 < score < _STRENGTH_THRESHOLD).
        detail is None when no meaningful partial description exists.
        """
        if name == "option_identified":
            sc = 1.0 if theory.recommended_option_id else 0.0
            return sc, theory.recommended_option_id or None

        if name == "position_articulated":
            sc = 1.0 if theory.winning_position else 0.0
            return sc, theory.winning_position or None

        if name == "mechanism_defined":
            sc = 1.0 if theory.winning_mechanism else 0.0
            return sc, theory.winning_mechanism or None

        if name == "choice_completeness":
            n = len(theory.strategic_choices)
            if n_dims == 0:
                return 1.0, "no dimensions required"
            sc = min(float(n) / float(n_dims), 1.0)
            return sc, f"{n} of {n_dims} dimension(s) covered"

        if name == "evidence_quality":
            n = len(theory.evidence)
            if vp.require_evidence and n == 0:
                sc = 0.0
            else:
                sc = min(float(n) / 3.0, 1.0)
            # Always supply a detail so partial scores have accurate rationale
            detail = f"{n} evidence item(s) cited" if n > 0 else None
            return sc, detail

        if name == "assumption_coverage":
            n = len(theory.assumptions)
            if vp.require_assumptions and n == 0:
                sc = 0.0
            else:
                sc = min(float(n) / 2.0, 1.0)
            detail = f"{n} assumption(s) documented" if n > 0 else None
            return sc, detail

        if name == "risk_awareness":
            n = len(theory.failure_modes)
            sc = 1.0 if n > 0 else 0.5
            detail = (
                f"{n} failure mode(s) identified"
                if n > 0
                else "no failure modes identified"
            )
            return sc, detail

        # PH12.0 criteria -------------------------------------------------

        if name == "strategic_fit":
            pos = 1.0 if theory.winning_position else 0.0
            mech = 1.0 if theory.winning_mechanism else 0.0
            n = len(theory.strategic_choices)
            comp = min(float(n) / float(n_dims), 1.0) if n_dims > 0 else 1.0
            sc = round(0.5 * pos + 0.3 * mech + 0.2 * comp, 6)
            detail = (
                f"position={'yes' if pos else 'no'}, "
                f"mechanism={'yes' if mech else 'no'}, "
                f"coverage={n}/{n_dims if n_dims else 'n/a'}"
            )
            return sc, detail

        if name == "assumption_robustness":
            n = len(theory.assumptions)
            if vp.require_assumptions and n == 0:
                sc = 0.0
            else:
                sc = min(float(n) / 2.0, 1.0)
            detail = f"{n} assumption(s) documented" if n > 0 else None
            return sc, detail

        if name == "execution_feasibility":
            n = len(theory.strategic_choices)
            sc = min(float(n) / float(n_dims), 1.0) if n_dims > 0 else 1.0
            detail = f"{n} of {n_dims} dimension(s) covered"
            return sc, detail

        if name == "risk_resilience":
            n = len(theory.failure_modes)
            if n == 0:
                sc = 0.3
            elif n == 1:
                sc = 0.6
            elif n == 2:
                sc = 0.8
            else:
                sc = 1.0
            detail = f"{n} failure mode(s) identified"
            return sc, detail

        if name == "opportunity_capture":
            n = len(theory.success_conditions)
            if n == 0:
                sc = 0.3
            elif n == 1:
                sc = 0.6
            elif n == 2:
                sc = 0.8
            else:
                sc = 1.0
            detail = f"{n} success condition(s) identified"
            return sc, detail

        # Unreachable: all _CRITERION_META keys are handled above
        return 0.0, None  # pragma: no cover

    @staticmethod
    def _make_score(
        name: str,
        score: float,
        weight: float,
        detail: str | None,
    ) -> CriterionScore:
        """Build a CriterionScore with a three-tier rationale.

        Tiers:
          score >= _STRENGTH_THRESHOLD  → high_rationale [+ (detail)]
          0 < score < _STRENGTH_THRESHOLD → detail (accurate partial state)
          score == 0.0                  → low_rationale
        """
        high_rationale, low_rationale, _ = _ALL_CRITERION_META[name]

        if score >= _STRENGTH_THRESHOLD:
            rationale = f"{high_rationale} ({detail})" if detail else high_rationale
        elif score == 0.0:
            rationale = low_rationale
        else:
            # Partial: detail accurately describes what was found
            rationale = detail if detail else low_rationale

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
        """Return rationales for criteria scoring >= _STRENGTH_THRESHOLD,
        plus any success_conditions from the theory."""
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
        """Return rationales for criteria scoring < _WEAKNESS_THRESHOLD."""
        return [
            cs.rationale
            for cs in criteria_scores.values()
            if cs.score < _WEAKNESS_THRESHOLD
        ]

    @staticmethod
    def _derive_confidence(theory: TheoryOfWinning, overall_score: float) -> str:
        """Derive evaluation confidence from theory.confidence then overall_score."""
        theory_conf = (theory.confidence or "").strip().lower()
        if theory_conf in {"high", "medium", "low"}:
            return theory_conf.capitalize()
        if overall_score >= 0.75:
            return "High"
        if overall_score >= 0.50:
            return "Medium"
        return "Low"
