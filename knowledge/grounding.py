"""Evidence Grounding — PH5.5e.

Populates subquestion assignments, investigation area assignments, grounding
strength, and coverage contribution on retrieved evidence items.

Design:
- Pure computation on existing retrieval and assembly data — no new retrieval
  or inference.
- Conservative: only assigns evidence that the existing mapping functions
  placed in evidence_by_subquestion or evidence_by_area.
- Deterministic: identical inputs produce identical grounding outputs.
- No Evidence schema changes — grounding fields are already defined in v2.

Grounding strength thresholds (based on retrieval hybrid_score):
    STRONG   >= _STRONG_SCORE
    MODERATE >= _MODERATE_SCORE
    WEAK     <  _MODERATE_SCORE

Coverage contribution thresholds (based on subquestion evidence count):
    STRONG   count <= 1  (sole evidence — subquestion has no other support)
    MODERATE 1 < count < _COMPLETE_THRESHOLD  (partial coverage)
    WEAK     count >= _COMPLETE_THRESHOLD  (subquestion already complete)
"""

from __future__ import annotations

from .models import Evidence, GroundingStrength

# Retrieval score thresholds — tune here only; never hard-code in callers.
_STRONG_SCORE: float = 0.6
_MODERATE_SCORE: float = 0.3

# Must match assembly._COMPLETE_THRESHOLD for consistent status semantics.
_COMPLETE_THRESHOLD: int = 4


# ---------------------------------------------------------------------------
# Internal helpers — all pure functions, no I/O
# ---------------------------------------------------------------------------


def _grounding_strength(hybrid_score: float) -> GroundingStrength:
    """Map a retrieval hybrid_score to a GroundingStrength value."""
    if hybrid_score >= _STRONG_SCORE:
        return "STRONG"
    if hybrid_score >= _MODERATE_SCORE:
        return "MODERATE"
    return "WEAK"


def _coverage_contribution(evidence_count: int) -> GroundingStrength:
    """Map a subquestion's total evidence count to a coverage contribution strength.

    Reflects how uniquely this evidence item contributes to the subquestion's
    coverage.  Sole item → STRONG; partial → MODERATE; already complete → WEAK.
    """
    if evidence_count <= 1:
        return "STRONG"
    if evidence_count < _COMPLETE_THRESHOLD:
        return "MODERATE"
    return "WEAK"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ground_evidence(
    evidence: Evidence,
    *,
    hybrid_score: float,
    subquestion_assignments: list[str],
    area_assignments: list[str],
    evidence_counts: dict[str, int],
) -> Evidence:
    """Create a grounded copy of an Evidence item with grounding fields populated.

    Returns a new Evidence instance (the original is unchanged — Evidence is
    frozen by design).

    Parameters
    ----------
    evidence:
        The source Evidence item.
    hybrid_score:
        Retrieval hybrid_score for this item; determines grounding_strength.
    subquestion_assignments:
        Subquestion texts this evidence is mapped to by evidence_by_subquestion.
    area_assignments:
        Investigation area names from evidence_by_area.
    evidence_counts:
        Mapping of subquestion_text → total evidence count for that subquestion
        (from SubquestionCompleteness.evidence_count).  Used to compute
        coverage_contribution: the evidence contributes most when its assigned
        subquestion has the fewest items.
    """
    strength = _grounding_strength(hybrid_score)

    if subquestion_assignments:
        # Contribution is determined by the least-covered assigned subquestion.
        min_count = min(
            evidence_counts.get(sq, 0) for sq in subquestion_assignments
        )
        contribution: GroundingStrength = _coverage_contribution(min_count)
    else:
        # Evidence mapped only to areas (or unmapped): marginal coverage value.
        contribution = "WEAK"

    return evidence.model_copy(update={
        "subquestion_assignments": list(subquestion_assignments),
        "investigation_area_assignments": list(area_assignments),
        "grounding_strength": strength,
        "coverage_contribution": contribution,
    })
