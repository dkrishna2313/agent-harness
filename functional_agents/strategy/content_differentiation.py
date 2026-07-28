"""PH12.2b — ContentDifferentiation: multi-state homogenization guard.

Computes pairwise Jaccard similarity across theories for each content dimension,
applies multi-state homogenization detection (none/partial/substantial/full),
and produces the differentiation block for StrategyTrace.

Deterministic. Input-order invariant. No LLM calls.
No explicit-link exemption (removed in PH12.2b).
"""

from __future__ import annotations

import logging
from typing import Any

from .theory_content import TheoryContent

LOGGER = logging.getLogger(__name__)

_DIM_KEYS = [
    "assumption_similarity",
    "risk_similarity",
    "opportunity_similarity",
    "evidence_similarity",
    "recommendation_similarity",
]

_DIM_NAMES = ["assumptions", "risks", "opportunities", "evidence", "recommendations"]


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return round(len(a & b) / len(union), 4)


def compute_differentiation(
    contents: list[TheoryContent],
    partial_threshold: float = 0.75,
    full_threshold: float = 0.95,
    maximum_identical_dimensions: int = 2,
) -> dict[str, Any]:
    """Compute pairwise similarity and multi-state homogenization detection.

    Homogenization states (in priority order):
      full:        ALL 5 dims >= full_threshold
      substantial: avg(dims) >= full_threshold  OR  >=4 dims >= partial_threshold
      partial:     avg(dims) >= partial_threshold  OR  avg_pairwise >= partial_threshold
                   OR  count(identical_dims) >= maximum_identical_dimensions
      none:        otherwise

    detected=True for partial/substantial/full states.

    Returns a dict shaped per PH12.2b spec §21/§22:
    {
        "theory_differentiation": { "TH-A::TH-B": {dim sims + overall}, ... },
        "content_homogenization_detected": bool,
        "homogenization_state": str,
        "homogenization_details": {
            "state": str,
            "detected": bool,
            "affected_theory_ids": [...],
            "identical_dimensions": [...],
            "dimension_overlaps": {...},
            "pairwise_similarity": {"TH-A::TH-B": 0.82},
            "rationale": "...",
        },
    }
    """
    if len(contents) < 2:
        return {
            "theory_differentiation": {},
            "content_homogenization_detected": False,
            "homogenization_state": "none",
            "homogenization_details": {
                "state": "none",
                "detected": False,
                "affected_theory_ids": [],
                "identical_dimensions": [],
                "dimension_overlaps": {},
                "pairwise_similarity": {},
                "rationale": "Only one theory; no comparison possible.",
            },
        }

    pairwise: dict[str, dict[str, float]] = {}

    _pair_attr_map = [
        ("assumption_similarity",    "assumption_ids"),
        ("risk_similarity",          "risk_ids"),
        ("opportunity_similarity",   "opportunity_ids"),
        ("evidence_similarity",      "evidence_ids"),
        ("recommendation_similarity","recommendation_ids"),
    ]

    for i, tc_a in enumerate(contents):
        for j, tc_b in enumerate(contents):
            if j <= i:
                continue
            key = f"{tc_a.theory_id}::{tc_b.theory_id}"

            sims: dict[str, float] = {}
            populated_sims: list[float] = []
            for dk, attr in _pair_attr_map:
                sa = set(getattr(tc_a, attr, []))
                sb = set(getattr(tc_b, attr, []))
                sim = _jaccard(sa, sb)
                sims[dk] = sim
                # Only include in overall if at least one side is non-empty
                if sa or sb:
                    populated_sims.append(sim)

            # overall_similarity: average over populated dims only (not empty-vs-empty)
            sims["overall_similarity"] = round(
                sum(populated_sims) / len(populated_sims) if populated_sims else 0.0, 4
            )
            pairwise[key] = sims

    # Average similarity per dimension across all pairs
    dimension_overlaps: dict[str, float] = {}
    for dk in _DIM_KEYS:
        vals = [v[dk] for v in pairwise.values()]
        dimension_overlaps[dk] = round(sum(vals) / len(vals), 4)

    # Pairwise summary (pair_key → overall_similarity)
    pairwise_similarity = {k: v["overall_similarity"] for k, v in pairwise.items()}

    # "Populated" dimensions: at least one theory has non-empty content in this category
    # Empty-vs-empty dimensions are mathematically 1.0 but don't represent shared content.
    _dim_to_attr = {
        "assumption_similarity": "assumption_ids",
        "risk_similarity": "risk_ids",
        "opportunity_similarity": "opportunity_ids",
        "evidence_similarity": "evidence_ids",
        "recommendation_similarity": "recommendation_ids",
    }
    populated_dims: set[str] = {
        dk for dk, attr in _dim_to_attr.items()
        if any(len(getattr(tc, attr, [])) > 0 for tc in contents)
    }

    # Dimensions where the average overlap == 1.0, restricted to populated dimensions
    identical_dimensions = [
        _DIM_NAMES[i]
        for i, dk in enumerate(_DIM_KEYS)
        if dimension_overlaps[dk] == 1.0 and dk in populated_dims
    ]

    # Aggregate metrics for state determination — computed over populated dims only
    populated_overlap_vals = [dimension_overlaps[dk] for dk in _DIM_KEYS if dk in populated_dims]
    avg_dims = (
        sum(populated_overlap_vals) / len(populated_overlap_vals)
        if populated_overlap_vals else 0.0
    )
    avg_pairwise = (
        sum(pairwise_similarity.values()) / len(pairwise_similarity)
        if pairwise_similarity else 0.0
    )
    count_identical = len(identical_dimensions)
    count_high_dims = sum(
        1 for dk in _DIM_KEYS
        if dk in populated_dims and dimension_overlaps[dk] >= partial_threshold
    )

    # State determination (priority: full > substantial > partial > none)
    # Only evaluate populated dimensions for full/substantial checks
    if populated_dims and all(
        dimension_overlaps[dk] >= full_threshold for dk in _DIM_KEYS if dk in populated_dims
    ):
        state = "full"
    elif avg_dims >= full_threshold or count_high_dims >= 4:
        state = "substantial"
    elif (
        avg_dims >= partial_threshold
        or avg_pairwise >= partial_threshold
        or count_identical >= maximum_identical_dimensions
    ):
        state = "partial"
    else:
        state = "none"

    detected = state != "none"
    affected_ids = sorted(tc.theory_id for tc in contents) if detected else []

    # Rationale string
    if state == "full":
        rationale = (
            f"Full homogenization: all 5 dimensions exceed "
            f"{full_threshold:.0%} overlap threshold."
        )
    elif state == "substantial":
        if avg_dims >= full_threshold:
            rationale = (
                f"Substantial homogenization: average dimension overlap "
                f"{avg_dims:.2f} >= full threshold {full_threshold:.0%}."
            )
        else:
            rationale = (
                f"Substantial homogenization: {count_high_dims}/5 dimensions "
                f">= partial threshold {partial_threshold:.0%}."
            )
    elif state == "partial":
        reasons = []
        if avg_dims >= partial_threshold:
            reasons.append(
                f"avg_dim_overlap={avg_dims:.2f} >= {partial_threshold:.0%}"
            )
        if avg_pairwise >= partial_threshold:
            reasons.append(
                f"avg_pairwise_similarity={avg_pairwise:.2f} >= {partial_threshold:.0%}"
            )
        if count_identical >= maximum_identical_dimensions:
            reasons.append(
                f"{count_identical} identical dimensions "
                f"(threshold={maximum_identical_dimensions})"
            )
        rationale = "Partial homogenization: " + "; ".join(reasons) + "."
    else:
        rationale = "Content differentiation is within acceptable bounds."

    if detected:
        LOGGER.warning(
            "[ContentDifferentiation] homogenization state=%s across theories: %s",
            state,
            affected_ids,
        )

    return {
        "theory_differentiation": pairwise,
        "content_homogenization_detected": detected,
        "homogenization_state": state,
        "homogenization_details": {
            "state": state,
            "detected": detected,
            "affected_theory_ids": affected_ids,
            "identical_dimensions": identical_dimensions,
            "dimension_overlaps": dimension_overlaps,
            "pairwise_similarity": pairwise_similarity,
            "rationale": rationale,
        },
    }
