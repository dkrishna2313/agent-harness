"""PH12.2 — ContentDifferentiation: theory differentiation metrics and homogenization guard.

Computes pairwise Jaccard similarity across theories for each content dimension,
detects content homogenization, and produces the differentiation block for StrategyTrace.

Deterministic. Input-order invariant. No LLM calls.
"""

from __future__ import annotations

import logging
from typing import Any

from .theory_content import TheoryContent

LOGGER = logging.getLogger(__name__)

# Homogenization threshold: if all theories share > this fraction of content
# in every dimension with no explicit justification, flag homogenization.
_HOMOGENIZATION_THRESHOLD = 0.90


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
) -> dict[str, Any]:
    """Compute pairwise similarity and homogenization detection.

    Returns a dict shaped per spec §21/§22:
    {
        "theory_differentiation": {
            "TH-A::TH-B": {
                "assumption_similarity": 0.40,
                "risk_similarity": 0.20,
                "opportunity_similarity": 0.33,
                "evidence_similarity": 0.50,
                "recommendation_similarity": 0.60,
                "overall_similarity": 0.41,
            },
        },
        "content_homogenization_detected": bool,
        "homogenization_details": {
            "detected": bool,
            "affected_theory_ids": [...],
            "dimension_overlaps": {...},
            "message": "...",
        },
    }
    """
    if len(contents) < 2:
        return {
            "theory_differentiation": {},
            "content_homogenization_detected": False,
            "homogenization_details": {
                "detected": False,
                "affected_theory_ids": [],
                "dimension_overlaps": {},
                "message": "Only one theory; no comparison possible.",
            },
        }

    pairwise: dict[str, dict[str, float]] = {}

    for i, tc_a in enumerate(contents):
        for j, tc_b in enumerate(contents):
            if j <= i:
                continue
            key = f"{tc_a.theory_id}::{tc_b.theory_id}"

            a_asms = set(tc_a.assumption_ids)
            b_asms = set(tc_b.assumption_ids)
            a_rsk  = set(tc_a.risk_ids)
            b_rsk  = set(tc_b.risk_ids)
            a_opp  = set(tc_a.opportunity_ids)
            b_opp  = set(tc_b.opportunity_ids)
            a_ev   = set(tc_a.evidence_ids)
            b_ev   = set(tc_b.evidence_ids)
            a_rec  = set(tc_a.recommendation_ids)
            b_rec  = set(tc_b.recommendation_ids)

            asm_sim = _jaccard(a_asms, b_asms)
            rsk_sim = _jaccard(a_rsk,  b_rsk)
            opp_sim = _jaccard(a_opp,  b_opp)
            ev_sim  = _jaccard(a_ev,   b_ev)
            rec_sim = _jaccard(a_rec,  b_rec)

            dims = [asm_sim, rsk_sim, opp_sim, ev_sim, rec_sim]
            overall = round(sum(dims) / len(dims), 4)

            pairwise[key] = {
                "assumption_similarity": asm_sim,
                "risk_similarity": rsk_sim,
                "opportunity_similarity": opp_sim,
                "evidence_similarity": ev_sim,
                "recommendation_similarity": rec_sim,
                "overall_similarity": overall,
            }

    # Homogenization detection
    homogenization_detected = False
    affected_ids: list[str] = []
    dimension_overlaps: dict[str, float] = {}
    message = ""

    if pairwise:
        # Compute average similarity per dimension across all pairs
        dim_keys = [
            "assumption_similarity", "risk_similarity", "opportunity_similarity",
            "evidence_similarity", "recommendation_similarity",
        ]
        for dk in dim_keys:
            vals = [v[dk] for v in pairwise.values()]
            dimension_overlaps[dk] = round(sum(vals) / len(vals), 4)

        # Homogenization: all dimensions exceed threshold AND fallback is not the cause
        all_high = all(dimension_overlaps[dk] >= _HOMOGENIZATION_THRESHOLD for dk in dim_keys)
        if all_high:
            # Check whether the overlap is justified by shared explicit links
            # (high overlap is OK when upstream canonical links genuinely support it)
            explicit_shares = [tc.coverage.explicit_count for tc in contents]
            avg_explicit = sum(explicit_shares) / len(explicit_shares) if explicit_shares else 0
            if avg_explicit == 0:
                # No explicit links justifying homogenization → flag it
                homogenization_detected = True
                affected_ids = sorted(tc.theory_id for tc in contents)
                message = (
                    "Content homogenization detected: materially distinct theories "
                    "share >{:.0%} of content across all dimensions with no explicit "
                    "canonical relationship justification.".format(_HOMOGENIZATION_THRESHOLD)
                )
                LOGGER.warning(
                    "[ContentDifferentiation] homogenization detected across theories: %s",
                    affected_ids,
                )
            else:
                message = (
                    "High overlap across theories, but justified by shared explicit "
                    "canonical relationship links."
                )
        else:
            message = "Content differentiation is within acceptable bounds."

    return {
        "theory_differentiation": pairwise,
        "content_homogenization_detected": homogenization_detected,
        "homogenization_details": {
            "detected": homogenization_detected,
            "affected_theory_ids": affected_ids,
            "dimension_overlaps": dimension_overlaps,
            "message": message,
        },
    }
