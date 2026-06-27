"""Recommendation Linkage – post-processing step that wires assumptions to recommendations (J7.2).

Runs inline in the orchestrator immediately after RecommendationAgent, before QA.
No LLM call. Pure deterministic linkage based on shared evidence IDs and
hypothesis IDs.

Public API
----------
build_recommendation_linkage()  – links assumptions ↔ recommendations, returns
                                   (updated_assumptions, updated_recommendations)
normalise_recommendation_id()   – derives stable REC-001 id from any id string
persist_linkage()               – re-writes DM (assumptions) and RO (recommendations)
"""

from __future__ import annotations

import logging
import re
from typing import Any

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ID normalisation
# ---------------------------------------------------------------------------

def normalise_recommendation_id(raw_id: str, index: int) -> str:
    """Return a stable REC-NNN id.

    If the raw id already matches REC-NNN it is returned unchanged.
    Otherwise the numeric part is extracted (R1 → REC-001) or the index is used.
    """
    if re.match(r"^REC-\d+$", raw_id, re.IGNORECASE):
        return raw_id.upper()
    match = re.search(r"\d+", raw_id)
    if match:
        return f"REC-{int(match.group()):03d}"
    return f"REC-{index + 1:03d}"


# ---------------------------------------------------------------------------
# Core linkage logic
# ---------------------------------------------------------------------------

def build_recommendation_linkage(
    assumptions: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Determine which assumptions support which recommendations and populate both sides.

    Linkage criteria (OR — any match is sufficient):
      1. Evidence overlap: assumption.evidence_ids ∩ recommendation.supporting_evidence
      2. Hypothesis overlap: assumption is derived from hypothesis H-x AND
                             recommendation.supported_by_hypotheses contains H-x
         (detected via assumption.rationale containing the hypothesis id)

    Returns updated copies of both lists with IDs populated.
    """
    if not assumptions or not recommendations:
        return assumptions, recommendations

    # Deep-copy to avoid mutating in-place
    import copy
    assumptions = copy.deepcopy(assumptions)
    recommendations = copy.deepcopy(recommendations)

    # --- Normalise recommendation_id ---
    for i, rec in enumerate(recommendations):
        if not rec.get("recommendation_id"):
            rec["recommendation_id"] = normalise_recommendation_id(rec.get("id", ""), i)

    # --- Build lookup sets ---
    # assumption evidence sets
    assumption_ev: dict[str, set[str]] = {
        a["assumption_id"]: set(a.get("evidence_ids") or [])
        for a in assumptions
    }

    # recommendation evidence + hypothesis sets
    rec_ev: dict[str, set[str]] = {
        rec["recommendation_id"]: set(rec.get("supporting_evidence") or [])
        for rec in recommendations
    }
    rec_hyp: dict[str, set[str]] = {
        rec["recommendation_id"]: set(rec.get("supported_by_hypotheses") or [])
        for rec in recommendations
    }

    # assumption hypothesis hints (parse rationale for hypothesis ids like H1, H2 …)
    _HYP_RE = re.compile(r"\bH\d+\b")
    assumption_hyp: dict[str, set[str]] = {
        a["assumption_id"]: set(_HYP_RE.findall(a.get("rationale", "")))
        for a in assumptions
    }

    # --- Build bi-directional links ---
    # assumption_id → set of recommendation_ids
    a_to_recs: dict[str, set[str]] = {a["assumption_id"]: set() for a in assumptions}
    # recommendation_id → set of assumption_ids
    r_to_assumptions: dict[str, set[str]] = {rec["recommendation_id"]: set() for rec in recommendations}

    for a in assumptions:
        a_id = a["assumption_id"]
        a_ev = assumption_ev[a_id]
        a_hyp = assumption_hyp[a_id]
        for rec in recommendations:
            r_id = rec["recommendation_id"]
            r_ev = rec_ev[r_id]
            r_hyp = rec_hyp[r_id]
            linked = bool(
                (a_ev and r_ev and a_ev & r_ev)   # shared evidence
                or (a_hyp and r_hyp and a_hyp & r_hyp)  # shared hypothesis
            )
            if linked:
                a_to_recs[a_id].add(r_id)
                r_to_assumptions[r_id].add(a_id)

    # --- Apply back ---
    link_count = 0
    for a in assumptions:
        a_id = a["assumption_id"]
        new_rec_ids = sorted(a_to_recs[a_id])
        a["supported_recommendation_ids"] = new_rec_ids
        link_count += len(new_rec_ids)

    for rec in recommendations:
        r_id = rec["recommendation_id"]
        rec["supported_assumption_ids"] = sorted(r_to_assumptions[r_id])

    LOGGER.info(
        "[RecommendationLinkage] %d assumptions × %d recommendations → %d links created",
        len(assumptions), len(recommendations), link_count,
    )
    return assumptions, recommendations


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_linkage(
    decision_model_id: str | None,
    assumptions: list[dict[str, Any]],
    recommendations: list[dict[str, Any]],
    research_object: dict[str, Any],
) -> tuple[bool, bool]:
    """Re-persist the Decision Model (assumptions) and Research Object (recommendations).

    Returns (dm_ok, ro_ok) booleans for observability.
    """
    from research_agent.log import PROGRESS

    dm_ok = _persist_dm(decision_model_id, assumptions)
    ro_ok = _persist_ro(research_object, recommendations)

    LOGGER.log(
        PROGRESS,
        "[RecommendationLinkage] persistence — DM ok=%s  RO ok=%s",
        dm_ok, ro_ok,
    )
    return dm_ok, ro_ok


def _persist_dm(decision_model_id: str | None, assumptions: list[dict[str, Any]]) -> bool:
    if not decision_model_id:
        LOGGER.warning("[RecommendationLinkage] no decision_model_id — skipping DM persistence")
        return False
    try:
        from research_agent.decision_model import (
            DecisionAssumption, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = [DecisionAssumption.model_validate(a) for a in assumptions]
        updated = dm.model_copy(update={"strategic_assumptions": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning("[RecommendationLinkage] DM persistence failed: %s", exc)
        return False


def _persist_ro(research_object: dict[str, Any], recommendations: list[dict[str, Any]]) -> bool:
    try:
        from research_agent.research_object import write_research_object
        research_object["recommendations"] = recommendations
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning("[RecommendationLinkage] RO persistence failed: %s", exc)
        return False
