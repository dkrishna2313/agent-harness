"""OpportunityAgent – identifies strategic opportunities from upside assumption scenarios (J7.4).

Pipeline position: after RiskAgent, before MULTI_PROFILE / SCENARIO / QA.

Reads:
  - context.assumptions              (linked, with supported_recommendation_ids)
  - context.recommendations          (linked, with supported_assumption_ids)
  - context.risks                    (from RiskAgent — provides downside context)
  - context.evidence_notes
  - context.decision_model
  - context.research_object

Writes:
  - context.opportunities                         (list of opportunity dicts)
  - context.research_object["strategic_opportunities"]
  - context.trace["_strategic_opportunities"]

Also persists opportunities into the Decision Model artifact via
research_agent.decision_model.write_decision_model().
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class OpportunityAgent(FunctionalAgent):
    """Produces 5–10 strategic opportunities from upside assumption scenarios.

    Each StrategicOpportunity:
      - describes what additional value becomes possible when an assumption
        is exceeded rather than merely satisfied
      - links to one or more related assumption IDs
      - links to enabled recommendation IDs via assumption→rec traversal
      - carries its own impact, likelihood, and evidence support ratings
    """

    def __init__(
        self,
        *,
        client: Any = None,
        domain_profiles: list[Any] | None = None,
    ) -> None:
        self._client = client
        self._domain_profiles = domain_profiles or []

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        assumptions: list[dict] = context.assumptions
        recommendations: list[dict] = context.recommendations
        risks: list[dict] = context.risks

        if not assumptions:
            LOGGER.warning("[OpportunityAgent] no assumptions available — skipping opportunity generation")
            context.trace["_strategic_opportunities"] = {
                "skipped": True,
                "reason": "no_assumptions",
            }
            self._record(context, status="skipped", summary="No assumptions — opportunity generation skipped.")
            return context

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        payload = self._generate_opportunities(
            assumptions=assumptions,
            recommendations=recommendations,
            risks=risks,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
        )

        opps_as_dicts = [o.model_dump() for o in payload.opportunities]

        # Store in context
        context.opportunities = opps_as_dicts

        if context.research_object is not None:
            context.research_object["strategic_opportunities"] = opps_as_dicts

        # Observability counts
        opp_assumption_links = sum(len(o.get("related_assumption_ids", [])) for o in opps_as_dicts)
        opp_recommendation_links = sum(len(o.get("enabled_recommendation_ids", [])) for o in opps_as_dicts)

        context.trace["_strategic_opportunities"] = {
            "opportunity_count": len(opps_as_dicts),
            "assumption_links": opp_assumption_links,
            "recommendation_links": opp_recommendation_links,
            "opportunities": opps_as_dicts,
        }

        # Persist into Decision Model artifact
        dm_id: str | None = (
            context.research_object.get("decision_model_id")
            if context.research_object
            else None
        )
        dm_persisted = False
        if dm_id:
            LOGGER.log(PROGRESS, "[OpportunityAgent] persisting %d opportunities → DM %s", len(opps_as_dicts), dm_id)
            dm_persisted = _persist_opportunities_to_dm(dm_id, opps_as_dicts)
            if dm_persisted:
                LOGGER.log(PROGRESS, "[OpportunityAgent] Decision Model written with %d opportunities", len(opps_as_dicts))
            else:
                LOGGER.warning("[OpportunityAgent] DM persistence failed — opportunities stored in context only")
        else:
            LOGGER.warning("[OpportunityAgent] no decision_model_id on RO — skipping DM persistence")

        ro_persisted = _persist_opportunities_to_ro(context.research_object, opps_as_dicts)

        context.trace["_strategic_opportunities"]["dm_persisted"] = dm_persisted
        context.trace["_strategic_opportunities"]["ro_persisted"] = ro_persisted

        # Opportunity persistence observability (J7.5c)
        persisted_opp_ids = _load_persisted_opportunity_ids(dm_id) if dm_persisted else set()
        context_opp_ids = {o["opportunity_id"] for o in opps_as_dicts}
        orphan_ids = sorted(context_opp_ids - persisted_opp_ids)
        linkage_verified = len(orphan_ids) == 0
        if orphan_ids:
            LOGGER.warning(
                "[OpportunityAgent] %d orphan opportunity ID(s) not persisted to DM: %s",
                len(orphan_ids), orphan_ids,
            )
        context.trace["_opportunity_persistence"] = {
            "opportunity_count": len(opps_as_dicts),
            "dm_persisted": dm_persisted,
            "ro_persisted": ro_persisted,
            "orphan_ids": orphan_ids,
            "linkage_verified": linkage_verified,
        }

        LOGGER.log(
            PROGRESS,
            "[OpportunityAgent] %d opportunities generated; a_links=%d r_links=%d dm=%s ro=%s",
            len(opps_as_dicts),
            opp_assumption_links,
            opp_recommendation_links,
            dm_persisted,
            ro_persisted,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(opps_as_dicts)} strategic opportunities generated; "
                f"{opp_assumption_links} assumption links; "
                f"{opp_recommendation_links} recommendation links; "
                f"DM persisted={dm_persisted}."
            ),
            opportunity_count=len(opps_as_dicts),
            assumption_links=opp_assumption_links,
            recommendation_links=opp_recommendation_links,
            dm_persisted=dm_persisted,
            ro_persisted=ro_persisted,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_opportunities(
        self,
        *,
        assumptions: list[dict],
        recommendations: list[dict],
        risks: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ):
        from research_agent.claude_client import OpportunityPayload

        if self._client is None:
            LOGGER.warning("[OpportunityAgent] no client — using mock opportunities")
            return _mock_opportunities(assumptions, recommendations, risks, evidence_items, decision_model)

        if hasattr(self._client, "generate_opportunities"):
            return self._client.generate_opportunities(
                assumptions=assumptions,
                recommendations=recommendations,
                risks=risks,
                evidence_items=evidence_items,
                decision_model=decision_model,
            )

        LOGGER.warning("[OpportunityAgent] client lacks generate_opportunities — using mock")
        return _mock_opportunities(assumptions, recommendations, risks, evidence_items, decision_model)


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_opportunities_to_dm(decision_model_id: str, opportunities: list[dict]) -> bool:
    """Load the persisted DecisionModel, inject opportunities, re-write it."""
    try:
        from research_agent.decision_model import (
            StrategicOpportunity, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = [StrategicOpportunity.model_validate(o) for o in opportunities]
        updated = dm.model_copy(update={"strategic_opportunities": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning(
            "[OpportunityAgent] could not persist opportunities to DM %s: %s — %s",
            decision_model_id, type(exc).__name__, exc,
        )
        return False


def _load_persisted_opportunity_ids(decision_model_id: str | None) -> set[str]:
    """Return the set of opportunity_ids stored in the persisted DM (empty on any error)."""
    if not decision_model_id:
        return set()
    try:
        from research_agent.decision_model import load_decision_model
        dm = load_decision_model(decision_model_id)
        return {o.opportunity_id for o in dm.strategic_opportunities}
    except Exception:
        return set()


def _persist_opportunities_to_ro(research_object: dict | None, opportunities: list[dict]) -> bool:
    """Re-persist the Research Object with the opportunities injected."""
    if not research_object:
        return False
    try:
        from research_agent.research_object import write_research_object
        research_object["strategic_opportunities"] = opportunities
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning("[OpportunityAgent] could not persist opportunities to RO: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_opportunities(
    assumptions: list[dict],
    recommendations: list[dict],
    risks: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import OpportunityItem, OpportunityPayload

    ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
    question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

    templates = [
        ("Technology matures faster than expected, enabling earlier deployment and first-mover advantage", "Technology", "High", "Medium"),
        ("Market demand accelerates above projections, creating a larger addressable opportunity", "Market", "High", "Medium"),
        ("Capital costs decline faster than modelled, improving project economics materially", "Economics", "High", "Low"),
        ("Regulatory environment becomes more favourable, reducing compliance burden and opening new markets", "Regulation", "Medium", "Medium"),
        ("Supply chain innovations reduce lead times, enabling faster scale-up than planned", "Supply Chain", "Medium", "Low"),
    ]

    opportunities = []
    for i, (stmt, cat, imp, lik) in enumerate(templates):
        related_a = [assumptions[i]["assumption_id"]] if i < len(assumptions) else []
        enabled_r: list[str] = []
        if related_a and i < len(assumptions):
            enabled_r = list(assumptions[i].get("supported_recommendation_ids", []))
        sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
        opportunities.append(OpportunityItem(
            opportunity_id=f"OPP-{i+1:03d}",
            statement=stmt,
            category=cat,
            impact=imp,
            likelihood=lik,
            evidence_support="Moderate",
            confidence="Medium",
            rationale=f"Strategic opportunity relevant to: {question[:80]}",
            related_assumption_ids=related_a,
            enabled_recommendation_ids=enabled_r,
            evidence_ids=sup_ev,
            exploitation_notes="",
            status="Active",
        ))

    return OpportunityPayload(opportunities=opportunities)
