"""RiskAgent – identifies strategic risks from assumptions and linked recommendations (J7.3).

Pipeline position: after Recommendation Linkage, before QA.

Reads:
  - context.assumptions              (linked, with supported_recommendation_ids)
  - context.recommendations          (linked, with supported_assumption_ids)
  - context.evidence_notes
  - context.decision_model
  - context.research_object

Writes:
  - context.risks                               (list of risk dicts)
  - context.research_object["strategic_risks"]
  - context.trace["_strategic_risks"]

Also persists risks into the Decision Model artifact via
research_agent.decision_model.write_decision_model().
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class RiskAgent(FunctionalAgent):
    """Produces 5–10 strategic risks derived from strategic assumptions.

    Each StrategicRisk:
      - identifies what could cause an assumption to fail
      - links to one or more related assumption IDs
      - propagates to affected recommendation IDs via assumption→rec links
      - carries its own severity, likelihood, and evidence support ratings
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

        if not assumptions:
            LOGGER.warning("[RiskAgent] no assumptions available — skipping risk generation")
            context.trace["_strategic_risks"] = {
                "skipped": True,
                "reason": "no_assumptions",
            }
            self._record(context, status="skipped", summary="No assumptions — risk generation skipped.")
            return context

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        payload = self._generate_risks(
            assumptions=assumptions,
            recommendations=recommendations,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
        )

        risks_as_dicts = [r.model_dump() for r in payload.risks]

        # Store in context
        context.risks = risks_as_dicts

        if context.research_object is not None:
            context.research_object["strategic_risks"] = risks_as_dicts

        # Compute link counts for observability
        risk_assumption_links = sum(len(r.get("related_assumption_ids", [])) for r in risks_as_dicts)
        risk_recommendation_links = sum(len(r.get("affected_recommendation_ids", [])) for r in risks_as_dicts)

        context.trace["_strategic_risks"] = {
            "assumption_count": len(assumptions),
            "recommendation_count": len(recommendations),
            "risk_count": len(risks_as_dicts),
            "risk_assumption_link_count": risk_assumption_links,
            "risk_recommendation_link_count": risk_recommendation_links,
            "risks": risks_as_dicts,
        }

        # Persist into Decision Model artifact
        dm_id: str | None = (
            context.research_object.get("decision_model_id")
            if context.research_object
            else None
        )
        dm_persisted = False
        if dm_id:
            LOGGER.log(PROGRESS, "[RiskAgent] persisting %d risks → DM %s", len(risks_as_dicts), dm_id)
            dm_persisted = _persist_risks_to_dm(dm_id, risks_as_dicts)
            if dm_persisted:
                LOGGER.log(PROGRESS, "[RiskAgent] Decision Model written with %d risks", len(risks_as_dicts))
            else:
                LOGGER.warning("[RiskAgent] DM persistence failed — risks stored in context only")
        else:
            LOGGER.warning("[RiskAgent] no decision_model_id on RO — skipping DM persistence")

        ro_persisted = _persist_risks_to_ro(context.research_object, risks_as_dicts)

        context.trace["_strategic_risks"]["dm_persisted"] = dm_persisted
        context.trace["_strategic_risks"]["ro_persisted"] = ro_persisted

        LOGGER.log(
            PROGRESS,
            "[RiskAgent] %d risks generated; a_links=%d r_links=%d dm=%s ro=%s",
            len(risks_as_dicts),
            risk_assumption_links,
            risk_recommendation_links,
            dm_persisted,
            ro_persisted,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(risks_as_dicts)} strategic risks generated; "
                f"{risk_assumption_links} assumption links; "
                f"{risk_recommendation_links} recommendation links; "
                f"DM persisted={dm_persisted}."
            ),
            risk_count=len(risks_as_dicts),
            risk_assumption_link_count=risk_assumption_links,
            risk_recommendation_link_count=risk_recommendation_links,
            dm_persisted=dm_persisted,
            ro_persisted=ro_persisted,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_risks(
        self,
        *,
        assumptions: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ):
        from research_agent.claude_client import RiskPayload

        if self._client is None:
            LOGGER.warning("[RiskAgent] no client — using mock risks")
            return _mock_risks(assumptions, recommendations, evidence_items, decision_model)

        if hasattr(self._client, "generate_risks"):
            return self._client.generate_risks(
                assumptions=assumptions,
                recommendations=recommendations,
                evidence_items=evidence_items,
                decision_model=decision_model,
            )

        LOGGER.warning("[RiskAgent] client lacks generate_risks — using mock")
        return _mock_risks(assumptions, recommendations, evidence_items, decision_model)


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_risks_to_dm(decision_model_id: str, risks: list[dict]) -> bool:
    """Load the persisted DecisionModel, inject risks, re-write it."""
    try:
        from research_agent.decision_model import (
            StrategicRisk, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = [StrategicRisk.model_validate(r) for r in risks]
        updated = dm.model_copy(update={"strategic_risks": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning("[RiskAgent] could not persist risks to DM %s: %s", decision_model_id, exc)
        return False


def _persist_risks_to_ro(research_object: dict | None, risks: list[dict]) -> bool:
    """Re-persist the Research Object with the risks injected."""
    if not research_object:
        return False
    try:
        from research_agent.research_object import write_research_object
        research_object["strategic_risks"] = risks
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning("[RiskAgent] could not persist risks to RO: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_risks(
    assumptions: list[dict],
    recommendations: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import RiskItem, RiskPayload

    ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
    question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

    templates = [
        ("Technology maturity proves insufficient, causing deployment delays or failures", "Technology", "High", "Medium"),
        ("Market demand shifts materially below projections, undermining the economic case", "Market", "High", "Medium"),
        ("Capital costs escalate beyond estimates, impairing project viability", "Economics", "High", "Low"),
        ("Regulatory changes impose unforeseen restrictions or obligations", "Regulation", "Medium", "Medium"),
        ("Supply chain disruptions delay or prevent timely execution", "Supply Chain", "Medium", "Low"),
    ]

    risks = []
    for i, (stmt, cat, sev, lik) in enumerate(templates):
        related_a = [assumptions[i]["assumption_id"]] if i < len(assumptions) else []
        affected_r: list[str] = []
        if related_a and i < len(assumptions):
            affected_r = list(assumptions[i].get("supported_recommendation_ids", []))
        sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
        risks.append(RiskItem(
            risk_id=f"RSK-{i+1:03d}",
            statement=stmt,
            category=cat,
            severity=sev,
            likelihood=lik,
            evidence_support="Moderate",
            confidence="Medium",
            rationale=f"Strategic risk relevant to: {question[:80]}",
            related_assumption_ids=related_a,
            affected_recommendation_ids=affected_r,
            evidence_ids=sup_ev,
            mitigation_notes="",
            status="Active",
        ))

    return RiskPayload(risks=risks)
