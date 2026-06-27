"""StrategicOptionAgent – synthesises the J7 graph into investable strategic options (J7.5).

Pipeline position: after OpportunityAgent, before MULTI_PROFILE / SCENARIO / QA.

Reads:
  - context.assumptions              (linked assumptions from AssumptionAgent)
  - context.risks                    (from RiskAgent)
  - context.opportunities            (from OpportunityAgent)
  - context.recommendations          (linked recommendations)
  - context.evidence_notes
  - context.decision_model
  - context.research_object

Writes:
  - context.strategic_options                          (list of option dicts)
  - context.preferred_option                           (the single recommended option)
  - context.research_object["strategic_options"]
  - context.trace["_strategic_options"]

Also persists options into the Decision Model artifact via
research_agent.decision_model.write_decision_model().
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class StrategicOptionAgent(FunctionalAgent):
    """Produces ~3 genuinely different strategic options from the full J7 graph.

    Each StrategicOption:
      - is a coherent strategic posture (not a list of tasks)
      - references supporting assumptions, risks, opportunities, and recommendations
      - carries implementation complexity, time horizon, and capital intensity ratings
    Exactly one option has recommended=True.
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
        risks: list[dict] = context.risks
        opportunities: list[dict] = context.opportunities
        recommendations: list[dict] = context.recommendations

        if not assumptions:
            LOGGER.warning("[StrategicOptionAgent] no assumptions available — skipping option generation")
            context.trace["_strategic_options"] = {
                "skipped": True,
                "reason": "no_assumptions",
            }
            self._record(context, status="skipped", summary="No assumptions — strategic option generation skipped.")
            return context

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        payload = self._generate_options(
            assumptions=assumptions,
            risks=risks,
            opportunities=opportunities,
            recommendations=recommendations,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
        )

        options_as_dicts = [o.model_dump() for o in payload.options]

        # Cardinality validation (2–5 options required)
        n = len(options_as_dicts)
        if n < 2:
            LOGGER.warning("[StrategicOptionAgent] LLM returned %d option(s) — minimum is 2; duplicating first to satisfy constraint", n)
            if n == 1:
                import copy
                extra = copy.deepcopy(options_as_dicts[0])
                extra["option_id"] = extra["option_id"] + "-B"
                extra["recommended"] = False
                options_as_dicts.append(extra)
        elif n > 5:
            LOGGER.warning("[StrategicOptionAgent] LLM returned %d options — truncating to 5", n)
            # Preserve the recommended option if it falls outside the first 5
            rec_idx = next((i for i, o in enumerate(options_as_dicts) if o.get("recommended")), None)
            if rec_idx is not None and rec_idx >= 5:
                options_as_dicts[4] = options_as_dicts[rec_idx]
            options_as_dicts = options_as_dicts[:5]

        # Exactly one recommended — enforce single recommended=True
        rec_options = [o for o in options_as_dicts if o.get("recommended")]
        if len(rec_options) == 0:
            options_as_dicts[0]["recommended"] = True
            LOGGER.warning("[StrategicOptionAgent] no option had recommended=True — defaulting to first option")
        elif len(rec_options) > 1:
            # Keep only the first recommended; clear the rest
            first_rec = rec_options[0]["option_id"]
            for o in options_as_dicts:
                if o.get("recommended") and o["option_id"] != first_rec:
                    o["recommended"] = False
            LOGGER.warning("[StrategicOptionAgent] %d options had recommended=True — keeping only %s", len(rec_options), first_rec)

        recommended = next((o for o in options_as_dicts if o.get("recommended")), None)

        # Store in context
        context.strategic_options = options_as_dicts
        context.preferred_option = recommended or {}

        if context.research_object is not None:
            context.research_object["strategic_options"] = options_as_dicts

        # Observability counts
        avg_assumptions = (
            sum(len(o.get("supporting_assumption_ids", [])) for o in options_as_dicts) / len(options_as_dicts)
            if options_as_dicts else 0.0
        )
        avg_risks = (
            sum(len(o.get("associated_risk_ids", [])) for o in options_as_dicts) / len(options_as_dicts)
            if options_as_dicts else 0.0
        )
        avg_opportunities = (
            sum(len(o.get("associated_opportunity_ids", [])) for o in options_as_dicts) / len(options_as_dicts)
            if options_as_dicts else 0.0
        )

        context.trace["_strategic_options"] = {
            "option_count": len(options_as_dicts),
            "recommended_option": recommended.get("option_id") if recommended else None,
            "average_assumptions_per_option": round(avg_assumptions, 2),
            "average_risks_per_option": round(avg_risks, 2),
            "average_opportunities_per_option": round(avg_opportunities, 2),
        }

        # Persist into Decision Model artifact
        dm_id: str | None = (
            context.research_object.get("decision_model_id")
            if context.research_object
            else None
        )
        dm_persisted = False
        if dm_id:
            LOGGER.log(PROGRESS, "[StrategicOptionAgent] persisting %d options → DM %s", len(options_as_dicts), dm_id)
            dm_persisted = _persist_options_to_dm(dm_id, options_as_dicts)
            if dm_persisted:
                LOGGER.log(PROGRESS, "[StrategicOptionAgent] Decision Model written with %d options", len(options_as_dicts))
            else:
                LOGGER.warning("[StrategicOptionAgent] DM persistence failed — options stored in context only")
        else:
            LOGGER.warning("[StrategicOptionAgent] no decision_model_id on RO — skipping DM persistence")

        ro_persisted = _persist_options_to_ro(context.research_object, options_as_dicts)

        context.trace["_strategic_options"]["dm_persisted"] = dm_persisted
        context.trace["_strategic_options"]["ro_persisted"] = ro_persisted

        recommended_id = recommended.get("option_id", "none") if recommended else "none"
        LOGGER.log(
            PROGRESS,
            "[StrategicOptionAgent] %d options generated; recommended=%s dm=%s ro=%s",
            len(options_as_dicts),
            recommended_id,
            dm_persisted,
            ro_persisted,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(options_as_dicts)} strategic options generated; "
                f"recommended={recommended_id}; "
                f"DM persisted={dm_persisted}."
            ),
            option_count=len(options_as_dicts),
            recommended_option_id=recommended_id,
            dm_persisted=dm_persisted,
            ro_persisted=ro_persisted,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_options(
        self,
        *,
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ):
        from research_agent.claude_client import StrategicOptionPayload

        if self._client is None:
            LOGGER.warning("[StrategicOptionAgent] no client — using mock options")
            return _mock_options(assumptions, risks, opportunities, recommendations, evidence_items, decision_model)

        if hasattr(self._client, "generate_strategic_options"):
            return self._client.generate_strategic_options(
                assumptions=assumptions,
                risks=risks,
                opportunities=opportunities,
                recommendations=recommendations,
                evidence_items=evidence_items,
                decision_model=decision_model,
            )

        LOGGER.warning("[StrategicOptionAgent] client lacks generate_strategic_options — using mock")
        return _mock_options(assumptions, risks, opportunities, recommendations, evidence_items, decision_model)


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_options_to_dm(decision_model_id: str, options: list[dict]) -> bool:
    """Load the persisted DecisionModel, inject options, re-write it."""
    try:
        from research_agent.decision_model import (
            StrategicOption, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = [StrategicOption.model_validate(o) for o in options]
        updated = dm.model_copy(update={"strategic_options": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning("[StrategicOptionAgent] could not persist options to DM %s: %s", decision_model_id, exc)
        return False


def _persist_options_to_ro(research_object: dict | None, options: list[dict]) -> bool:
    """Re-persist the Research Object with the options injected."""
    if not research_object:
        return False
    try:
        from research_agent.research_object import write_research_object
        research_object["strategic_options"] = options
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning("[StrategicOptionAgent] could not persist options to RO: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_options(
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import MockClaudeClient
    return MockClaudeClient().generate_strategic_options(
        assumptions=assumptions,
        risks=risks,
        opportunities=opportunities,
        recommendations=recommendations,
        evidence_items=evidence_items,
        decision_model=decision_model,
    )
