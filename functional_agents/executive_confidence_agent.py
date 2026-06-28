"""ExecutiveConfidenceAgent – synthesis over the completed J7 Decision Graph (J7.7).

Pipeline position: after DecisionAnalysisAgent, before MULTI_PROFILE / SCENARIO / QA.

Reads:
  - context.decision_analysis        (from DecisionAnalysisAgent)
  - context.strategic_options        (from StrategicOptionAgent)
  - context.assumptions              (from AssumptionAgent)
  - context.risks                    (from RiskAgent)
  - context.opportunities            (from OpportunityAgent)
  - context.recommendations          (from RecommendationAgent)
  - context.scenarios                (from ScenarioAgent — empty when not yet run)
  - context.decision_model

Writes:
  - context.executive_confidence                          (dict)
  - context.research_object["executive_confidence"]
  - context.trace["_executive_confidence"]

Also persists ExecutiveConfidence into the Decision Model artifact via
research_agent.decision_model.write_decision_model().

ExecutiveConfidence is evaluative, not generative.
It answers "Should an executive approve this recommendation today?"
by synthesising the existing graph. It does NOT generate new options,
evidence, risks, or recommendations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ExecutiveConfidenceAgent(FunctionalAgent):
    """Produces an executive confidence assessment from the full J7 graph.

    ExecutiveConfidence:
      - rates overall confidence (High / Medium / Low)
      - determines decision readiness
      - issues a board recommendation
      - produces a validation-priorities checklist (due diligence)
      - identifies critical unknowns
      - provides conditional confidence (if assumptions hold / fail)
    It reasons over existing objects only — nothing is generated from scratch.
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

        decision_analysis: dict = context.decision_analysis
        strategic_options: list[dict] = context.strategic_options

        if not strategic_options:
            LOGGER.warning(
                "[ExecutiveConfidenceAgent] no strategic options — skipping executive confidence"
            )
            context.trace["_executive_confidence"] = {
                "skipped": True,
                "reason": "no_strategic_options",
            }
            self._record(
                context, status="skipped",
                summary="No strategic options — executive confidence skipped.",
            )
            return context

        if not decision_analysis:
            LOGGER.warning(
                "[ExecutiveConfidenceAgent] no decision analysis — skipping executive confidence"
            )
            context.trace["_executive_confidence"] = {
                "skipped": True,
                "reason": "no_decision_analysis",
            }
            self._record(
                context, status="skipped",
                summary="No decision analysis — executive confidence skipped.",
            )
            return context

        payload = self._generate_confidence(
            decision_analysis=decision_analysis,
            strategic_options=strategic_options,
            assumptions=context.assumptions,
            risks=context.risks,
            opportunities=context.opportunities,
            recommendations=context.recommendations,
            scenarios=context.scenarios,
            decision_model=context.decision_model,
        )

        confidence = payload.confidence
        # Stamp last_updated
        confidence_dict = confidence.model_dump()
        confidence_dict["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Store in context
        context.executive_confidence = confidence_dict

        if context.research_object is not None:
            context.research_object["executive_confidence"] = confidence_dict

        # Persist into Decision Model artifact
        dm_id: str | None = (
            context.research_object.get("decision_model_id")
            if context.research_object
            else None
        )
        dm_persisted = False
        if dm_id:
            LOGGER.log(
                PROGRESS,
                "[ExecutiveConfidenceAgent] persisting executive confidence → DM %s",
                dm_id,
            )
            dm_persisted = _persist_confidence_to_dm(dm_id, confidence_dict)
            if dm_persisted:
                LOGGER.log(
                    PROGRESS,
                    "[ExecutiveConfidenceAgent] Decision Model written with executive confidence",
                )
            else:
                LOGGER.warning(
                    "[ExecutiveConfidenceAgent] DM persistence failed — confidence stored in context only"
                )
        else:
            LOGGER.warning(
                "[ExecutiveConfidenceAgent] no decision_model_id on RO — skipping DM persistence"
            )

        ro_persisted = _persist_confidence_to_ro(context.research_object, confidence_dict)

        vp_count = len(confidence.validation_priorities)
        cu_count = len(confidence.critical_unknowns)

        context.trace["_executive_confidence"] = {
            "overall_confidence": confidence.overall_confidence,
            "decision_readiness": confidence.decision_readiness,
            "board_recommendation": confidence.board_recommendation,
            "critical_unknown_count": cu_count,
            "validation_priority_count": vp_count,
            "persisted": dm_persisted,
        }

        LOGGER.log(
            PROGRESS,
            "[ExecutiveConfidenceAgent] complete; confidence=%s readiness=%s "
            "board=%s unknowns=%d priorities=%d dm=%s ro=%s",
            confidence.overall_confidence,
            confidence.decision_readiness,
            confidence.board_recommendation,
            cu_count,
            vp_count,
            dm_persisted,
            ro_persisted,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Executive confidence complete; overall={confidence.overall_confidence}; "
                f"readiness={confidence.decision_readiness}; "
                f"board={confidence.board_recommendation}; "
                f"{cu_count} critical unknowns; {vp_count} validation priorities; "
                f"DM persisted={dm_persisted}."
            ),
            overall_confidence=confidence.overall_confidence,
            decision_readiness=confidence.decision_readiness,
            board_recommendation=confidence.board_recommendation,
            critical_unknown_count=cu_count,
            validation_priority_count=vp_count,
            dm_persisted=dm_persisted,
            ro_persisted=ro_persisted,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_confidence(
        self,
        *,
        decision_analysis: dict,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        scenarios: list[dict],
        decision_model: dict,
    ):
        if self._client is None:
            LOGGER.warning("[ExecutiveConfidenceAgent] no client — using mock confidence")
            return _mock_confidence(
                decision_analysis, strategic_options, assumptions,
                risks, opportunities, recommendations, scenarios, decision_model,
            )

        if hasattr(self._client, "generate_executive_confidence"):
            return self._client.generate_executive_confidence(
                decision_analysis=decision_analysis,
                strategic_options=strategic_options,
                assumptions=assumptions,
                risks=risks,
                opportunities=opportunities,
                recommendations=recommendations,
                scenarios=scenarios,
                decision_model=decision_model,
            )

        LOGGER.warning(
            "[ExecutiveConfidenceAgent] client lacks generate_executive_confidence — using mock"
        )
        return _mock_confidence(
            decision_analysis, strategic_options, assumptions,
            risks, opportunities, recommendations, scenarios, decision_model,
        )


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_confidence_to_dm(decision_model_id: str, confidence: dict) -> bool:
    """Load the persisted DecisionModel, inject executive_confidence, re-write it."""
    try:
        from research_agent.decision_model import (
            ExecutiveConfidence, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = ExecutiveConfidence.model_validate(confidence)
        updated = dm.model_copy(update={"executive_confidence": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning(
            "[ExecutiveConfidenceAgent] could not persist confidence to DM %s: %s — %s",
            decision_model_id, type(exc).__name__, exc,
        )
        return False


def _persist_confidence_to_ro(research_object: dict | None, confidence: dict) -> bool:
    """Re-persist the Research Object with executive confidence injected."""
    if not research_object:
        return False
    try:
        from research_agent.research_object import write_research_object
        research_object["executive_confidence"] = confidence
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning(
            "[ExecutiveConfidenceAgent] could not persist confidence to RO: %s", exc
        )
        return False


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_confidence(
    decision_analysis: dict,
    strategic_options: list[dict],
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    scenarios: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import MockClaudeClient
    return MockClaudeClient().generate_executive_confidence(
        decision_analysis=decision_analysis,
        strategic_options=strategic_options,
        assumptions=assumptions,
        risks=risks,
        opportunities=opportunities,
        recommendations=recommendations,
        scenarios=scenarios,
        decision_model=decision_model,
    )
