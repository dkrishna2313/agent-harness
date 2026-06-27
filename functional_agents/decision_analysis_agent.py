"""DecisionAnalysisAgent – explicit comparison of Strategic Options using the J7 graph (J7.6).

Pipeline position: after StrategicOptionAgent, before MULTI_PROFILE / SCENARIO / QA.

Reads:
  - context.strategic_options       (from StrategicOptionAgent)
  - context.assumptions              (from AssumptionAgent)
  - context.risks                    (from RiskAgent)
  - context.opportunities            (from OpportunityAgent)
  - context.recommendations          (from RecommendationAgent)
  - context.decision_model
  - context.research_object

Writes:
  - context.decision_analysis                         (dict)
  - context.research_object["decision_analysis"]
  - context.trace["_decision_analysis"]

Also persists DecisionAnalysis into the Decision Model artifact via
research_agent.decision_model.write_decision_model().

DecisionAnalysis is an explanation, not a recommendation generator.
It answers 'Why is Option B preferred over Option A?' using the existing graph.
It does NOT generate new options, evidence, or scenarios.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class DecisionAnalysisAgent(FunctionalAgent):
    """Produces an explicit decision analysis from the full J7 graph.

    DecisionAnalysis:
      - rates each Strategic Option across 10 decision dimensions
      - ranks all options explicitly
      - surfaces key tradeoffs as first-class objects
      - explains sensitivity to assumption failures
      - justifies the preferred option against each alternative
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

        strategic_options: list[dict] = context.strategic_options
        assumptions: list[dict] = context.assumptions
        risks: list[dict] = context.risks
        opportunities: list[dict] = context.opportunities
        recommendations: list[dict] = context.recommendations

        if not strategic_options:
            LOGGER.warning("[DecisionAnalysisAgent] no strategic options — skipping decision analysis")
            context.trace["_decision_analysis"] = {
                "skipped": True,
                "reason": "no_strategic_options",
            }
            self._record(context, status="skipped", summary="No strategic options — decision analysis skipped.")
            return context

        payload = self._generate_analysis(
            strategic_options=strategic_options,
            assumptions=assumptions,
            risks=risks,
            opportunities=opportunities,
            recommendations=recommendations,
            decision_model=context.decision_model,
        )

        analysis = payload.analysis
        analysis_dict = analysis.model_dump()

        # Store in context
        context.decision_analysis = analysis_dict

        if context.research_object is not None:
            context.research_object["decision_analysis"] = analysis_dict

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
                "[DecisionAnalysisAgent] persisting decision analysis → DM %s",
                dm_id,
            )
            dm_persisted = _persist_analysis_to_dm(dm_id, analysis_dict)
            if dm_persisted:
                LOGGER.log(PROGRESS, "[DecisionAnalysisAgent] Decision Model written with decision analysis")
            else:
                LOGGER.warning("[DecisionAnalysisAgent] DM persistence failed — analysis stored in context only")
        else:
            LOGGER.warning("[DecisionAnalysisAgent] no decision_model_id on RO — skipping DM persistence")

        ro_persisted = _persist_analysis_to_ro(context.research_object, analysis_dict)

        option_count = len(strategic_options)
        tradeoff_count = len(analysis.key_tradeoffs)
        dimensions = analysis.comparison_dimensions

        context.trace["_decision_analysis"] = {
            "option_count": option_count,
            "comparison_dimensions": dimensions,
            "tradeoff_count": tradeoff_count,
            "recommended_option": analysis.recommended_option_id,
            "decision_persisted": dm_persisted,
            "analysis_persisted": ro_persisted,
        }

        LOGGER.log(
            PROGRESS,
            "[DecisionAnalysisAgent] analysis complete; options=%d tradeoffs=%d "
            "recommended=%s dm=%s ro=%s",
            option_count,
            tradeoff_count,
            analysis.recommended_option_id,
            dm_persisted,
            ro_persisted,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Decision analysis complete; {option_count} options compared across "
                f"{len(dimensions)} dimensions; {tradeoff_count} explicit tradeoffs; "
                f"recommended={analysis.recommended_option_id}; DM persisted={dm_persisted}."
            ),
            option_count=option_count,
            tradeoff_count=tradeoff_count,
            recommended_option_id=analysis.recommended_option_id,
            dm_persisted=dm_persisted,
            ro_persisted=ro_persisted,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_analysis(
        self,
        *,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        decision_model: dict,
    ):
        if self._client is None:
            LOGGER.warning("[DecisionAnalysisAgent] no client — using mock analysis")
            return _mock_analysis(strategic_options, assumptions, risks, opportunities, recommendations, decision_model)

        if hasattr(self._client, "generate_decision_analysis"):
            return self._client.generate_decision_analysis(
                strategic_options=strategic_options,
                assumptions=assumptions,
                risks=risks,
                opportunities=opportunities,
                recommendations=recommendations,
                decision_model=decision_model,
            )

        LOGGER.warning("[DecisionAnalysisAgent] client lacks generate_decision_analysis — using mock")
        return _mock_analysis(strategic_options, assumptions, risks, opportunities, recommendations, decision_model)


# ---------------------------------------------------------------------------
# Decision Model persistence
# ---------------------------------------------------------------------------

def _persist_analysis_to_dm(decision_model_id: str, analysis: dict) -> bool:
    """Load the persisted DecisionModel, inject decision_analysis, re-write it."""
    try:
        from research_agent.decision_model import (
            DecisionAnalysis, load_decision_model, write_decision_model,
        )
        dm = load_decision_model(decision_model_id)
        parsed = DecisionAnalysis.model_validate(analysis)
        updated = dm.model_copy(update={"decision_analysis": parsed})
        write_decision_model(updated, write_latest=True)
        return True
    except Exception as exc:
        LOGGER.warning(
            "[DecisionAnalysisAgent] could not persist analysis to DM %s: %s — %s",
            decision_model_id, type(exc).__name__, exc,
        )
        return False


def _persist_analysis_to_ro(research_object: dict | None, analysis: dict) -> bool:
    """Re-persist the Research Object with the decision analysis injected."""
    if not research_object:
        return False
    try:
        from research_agent.research_object import write_research_object
        research_object["decision_analysis"] = analysis
        write_research_object(research_object)
        return True
    except Exception as exc:
        LOGGER.warning("[DecisionAnalysisAgent] could not persist analysis to RO: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Mock fallback
# ---------------------------------------------------------------------------

def _mock_analysis(
    strategic_options: list[dict],
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    decision_model: dict,
):
    from research_agent.claude_client import MockClaudeClient
    return MockClaudeClient().generate_decision_analysis(
        strategic_options=strategic_options,
        assumptions=assumptions,
        risks=risks,
        opportunities=opportunities,
        recommendations=recommendations,
        decision_model=decision_model,
    )
