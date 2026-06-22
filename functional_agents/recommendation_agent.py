"""RecommendationAgent – transforms challenged hypotheses into actionable recommendations (J6.5).

Runs between ChallengeAgent and QAAgent. Reads hypotheses, challenge results,
and surviving hypotheses, then generates 3-5 recommendations that are:
  - grounded in surviving hypotheses
  - linked to specific evidence IDs
  - classified by time horizon (near_term / medium_term / long_term)
  - accompanied by key risks and trigger conditions
  - confidence-rated based on hypothesis robustness

Writes to:
  - context.recommendations               (list of recommendation dicts)
  - context.recommendation_portfolio      (dict grouped by time horizon)
  - context.research_object["recommendations"]
  - context.research_object["recommendation_portfolio"]
  - context.trace["_recommendations"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class RecommendationAgent(FunctionalAgent):
    """Derives actionable recommendations from challenged and surviving hypotheses.

    Each RecommendationItem contains:
      - id, title, summary
      - priority               : "high" | "medium" | "low"
      - time_horizon           : "near_term" | "medium_term" | "long_term"
      - supported_by_hypotheses: hypothesis IDs that justify this recommendation
      - supporting_evidence    : evidence IDs that ground the recommendation
      - key_risks              : risks that could undermine it
      - trigger_conditions     : future events that change or activate it
      - confidence             : "high" | "medium" | "low"
      - confidence_rationale   : explanation of confidence level

    recommendation_portfolio groups recommendation IDs by time horizon.
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

        hypotheses: list[dict] = context.hypotheses
        if not hypotheses:
            LOGGER.warning("[RecommendationAgent] no hypotheses available — skipping")
            self._record(context, status="warning", summary="No hypotheses available for recommendations.")
            return context

        evidence_note = context.evidence_notes[0] if context.evidence_notes else {}
        evidence_items: list[dict] = evidence_note.get("evidence_items", [])

        rec_payload = self._generate_recommendations(
            hypotheses=hypotheses,
            surviving_hypotheses=context.surviving_hypotheses,
            hypothesis_challenges=context.hypothesis_challenges,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
            research_strategy=context.research_strategy,
        )

        recs_as_dicts = [r.model_dump() for r in rec_payload.recommendations]
        portfolio_as_dict = rec_payload.recommendation_portfolio.model_dump()

        context.recommendations = recs_as_dicts
        context.recommendation_portfolio = portfolio_as_dict

        if context.research_object:
            context.research_object["recommendations"] = recs_as_dicts
            context.research_object["recommendation_portfolio"] = portfolio_as_dict

        context.trace["_recommendations"] = {
            "recommendations": recs_as_dicts,
            "recommendation_portfolio": portfolio_as_dict,
            "synthesis_note": rec_payload.synthesis_note,
        }

        high_count = sum(1 for r in recs_as_dicts if r.get("priority") == "high")
        near_count = len(portfolio_as_dict.get("near_term", []))
        mid_count = len(portfolio_as_dict.get("medium_term", []))
        lng_count = len(portfolio_as_dict.get("long_term", []))

        LOGGER.log(
            PROGRESS,
            "[RecommendationAgent] recommendations=%d  high_priority=%d  "
            "portfolio: near=%d mid=%d long=%d",
            len(recs_as_dicts), high_count, near_count, mid_count, lng_count,
        )

        self._record(
            context,
            status="success",
            summary=(
                f"{len(recs_as_dicts)} recommendations generated. "
                f"high_priority={high_count}. "
                + rec_payload.synthesis_note[:100]
            ),
            recommendation_count=len(recs_as_dicts),
            high_priority=high_count,
            near_term_count=near_count,
            medium_term_count=mid_count,
            long_term_count=lng_count,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        hypotheses: list[dict],
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
    ):
        """Call the LLM client to generate recommendations."""
        if self._client is None:
            LOGGER.warning("[RecommendationAgent] no client — using mock recommendations")
            return self._mock_recommendations(hypotheses, surviving_hypotheses, hypothesis_challenges, evidence_items, decision_model)

        if hasattr(self._client, "generate_recommendations"):
            return self._client.generate_recommendations(
                hypotheses, surviving_hypotheses, hypothesis_challenges,
                evidence_items, decision_model, research_strategy,
            )

        LOGGER.warning("[RecommendationAgent] client does not support generate_recommendations — using mock")
        return self._mock_recommendations(hypotheses, surviving_hypotheses, hypothesis_challenges, evidence_items, decision_model)

    def _mock_recommendations(
        self,
        hypotheses: list[dict],
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ):
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().generate_recommendations(
            hypotheses=hypotheses,
            surviving_hypotheses=surviving_hypotheses,
            hypothesis_challenges=hypothesis_challenges,
            evidence_items=evidence_items,
            decision_model=decision_model,
            research_strategy={},
        )
