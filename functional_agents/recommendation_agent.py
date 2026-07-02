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

        # J6.5a — prefer validated (post-suppression) contradictions when available
        validated_contradictions: list[dict] = (
            context.validated_contradictions
            or context.research_object.get("validated_contradictions", [])
            if context.research_object else context.validated_contradictions
        )

        # J10.8 — Strategic Synthesis (J10.7) shapes recommendation reasoning and
        # prioritisation when present; absent → legacy hypothesis-driven behaviour.
        strategic_synthesis: dict = context.strategic_synthesis or {}

        rec_payload = self._generate_recommendations(
            hypotheses=hypotheses,
            surviving_hypotheses=context.surviving_hypotheses,
            hypothesis_challenges=context.hypothesis_challenges,
            evidence_items=evidence_items,
            decision_model=context.decision_model,
            research_strategy=context.research_strategy,
            validated_contradictions=validated_contradictions,
            strategic_synthesis=strategic_synthesis,
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

        # J10.8 — additive diagnostics: how much Strategic Synthesis context was
        # available and used (capped identically to the prompt). Trace-only; the
        # recommendation schema is unchanged.
        _cap = 5
        def _used(key: str) -> int:
            return min(len(strategic_synthesis.get(key) or []), _cap)
        synthesis_available = bool(strategic_synthesis)
        context.trace["_recommendation_strategy_context"] = {
            "strategic_synthesis_available": synthesis_available,
            "strategic_synthesis_used": synthesis_available,
            "cross_domain_findings_used": _used("cross_domain_findings"),
            "dependencies_used": _used("cross_domain_dependencies"),
            "conflicts_used": _used("cross_domain_conflicts"),
            "strategic_levers_used": _used("strategic_levers"),
            "dominant_constraints_used": _used("dominant_constraints"),
            "emerging_themes_used": _used("emerging_themes"),
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
        validated_contradictions: list[dict] | None = None,
        strategic_synthesis: dict | None = None,
    ):
        """Call the LLM client to generate recommendations (J10.8: + synthesis)."""
        if self._client is None:
            LOGGER.warning("[RecommendationAgent] no client — using mock recommendations")
            return self._mock_recommendations(hypotheses, surviving_hypotheses, hypothesis_challenges, evidence_items, decision_model)

        if hasattr(self._client, "generate_recommendations"):
            return self._client.generate_recommendations(
                hypotheses, surviving_hypotheses, hypothesis_challenges,
                evidence_items, decision_model, research_strategy,
                validated_contradictions=validated_contradictions or [],
                strategic_synthesis=strategic_synthesis or {},
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
