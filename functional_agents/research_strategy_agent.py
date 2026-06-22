"""ResearchStrategyAgent – transforms a Decision Model into an executable research plan (J6.2).

Runs between ProblemFramingAgent and PlannerAgent in goal-driven workflows.
Reads context.decision_model, calls Claude to produce a ResearchStrategyPayload,
and writes it to:
  - context.research_strategy
  - context.research_object["research_strategy"]
  - context.trace["_research_strategy"]
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ResearchStrategyAgent(FunctionalAgent):
    """Converts a Decision Model into a prioritised research strategy.

    The ResearchStrategyPayload contains:
      - profile_priorities         : {profile_name: rank} — 1 = highest priority
      - research_question_priorities: [{question, priority}] ordered by decision impact
      - required_evidence          : specific evidence items needed
      - source_priorities          : source types in priority order
      - coverage_targets           : {area: "strong"|"moderate"|"light"}
      - strategy_rationale         : 2-3 sentence explanation
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

        if not context.decision_model:
            LOGGER.warning("[ResearchStrategyAgent] called with empty decision_model — skipping")
            self._record(context, status="warning", summary="No decision model; strategy skipped.")
            return context

        profiles_context = self._build_profiles_context(context)
        strategy = self._generate_strategy(context.decision_model, profiles_context)

        rs_dict = strategy.model_dump()
        context.research_strategy = rs_dict

        if context.research_object:
            context.research_object["research_strategy"] = rs_dict

        context.trace["_research_strategy"] = rs_dict

        LOGGER.log(
            PROGRESS,
            "[ResearchStrategyAgent] profiles=%d  questions=%d  coverage_targets=%d",
            len(strategy.profile_priorities),
            len(strategy.research_question_priorities),
            len(strategy.coverage_targets),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Research strategy generated: {len(strategy.profile_priorities)} profile priorities, "
                f"{len(strategy.research_question_priorities)} question priorities, "
                f"{len(strategy.coverage_targets)} coverage targets."
            ),
            profile_priorities_count=len(strategy.profile_priorities),
            research_question_priorities_count=len(strategy.research_question_priorities),
            required_evidence_count=len(strategy.required_evidence),
            coverage_targets_count=len(strategy.coverage_targets),
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_profiles_context(self, context: AgentContext) -> list[dict]:
        """Build a lightweight profile summary list for the strategy prompt."""
        result: list[dict] = []
        profile_map: dict[str, Any] = {
            p.name: p for p in self._domain_profiles if hasattr(p, "name")
        }
        for name in context.profiles:
            if name in profile_map:
                p = profile_map[name]
                result.append({
                    "name": name,
                    "description": getattr(p, "description", ""),
                    "key_topics": list(getattr(p, "evaluator_topic_terms", {}).keys())[:8],
                })
            else:
                result.append({"name": name, "description": "", "key_topics": []})
        return result

    def _generate_strategy(self, decision_model: dict, profiles_context: list[dict]):
        """Call the LLM client to generate the Research Strategy."""
        from research_agent.claude_client import ResearchStrategyPayload

        if self._client is None:
            LOGGER.warning("[ResearchStrategyAgent] no client provided — using mock strategy")
            profiles = [p.get("name", "") for p in profiles_context if p.get("name")]
            rqs = decision_model.get("research_questions", [])
            areas = decision_model.get("decision_areas", [])
            uncertainties = decision_model.get("critical_uncertainties", [])
            evidence_reqs = decision_model.get("evidence_requirements", [])
            return ResearchStrategyPayload(
                profile_priorities={p: i + 1 for i, p in enumerate(profiles)},
                research_question_priorities=[
                    {"question": q, "priority": i + 1} for i, q in enumerate(rqs)
                ],
                required_evidence=evidence_reqs or [
                    "Primary data sources",
                    "Expert assessments",
                    "Quantitative benchmarks",
                ],
                source_priorities=["primary research", "industry reports", "expert analysis", "case studies"],
                coverage_targets={
                    **{area: "strong" for area in areas[:2]},
                    **{area: "moderate" for area in areas[2:]},
                    **{u: "strong" for u in uncertainties[:1]},
                },
                strategy_rationale="Mock strategy: profiles ranked by order, questions ranked by position.",
            )

        if hasattr(self._client, "generate_research_strategy"):
            return self._client.generate_research_strategy(decision_model, profiles_context)

        LOGGER.warning("[ResearchStrategyAgent] client does not support generate_research_strategy — using mock")
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().generate_research_strategy(decision_model, profiles_context)
