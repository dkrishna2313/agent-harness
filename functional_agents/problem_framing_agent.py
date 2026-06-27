"""ProblemFramingAgent – transforms a business goal into a Decision Model (J6.1 / J7.0b).

Runs before PlannerAgent in goal-driven workflows.  Reads context.goal, calls
Claude to produce a structured Decision Model, and writes it to:
  - context.decision_model        (v2 dict — superset of old v1 fields)
  - context.research_object["decision_model"]
  - context.research_object["decision_model_id"]
  - context.question  (first research question, used by all downstream agents)

J7.0b: also produces and persists a DecisionModel v2 object, links it to the
engagement stored in context.trace["_engagement_id"] (when available).
"""

from __future__ import annotations

import logging
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ProblemFramingAgent(FunctionalAgent):
    """Converts a high-level business goal into a structured Decision Model.

    The Decision Model contains:
      - objective              : precise research objective
      - decision_areas         : key dimensions to investigate
      - critical_uncertainties : unknowns that most affect the decision
      - research_questions     : specific answerable questions
      - evidence_requirements  : types of evidence needed

    The first research question is written to context.question so that
    PlannerAgent, EvidenceAgent, QAAgent, and ReportAgent work unchanged.
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

        if not context.goal:
            LOGGER.warning("[ProblemFramingAgent] called with empty goal — skipping")
            self._record(context, status="warning", summary="No goal provided; framing skipped.")
            return context

        profiles_context = self._build_profiles_context(context)
        decision_model = self._generate_decision_model(context.goal, profiles_context)

        # J7.0b – build and persist Decision Model v2
        engagement_id: str | None = context.trace.get("_engagement_id")
        dm_v2 = self._build_decision_model_v2(
            decision_model,
            goal=context.goal,
            engagement_id=engagement_id,
        )

        # Store the full decision model (v2 dict is a superset of v1 fields)
        dm_dict = decision_model.model_dump()
        context.decision_model = dm_dict

        # Write into Research Object
        if context.research_object:
            context.research_object["decision_model"] = dm_dict
            context.research_object["goal"] = context.goal
            context.research_object["decision_model_id"] = dm_v2.decision_model_id

        # Stash in trace so ReportAgent can emit the problem_framing block
        context.trace["_problem_framing"] = dm_dict
        # J7.0b – surface decision_model_id for downstream linkage
        context.trace["_decision_model_id"] = dm_v2.decision_model_id

        # Populate question from the first research question (enables downstream agents)
        if decision_model.research_questions and not context.question.strip():
            context.question = decision_model.research_questions[0]
            if context.research_object:
                context.research_object["question"] = context.question
            LOGGER.log(
                PROGRESS,
                "[ProblemFramingAgent] primary question set: %s",
                context.question[:80],
            )

        LOGGER.log(
            PROGRESS,
            "[ProblemFramingAgent] goal=%r  decision_areas=%d  research_questions=%d",
            context.goal[:60],
            len(decision_model.decision_areas),
            len(decision_model.research_questions),
        )

        self._record(
            context,
            status="success",
            summary=(
                f"Decision model generated: {len(decision_model.decision_areas)} decision areas, "
                f"{len(decision_model.research_questions)} research questions."
            ),
            decision_areas_count=len(decision_model.decision_areas),
            critical_uncertainties_count=len(decision_model.critical_uncertainties),
            research_questions_count=len(decision_model.research_questions),
            evidence_requirements_count=len(decision_model.evidence_requirements),
            decision_model_id=dm_v2.decision_model_id,
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_decision_model_v2(self, payload: Any, *, goal: str, engagement_id: str | None):
        """Produce, persist, and optionally engagement-link a DecisionModel v2."""
        from research_agent.decision_model import from_framing_payload, write_decision_model
        dm_v2 = from_framing_payload(payload, strategic_question=goal, engagement_id=engagement_id)
        try:
            write_decision_model(dm_v2)
        except Exception:
            pass  # persistence failure must never block a research run
        if engagement_id:
            try:
                from research_agent.engagement import load_engagement, link_decision_model
                eng = load_engagement(engagement_id)
                link_decision_model(eng, dm_v2.decision_model_id)
            except Exception:
                pass
        return dm_v2

    def _build_profiles_context(self, context: AgentContext) -> list[dict]:
        """Build a lightweight profile summary list for the framing prompt."""
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

    def _generate_decision_model(self, goal: str, profiles_context: list[dict]):
        """Call the LLM client to generate the Decision Model."""
        from research_agent.claude_client import DecisionModelPayload

        if self._client is None:
            LOGGER.warning("[ProblemFramingAgent] no client provided — using mock decision model")
            return DecisionModelPayload(
                objective=f"Research and analyse: {goal}",
                decision_areas=["Context", "Key Factors", "Options", "Risks"],
                critical_uncertainties=["Data availability", "Scope boundaries"],
                research_questions=[
                    f"What is the current state of: {goal}?",
                    "What are the main constraints and opportunities?",
                    "What evidence exists on outcomes and trade-offs?",
                ],
                evidence_requirements=["Primary sources", "Expert assessments", "Case studies"],
            )

        if hasattr(self._client, "frame_problem"):
            return self._client.frame_problem(goal, profiles_context)

        LOGGER.warning("[ProblemFramingAgent] client does not support frame_problem — using mock")
        from research_agent.claude_client import MockClaudeClient
        return MockClaudeClient().frame_problem(goal, profiles_context)
