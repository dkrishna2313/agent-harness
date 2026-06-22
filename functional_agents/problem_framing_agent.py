"""ProblemFramingAgent – transforms a business goal into a Decision Model (J6.1).

Runs before PlannerAgent in goal-driven workflows.  Reads context.goal, calls
Claude to produce a structured Decision Model, and writes it to:
  - context.decision_model
  - context.research_object["decision_model"]
  - context.question  (first research question, used by all downstream agents)
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

        # Store the full decision model
        dm_dict = decision_model.model_dump()
        context.decision_model = dm_dict

        # Write into Research Object
        if context.research_object:
            context.research_object["decision_model"] = dm_dict
            context.research_object["goal"] = context.goal

        # Stash in trace so ReportAgent can emit the problem_framing block
        context.trace["_problem_framing"] = dm_dict

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
        )
        return context

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
