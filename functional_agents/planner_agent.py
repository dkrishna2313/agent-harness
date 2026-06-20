"""PlannerAgent – creates the initial research plan (J5.0b)."""

from __future__ import annotations

from .base import FunctionalAgent
from .context import AgentContext


class PlannerAgent(FunctionalAgent):
    """Skeleton planner: records profiles and question into a plan."""

    def _execute(self, context: AgentContext) -> AgentContext:
        context.plan = {
            "question": context.question,
            "execution_profile": context.execution_profile,
            "supporting_profiles": context.profiles[1:],
            "strategy": "single-pass retrieval and synthesis via research_agent engine",
        }
        self._record(
            context,
            status="success",
            summary="Generated initial research plan.",
            execution_profile=context.execution_profile,
            supporting_profiles=context.profiles[1:],
        )
        return context
