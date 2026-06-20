"""PlannerAgent – creates the initial research plan (J5.0a.4)."""

from __future__ import annotations

from .base import FunctionalAgent
from .context import AgentContext


class PlannerAgent(FunctionalAgent):
    """Skeleton planner: records profiles and question into a plan."""

    def _execute(self, ctx: AgentContext) -> AgentContext:
        ctx.plan = {
            "question": ctx.question,
            "execution_profile": ctx.execution_profile,
            "supporting_profiles": ctx.profiles[1:],
            "strategy": "single-pass retrieval and synthesis via research_agent engine",
        }
        note = self._make_note(
            status="completed",
            summary="Created initial functional-agent plan.",
            plan=ctx.plan,
        )
        ctx.record_agent(note)
        return ctx
