"""Base class for functional agents (J5.0b.6)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class FunctionalAgent(ABC):
    """Base class for all functional agents.

    Standardized interface (J5.0b.6):
        run(context: AgentContext) -> AgentContext

    Each agent:
      - receives the shared AgentContext
      - performs its function
      - calls _record(ctx, status, summary) to append to agent_history
      - returns the updated context
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, context: AgentContext) -> AgentContext:
        """Standardized entry point — all agents use this exact signature."""
        from research_agent.log import PROGRESS
        LOGGER.log(PROGRESS, "[%s] starting", self.name)
        context = self._execute(context)
        LOGGER.log(PROGRESS, "[%s] completed", self.name)
        return context

    @abstractmethod
    def _execute(self, context: AgentContext) -> AgentContext:
        """Implement agent logic. Must return the updated context."""

    def _record(
        self,
        ctx: AgentContext,
        status: str,
        summary: str,
        **kwargs: Any,
    ) -> None:
        """Append a structured entry to ctx.agent_history (J5.0b.3)."""
        ctx.append_history(
            {"agent": self.name, "status": status, "summary": summary, **kwargs}
        )

    def _make_note(self, status: str, summary: str, **kwargs: Any) -> dict[str, Any]:
        """Build a note dict without appending it (for per-agent detail lists)."""
        return {"agent": self.name, "status": status, "summary": summary, **kwargs}
