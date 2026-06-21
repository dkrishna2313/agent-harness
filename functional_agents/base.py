"""Base class for functional agents (J5.0b.6 / J5.5a)."""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from .context import AgentContext, AgentResult, NextAction

LOGGER = logging.getLogger(__name__)


class FunctionalAgent(ABC):
    """Base class for all functional agents.

    Standardized contract (J5.5a):
        run(context: AgentContext) -> AgentResult

    Each agent:
      - receives the shared AgentContext
      - calls _execute() to perform its work (returns updated context)
      - run() wraps the result in AgentResult with timing metrics and trace block
      - calls _record(ctx, ...) inside _execute() to append to agent_history
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, context: AgentContext) -> AgentResult:
        """Standardized entry point — returns AgentResult with outputs/metrics/trace."""
        from research_agent.log import PROGRESS
        LOGGER.log(PROGRESS, "[%s] starting", self.name)
        t0 = time.monotonic()
        context = self._execute(context)
        duration = round(time.monotonic() - t0, 3)
        LOGGER.log(PROGRESS, "[%s] completed in %.3fs", self.name, duration)

        last = context.agent_history[-1] if context.agent_history else {}
        status = last.get("status", "success")
        next_action = last.get("next_action", NextAction.CONTINUE)
        summary = last.get("summary", "")

        return AgentResult(
            status=status,
            next_action=next_action,
            summary=summary,
            context=context,
            outputs=self._extract_outputs(context),
            metrics={"duration_seconds": duration},
            trace={
                "agent": self.name,
                "run_id": context.run_id,
                "duration_seconds": duration,
                "status": status,
            },
        )

    @abstractmethod
    def _execute(self, context: AgentContext) -> AgentContext:
        """Implement agent logic. Must return the updated context."""

    def _extract_outputs(self, context: AgentContext) -> dict[str, Any]:
        """Return agent-specific output data. Subclasses may override."""
        return {}

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
