"""Base class for functional agents (J5.0a.4)."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class FunctionalAgent(ABC):
    """Base class for all functional agents.

    Each agent:
      - accepts an AgentContext
      - performs its function
      - appends a structured note
      - returns the updated context
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def run(self, ctx: AgentContext) -> AgentContext:
        LOGGER.info("[%s] starting", self.name)
        ctx = self._execute(ctx)
        LOGGER.info("[%s] completed", self.name)
        return ctx

    @abstractmethod
    def _execute(self, ctx: AgentContext) -> AgentContext:
        """Implement agent logic. Must return the updated context."""

    def _make_note(self, status: str, summary: str, **kwargs: Any) -> dict[str, Any]:
        return {"agent": self.name, "status": status, "summary": summary, **kwargs}
