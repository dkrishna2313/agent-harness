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
        from research_agent.schemas import ClaudeCallTrace

        LOGGER.log(PROGRESS, "[%s] starting", self.name)

        # Snap call_traces index so we can slice per-agent LLM calls after _execute()
        client = context.trace.get("_client")
        traces_start = len(client.call_traces) if client is not None else 0

        t0 = time.monotonic()
        context = self._execute(context)
        wall_ms = (time.monotonic() - t0) * 1000
        duration = round(wall_ms / 1000, 3)

        LOGGER.log(PROGRESS, "[%s] completed in %.3fs", self.name, duration)

        # Record per-agent performance if tracker is present
        tracker = context.trace.get("_perf_tracker")
        if tracker is not None and client is not None:
            from .performance import AgentPerfRecord, LLMCallRecord
            agent_traces: list[ClaudeCallTrace] = client.call_traces[traces_start:]
            llm_calls = [
                LLMCallRecord(
                    operation=t.operation,
                    model=t.model_name,
                    duration_ms=t.duration_ms,
                    prompt_tokens=t.token_usage.get("input_tokens", 0),
                    completion_tokens=t.token_usage.get("output_tokens", 0),
                    total_tokens=t.token_usage.get("input_tokens", 0) + t.token_usage.get("output_tokens", 0),
                    success=t.success,
                    error=t.error,
                )
                for t in agent_traces
            ]
            sub_phases = tracker.flush_sub_phases()
            rec = AgentPerfRecord(
                agent_name=self.name,
                wall_ms=wall_ms,
                llm_calls=llm_calls,
                sub_phases=sub_phases,
            )
            tracker.record(rec)

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
