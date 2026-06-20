"""ReportAgent – writes the Markdown memo and trace (J5.0b)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ReportAgent(FunctionalAgent):
    """Writes the Markdown output and JSON trace. Surfaces agent_history in both."""

    def __init__(self, *, out_path: Path, domain_profile: Any = None) -> None:
        self._out_path = out_path
        self._domain_profile = domain_profile

    def _execute(self, context: AgentContext) -> AgentContext:
        from research_agent.markdown import memo_to_markdown, write_markdown
        from research_agent.trace import build_trace, write_trace

        memo = context.trace.get("_memo")
        documents = context.trace.get("_documents", [])

        if memo is None:
            LOGGER.error("ReportAgent: no memo on context — cannot write report")
            self._record(context, status="error", summary="No memo available; report not written.")
            return context

        output_path = write_markdown(memo_to_markdown(memo), self._out_path)
        context.artifacts["report_path"] = str(output_path)
        context.artifacts["trace_path"] = str(output_path.with_suffix(".trace.json"))

        # Record ReportAgent in history before building trace so it appears in agents_run
        self._record(
            context,
            status="success",
            summary=f"Report written to {output_path}",
            report_path=str(output_path),
        )

        # Build standard trace then inject functional_agents block (J5.0b.5)
        trace_payload = build_trace(
            question=context.question,
            source_directory=Path("sources"),
            output_path=output_path,
            documents=documents,
            memo=memo,
            mock_mode=False,
            profile=self._domain_profile,
        )
        trace_payload["functional_agents"] = context.to_functional_trace()

        # Update Research Object and surface agent_history in it (J5.0b.4)
        if context.research_object:
            from research_agent.research_object import (
                research_object_trace_stub,
                update_research_object,
                write_research_object,
            )

            ro = update_research_object(
                context.research_object,
                memo=memo,
                output_path=output_path,
                trace_path=context.artifacts["trace_path"],
            )
            # Inject agent_history into the research object outputs
            ro.setdefault("outputs", {})["agent_history"] = context.agent_history

            ro_path = write_research_object(ro, out_dir=output_path.parent)
            trace_payload["research_object"] = research_object_trace_stub(ro, ro_path)
            context.research_object = ro
            context.artifacts["research_object_path"] = str(ro_path)

        write_trace(trace_payload, output_path)
        return context
