"""ReportAgent – writes the Markdown memo and trace (J5.0a.4/8)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class ReportAgent(FunctionalAgent):
    """Writes the Markdown output and JSON trace from the memo on context."""

    def __init__(self, *, out_path: Path, domain_profile: Any = None) -> None:
        self._out_path = out_path
        self._domain_profile = domain_profile

    def _execute(self, ctx: AgentContext) -> AgentContext:
        from research_agent.markdown import memo_to_markdown, write_markdown
        from research_agent.trace import build_trace, write_trace

        memo = ctx.trace.get("_memo")
        documents = ctx.trace.get("_documents", [])

        if memo is None:
            LOGGER.error("ReportAgent: no memo on context — cannot write report")
            ctx.record_agent({"agent": self.name})
            return ctx

        output_path = write_markdown(memo_to_markdown(memo), self._out_path)
        ctx.report_path = str(output_path)

        # Build standard trace then inject functional_agents block
        trace_payload = build_trace(
            question=ctx.question,
            source_directory=Path("sources"),
            output_path=output_path,
            documents=documents,
            memo=memo,
            mock_mode=False,
            profile=self._domain_profile,
        )
        # Record ReportAgent before building the trace so it appears in agents_run
        ctx.record_agent(
            self._make_note(
                status="completed",
                summary=f"Report written to {output_path}",
                report_path=str(output_path),
            )
        )
        trace_payload["functional_agents"] = ctx.to_functional_trace()

        # Inject research object stub if present
        if ctx.research_object:
            from research_agent.research_object import research_object_trace_stub, write_research_object
            from research_agent.research_object import update_research_object

            ro = update_research_object(
                ctx.research_object,
                memo=memo,
                output_path=output_path,
                trace_path=str(output_path.with_suffix(".trace.json")),
            )
            ro_path = write_research_object(ro, out_dir=output_path.parent)
            trace_payload["research_object"] = research_object_trace_stub(ro, ro_path)
            ctx.research_object = ro

        write_trace(trace_payload, output_path)
        return ctx
