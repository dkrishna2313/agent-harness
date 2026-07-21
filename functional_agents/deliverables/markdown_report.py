"""MarkdownReportGenerator — PH7 (upgraded from J11.0).

When an EditorialManuscript is available in the trace (placed there by the
orchestrator's editorial hook), uses MarkdownRenderer to produce the report
from editorial prose — making the Editorial Platform the authoritative source
of all rendered Markdown output.

Falls back to the legacy ``build_markdown_report_content()`` path when no
manuscript is in the trace, preserving backward compatibility with runs that
pre-date PH7 or where the editorial hook failed.

Invokes no Functional Agent, calls no LLM, and mutates no reasoning field.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator

if TYPE_CHECKING:
    from ..context import AgentContext

LOGGER = logging.getLogger(__name__)


class MarkdownReportGenerator(DeliverableGenerator):
    deliverable_type = "markdown"

    def generate(self, context: "AgentContext", output_path: Path) -> DeliverableArtifact:
        from research_agent.markdown import write_markdown

        manuscript = context.trace.get("_editorial_manuscript")
        brief = context.trace.get("_editorial_brief")

        if manuscript is not None:
            # PH7 path — use editorial prose
            try:
                from ..editorial.markdown_renderer import MarkdownRenderer
                report_content = MarkdownRenderer().render(manuscript, brief=brief)
                LOGGER.debug("[MarkdownReportGenerator] rendered via MarkdownRenderer (PH7)")
            except Exception as exc:
                LOGGER.warning(
                    "[MarkdownReportGenerator] MarkdownRenderer failed (%s: %s) — falling back to legacy",
                    type(exc).__name__, exc,
                )
                manuscript = None  # trigger fallback below

        if manuscript is None:
            # Legacy fallback path
            from ..report_agent import build_markdown_report_content
            memo = context.trace.get("_report_memo", {})
            report_content = build_markdown_report_content(context, memo)
            LOGGER.debug("[MarkdownReportGenerator] rendered via legacy build_markdown_report_content")

        written_path = write_markdown(report_content, output_path)

        return DeliverableArtifact(
            type=self.deliverable_type,
            path=str(written_path),
            mime_type="text/markdown",
            metadata={"status": "generated"},
        )
