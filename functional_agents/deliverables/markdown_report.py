"""MarkdownReportGenerator — the first concrete DeliverableGenerator (J11.0).

This generator changes nothing about *what* markdown gets produced. It calls
``report_agent.build_markdown_report_content()`` — the exact J7.6b/legacy
section-assembly logic that used to run inline inside ``ReportAgent._execute``
— and then writes it to disk with the same ``write_markdown()`` helper
ReportAgent always used. Behaviour is byte-for-byte identical to pre-J11.0.

Invokes no Functional Agent, calls no LLM, and mutates no reasoning field.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator

if TYPE_CHECKING:
    from ..context import AgentContext


class MarkdownReportGenerator(DeliverableGenerator):
    deliverable_type = "markdown"

    def generate(self, context: "AgentContext", output_path: Path) -> DeliverableArtifact:
        from research_agent.markdown import write_markdown

        from ..report_agent import build_markdown_report_content

        memo = context.trace.get("_report_memo", {})
        report_content = build_markdown_report_content(context, memo)

        written_path = write_markdown(report_content, output_path)

        return DeliverableArtifact(
            type=self.deliverable_type,
            path=str(written_path),
            mime_type="text/markdown",
            metadata={"status": "generated"},
        )
