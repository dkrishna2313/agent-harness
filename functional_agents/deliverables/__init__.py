"""Strategic Deliverables Framework (J11.0).

Separates reasoning from presentation: multiple consulting deliverables
(markdown reports today; executive briefs, board papers, investment memos
in future milestones) can be generated from the same Strategic Reasoning
Graph on ``AgentContext`` without changing the reasoning engine.

    DeliverableRequest   -- what to generate, and for whom
    DeliverableArtifact  -- what was produced
    DeliverableGenerator -- one generator per deliverable type
    DeliverableRegistry  -- dispatches a request to its generator

``default_registry`` is the registry ReportAgent uses; it comes pre-registered
with ``MarkdownReportGenerator`` under ``"markdown"``.
"""

from __future__ import annotations

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator
from .markdown_report import MarkdownReportGenerator
from .registry import DeliverableRegistry
from .request import DeliverableRequest

default_registry = DeliverableRegistry()
default_registry.register(MarkdownReportGenerator())

__all__ = [
    "DeliverableRequest",
    "DeliverableArtifact",
    "DeliverableGenerator",
    "DeliverableRegistry",
    "MarkdownReportGenerator",
    "default_registry",
]
