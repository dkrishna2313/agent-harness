"""Strategic Deliverables Framework (J11.0 / J11.1 / J11.2).

Separates reasoning from presentation: multiple consulting deliverables
(markdown reports and executive briefs today; board papers, investment
memos in future milestones) can be generated from the same Strategic
Reasoning Graph on ``AgentContext`` without changing the reasoning engine.

    DeliverableRequest          -- what to generate, and for whom
    DeliverableArtifact         -- what was produced (one deliverable)
    DeliverableGenerator        -- one generator per deliverable type
    DeliverableRegistry         -- dispatches a request to its generator
    DeliverableBundle           -- complete output of one engagement (J11.2)
    DeliverableBundleGenerator  -- orchestrates N generators, builds bundle (J11.2)

``default_registry`` is the registry ReportAgent uses; it comes pre-registered
with ``MarkdownReportGenerator`` under ``"markdown"``,
``ExecutiveBriefGenerator`` under ``"executive_brief"``, and
``StrategyDeckGenerator`` under ``"strategy_deck"``. ReportAgent's default
run only ever requests ``"markdown"`` — registering additional generators
here does not change what a normal pipeline run produces.
"""

from __future__ import annotations

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator
from .bundle import DeliverableBundle, DeliverableBundleGenerator
from .executive_brief import ExecutiveBriefGenerator
from .markdown_report import MarkdownReportGenerator
from .registry import DeliverableRegistry
from .request import DeliverableRequest
from .strategy_deck import StrategyDeckGenerator

default_registry = DeliverableRegistry()
default_registry.register(MarkdownReportGenerator())
default_registry.register(ExecutiveBriefGenerator())
default_registry.register(StrategyDeckGenerator())

__all__ = [
    "DeliverableRequest",
    "DeliverableArtifact",
    "DeliverableGenerator",
    "DeliverableRegistry",
    "DeliverableBundle",
    "DeliverableBundleGenerator",
    "MarkdownReportGenerator",
    "ExecutiveBriefGenerator",
    "StrategyDeckGenerator",
    "default_registry",
]
