"""DeliverableBundle and DeliverableBundleGenerator (J11.2).

DeliverableBundleGenerator is a presentation-layer orchestrator. It drives
multiple DeliverableGenerators over one already-completed AgentContext,
collecting their artifacts into a DeliverableBundle. No Functional Agent is
invoked, no LLM call is made, and no reasoning field on AgentContext is
mutated — the only context mutation is setting ``context.deliverable_bundle``
(presentation state, equivalent to how ReportAgent sets ``context.deliverables``).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact import DeliverableArtifact
from .registry import DeliverableRegistry

if TYPE_CHECKING:
    from ..context import AgentContext


@dataclass
class DeliverableBundle:
    """Complete output of one engagement's presentation phase.

    ``deliverables`` holds the :class:`DeliverableArtifact` objects
    produced during this bundle run. :meth:`to_dict` serialises them via
    :meth:`~DeliverableArtifact.to_dict` under the ``generated`` key — the
    shape the pipeline trace uses for the ``deliverable_bundle`` section.
    """

    bundle_id: str = ""
    engagement_id: str = ""
    reasoning_graph_id: str = ""
    created_at: str = ""
    deliverables: list[DeliverableArtifact] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.bundle_id:
            out["bundle_id"] = self.bundle_id
        if self.engagement_id:
            out["engagement_id"] = self.engagement_id
        if self.reasoning_graph_id:
            out["reasoning_graph_id"] = self.reasoning_graph_id
        if self.created_at:
            out["created_at"] = self.created_at
        out["generated"] = [a.to_dict() for a in self.deliverables]
        if self.metadata:
            out["metadata"] = self.metadata
        return out


class DeliverableBundleGenerator:
    """Orchestrates multiple DeliverableGenerators over one completed AgentContext.

    Responsibilities:
    - invoke each requested generator independently
    - assign a bundle-scoped ``id`` to each resulting artifact
    - assemble and return a :class:`DeliverableBundle`
    - set ``context.deliverable_bundle`` so the pipeline trace captures it

    No Functional Agents are invoked; no LLM calls are made; no reasoning
    fields on AgentContext are mutated.
    """

    def __init__(self, registry: DeliverableRegistry | None = None) -> None:
        self._registry = registry  # resolved lazily to avoid circular import

    def _resolved_registry(self) -> DeliverableRegistry:
        if self._registry is not None:
            return self._registry
        from . import default_registry  # late import breaks the import cycle
        return default_registry

    def generate(
        self,
        context: "AgentContext",
        deliverable_types: list[str],
        output_dir: Path,
        *,
        bundle_id: str = "",
        engagement_id: str = "",
        created_at: str = "",
    ) -> DeliverableBundle:
        """Generate each requested deliverable type and return a DeliverableBundle.

        ``output_dir`` is the directory where each artifact is written.
        Each artifact is named ``<type>.md`` (underscores replaced by hyphens,
        e.g. ``executive_brief`` -> ``executive-brief.md``).

        ``context.deliverable_bundle`` is set to ``bundle.to_dict()`` on
        return so the pipeline trace captures the bundle.
        """
        if not bundle_id:
            bundle_id = str(uuid.uuid4())

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        registry = self._resolved_registry()
        artifacts: list[DeliverableArtifact] = []
        for deliverable_type in deliverable_types:
            filename = deliverable_type.replace("_", "-") + ".md"
            artifact = registry.get(deliverable_type).generate(
                context, output_dir / filename
            )
            artifact.id = f"{bundle_id[:8]}-{deliverable_type}"
            artifacts.append(artifact)

        ro = context.research_object or {}
        reasoning_graph_id = ro.get("research_id", "")

        bundle = DeliverableBundle(
            bundle_id=bundle_id,
            engagement_id=engagement_id,
            reasoning_graph_id=reasoning_graph_id,
            created_at=created_at,
            deliverables=artifacts,
        )
        context.deliverable_bundle = bundle.to_dict()
        return bundle
