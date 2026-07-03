"""DeliverableRegistry — dispatch a DeliverableRequest to its generator (J11.0)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact
from .base import DeliverableGenerator
from .request import DeliverableRequest

if TYPE_CHECKING:
    from ..context import AgentContext


class DeliverableRegistry:
    """Maps ``deliverable_type`` -> :class:`DeliverableGenerator` instance."""

    def __init__(self) -> None:
        self._generators: dict[str, DeliverableGenerator] = {}

    def register(self, generator: DeliverableGenerator) -> None:
        if not generator.deliverable_type:
            raise ValueError(f"{generator.__class__.__name__} must set deliverable_type")
        self._generators[generator.deliverable_type] = generator

    def get(self, deliverable_type: str) -> DeliverableGenerator:
        try:
            return self._generators[deliverable_type]
        except KeyError:
            raise KeyError(
                f"No deliverable generator registered for type {deliverable_type!r}. "
                f"Registered types: {sorted(self._generators)}"
            ) from None

    def generate(
        self, context: "AgentContext", request: DeliverableRequest, output_path: Path
    ) -> DeliverableArtifact:
        """Dispatch ``request`` (by ``request.type``) to its generator."""
        return self.get(request.type).generate(context, output_path)
