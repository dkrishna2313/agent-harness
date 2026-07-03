"""DeliverableGenerator — abstract base for the Deliverables Framework (J11.0).

The platform's reasoning stack (Planner -> Evidence -> Hypothesis -> ...
-> Report) produces a Strategic Reasoning Graph on AgentContext, independent
of presentation. A DeliverableGenerator consumes that completed graph and
produces one DeliverableArtifact — it never runs new reasoning, never calls
an LLM, and never invokes a Functional Agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

from .artifact import DeliverableArtifact

if TYPE_CHECKING:
    from ..context import AgentContext


class DeliverableGenerator(ABC):
    """Base class for every deliverable generator.

    A generator consumes the shared Strategic Reasoning Graph on
    ``AgentContext`` — it must not run new reasoning, call an LLM, mutate
    reasoning fields (assumptions, risks, recommendations, ...), or invoke a
    Functional Agent. It may only read from context and write presentation
    output to ``output_path``.
    """

    deliverable_type: str = ""

    @abstractmethod
    def generate(self, context: "AgentContext", output_path: Path) -> DeliverableArtifact:
        """Produce a DeliverableArtifact from ``context``, writing to ``output_path``."""
