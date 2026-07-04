"""Executive Narrative Layer (J12.0).

Introduces the canonical executive communication model that sits between the
Strategic Reasoning Graph (what is true) and executive-facing deliverables
(how reasoning is communicated). All future executive deliverables will consume
ExecutiveNarrative instead of independently assembling narrative content from
AgentContext fields.

    ExecutiveNarrative        -- canonical executive storyline (business object)
    ExecutiveNarrativeBuilder -- assembles ExecutiveNarrative from AgentContext
"""

from __future__ import annotations

from .builder import ExecutiveNarrativeBuilder
from .executive_narrative import ExecutiveNarrative

__all__ = [
    "ExecutiveNarrative",
    "ExecutiveNarrativeBuilder",
]
