"""Executive Narrative Layer (J12.0–J12.4).

Introduces the canonical executive communication model that sits between the
Strategic Reasoning Graph (what is true) and executive-facing deliverables
(how reasoning is communicated).

    ExecutiveNarrative         -- canonical executive storyline (business object, v1.1)
    ExecutiveNarrativeBuilder  -- extracts ExecutiveNarrative fields from AgentContext
    ExecutiveNarrativeComposer -- enriches why_this_option; composes story fields (J12.4)
"""

from __future__ import annotations

from .builder import ExecutiveNarrativeBuilder
from .composer import ExecutiveNarrativeComposer
from .executive_narrative import ExecutiveNarrative

__all__ = [
    "ExecutiveNarrative",
    "ExecutiveNarrativeBuilder",
    "ExecutiveNarrativeComposer",
]
