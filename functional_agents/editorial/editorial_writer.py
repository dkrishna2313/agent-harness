"""EditorialWriter — abstract base class for all editorial writers (PH6.5).

Every writer:
  - consumes an EditorialBrief section (plus explicit linked sections)
  - populates the corresponding EditorialManuscript section
  - returns the updated manuscript
  - never calls reasoning agents
  - never mutates AgentContext
  - never produces Markdown, DOCX, or PPTX formatting
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript


class EditorialWriter(ABC):
    """Abstract base for editorial writer agents.

    Subclasses implement write() to populate one manuscript section
    from the corresponding brief section. Provenance is always retained.
    """

    @abstractmethod
    def write(
        self,
        brief: EditorialBrief,
        manuscript: EditorialManuscript,
    ) -> EditorialManuscript:
        """Populate the writer's target section in manuscript and return it.

        The manuscript is mutated in place; the return value is the same
        object (allows chaining).
        """
        ...
