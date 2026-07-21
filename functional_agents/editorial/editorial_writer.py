"""EditorialWriter — abstract base class for all editorial writers (PH6.5).

Every writer:
  - declares section_name: the EditorialManuscript attribute it owns
  - consumes an EditorialBrief section (plus explicit linked sections)
  - populates the corresponding EditorialManuscript section
  - returns the updated manuscript
  - never calls reasoning agents
  - never mutates AgentContext
  - never produces Markdown, DOCX, or PPTX formatting
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .editorial_brief import EditorialBrief
from .editorial_manuscript import EditorialManuscript


class EditorialWriter(ABC):
    """Abstract base for editorial writer agents.

    Each subclass must define:
      section_name: ClassVar[str]   — the EditorialManuscript attribute this writer owns.
          Used by EditorialCoordinator for registry validation and completeness checks.

    Subclasses implement write() to populate one manuscript section
    from the corresponding brief section. Provenance is always retained.
    """

    section_name: ClassVar[str]

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
