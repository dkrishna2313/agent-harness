"""Canonical intermediate representation for executive presentations.

A ``Presentation`` is the single source of truth produced by
``ExecutivePresentationGenerator``.  Downstream renderers convert it to
Markdown, PPTX, Google Slides, or Keynote without touching the generator.

Hierarchy::

    Presentation
        title, subtitle, client, date, metadata
        slides: list[Slide]
            slide_number, slide_type, title
            key_message          (one-sentence executive takeaway)
            bullets: list[str]   (max 5; max 15 words each)
            table: SlideTable    (optional)
                headers: list[str]
                rows: list[TableRow]
                    cells: list[str]
            notes: str           (speaker context; not displayed on slide)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TableRow:
    """One row of a slide table."""

    cells: list[str]


@dataclass
class SlideTable:
    """Optional tabular data block within a slide."""

    headers: list[str]
    rows: list[TableRow]
    caption: str = ""


@dataclass
class Slide:
    """One slide in the executive deck.

    ``slide_type`` is a hint for renderers:
    ``"title"`` | ``"content"`` | ``"comparison"`` | ``"appendix"``
    """

    slide_number: int
    slide_type: str
    title: str
    key_message: str = ""
    bullets: list[str] = field(default_factory=list)
    table: Optional[SlideTable] = None
    notes: str = ""


@dataclass
class Presentation:
    """Canonical intermediate representation of an executive presentation.

    Renderers should read from this model only — never from AgentContext
    directly — so that PPTX, Markdown, and other outputs are consistent.
    """

    title: str
    subtitle: str
    client: str = ""
    date: str = ""
    slides: list[Slide] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
