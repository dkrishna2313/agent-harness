"""DeliverableRequest — canonical business object (J11.0).

Represents a request to generate one or more deliverables. Intentionally
minimal: only ``type`` drives dispatch today (via DeliverableRegistry).
``audience``/``title``/``format`` are inert for MarkdownReportGenerator; they
exist so future generators (executive briefs, board papers) can read caller
intent without another contract change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeliverableRequest:
    """Describes which deliverable to generate and with what options.

    Stored on ``AgentContext.deliverable_request`` as a plain dict (via
    :meth:`to_dict`) to match the JSON-serializable convention every other
    AgentContext field follows.
    """

    id: str = ""
    type: str = "markdown"
    audience: str = ""
    title: str = ""
    format: str = "markdown"
    options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "audience": self.audience,
            "title": self.title,
            "format": self.format,
            "options": dict(self.options),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "DeliverableRequest":
        data = data or {}
        return cls(
            id=data.get("id", ""),
            type=data.get("type", "markdown"),
            audience=data.get("audience", ""),
            title=data.get("title", ""),
            format=data.get("format", "markdown"),
            options=dict(data.get("options") or {}),
        )
