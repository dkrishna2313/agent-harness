"""DeliverableArtifact — canonical output object (J11.0).

Represents one generated deliverable. This becomes the canonical output
object for future deliverables (executive briefs, board papers, ...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeliverableArtifact:
    """Result of generating one deliverable.

    ``status`` is not a dataclass field (the canonical field list is
    ``id``/``type``/``path``/``mime_type``/``metadata``) — generators report
    it via ``metadata["status"]``, and :meth:`to_dict` projects it back out
    as a top-level ``"status"`` key (default ``"generated"``), matching the
    canonical pipeline trace's ``deliverables`` section shape:
    ``{"type": ..., "status": ...}``.
    """

    type: str
    id: str = ""
    path: str | None = None
    mime_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type,
            "status": self.metadata.get("status", "generated"),
        }
        if self.id:
            out["id"] = self.id
        if self.path is not None:
            out["path"] = self.path
        if self.mime_type:
            out["mime_type"] = self.mime_type
        extra_metadata = {k: v for k, v in self.metadata.items() if k != "status"}
        if extra_metadata:
            out["metadata"] = extra_metadata
        return out
