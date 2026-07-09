"""StateChange — first-class model representing a mutation of ResearchState (J13.1).

Each StateChange records the who/what/when of a single state mutation. The
`affected_paths` list uses the canonical logical path vocabulary (e.g.
"research_object.evidence") rather than Python implementation details.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


class ChangeType:
    """Enumeration of state change types."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    REPLACE = "REPLACE"
    APPEND = "APPEND"
    EXTERNAL_EVIDENCE_ADDED = "EXTERNAL_EVIDENCE_ADDED"
    SESSION_CONTINUED = "SESSION_CONTINUED"
    MANUAL_OVERRIDE = "MANUAL_OVERRIDE"

    _ALL: frozenset[str] = frozenset({
        "CREATE", "UPDATE", "DELETE", "REPLACE", "APPEND",
        "EXTERNAL_EVIDENCE_ADDED", "SESSION_CONTINUED", "MANUAL_OVERRIDE",
    })


@dataclass
class StateChange:
    """First-class record of a single ResearchState mutation.

    Fields
    ------
    change_id      : unique identifier, SC-{YYYYMMDD}-{HHMMSS}-{hex6}
    timestamp      : ISO 8601 UTC timestamp
    source         : originating component ("orchestrator", "cli", "manual")
    change_type    : one of the ChangeType constants
    affected_paths : logical paths mutated (e.g. ["research_state"])
    description    : human-readable description of the change
    metadata       : arbitrary key-value pairs for future extension
    """

    change_id: str
    timestamp: str
    source: str
    change_type: str
    affected_paths: list[str]
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        source: str,
        change_type: str,
        affected_paths: list[str],
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> "StateChange":
        """Construct a new StateChange with an auto-generated ID and timestamp."""
        ts = datetime.now(timezone.utc)
        change_id = f"SC-{ts.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        return cls(
            change_id=change_id,
            timestamp=ts.isoformat(),
            source=source,
            change_type=change_type,
            affected_paths=list(affected_paths),
            description=description,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "change_type": self.change_type,
            "affected_paths": list(self.affected_paths),
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StateChange":
        return cls(
            change_id=d.get("change_id") or "",
            timestamp=d.get("timestamp") or "",
            source=d.get("source") or "",
            change_type=d.get("change_type") or "",
            affected_paths=list(d.get("affected_paths") or []),
            description=d.get("description") or "",
            metadata=dict(d.get("metadata") or {}),
        )
