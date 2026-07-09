"""Snapshot — complete ResearchState capture at a point in time (J13.0).

Snapshots are append-only. The current implementation serializes the full
ResearchState dict. The interface is designed so J13.1+ can replace the
body with delta-based snapshots without changing public APIs.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .research_state import ResearchState


@dataclass
class Snapshot:
    """Full serialized capture of ResearchState at one point in time.

    Fields
    ------
    snapshot_id      : unique identifier, SNAP-{iteration:03d}-{hex8}
    created_at       : ISO 8601 UTC timestamp
    iteration_number : iteration number at snapshot time
    state            : serialized ResearchState (from ResearchState.to_dict())
    """

    snapshot_id: str
    created_at: str
    iteration_number: int
    state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: ResearchState, *, iteration_number: int) -> "Snapshot":
        """Create a snapshot from a ResearchState."""
        snap_id = f"SNAP-{iteration_number:03d}-{uuid.uuid4().hex[:8]}"
        return cls(
            snapshot_id=snap_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            iteration_number=iteration_number,
            state=state.to_dict(),
        )

    def to_research_state(self) -> ResearchState:
        """Reconstruct the ResearchState from this snapshot."""
        return ResearchState.from_dict(self.state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "iteration_number": self.iteration_number,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Snapshot":
        return cls(
            snapshot_id=d.get("snapshot_id") or "",
            created_at=d.get("created_at") or "",
            iteration_number=d.get("iteration_number") or 0,
            state=d.get("state") or {},
        )
