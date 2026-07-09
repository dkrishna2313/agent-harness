"""ResearchSession — first-class persistent unit of work (J13.0).

A ResearchSession owns one ongoing investigation from initial execution
through any subsequent iterations. It is the primary unit of persistence
in the Strategic Research Harness from J13.0 onward.

The session is NOT the reasoning object. It owns the reasoning artifacts
(via ResearchState) but does not itself contain reasoning logic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .iteration_record import IterationRecord
from .research_state import ResearchState
from .snapshot import Snapshot


class SessionStatus:
    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ResearchSession:
    """Persistent unit of work representing one ongoing investigation.

    Lifecycle
    ---------
    1. ResearchSession.create() — initialise with metadata and initial state
    2. add_iteration()          — record each pipeline execution
    3. take_snapshot()          — capture ResearchState at key moments
    4. complete() / archive()   — terminal transitions

    Fields
    ------
    session_id        : unique identifier, SS-{YYYYMMDD}-{HHMMSS}-{hex6}
    created_at        : ISO 8601 UTC timestamp
    updated_at        : ISO 8601 UTC timestamp, updated on every mutation
    status            : SessionStatus constant
    metadata          : run_id, profiles, execution_profile, run_mode, engagement_id
    research_state    : current ResearchState (mutable; replaced after each run)
    iteration_history : ordered list of IterationRecord (append-only)
    snapshots         : ordered list of Snapshot (append-only)
    """

    session_id: str
    created_at: str
    updated_at: str
    status: str
    metadata: dict[str, Any]
    research_state: ResearchState
    iteration_history: list[IterationRecord] = field(default_factory=list)
    snapshots: list[Snapshot] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        *,
        metadata: dict[str, Any],
        research_state: ResearchState,
    ) -> "ResearchSession":
        """Create a new active ResearchSession."""
        ts = datetime.now(timezone.utc)
        stamp = ts.strftime("%Y%m%d-%H%M%S")
        session_id = f"SS-{stamp}-{uuid.uuid4().hex[:6]}"
        now_iso = ts.isoformat()
        return cls(
            session_id=session_id,
            created_at=now_iso,
            updated_at=now_iso,
            status=SessionStatus.ACTIVE,
            metadata=dict(metadata),
            research_state=research_state,
            iteration_history=[],
            snapshots=[],
        )

    def _touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_iteration(self, record: IterationRecord) -> None:
        """Append an IterationRecord to the iteration history."""
        self.iteration_history.append(record)
        self._touch()

    def take_snapshot(self) -> Snapshot:
        """Take a snapshot of the current ResearchState and append it."""
        snap = Snapshot.from_state(
            self.research_state,
            iteration_number=len(self.iteration_history),
        )
        self.snapshots.append(snap)
        self._touch()
        return snap

    def complete(self) -> None:
        """Transition to COMPLETED status."""
        self.status = SessionStatus.COMPLETED
        self._touch()

    def archive(self) -> None:
        """Transition to ARCHIVED status."""
        self.status = SessionStatus.ARCHIVED
        self._touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "metadata": self.metadata,
            "research_state": self.research_state.to_dict(),
            "iteration_history": [r.to_dict() for r in self.iteration_history],
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchSession":
        return cls(
            session_id=d.get("session_id") or "",
            created_at=d.get("created_at") or "",
            updated_at=d.get("updated_at") or "",
            status=d.get("status") or SessionStatus.ACTIVE,
            metadata=d.get("metadata") or {},
            research_state=ResearchState.from_dict(d.get("research_state") or {}),
            iteration_history=[
                IterationRecord.from_dict(r) for r in (d.get("iteration_history") or [])
            ],
            snapshots=[
                Snapshot.from_dict(s) for s in (d.get("snapshots") or [])
            ],
        )
