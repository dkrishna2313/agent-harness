"""Research Session package — persistent investigation management (J13.0).

Public API
----------
ResearchSession  — first-class persistent unit of work
SessionStatus    — session lifecycle constants (ACTIVE / COMPLETED / ARCHIVED)
ResearchState    — canonical mutable state owning reasoning artifacts
IterationRecord  — immutable record of one pipeline execution
Snapshot         — full ResearchState capture at a point in time
SessionStore     — JSON-backed persistence layer
SessionNotFoundError — raised by SessionStore.load() when session is absent
"""

from .iteration_record import IterationRecord
from .research_session import ResearchSession, SessionStatus
from .research_state import ResearchState
from .session_store import (
    SessionNotFoundError,
    SessionStore,
    load_session_file,
    save_session_file,
)
from .snapshot import Snapshot

__all__ = [
    "ResearchSession",
    "SessionStatus",
    "ResearchState",
    "IterationRecord",
    "Snapshot",
    "SessionStore",
    "SessionNotFoundError",
    "load_session_file",
    "save_session_file",
]
