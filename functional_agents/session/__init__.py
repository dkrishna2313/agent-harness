"""Research Session package — persistent investigation management (J13.0+J13.1).

Public API
----------
ResearchSession  — first-class persistent unit of work
SessionStatus    — session lifecycle constants (ACTIVE / COMPLETED / ARCHIVED)
ResearchState    — canonical mutable state owning reasoning artifacts
IterationRecord  — immutable record of one pipeline execution
Snapshot         — full ResearchState capture at a point in time
StateChange      — first-class record of a single ResearchState mutation (J13.1)
ChangeType       — StateChange type constants (J13.1)
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
from .state_change import ChangeType, StateChange

__all__ = [
    "ResearchSession",
    "SessionStatus",
    "ResearchState",
    "IterationRecord",
    "Snapshot",
    "StateChange",
    "ChangeType",
    "SessionStore",
    "SessionNotFoundError",
    "load_session_file",
    "save_session_file",
]
