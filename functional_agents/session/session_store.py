"""SessionStore — JSON-backed persistence for ResearchSession (J13.0).

Supports: create, load, save, archive, continue.

One JSON file per session: {base_dir}/{session_id}.json

The format is deterministic and human-inspectable. JSON is chosen for
compatibility with the existing research object persistence layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .research_session import ResearchSession, SessionStatus

LOGGER = logging.getLogger(__name__)

_DEFAULT_DIR = Path("outputs/sessions")


class SessionNotFoundError(KeyError):
    """Raised when a requested session cannot be found in the store."""


class SessionStore:
    """Persist and retrieve ResearchSession objects as JSON files.

    All write operations are atomic at the Python level (write_text).
    Directory is created on first use.

    Parameters
    ----------
    base_dir : storage directory for session files (default: outputs/sessions/)
    """

    def __init__(self, base_dir: Path | str | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_DIR

    def _session_path(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.json"

    def _ensure_dir(self) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def create(self, session: ResearchSession) -> Path:
        """Write a new session file. Overwrites if a file with the same ID exists."""
        self._ensure_dir()
        path = self._session_path(session.session_id)
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.debug("[SessionStore] created %s → %s", session.session_id, path)
        return path

    def save(self, session: ResearchSession) -> Path:
        """Overwrite the session file with the current state."""
        self._ensure_dir()
        path = self._session_path(session.session_id)
        path.write_text(
            json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        LOGGER.debug("[SessionStore] saved %s → %s", session.session_id, path)
        return path

    def load(self, session_id: str) -> ResearchSession:
        """Load a session by ID. Raises SessionNotFoundError if absent."""
        path = self._session_path(session_id)
        if not path.exists():
            raise SessionNotFoundError(
                f"Session {session_id!r} not found at {path}"
            )
        raw = path.read_text(encoding="utf-8")
        data: dict[str, Any] = json.loads(raw)
        return ResearchSession.from_dict(data)

    def archive(self, session_id: str) -> None:
        """Load, transition to ARCHIVED, and re-save."""
        session = self.load(session_id)
        session.archive()
        self.save(session)
        LOGGER.debug("[SessionStore] archived %s", session_id)

    def continue_session(self, session_id: str) -> ResearchSession:
        """Load and re-activate an archived or completed session.

        Returns the in-memory session with status=ACTIVE. The caller
        is responsible for saving after making further changes.
        """
        session = self.load(session_id)
        session.status = SessionStatus.ACTIVE
        session._touch()
        return session

    def exists(self, session_id: str) -> bool:
        """Return True if a file for session_id is present."""
        return self._session_path(session_id).exists()

    def list_sessions(self) -> list[str]:
        """Return sorted list of session IDs present in the store."""
        if not self._base_dir.exists():
            return []
        return sorted(p.stem for p in self._base_dir.glob("SS-*.json"))


# ---------------------------------------------------------------------------
# Path-based helpers (J13.1) — load/save at an explicit file path rather
# than through the auto-ID store directory.
# ---------------------------------------------------------------------------

def load_session_file(path: Path | str) -> ResearchSession:
    """Load a ResearchSession from an explicit file path."""
    path = Path(path)
    if not path.exists():
        raise SessionNotFoundError(f"Session file not found: {path}")
    return ResearchSession.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_session_file(session: ResearchSession, path: Path | str) -> None:
    """Write a ResearchSession to an explicit file path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(session.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.debug("[SessionStore] saved %s → %s", session.session_id, path)
