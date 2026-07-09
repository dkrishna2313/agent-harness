"""IterationRecord — immutable record of a single research iteration (J13.0).

Each iteration captures metadata about one pipeline execution cycle.
Records are appended to ResearchSession.iteration_history and are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Valid trigger values
TRIGGER_INITIAL = "initial"
TRIGGER_EVIDENCE_REQUEST = "evidence_request"
TRIGGER_REPLAN = "replan"
TRIGGER_CONTINUATION = "continuation"


@dataclass
class IterationRecord:
    """Immutable metadata record for one research iteration.

    Fields
    ------
    iteration_number : 0-based index of this iteration within the session
    timestamp        : ISO 8601 UTC timestamp when the iteration started
    trigger          : what caused this iteration (see TRIGGER_* constants)
    summary          : one-line human-readable description
    completed_tasks  : IRT task IDs from the IterationPlan completed this iteration
    notes            : free-text annotation
    """

    iteration_number: int
    timestamp: str
    trigger: str
    summary: str
    completed_tasks: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration_number": self.iteration_number,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "summary": self.summary,
            "completed_tasks": list(self.completed_tasks),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IterationRecord":
        return cls(
            iteration_number=d.get("iteration_number") or 0,
            timestamp=d.get("timestamp") or "",
            trigger=d.get("trigger") or TRIGGER_INITIAL,
            summary=d.get("summary") or "",
            completed_tasks=list(d.get("completed_tasks") or []),
            notes=d.get("notes") or "",
        )
