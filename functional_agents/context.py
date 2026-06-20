"""Shared context object passed through functional agents (J5.0a.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentContext:
    """Mutable shared state threaded through each functional agent.

    Each agent receives the context, appends its notes, and returns it.
    The orchestrator owns the lifecycle.
    """

    # Core research intent
    question: str
    profiles: list[str] = field(default_factory=list)

    # Execution profile is the first entry in profiles
    @property
    def execution_profile(self) -> str | None:
        return self.profiles[0] if self.profiles else None

    # Shared durable state
    research_object: dict[str, Any] = field(default_factory=dict)

    # Agent outputs
    plan: dict[str, Any] = field(default_factory=dict)
    evidence_notes: list[dict[str, Any]] = field(default_factory=list)
    qa_notes: list[dict[str, Any]] = field(default_factory=list)

    # Final outputs
    report_path: str | None = None
    trace: dict[str, Any] = field(default_factory=dict)

    # Agent run log
    agents_run: list[str] = field(default_factory=list)

    def record_agent(self, note: dict[str, Any]) -> None:
        """Append an agent completion note and record the agent name."""
        agent_name = note.get("agent", "unknown")
        if agent_name not in self.agents_run:
            self.agents_run.append(agent_name)

    def to_functional_trace(self) -> dict[str, Any]:
        """Return the functional_agents block for the execution trace."""
        return {
            "enabled": True,
            "agents_run": self.agents_run,
            "profiles": self.profiles,
            "execution_profile": self.execution_profile,
        }
