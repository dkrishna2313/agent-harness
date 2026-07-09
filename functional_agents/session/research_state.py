"""ResearchState — canonical persistent state for a research investigation (J13.0).

Owns the current reasoning artifacts produced by the pipeline.
No execution logic lives here; this is purely persistent state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from functional_agents.context import AgentContext


@dataclass
class ResearchState:
    """Canonical mutable state owning the current reasoning artifacts.

    Fields map to the primary artifact slots on AgentContext:
      engagement           ← context.engagement
      research_object      ← context.research_object
      decision_model       ← context.decision_model
      research_gap_analysis← context.research_gap_analysis
      executive_confidence ← context.executive_confidence
      iteration_plan       ← context.iteration_plan
    """

    engagement: dict[str, Any] = field(default_factory=dict)
    research_object: dict[str, Any] = field(default_factory=dict)
    decision_model: dict[str, Any] = field(default_factory=dict)
    research_gap_analysis: dict[str, Any] = field(default_factory=dict)
    executive_confidence: dict[str, Any] = field(default_factory=dict)
    iteration_plan: dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def from_context(cls, ctx: "AgentContext") -> "ResearchState":
        """Build a ResearchState snapshot from an AgentContext."""
        return cls(
            engagement=dict(ctx.engagement or {}),
            research_object=dict(ctx.research_object or {}),
            decision_model=dict(ctx.decision_model or {}),
            research_gap_analysis=dict(ctx.research_gap_analysis or {}),
            executive_confidence=dict(ctx.executive_confidence or {}),
            iteration_plan=dict(ctx.iteration_plan or {}),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "engagement": self.engagement,
            "research_object": self.research_object,
            "decision_model": self.decision_model,
            "research_gap_analysis": self.research_gap_analysis,
            "executive_confidence": self.executive_confidence,
            "iteration_plan": self.iteration_plan,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ResearchState":
        return cls(
            engagement=d.get("engagement") or {},
            research_object=d.get("research_object") or {},
            decision_model=d.get("decision_model") or {},
            research_gap_analysis=d.get("research_gap_analysis") or {},
            executive_confidence=d.get("executive_confidence") or {},
            iteration_plan=d.get("iteration_plan") or {},
            updated_at=d.get("updated_at") or "",
        )
