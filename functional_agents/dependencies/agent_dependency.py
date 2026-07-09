"""AgentDependency — declarative dependency contract for one pipeline agent (J13.1).

Paths use the canonical logical vocabulary:
    engagement, decision_model, decision_architecture, research_strategy, planner,
    research_object.evidence, research_object.hypotheses, research_gap_analysis,
    strategic_synthesis, challenge_results, decision_model.assumptions,
    decision_model.recommendations, decision_model.risks, decision_model.opportunities,
    decision_model.strategic_options, decision_model.decision_analysis,
    executive_confidence, iteration_plan, multi_profile_analysis, scenario_analysis,
    qa, recommendation_improvement, recommendation_synthesis, report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentDependency:
    """Declarative dependency contract for a single pipeline agent.

    Fields
    ------
    agent_name  : canonical Python class name (e.g. "EvidenceAgent")
    consumes    : logical paths this agent reads
    produces    : logical paths this agent writes
    invalidates : downstream logical paths made stale when this agent's outputs change
    """

    agent_name: str
    consumes: list[str] = field(default_factory=list)
    produces: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "consumes": list(self.consumes),
            "produces": list(self.produces),
            "invalidates": list(self.invalidates),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AgentDependency":
        return cls(
            agent_name=d.get("agent_name") or "",
            consumes=list(d.get("consumes") or []),
            produces=list(d.get("produces") or []),
            invalidates=list(d.get("invalidates") or []),
        )
