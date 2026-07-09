"""ExecutionPlan — output of ExecutionPlanner (J13.3).

An ExecutionPlan is analysis-only: it describes which agents must run, in what
topological order, and which can execute in parallel.  It never triggers
execution itself.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ExecutionPlan:
    """Describes how to execute the agents needed to restore a stale ResearchState.

    Fields
    ------
    plan_id                  EP-{YYYYMMDD}-{HHMMSS}-{hex6} identifier
    created_at               ISO 8601 UTC creation timestamp
    triggering_state_changes change_ids of the StateChanges that prompted this plan
    staleness_plan_id        plan_id of the StalenessPlan this was derived from
    confidence               propagated from the StalenessPlan (HIGH / MEDIUM / LOW)
    required_agents          agents needed to restore all stale PERSISTED paths,
                             including any EXECUTION_ONLY prerequisite producers
    optional_agents          stale agents whose products are EXECUTION_ONLY and are
                             not prerequisites of any required_agent
    blocked_agents           agents that cannot run because a consumed EXECUTION_ONLY
                             path has no known producer in the registry
    blocked_reasons          agent_name → human-readable reason for being blocked
    execution_order          topological ordering of required + optional agents
    execution_groups         agents grouped by topological level; all agents within
                             one group can execute in parallel
    estimated_steps          number of sequential steps (len(execution_groups))
    reasoning                agent_name → why it appears in this plan
    """

    plan_id: str
    created_at: str
    triggering_state_changes: list[str]
    staleness_plan_id: str
    confidence: str

    required_agents: list[str]
    optional_agents: list[str]
    blocked_agents: list[str]
    blocked_reasons: dict[str, str]

    execution_order: list[str]
    execution_groups: list[list[str]]
    estimated_steps: int

    reasoning: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        triggering_state_changes: list[str],
        staleness_plan_id: str,
        confidence: str,
        required_agents: list[str],
        optional_agents: list[str],
        blocked_agents: list[str],
        blocked_reasons: dict[str, str],
        execution_order: list[str],
        execution_groups: list[list[str]],
        estimated_steps: int,
        reasoning: dict[str, str] | None = None,
    ) -> "ExecutionPlan":
        ts = datetime.now(timezone.utc)
        plan_id = f"EP-{ts.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        return cls(
            plan_id=plan_id,
            created_at=ts.isoformat(),
            triggering_state_changes=list(triggering_state_changes),
            staleness_plan_id=staleness_plan_id,
            confidence=confidence,
            required_agents=list(required_agents),
            optional_agents=list(optional_agents),
            blocked_agents=list(blocked_agents),
            blocked_reasons=dict(blocked_reasons),
            execution_order=list(execution_order),
            execution_groups=[list(g) for g in execution_groups],
            estimated_steps=estimated_steps,
            reasoning=dict(reasoning) if reasoning else {},
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "triggering_state_changes": self.triggering_state_changes,
            "staleness_plan_id": self.staleness_plan_id,
            "confidence": self.confidence,
            "required_agents": self.required_agents,
            "optional_agents": self.optional_agents,
            "blocked_agents": self.blocked_agents,
            "blocked_reasons": self.blocked_reasons,
            "execution_order": self.execution_order,
            "execution_groups": self.execution_groups,
            "estimated_steps": self.estimated_steps,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionPlan":
        return cls(
            plan_id=d["plan_id"],
            created_at=d["created_at"],
            triggering_state_changes=list(d.get("triggering_state_changes") or []),
            staleness_plan_id=d.get("staleness_plan_id", ""),
            confidence=d.get("confidence", "LOW"),
            required_agents=list(d.get("required_agents") or []),
            optional_agents=list(d.get("optional_agents") or []),
            blocked_agents=list(d.get("blocked_agents") or []),
            blocked_reasons=dict(d.get("blocked_reasons") or {}),
            execution_order=list(d.get("execution_order") or []),
            execution_groups=[list(g) for g in (d.get("execution_groups") or [])],
            estimated_steps=int(d.get("estimated_steps", 0)),
            reasoning=dict(d.get("reasoning") or {}),
        )
