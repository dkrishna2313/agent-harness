"""StalenessPlan — the output of DependencyReasoner (J13.2).

Represents the complete result of a staleness analysis: what changed, what is
now stale, which agents need to re-run, and why. Immutable once created.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StalenessPlan:
    """Deterministic analysis of what has become stale given a set of StateChanges.

    Produced exclusively by DependencyReasoner.analyze(). Never produces side
    effects. The pipeline is unchanged; this plan describes what WOULD need to
    happen, not what DOES happen.

    Fields
    ------
    plan_id              : unique identifier, SP-{YYYYMMDD}-{HHMMSS}-{hex6}
    created_at           : ISO 8601 UTC
    source_changes       : change_ids of the StateChanges that triggered this analysis
    changed_paths        : direct change paths expanded from StateChange.affected_paths
    stale_paths          : all paths that need recomputation (PERSISTED + EXECUTION_ONLY)
    stale_agents         : agents (declaration order) whose produces overlap stale_paths
    required_producers   : agents needed specifically to restore stale PERSISTED paths
    persisted_paths      : subset of stale_paths with PathKind.PERSISTED
    execution_only_paths : subset of stale_paths with PathKind.EXECUTION_ONLY
    external_dependencies: EXTERNAL paths that triggered changes (never recomputed)
    reasoning            : path → human-readable explanation of why it is stale
    confidence           : "HIGH" | "MEDIUM" | "LOW" — reliability of the analysis
    """

    plan_id: str
    created_at: str
    source_changes: list[str]
    changed_paths: list[str]
    stale_paths: list[str]
    stale_agents: list[str]
    required_producers: list[str]
    persisted_paths: list[str]
    execution_only_paths: list[str]
    external_dependencies: list[str]
    reasoning: dict[str, str]
    confidence: str

    @classmethod
    def create(
        cls,
        *,
        source_changes: list[str],
        changed_paths: list[str],
        stale_paths: list[str],
        stale_agents: list[str],
        required_producers: list[str],
        persisted_paths: list[str],
        execution_only_paths: list[str],
        external_dependencies: list[str],
        reasoning: dict[str, str],
        confidence: str,
    ) -> "StalenessPlan":
        """Construct a StalenessPlan with an auto-generated ID and timestamp."""
        ts = datetime.now(timezone.utc)
        plan_id = f"SP-{ts.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        return cls(
            plan_id=plan_id,
            created_at=ts.isoformat(),
            source_changes=list(source_changes),
            changed_paths=list(changed_paths),
            stale_paths=list(stale_paths),
            stale_agents=list(stale_agents),
            required_producers=list(required_producers),
            persisted_paths=list(persisted_paths),
            execution_only_paths=list(execution_only_paths),
            external_dependencies=list(external_dependencies),
            reasoning=dict(reasoning),
            confidence=str(confidence),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "created_at": self.created_at,
            "source_changes": list(self.source_changes),
            "changed_paths": list(self.changed_paths),
            "stale_paths": list(self.stale_paths),
            "stale_agents": list(self.stale_agents),
            "required_producers": list(self.required_producers),
            "persisted_paths": list(self.persisted_paths),
            "execution_only_paths": list(self.execution_only_paths),
            "external_dependencies": list(self.external_dependencies),
            "reasoning": dict(self.reasoning),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StalenessPlan":
        return cls(
            plan_id=d.get("plan_id") or "",
            created_at=d.get("created_at") or "",
            source_changes=list(d.get("source_changes") or []),
            changed_paths=list(d.get("changed_paths") or []),
            stale_paths=list(d.get("stale_paths") or []),
            stale_agents=list(d.get("stale_agents") or []),
            required_producers=list(d.get("required_producers") or []),
            persisted_paths=list(d.get("persisted_paths") or []),
            execution_only_paths=list(d.get("execution_only_paths") or []),
            external_dependencies=list(d.get("external_dependencies") or []),
            reasoning=dict(d.get("reasoning") or {}),
            confidence=d.get("confidence") or "LOW",
        )
