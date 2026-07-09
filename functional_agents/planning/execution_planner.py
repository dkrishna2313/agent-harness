"""ExecutionPlanner — converts a StalenessPlan into an ExecutionPlan (J13.3).

Algorithm
---------
1. **required_agents** — start from StalenessPlan.required_producers (agents that
   produce stale PERSISTED paths).  For each agent, find any EXECUTION_ONLY consumed
   paths that are not themselves stale; add the producers of those paths to the
   required set.  Repeat until fixpoint.  This is always a superset of
   required_producers and may differ when a stale agent needs EXECUTION_ONLY
   intermediates that were not themselves marked stale (e.g. a MANUAL_OVERRIDE on a
   downstream persisted path).

2. **optional_agents** — stale_agents from StalenessPlan that are NOT in
   required_agents.  These produce EXECUTION_ONLY stale paths that are not
   prerequisites for any required_agent.

3. **blocked_agents** — agents whose consumed EXECUTION_ONLY path has no known
   producer in the registry.  In practice this should always be empty given the
   current registry, but is computed defensively.

4. **execution_groups** — topological levels of all planned agents (required +
   optional).  Within each level, agents share no dependency edges and may run in
   parallel.  Computed via Kahn's algorithm restricted to the planned agent set.

5. **execution_order** — the flattened execution_groups list.
"""

from __future__ import annotations

from collections import deque

from functional_agents.dependencies import DependencyRegistry
from functional_agents.staleness import PathKind, classify_path
from functional_agents.staleness.staleness_plan import StalenessPlan

from .execution_plan import ExecutionPlan


class ExecutionPlanner:
    """Deterministic, analysis-only execution planner.

    Call ``plan(staleness_plan)`` to obtain an ExecutionPlan.  Never triggers
    agent execution.
    """

    def plan(self, staleness_plan: StalenessPlan) -> ExecutionPlan:
        """Convert *staleness_plan* into a dependency-respecting ExecutionPlan."""
        if not staleness_plan.stale_agents:
            return ExecutionPlan.create(
                triggering_state_changes=staleness_plan.source_changes,
                staleness_plan_id=staleness_plan.plan_id,
                confidence=staleness_plan.confidence,
                required_agents=[],
                optional_agents=[],
                blocked_agents=[],
                blocked_reasons={},
                execution_order=[],
                execution_groups=[],
                estimated_steps=0,
                reasoning={},
            )

        required_agents, blocked_agents, blocked_reasons = self._expand_required_agents(
            staleness_plan
        )
        required_set = set(required_agents)

        # optional = stale agents whose products are EXECUTION_ONLY and not prerequisites
        optional_agents = [
            a for a in staleness_plan.stale_agents if a not in required_set
        ]

        all_planned = required_agents + optional_agents

        execution_groups = self._compute_execution_groups(all_planned)
        execution_order = [a for group in execution_groups for a in group]

        reasoning = self._build_reasoning(
            required_agents, optional_agents, staleness_plan
        )

        return ExecutionPlan.create(
            triggering_state_changes=staleness_plan.source_changes,
            staleness_plan_id=staleness_plan.plan_id,
            confidence=staleness_plan.confidence,
            required_agents=required_agents,
            optional_agents=optional_agents,
            blocked_agents=blocked_agents,
            blocked_reasons=blocked_reasons,
            execution_order=execution_order,
            execution_groups=execution_groups,
            estimated_steps=len(execution_groups),
            reasoning=reasoning,
        )

    # ------------------------------------------------------------------
    # Step 1 — expand required_agents from required_producers
    # ------------------------------------------------------------------

    def _expand_required_agents(
        self, staleness_plan: StalenessPlan
    ) -> tuple[list[str], list[str], dict[str, str]]:
        """Expand required_producers to include EXECUTION_ONLY prerequisites.

        Returns (required_ordered, blocked_agents, blocked_reasons).
        """
        required: set[str] = set(staleness_plan.required_producers)
        blocked: dict[str, str] = {}

        queue: deque[str] = deque(staleness_plan.required_producers)
        visited: set[str] = set()

        while queue:
            agent_name = queue.popleft()
            if agent_name in visited:
                continue
            visited.add(agent_name)

            try:
                dep = DependencyRegistry.get_dependency(agent_name)
            except KeyError:
                continue

            for consumed_path in dep.consumes:
                if classify_path(consumed_path) != PathKind.EXECUTION_ONLY:
                    continue
                # EXECUTION_ONLY paths are never persisted in ResearchState, so their
                # producer must always run — whether the path is stale or not.
                producers = DependencyRegistry.agents_producing(consumed_path)
                if not producers:
                    if agent_name not in blocked:
                        blocked[agent_name] = (
                            f"consumes '{consumed_path}' (EXECUTION_ONLY) "
                            f"which has no known producer in the registry"
                        )
                else:
                    for producer in producers:
                        if producer not in required:
                            required.add(producer)
                            queue.append(producer)

        # Return in declaration order for determinism
        required_ordered = [
            d.agent_name
            for d in DependencyRegistry.list_dependencies()
            if d.agent_name in required
        ]
        return required_ordered, sorted(blocked.keys()), dict(blocked)

    # ------------------------------------------------------------------
    # Step 2 — topological execution groups
    # ------------------------------------------------------------------

    def _compute_execution_groups(self, all_agents: list[str]) -> list[list[str]]:
        """Assign agents to topological levels using Kahn's algorithm.

        Within each level all agents may execute in parallel (no dependency edges
        exist between them in the restricted subgraph).

        Only edges between agents that are both within *all_agents* are considered;
        dependencies on agents outside the plan are ignored (those paths are read
        directly from ResearchState).
        """
        if not all_agents:
            return []

        agent_set = set(all_agents)

        # predecessors[A] = set of agents in the plan that must complete before A
        predecessors: dict[str, set[str]] = {}
        for agent_name in all_agents:
            try:
                dep = DependencyRegistry.get_dependency(agent_name)
            except KeyError:
                predecessors[agent_name] = set()
                continue
            preds: set[str] = set()
            for consumed_path in dep.consumes:
                for producer in DependencyRegistry.agents_producing(consumed_path):
                    if producer in agent_set and producer != agent_name:
                        preds.add(producer)
            predecessors[agent_name] = preds

        groups: list[list[str]] = []
        scheduled: set[str] = set()

        while len(scheduled) < len(all_agents):
            ready = sorted(
                a
                for a in all_agents
                if a not in scheduled and predecessors[a].issubset(scheduled)
            )
            if not ready:
                # Cycle in registry (should not happen) — emit remainder as one group
                remaining = sorted(set(all_agents) - scheduled)
                groups.append(remaining)
                break
            groups.append(ready)
            scheduled.update(ready)

        return groups

    # ------------------------------------------------------------------
    # Step 3 — per-agent reasoning
    # ------------------------------------------------------------------

    def _build_reasoning(
        self,
        required_agents: list[str],
        optional_agents: list[str],
        staleness_plan: StalenessPlan,
    ) -> dict[str, str]:
        stale_paths_set = set(staleness_plan.stale_paths)
        required_set = set(required_agents)
        reasoning: dict[str, str] = {}

        for agent_name in required_agents + optional_agents:
            try:
                dep = DependencyRegistry.get_dependency(agent_name)
            except KeyError:
                reasoning[agent_name] = "unknown agent"
                continue

            stale_produces = [p for p in dep.produces if p in stale_paths_set]

            if agent_name in required_set:
                if stale_produces:
                    persisted = [
                        p for p in stale_produces
                        if classify_path(p) == PathKind.PERSISTED
                    ]
                    eo = [
                        p for p in stale_produces
                        if classify_path(p) == PathKind.EXECUTION_ONLY
                    ]
                    if persisted:
                        paths_str = ", ".join(f"'{p}'" for p in persisted)
                        reasoning[agent_name] = (
                            f"required: produces stale PERSISTED path(s): {paths_str}"
                        )
                    else:
                        paths_str = ", ".join(f"'{p}'" for p in eo)
                        reasoning[agent_name] = (
                            f"required: produces stale EXECUTION_ONLY path(s): {paths_str}; "
                            f"needed as prerequisite for downstream PERSISTED restoration"
                        )
                else:
                    # Added as EXECUTION_ONLY prerequisite for another required agent
                    eo_produced = [
                        p for p in dep.produces
                        if classify_path(p) == PathKind.EXECUTION_ONLY
                    ]
                    paths_str = ", ".join(f"'{p}'" for p in eo_produced)
                    reasoning[agent_name] = (
                        f"prerequisite: produces EXECUTION_ONLY path(s): {paths_str}; "
                        f"consumed by downstream required agents"
                    )
            else:
                # optional
                paths_str = ", ".join(f"'{p}'" for p in stale_produces)
                reasoning[agent_name] = (
                    f"optional: produces stale EXECUTION_ONLY path(s): {paths_str}; "
                    f"not needed for PERSISTED state restoration"
                )

        return reasoning
