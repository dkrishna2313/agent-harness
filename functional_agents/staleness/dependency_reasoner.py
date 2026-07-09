"""DependencyReasoner — deterministic staleness engine (J13.2).

Analyzes a list of StateChanges against the DependencyRegistry to produce a
StalenessPlan. No LLM. No heuristics. No probabilistic reasoning. Pure BFS
over the consumption graph.

Algorithm
---------
1. Expand changed paths from StateChanges (container paths unfold to sub-paths).
2. BFS from each changed path through agents that CONSUME it.
   For each consumer: its produces become stale → add to queue → continue.
3. Classify stale paths by PathKind.
4. Derive stale_agents, required_producers, confidence.
5. Record per-path reasoning throughout traversal.

EXTERNAL paths trigger BFS propagation but are placed in external_dependencies,
not stale_paths, because the pipeline cannot recompute them.
"""

from __future__ import annotations

from collections import deque

from ..dependencies import DependencyRegistry
from ..session.research_state import ResearchState
from ..session.state_change import StateChange
from .path_kind import (
    PATH_CLASSIFICATION,
    PathKind,
    classify_path,
    expand_path,
    is_container_path,
)
from .staleness_plan import StalenessPlan


class DependencyReasoner:
    """Deterministic engine that computes staleness from a list of StateChanges.

    Usage
    -----
    plan = DependencyReasoner().analyze(research_state, state_changes)
    """

    def analyze(
        self,
        research_state: ResearchState | None,
        state_changes: list[StateChange],
    ) -> StalenessPlan:
        """Analyze *state_changes* and return a StalenessPlan.

        Parameters
        ----------
        research_state : current ResearchState snapshot (accepted for context;
                         the analysis is conservative and does not require it)
        state_changes  : list of StateChange records to analyze
        """
        # 1. Extract and expand changed paths
        changed_paths, expansion_used, source_ids = self._extract_changed_paths(
            state_changes
        )

        # 2. BFS through consumption graph
        reasoning: dict[str, str] = {}
        stale_paths, external_deps = self._compute_stale_paths(
            changed_paths, state_changes, reasoning
        )

        # 3. Classify stale paths
        persisted = [p for p in stale_paths if classify_path(p) == PathKind.PERSISTED]
        execution_only = [
            p for p in stale_paths if classify_path(p) == PathKind.EXECUTION_ONLY
        ]

        # 4. Determine stale agents and required producers
        stale_agents = self._determine_stale_agents(stale_paths)
        required_producers = self._determine_required_producers(persisted)

        # 5. Determine analysis confidence
        confidence = self._determine_confidence(changed_paths, expansion_used)

        return StalenessPlan.create(
            source_changes=source_ids,
            changed_paths=changed_paths,
            stale_paths=stale_paths,
            stale_agents=stale_agents,
            required_producers=required_producers,
            persisted_paths=persisted,
            execution_only_paths=execution_only,
            external_dependencies=external_deps,
            reasoning=reasoning,
            confidence=confidence,
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _extract_changed_paths(
        self, state_changes: list[StateChange]
    ) -> tuple[list[str], bool, list[str]]:
        """Return ``(expanded_paths, expansion_was_used, source_change_ids)``.

        Container paths (e.g. ``research_state``) are expanded to their
        constituent sub-paths. Deduplication preserves first-seen order.
        """
        paths: list[str] = []
        expansion_used = False
        source_ids: list[str] = []

        for sc in state_changes:
            source_ids.append(sc.change_id)
            for path in sc.affected_paths:
                if is_container_path(path):
                    expansion_used = True
                    paths.extend(expand_path(path))
                else:
                    paths.append(path)

        # Deduplicate while preserving first-seen order
        seen: set[str] = set()
        result: list[str] = []
        for p in paths:
            if p not in seen:
                seen.add(p)
                result.append(p)

        return result, expansion_used, source_ids

    def _compute_stale_paths(
        self,
        changed_paths: list[str],
        state_changes: list[StateChange],
        reasoning: dict[str, str],
    ) -> tuple[list[str], list[str]]:
        """BFS through consumption graph.

        Returns ``(sorted_stale_paths, sorted_external_deps)``.

        EXTERNAL paths are placed in ``external_deps`` rather than
        ``stale_paths`` because they cannot be recomputed by the pipeline.
        They still propagate staleness to their consumers.
        """
        # Build per-path initial reasons from StateChange metadata
        initial_reasons: dict[str, str] = {}
        for sc in state_changes:
            label = f"{sc.change_type} [{sc.change_id}]"
            for raw_path in sc.affected_paths:
                for expanded in expand_path(raw_path):
                    if expanded not in initial_reasons:
                        initial_reasons[expanded] = label

        stale: set[str] = set()
        external: set[str] = set()
        visited: set[str] = set()
        queue: deque[str] = deque()

        def _enqueue(path: str, reason: str) -> None:
            if path in visited:
                return
            kind = classify_path(path)
            if kind == PathKind.EXTERNAL:
                external.add(path)
            else:
                stale.add(path)
            reasoning[path] = reason
            queue.append(path)

        for path in changed_paths:
            _enqueue(path, initial_reasons.get(path, "declared changed"))

        while queue:
            path = queue.popleft()
            if path in visited:
                continue
            visited.add(path)

            # Every agent that consumes this path now has stale inputs →
            # its products become stale.
            for consumer_name in DependencyRegistry.agents_consuming(path):
                consumer = DependencyRegistry.get_dependency(consumer_name)
                for produced in consumer.produces:
                    parent_reason = reasoning.get(path, "stale")
                    new_reason = (
                        f"produced by {consumer_name}, "
                        f"which consumes '{path}' "
                        f"({parent_reason})"
                    )
                    _enqueue(produced, new_reason)

        return sorted(stale), sorted(external)

    def _determine_stale_agents(self, stale_paths: list[str]) -> list[str]:
        """Return agent names (declaration order) that produce ≥1 stale path."""
        stale_set = set(stale_paths)
        agents: list[str] = []
        seen: set[str] = set()
        for dep in DependencyRegistry.list_dependencies():
            if dep.agent_name not in seen:
                for produced in dep.produces:
                    if produced in stale_set:
                        agents.append(dep.agent_name)
                        seen.add(dep.agent_name)
                        break
        return agents

    def _determine_required_producers(
        self, persisted_stale_paths: list[str]
    ) -> list[str]:
        """Return agent names (declaration order) that produce ≥1 stale PERSISTED path."""
        persisted_set = set(persisted_stale_paths)
        producers: list[str] = []
        seen: set[str] = set()
        for dep in DependencyRegistry.list_dependencies():
            if dep.agent_name not in seen:
                for produced in dep.produces:
                    if produced in persisted_set:
                        producers.append(dep.agent_name)
                        seen.add(dep.agent_name)
                        break
        return producers

    def _determine_confidence(
        self, changed_paths: list[str], expansion_used: bool
    ) -> str:
        """Return ``"HIGH"``, ``"MEDIUM"``, or ``"LOW"`` confidence.

        HIGH   — all changed paths are individually known in the classification map
        MEDIUM — all changed paths are known but some container expansion was used
        LOW    — some changed paths are completely unknown to the registry
        """
        if not changed_paths:
            return "LOW"
        unknown = [p for p in changed_paths if p not in PATH_CLASSIFICATION]
        if unknown:
            return "LOW"
        if expansion_used:
            return "MEDIUM"
        return "HIGH"
