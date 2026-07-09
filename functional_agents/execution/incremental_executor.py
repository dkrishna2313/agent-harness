"""IncrementalExecutor — executes an ExecutionPlan against a ResearchSession (J13.4).

Architecture
------------
The executor is analysis-free: it receives a fully-computed ExecutionPlan
and simply runs the required_agents in topological order.

It does NOT:
- reason about dependencies
- compute staleness
- build execution plans

Each agent is constructed from the same factories used by the full Orchestrator,
so production agent logic is reused without modification.

Session contract
----------------
Regardless of success or failure, the returned ResearchSession is always
consistent: research_state reflects the context after the last successful
agent, a StateChange is recorded, an IterationRecord is appended, and a
Snapshot is taken.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from functional_agents.context import AgentContext, AgentResult, NextAction
from functional_agents.dependencies import DependencyRegistry
from functional_agents.planning.execution_plan import ExecutionPlan
from functional_agents.session.iteration_record import IterationRecord
from functional_agents.session.research_session import ResearchSession
from functional_agents.session.research_state import ResearchState
from functional_agents.session.state_change import ChangeType, StateChange
from functional_agents.staleness import PathKind, classify_path

from .execution_result import ExecutionResult, ExecutionStatus

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent names that take no constructor arguments
# ---------------------------------------------------------------------------
_NO_ARG_AGENTS: frozenset[str] = frozenset({
    "ResearchGapAgent",
    "MultiProfileAgent",
    "ScenarioAgent",
    "RecommendationImprovementAgent",
    "RecommendationSynthesisAgent",
    "QAAgent",
    "IterationPlanAgent",
})

# ---------------------------------------------------------------------------
# Agent names that take (client, domain_profiles)
# ---------------------------------------------------------------------------
_CLIENT_AGENTS: frozenset[str] = frozenset({
    "ProblemFramingAgent",
    "ResearchStrategyAgent",
    "HypothesisAgent",
    "StrategicSynthesisAgent",
    "ChallengeAgent",
    "AssumptionAgent",
    "RiskAgent",
    "OpportunityAgent",
    "RecommendationAgent",
    "StrategicOptionAgent",
    "DecisionAnalysisAgent",
    "ExecutiveConfidenceAgent",
    "PlannerAgent",
})


class IncrementalExecutor:
    """Executes only the required_agents from an ExecutionPlan.

    Parameters mirror those of the full Orchestrator so that both execution
    modes can be configured identically from the CLI.

    Usage
    -----
    executor = IncrementalExecutor(
        client=client,
        profile_names=["ai_data_centers"],
        sources_dir=Path("sources"),
        out_path=Path("outputs/report.md"),
    )
    result = executor.execute(execution_plan, session)
    """

    def __init__(
        self,
        *,
        client: Any,
        profile_names: list[str],
        sources_dir: str | Path = "sources",
        out_path: Path,
        top_evidence: int = 50,
        top_chunks: int = 20,
        web_search: bool = False,
        knowledge_store: str | Path | None = None,
        use_reranker: bool = False,
    ) -> None:
        self._client = client
        self._profile_names = list(profile_names)
        self._sources_dir = Path(sources_dir)
        self._out_path = Path(out_path)
        self._top_evidence = top_evidence
        self._top_chunks = top_chunks
        self._web_search = web_search
        self._knowledge_store = Path(knowledge_store) if knowledge_store else None
        self._use_reranker = use_reranker

        # Load profiles once — same as Orchestrator
        from research_agent.profile import load_profile, DomainProfile
        self._loaded_profiles: list[DomainProfile] = []
        self._domain_profile = None
        for i, name in enumerate(self._profile_names):
            try:
                p = load_profile(name)
                self._loaded_profiles.append(p)
                if i == 0:
                    self._domain_profile = p
            except FileNotFoundError:
                LOGGER.warning("[IncrementalExecutor] profile not found: %r", name)

        # Initialise Knowledge Layer retriever (mirrors Orchestrator)
        self._retriever = None
        if self._knowledge_store and self._knowledge_store.exists():
            try:
                from knowledge.store import KnowledgeStore
                from knowledge.retriever import EvidenceRetriever
                from knowledge.embeddings import get_provider
                from knowledge.health import check_store_health
                _ks = KnowledgeStore(self._knowledge_store)
                _health = check_store_health(_ks)
                if _health.runtime_ready:
                    self._retriever = EvidenceRetriever(_ks, provider=get_provider())
            except Exception as exc:
                LOGGER.warning("[IncrementalExecutor] Knowledge Layer init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        execution_plan: ExecutionPlan,
        session: ResearchSession,
    ) -> ExecutionResult:
        """Execute required_agents from *execution_plan* against *session*.

        Returns an ExecutionResult containing an updated ResearchSession.
        The session is always consistent: if an agent fails, the session
        reflects the state after the last successful agent.
        """
        if not execution_plan.required_agents:
            LOGGER.info("[IncrementalExecutor] nothing to execute (empty plan)")
            return ExecutionResult(
                status=ExecutionStatus.EMPTY,
                session=session,
                completed_agents=[],
                failed_agent=None,
                failure_reason=None,
                execution_plan_id=execution_plan.plan_id,
            )

        # Determine execution sequence: topological order restricted to required agents
        required_set = set(execution_plan.required_agents)
        to_run = [
            a for a in execution_plan.execution_order
            if a in required_set
        ]

        ctx = self._build_context(session)
        completed: list[str] = []

        for agent_name in to_run:
            try:
                LOGGER.info("[IncrementalExecutor] running %s", agent_name)
                agent = self._build_agent(agent_name)
                result = self._run_agent(agent, ctx)
                ctx = result.context
                completed.append(agent_name)
                # Post-step hooks (mirrors orchestrator post-processing)
                ctx = self._post_hook(agent_name, ctx)
            except Exception as exc:
                LOGGER.error(
                    "[IncrementalExecutor] agent %s failed: %s", agent_name, exc
                )
                updated = self._finalize_session(
                    session, ctx, completed, execution_plan,
                    failed_agent=agent_name,
                    failure_reason=str(exc),
                )
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    session=updated,
                    completed_agents=completed,
                    failed_agent=agent_name,
                    failure_reason=str(exc),
                    execution_plan_id=execution_plan.plan_id,
                    trace={"failed_at": agent_name, "error": str(exc)},
                )

        updated = self._finalize_session(
            session, ctx, completed, execution_plan,
        )
        LOGGER.info(
            "[IncrementalExecutor] complete — %d/%d agents ran",
            len(completed), len(execution_plan.required_agents),
        )
        return ExecutionResult(
            status=ExecutionStatus.COMPLETE,
            session=updated,
            completed_agents=completed,
            failed_agent=None,
            failure_reason=None,
            execution_plan_id=execution_plan.plan_id,
            trace={"completed_agents": completed},
        )

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def _build_context(self, session: ResearchSession) -> AgentContext:
        """Reconstruct an AgentContext from the session's persisted ResearchState."""
        state = session.research_state
        meta = session.metadata

        # Restore PERSISTED fields directly from ResearchState
        engagement = dict(state.engagement or {})
        research_object = dict(state.research_object or {})
        decision_model = dict(state.decision_model or {})
        research_gap_analysis = dict(state.research_gap_analysis or {})
        executive_confidence = dict(state.executive_confidence or {})
        iteration_plan = dict(state.iteration_plan or {})

        # Recover question from research_object if available
        question = research_object.get("question", "")

        # Recover goal from session metadata engagement spec
        goal = ""
        if "engagement_spec" in meta:
            try:
                from functional_agents.engagement_spec import EngagementSpec
                spec = EngagementSpec.model_validate(meta["engagement_spec"])
                goal = spec.to_framing_brief()
            except Exception:
                pass

        profiles = list(meta.get("profiles") or self._profile_names)
        execution_profile = meta.get("execution_profile") or (profiles[0] if profiles else "")

        ctx = AgentContext(
            question=question,
            goal=goal,
            engagement=engagement,
            research_object=research_object,
            decision_model=decision_model,
            research_gap_analysis=research_gap_analysis,
            executive_confidence=executive_confidence,
            iteration_plan=iteration_plan,
            profiles=profiles,
            execution_profile=execution_profile,
            run_id=uuid.uuid4().hex[:12],
        )

        # Attach client and performance tracker (mirrors Orchestrator)
        from functional_agents.performance import PerformanceTracker
        ctx.trace["_client"] = self._client
        ctx.trace["_perf_tracker"] = PerformanceTracker()
        ctx.trace["_incremental"] = True
        ctx.trace["_session_id"] = session.session_id

        return ctx

    # ------------------------------------------------------------------
    # Agent construction
    # ------------------------------------------------------------------

    def _build_agent(self, agent_name: str) -> Any:
        """Construct the production agent instance for *agent_name*."""
        if agent_name in _NO_ARG_AGENTS:
            return self._build_no_arg_agent(agent_name)
        if agent_name in _CLIENT_AGENTS:
            return self._build_client_agent(agent_name)
        if agent_name == "EvidenceAgent":
            return self._build_evidence_agent()
        if agent_name == "ReportAgent":
            return self._build_report_agent()
        raise ValueError(
            f"[IncrementalExecutor] no factory registered for agent {agent_name!r}"
        )

    def _build_no_arg_agent(self, agent_name: str) -> Any:
        import importlib
        module_names = {
            "ResearchGapAgent":             "functional_agents.research_gap_agent",
            "MultiProfileAgent":            "functional_agents.multi_profile_agent",
            "ScenarioAgent":                "functional_agents.scenario_agent",
            "RecommendationImprovementAgent": "functional_agents.recommendation_improvement_agent",
            "RecommendationSynthesisAgent": "functional_agents.recommendation_synthesis_agent",
            "QAAgent":                      "functional_agents.qa_agent",
            "IterationPlanAgent":           "functional_agents.iteration_plan_agent",
        }
        mod = importlib.import_module(module_names[agent_name])
        cls = getattr(mod, agent_name)
        return cls()

    def _build_client_agent(self, agent_name: str) -> Any:
        import importlib
        module_names = {
            "ProblemFramingAgent":      "functional_agents.problem_framing_agent",
            "ResearchStrategyAgent":    "functional_agents.research_strategy_agent",
            "HypothesisAgent":          "functional_agents.hypothesis_agent",
            "StrategicSynthesisAgent":  "functional_agents.strategic_synthesis_agent",
            "ChallengeAgent":           "functional_agents.challenge_agent",
            "AssumptionAgent":          "functional_agents.assumption_agent",
            "RiskAgent":                "functional_agents.risk_agent",
            "OpportunityAgent":         "functional_agents.opportunity_agent",
            "RecommendationAgent":      "functional_agents.recommendation_agent",
            "StrategicOptionAgent":     "functional_agents.strategic_option_agent",
            "DecisionAnalysisAgent":    "functional_agents.decision_analysis_agent",
            "ExecutiveConfidenceAgent": "functional_agents.executive_confidence_agent",
            "PlannerAgent":             "functional_agents.planner_agent",
        }
        mod = importlib.import_module(module_names[agent_name])
        cls = getattr(mod, agent_name)
        return cls(client=self._client, domain_profiles=self._loaded_profiles)

    def _build_evidence_agent(self) -> Any:
        from functional_agents.evidence_agent import EvidenceAgent
        return EvidenceAgent(
            sources_dir=self._sources_dir,
            client=self._client,
            top_evidence=self._top_evidence,
            top_chunks=self._top_chunks,
            domain_profile=self._domain_profile,
            domain_profiles=self._loaded_profiles,
            retriever=self._retriever,
            use_reranker=self._use_reranker,
        )

    def _build_report_agent(self) -> Any:
        from functional_agents.report_agent import ReportAgent
        return ReportAgent(
            out_path=self._out_path,
            domain_profile=self._domain_profile,
        )

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _run_agent(self, agent: Any, ctx: AgentContext) -> AgentResult:
        """Execute one agent.  Isolated as a method for test override."""
        from functional_agents.orchestrator import _step
        return _step(agent, ctx)

    def _post_hook(self, agent_name: str, ctx: AgentContext) -> AgentContext:
        """Apply any post-step processing required after a specific agent."""
        if agent_name == "RecommendationAgent":
            from functional_agents.orchestrator import _apply_recommendation_linkage
            ctx = _apply_recommendation_linkage(ctx)
        return ctx

    # ------------------------------------------------------------------
    # Session finalisation
    # ------------------------------------------------------------------

    def _finalize_session(
        self,
        session: ResearchSession,
        ctx: AgentContext,
        completed_agents: list[str],
        execution_plan: ExecutionPlan,
        *,
        failed_agent: str | None = None,
        failure_reason: str | None = None,
    ) -> ResearchSession:
        """Persist context back to session, append StateChange + IterationRecord + Snapshot."""
        # 1. Update ResearchState from the latest context
        session.research_state = ResearchState.from_context(ctx)

        # 2. Determine which PERSISTED paths were produced
        affected_paths: list[str] = []
        for name in completed_agents:
            try:
                dep = DependencyRegistry.get_dependency(name)
                for path in dep.produces:
                    if classify_path(path) == PathKind.PERSISTED and path not in affected_paths:
                        affected_paths.append(path)
            except KeyError:
                pass
        if not affected_paths:
            affected_paths = ["research_state"]

        # 3. Record StateChange
        desc_parts = [
            f"Incremental execution — {len(completed_agents)}/{len(execution_plan.required_agents)} agents",
            f"plan_id={execution_plan.plan_id}",
        ]
        if failed_agent:
            desc_parts.append(f"failed_at={failed_agent}")
        session.record_state_change(StateChange.create(
            source="incremental_executor",
            change_type=ChangeType.REPLACE,
            affected_paths=affected_paths,
            description=" — ".join(desc_parts),
            metadata={
                "execution_plan_id": execution_plan.plan_id,
                "completed_agents": completed_agents,
                "failed_agent": failed_agent,
            },
        ))

        # 4. Append IterationRecord
        if failed_agent:
            summary = (
                f"Incremental execution FAILED at {failed_agent}: {failure_reason} "
                f"({len(completed_agents)} agents ran)"
            )
        else:
            summary = (
                f"Incremental execution complete — "
                f"{len(completed_agents)} agents — plan_id={execution_plan.plan_id}"
            )
        session.add_iteration(IterationRecord(
            iteration_number=len(session.iteration_history),
            timestamp=datetime.now(timezone.utc).isoformat(),
            trigger="incremental",
            summary=summary,
            completed_tasks=list(completed_agents),
            notes=f"execution_plan_id={execution_plan.plan_id}",
        ))

        # 5. Take snapshot
        session.take_snapshot()

        return session
