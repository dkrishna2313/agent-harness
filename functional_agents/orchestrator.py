"""Orchestrator – adaptive workflow engine for the functional agent pipeline (J5.5).

Public API
----------
WorkflowState     – canonical state names (re-exported from context)
NextAction        – agent routing tokens (re-exported from context)
AgentResult       – standardised agent return type (re-exported from context)
AgentOrchestrator – state-machine orchestrator; replaces the fixed loop
Orchestrator      – thin compatibility wrapper (used by CLI)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from research_agent.decision_model import from_question as _dm_from_question, write_decision_model
from research_agent.engagement import from_question as _engagement_from_question, link_decision_model as _link_dm, write_engagement
from research_agent.profile import DomainProfile, load_profile
from research_agent.research_object import create_research_object

from .context import (
    AgentContext,
    AgentResult,
    ContextValidationError,
    NextAction,
    WorkflowState,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_recommendation_linkage(ctx: AgentContext) -> AgentContext:
    """J7.2 – link assumptions ↔ recommendations and re-persist both artifacts."""
    from research_agent.log import PROGRESS
    from .recommendation_linkage import build_recommendation_linkage, persist_linkage

    assumptions = ctx.assumptions
    recommendations = ctx.recommendations

    if not assumptions or not recommendations:
        LOGGER.log(
            PROGRESS,
            "[RecommendationLinkage] skipped — assumptions=%d recommendations=%d",
            len(assumptions), len(recommendations),
        )
        return ctx

    linked_assumptions, linked_recommendations = build_recommendation_linkage(
        assumptions, recommendations
    )

    # Count links for observability
    link_count = sum(len(a.get("supported_recommendation_ids", [])) for a in linked_assumptions)
    LOGGER.log(
        PROGRESS,
        "[RecommendationLinkage] %d assumptions × %d recommendations → %d links",
        len(linked_assumptions), len(linked_recommendations), link_count,
    )

    # Write back into context
    ctx.assumptions = linked_assumptions
    ctx.recommendations = linked_recommendations
    if ctx.research_object:
        ctx.research_object["strategic_assumptions"] = linked_assumptions
        ctx.research_object["recommendations"] = linked_recommendations

    ctx.trace["_recommendation_linkage"] = {
        "assumption_count": len(linked_assumptions),
        "recommendation_count": len(linked_recommendations),
        "link_count": link_count,
    }

    # Re-persist Decision Model and Research Object
    dm_id = ctx.research_object.get("decision_model_id") if ctx.research_object else None
    dm_ok, ro_ok = persist_linkage(dm_id, linked_assumptions, linked_recommendations, ctx.research_object or {})
    ctx.trace["_recommendation_linkage"]["dm_persisted"] = dm_ok
    ctx.trace["_recommendation_linkage"]["ro_persisted"] = ro_ok

    return ctx


def _step(agent: Any, ctx: AgentContext) -> AgentResult:
    """Run one agent, validate its AgentResult, and append to workflow_path."""
    from .contract import validate_agent_result
    result = agent.run(ctx)
    result.context.workflow_path.append(agent.name)
    # Accumulate runtime contract checks for the trace
    check = validate_agent_result(result, agent.name)
    runtime_checks: dict = result.context.trace.setdefault("_contract_runtime", {})
    runtime_checks[agent.name] = check
    if check.get("error"):
        LOGGER.warning("[contract] %s", check["error"])
    return result


# ---------------------------------------------------------------------------
# AgentOrchestrator (J5.5.1)
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """State-machine orchestrator.  Executes agents, reads their next_action,
    and drives the workflow — supporting iteration loops and re-planning.

    States (J5.5.3 / J6.1):
        [PROBLEM_FRAMING →] PLANNING → EVIDENCE → QA → REPORT → COMPLETE
        PROBLEM_FRAMING is prepended when problem_framing_factory is provided.
        QA may loop back to EVIDENCE (REQUEST_EVIDENCE) or PLANNING (REQUEST_REPLAN)
        until max_iterations is reached, then forces REPORT.

    Agents are constructed lazily via factory callables so each re-invocation
    gets a fresh agent instance (important for EvidenceAgent which owns internal
    pipeline state via DcPowerAgent).
    """

    def __init__(
        self,
        *,
        planner_factory: Any,
        evidence_factory: Any,
        qa_factory: Any,
        report_factory: Any,
        problem_framing_factory: Any = None,
        research_strategy_factory: Any = None,
        hypothesis_factory: Any = None,
        challenge_factory: Any = None,
        assumption_factory: Any = None,
        risk_factory: Any = None,
        opportunity_factory: Any = None,
        recommendation_factory: Any = None,
        multi_profile_factory: Any = None,
        scenario_factory: Any = None,
        improvement_factory: Any = None,
        synthesis_factory: Any = None,
        strategic_option_factory: Any = None,
        decision_analysis_factory: Any = None,
        max_iterations: int = 3,
    ) -> None:
        self._problem_framing_factory   = problem_framing_factory
        self._research_strategy_factory = research_strategy_factory
        self._hypothesis_factory        = hypothesis_factory
        self._challenge_factory         = challenge_factory
        self._assumption_factory        = assumption_factory
        self._risk_factory              = risk_factory
        self._opportunity_factory       = opportunity_factory
        self._recommendation_factory    = recommendation_factory
        self._multi_profile_factory     = multi_profile_factory
        self._scenario_factory          = scenario_factory
        self._improvement_factory       = improvement_factory
        self._synthesis_factory         = synthesis_factory
        self._strategic_option_factory  = strategic_option_factory
        self._decision_analysis_factory = decision_analysis_factory
        self._planner_factory  = planner_factory
        self._evidence_factory = evidence_factory
        self._qa_factory       = qa_factory
        self._report_factory   = report_factory
        self._max_iterations   = max_iterations

    def run(self, ctx: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        state = (
            WorkflowState.PROBLEM_FRAMING
            if self._problem_framing_factory is not None
            else WorkflowState.PLANNING
        )
        termination_reason = NextAction.COMPLETE
        ctx.iteration_count = 0

        while state != WorkflowState.COMPLETE:

            ctx.workflow_state = state
            LOGGER.log(PROGRESS, "[Orchestrator] state=%s  iteration=%d", state, ctx.iteration_count)

            # ---- PROBLEM FRAMING (J6.1) -------------------------------------
            if state == WorkflowState.PROBLEM_FRAMING:
                result = _step(self._problem_framing_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.RESEARCH_STRATEGY
                    if self._research_strategy_factory is not None
                    else WorkflowState.PLANNING
                )

            # ---- RESEARCH STRATEGY (J6.2) -----------------------------------
            elif state == WorkflowState.RESEARCH_STRATEGY:
                result = _step(self._research_strategy_factory(), ctx)
                ctx = result.context
                state = WorkflowState.PLANNING

            # ---- PLANNING ---------------------------------------------------
            elif state == WorkflowState.PLANNING:
                result = _step(self._planner_factory(), ctx)
                ctx = result.context
                state = WorkflowState.EVIDENCE

            # ---- EVIDENCE ---------------------------------------------------
            elif state == WorkflowState.EVIDENCE:
                result = _step(self._evidence_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.HYPOTHESIS
                    if self._hypothesis_factory is not None
                    else WorkflowState.QA
                )

            # ---- HYPOTHESIS (J6.3) ------------------------------------------
            elif state == WorkflowState.HYPOTHESIS:
                result = _step(self._hypothesis_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.CHALLENGE
                    if self._challenge_factory is not None
                    else WorkflowState.QA
                )

            # ---- CHALLENGE (J6.4) -------------------------------------------
            elif state == WorkflowState.CHALLENGE:
                result = _step(self._challenge_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.ASSUMPTION
                    if self._assumption_factory is not None
                    else (
                        WorkflowState.RECOMMENDATION
                        if self._recommendation_factory is not None
                        else WorkflowState.QA
                    )
                )

            # ---- ASSUMPTION (J7.1) ------------------------------------------
            elif state == WorkflowState.ASSUMPTION:
                result = _step(self._assumption_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.RECOMMENDATION
                    if self._recommendation_factory is not None
                    else WorkflowState.QA
                )

            # ---- RECOMMENDATION (J6.5) --------------------------------------
            elif state == WorkflowState.RECOMMENDATION:
                result = _step(self._recommendation_factory(), ctx)
                ctx = result.context
                # J7.2 – link assumptions ↔ recommendations immediately after generation
                ctx = _apply_recommendation_linkage(ctx)
                state = (
                    WorkflowState.RISK
                    if self._risk_factory is not None
                    else (
                        WorkflowState.MULTI_PROFILE
                        if self._multi_profile_factory is not None
                        else (
                            WorkflowState.SCENARIO
                            if self._scenario_factory is not None
                            else WorkflowState.QA
                        )
                    )
                )

            # ---- RISK (J7.3) ------------------------------------------------
            elif state == WorkflowState.RISK:
                result = _step(self._risk_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.OPPORTUNITY
                    if self._opportunity_factory is not None
                    else (
                        WorkflowState.MULTI_PROFILE
                        if self._multi_profile_factory is not None
                        else (
                            WorkflowState.SCENARIO
                            if self._scenario_factory is not None
                            else WorkflowState.QA
                        )
                    )
                )

            # ---- OPPORTUNITY (J7.4) -----------------------------------------
            elif state == WorkflowState.OPPORTUNITY:
                result = _step(self._opportunity_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.STRATEGIC_OPTIONS
                    if self._strategic_option_factory is not None
                    else (
                        WorkflowState.MULTI_PROFILE
                        if self._multi_profile_factory is not None
                        else (
                            WorkflowState.SCENARIO
                            if self._scenario_factory is not None
                            else WorkflowState.QA
                        )
                    )
                )

            # ---- MULTI-PROFILE (J5.6a) --------------------------------------
            elif state == WorkflowState.MULTI_PROFILE:
                result = _step(self._multi_profile_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.SCENARIO
                    if self._scenario_factory is not None
                    else WorkflowState.QA
                )

            # ---- SCENARIO (J6.8) --------------------------------------------
            elif state == WorkflowState.SCENARIO:
                result = _step(self._scenario_factory(), ctx)
                ctx = result.context
                state = WorkflowState.QA

            # ---- QA ---------------------------------------------------------
            elif state == WorkflowState.QA:
                result = _step(self._qa_factory(), ctx)
                ctx = result.context
                action = result.next_action

                if action == NextAction.REQUEST_EVIDENCE:
                    if ctx.iteration_count < self._max_iterations:
                        ctx.iteration_count += 1
                        LOGGER.log(
                            PROGRESS,
                            "[Orchestrator] QA requested more evidence — iteration %d/%d",
                            ctx.iteration_count, self._max_iterations,
                        )
                        state = WorkflowState.EVIDENCE
                    else:
                        LOGGER.warning(
                            "[Orchestrator] max_iterations=%d reached — forcing REPORT",
                            self._max_iterations,
                        )
                        termination_reason = "MAX_ITERATIONS_REACHED"
                        state = WorkflowState.REPORT

                elif action == NextAction.REQUEST_REPLAN:
                    if ctx.iteration_count < self._max_iterations:
                        ctx.iteration_count += 1
                        LOGGER.log(
                            PROGRESS,
                            "[Orchestrator] QA requested re-plan — iteration %d/%d",
                            ctx.iteration_count, self._max_iterations,
                        )
                        state = WorkflowState.PLANNING
                    else:
                        LOGGER.warning(
                            "[Orchestrator] max_iterations=%d reached — forcing REPORT",
                            self._max_iterations,
                        )
                        termination_reason = "MAX_ITERATIONS_REACHED"
                        state = WorkflowState.REPORT

                else:
                    if self._improvement_factory is not None:
                        state = WorkflowState.RECOMMENDATION_IMPROVEMENT
                    elif self._synthesis_factory is not None:
                        state = WorkflowState.RECOMMENDATION_SYNTHESIS
                    else:
                        state = WorkflowState.REPORT

            # ---- RECOMMENDATION IMPROVEMENT (J6.7) --------------------------
            elif state == WorkflowState.RECOMMENDATION_IMPROVEMENT:
                result = _step(self._improvement_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.RECOMMENDATION_SYNTHESIS
                    if self._synthesis_factory is not None
                    else WorkflowState.REPORT
                )

            # ---- RECOMMENDATION SYNTHESIS (J6.8c) ---------------------------
            elif state == WorkflowState.RECOMMENDATION_SYNTHESIS:
                result = _step(self._synthesis_factory(), ctx)
                ctx = result.context
                state = WorkflowState.REPORT

            # ---- STRATEGIC OPTIONS (J7.5) -----------------------------------
            elif state == WorkflowState.STRATEGIC_OPTIONS:
                result = _step(self._strategic_option_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.DECISION_ANALYSIS
                    if self._decision_analysis_factory is not None
                    else (
                        WorkflowState.MULTI_PROFILE
                        if self._multi_profile_factory is not None
                        else (
                            WorkflowState.SCENARIO
                            if self._scenario_factory is not None
                            else WorkflowState.QA
                        )
                    )
                )

            # ---- DECISION ANALYSIS (J7.6) -----------------------------------
            elif state == WorkflowState.DECISION_ANALYSIS:
                result = _step(self._decision_analysis_factory(), ctx)
                ctx = result.context
                state = (
                    WorkflowState.MULTI_PROFILE
                    if self._multi_profile_factory is not None
                    else (
                        WorkflowState.SCENARIO
                        if self._scenario_factory is not None
                        else WorkflowState.QA
                    )
                )

            # ---- REPORT -----------------------------------------------------
            elif state == WorkflowState.REPORT:
                ctx.workflow_state = WorkflowState.REPORT
                # Stash orchestrator summary for ReportAgent to inject
                ctx.trace["_orchestrator"] = {
                    "iterations": ctx.iteration_count,
                    "workflow_path": list(ctx.workflow_path) + ["ReportAgent"],
                    "termination_reason": termination_reason,
                    "max_iterations": self._max_iterations,
                }
                result = _step(self._report_factory(), ctx)
                ctx = result.context
                state = WorkflowState.COMPLETE

        ctx.workflow_state = WorkflowState.COMPLETE
        LOGGER.log(
            PROGRESS,
            "[Orchestrator] complete  path=%s  iterations=%d  reason=%s",
            "→".join(ctx.workflow_path), ctx.iteration_count, termination_reason,
        )
        return ctx


# ---------------------------------------------------------------------------
# Orchestrator – thin compatibility wrapper for CLI (J5.5.10)
# ---------------------------------------------------------------------------

class Orchestrator:
    """Thin wrapper that builds an AgentOrchestrator from config and runs it.

    The public interface (``run(question)``) is unchanged so the CLI needs no
    modifications.
    """

    def __init__(
        self,
        *,
        profile_names: list[str],
        sources_dir: str | Path = "sources",
        out_path: Path,
        client: Any = None,
        top_evidence: int = 50,
        top_chunks: int = 20,
        max_iterations: int = 3,
        web_search: bool = False,
    ) -> None:
        self._profile_names  = profile_names
        self._sources_dir    = Path(sources_dir)
        self._out_path       = out_path
        self._client         = client
        self._top_evidence   = top_evidence
        self._top_chunks     = top_chunks
        self._max_iterations = max_iterations
        self._web_search     = web_search

        from research_agent.log import PROGRESS

        # Load execution profile (first in list)
        self._domain_profile: DomainProfile | None = None
        if profile_names:
            try:
                self._domain_profile = load_profile(profile_names[0])
                LOGGER.log(PROGRESS, "Execution profile loaded: %s", profile_names[0])
            except FileNotFoundError as exc:
                LOGGER.warning("Could not load profile %r: %s", profile_names[0], exc)

        # Patch web search into the execution profile when requested via CLI flag
        if web_search and self._domain_profile is not None:
            from research_agent.profile import WebSearchConfig
            self._domain_profile = self._domain_profile.model_copy(update={
                "web_search": WebSearchConfig(enabled=True, max_results=5, max_pages=5)
            })

        # Verify supporting profiles (warn on missing; don't abort)
        for name in profile_names[1:]:
            try:
                load_profile(name)
                LOGGER.log(PROGRESS, "Supporting profile loaded: %s", name)
            except FileNotFoundError:
                LOGGER.warning("Supporting profile not found: %r", name)

    def run(self, question: str) -> AgentContext:
        """Build, validate, and execute the adaptive agent pipeline (question-driven)."""
        return self._run_internal(question=question, goal="")

    def run_from_goal(self, goal: str) -> AgentContext:
        """Build, validate, and execute the pipeline starting from a business goal (J6.1).

        ProblemFramingAgent runs first to derive research questions from the goal,
        then the standard PLANNING → EVIDENCE → QA → REPORT pipeline follows.
        """
        return self._run_internal(question="", goal=goal)

    def _run_internal(self, *, question: str, goal: str) -> AgentContext:
        """Shared implementation for question-driven and goal-driven runs."""
        from .planner_agent             import PlannerAgent
        from .evidence_agent            import EvidenceAgent
        from .qa_agent                  import QAAgent
        from .report_agent              import ReportAgent
        from .problem_framing_agent     import ProblemFramingAgent
        from .research_strategy_agent   import ResearchStrategyAgent
        from .hypothesis_agent          import HypothesisAgent
        from .challenge_agent           import ChallengeAgent
        from .assumption_agent          import AssumptionAgent
        from .risk_agent                import RiskAgent
        from .opportunity_agent         import OpportunityAgent
        from .recommendation_agent               import RecommendationAgent
        from .scenario_agent                     import ScenarioAgent
        from .recommendation_improvement_agent   import RecommendationImprovementAgent
        from .multi_profile_agent                import MultiProfileAgent
        from .recommendation_synthesis_agent    import RecommendationSynthesisAgent
        from .strategic_option_agent            import StrategicOptionAgent
        from .decision_analysis_agent           import DecisionAnalysisAgent

        execution_profile = self._profile_names[0] if self._profile_names else ""
        mock_mode = self._client is not None and getattr(self._client, "is_mock", False)

        # Collect all loaded DomainProfile objects for the planner
        loaded_profiles: list[DomainProfile] = []
        if self._domain_profile is not None:
            loaded_profiles.append(self._domain_profile)
        for name in self._profile_names[1:]:
            try:
                loaded_profiles.append(load_profile(name))
            except FileNotFoundError:
                pass

        # Agent factories — called fresh for each invocation in the loop
        def problem_framing_factory() -> ProblemFramingAgent:
            return ProblemFramingAgent(client=self._client, domain_profiles=loaded_profiles)

        def research_strategy_factory() -> ResearchStrategyAgent:
            return ResearchStrategyAgent(client=self._client, domain_profiles=loaded_profiles)

        def hypothesis_factory() -> HypothesisAgent:
            return HypothesisAgent(client=self._client, domain_profiles=loaded_profiles)

        def challenge_factory() -> ChallengeAgent:
            return ChallengeAgent(client=self._client, domain_profiles=loaded_profiles)

        def assumption_factory() -> AssumptionAgent:
            return AssumptionAgent(client=self._client, domain_profiles=loaded_profiles)

        def risk_factory() -> RiskAgent:
            return RiskAgent(client=self._client, domain_profiles=loaded_profiles)

        def opportunity_factory() -> OpportunityAgent:
            return OpportunityAgent(client=self._client, domain_profiles=loaded_profiles)

        def recommendation_factory() -> RecommendationAgent:
            return RecommendationAgent(client=self._client, domain_profiles=loaded_profiles)

        def multi_profile_factory() -> MultiProfileAgent:
            return MultiProfileAgent()

        def scenario_factory() -> ScenarioAgent:
            return ScenarioAgent()

        def improvement_factory() -> RecommendationImprovementAgent:
            return RecommendationImprovementAgent()

        def synthesis_factory() -> RecommendationSynthesisAgent:
            return RecommendationSynthesisAgent()

        def strategic_option_factory() -> StrategicOptionAgent:
            return StrategicOptionAgent(client=self._client, domain_profiles=loaded_profiles)

        def decision_analysis_factory() -> DecisionAnalysisAgent:
            return DecisionAnalysisAgent(client=self._client, domain_profiles=loaded_profiles)

        def planner_factory() -> PlannerAgent:
            return PlannerAgent(client=self._client, domain_profiles=loaded_profiles)

        def evidence_factory() -> EvidenceAgent:
            return EvidenceAgent(
                sources_dir=self._sources_dir,
                client=self._client,
                top_evidence=self._top_evidence,
                top_chunks=self._top_chunks,
                domain_profile=self._domain_profile,
                domain_profiles=loaded_profiles,
            )

        def qa_factory() -> QAAgent:
            return QAAgent()

        def report_factory() -> ReportAgent:
            return ReportAgent(
                out_path=self._out_path,
                domain_profile=self._domain_profile,
            )

        # For goal-driven runs the question is empty until ProblemFramingAgent runs;
        # use a placeholder so create_research_object gets a non-empty string.
        ro_question = question or goal

        # J7.0a – auto-create a minimal Strategic Engagement for every run.
        engagement = _engagement_from_question(ro_question)
        try:
            write_engagement(engagement)
        except Exception:
            pass  # persistence failure must never block a research run

        # J7.0b – auto-create a minimal Decision Model for question-driven runs.
        # Goal-driven runs have ProblemFramingAgent produce the full DM v2 instead.
        # J7.0b1 – also back-link the engagement so decision_model_id is non-null.
        # J7.1a – question-driven auto-DMs use write_latest=False so they don't
        # overwrite a richer assumption-populated DM from a prior functional run.
        # The functional pipeline's AssumptionAgent re-writes with write_latest=True.
        dm_id: str | None = None
        if not goal:
            dm = _dm_from_question(ro_question, engagement_id=engagement.engagement_id)
            try:
                write_decision_model(dm, write_latest=False)
                dm_id = dm.decision_model_id
                engagement = _link_dm(engagement, dm_id)  # persists updated engagement
            except Exception:
                pass

        research_object = create_research_object(
            question=ro_question,
            profile_name=execution_profile or None,
            profile_names=self._profile_names or None,
            profile_source="cli_argument",
            sources_dir=self._sources_dir,
            web_search=self._web_search,
            mock_mode=mock_mode,
            engagement_id=engagement.engagement_id,
            decision_model_id=dm_id,
        )

        # Build and validate context (J5.0b.1 / J5.0b.7 / J6.1)
        ctx = AgentContext(
            question=question,
            goal=goal,
            profiles=self._profile_names,
            execution_profile=execution_profile,
            research_object=research_object,
            run_id=uuid.uuid4().hex[:12],
        )
        # J7.0b – stash engagement_id in trace so ProblemFramingAgent can link
        # the DM v2 it produces back to the engagement.
        ctx.trace["_engagement_id"] = engagement.engagement_id

        try:
            ctx.validate()
        except ContextValidationError as exc:
            LOGGER.error("Context validation failed: %s", exc)
            raise

        from research_agent.log import PROGRESS
        LOGGER.log(
            PROGRESS,
            "AgentContext validated — profiles=%s execution=%s goal=%r",
            ctx.profiles, ctx.execution_profile, ctx.goal[:60] if ctx.goal else "",
        )

        # Hand off to the adaptive orchestrator
        orchestrator = AgentOrchestrator(
            problem_framing_factory=problem_framing_factory if goal else None,
            research_strategy_factory=research_strategy_factory if goal else None,
            hypothesis_factory=hypothesis_factory,
            challenge_factory=challenge_factory,
            assumption_factory=assumption_factory,
            risk_factory=risk_factory,
            opportunity_factory=opportunity_factory,
            recommendation_factory=recommendation_factory,
            multi_profile_factory=multi_profile_factory,
            scenario_factory=scenario_factory,
            improvement_factory=improvement_factory,
            synthesis_factory=synthesis_factory,
            strategic_option_factory=strategic_option_factory,
            decision_analysis_factory=decision_analysis_factory,
            planner_factory=planner_factory,
            evidence_factory=evidence_factory,
            qa_factory=qa_factory,
            report_factory=report_factory,
            max_iterations=self._max_iterations,
        )
        return orchestrator.run(ctx)
