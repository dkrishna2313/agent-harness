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

def _step(agent: Any, ctx: AgentContext) -> AgentResult:
    """Run one agent (returns AgentResult) and append its name to workflow_path."""
    result = agent.run(ctx)
    result.context.workflow_path.append(agent.name)
    return result


# ---------------------------------------------------------------------------
# AgentOrchestrator (J5.5.1)
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """State-machine orchestrator.  Executes agents, reads their next_action,
    and drives the workflow — supporting iteration loops and re-planning.

    States (J5.5.3):
        PLANNING → EVIDENCE → QA → REPORT → COMPLETE
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
        max_iterations: int = 3,
    ) -> None:
        self._planner_factory  = planner_factory
        self._evidence_factory = evidence_factory
        self._qa_factory       = qa_factory
        self._report_factory   = report_factory
        self._max_iterations   = max_iterations

    def run(self, ctx: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS

        state = WorkflowState.PLANNING
        termination_reason = NextAction.COMPLETE
        ctx.iteration_count = 0

        while state != WorkflowState.COMPLETE:

            ctx.workflow_state = state
            LOGGER.log(PROGRESS, "[Orchestrator] state=%s  iteration=%d", state, ctx.iteration_count)

            # ---- PLANNING ---------------------------------------------------
            if state == WorkflowState.PLANNING:
                result = _step(self._planner_factory(), ctx)
                ctx = result.context
                state = WorkflowState.EVIDENCE

            # ---- EVIDENCE ---------------------------------------------------
            elif state == WorkflowState.EVIDENCE:
                result = _step(self._evidence_factory(), ctx)
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
                    state = WorkflowState.REPORT

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
    ) -> None:
        self._profile_names  = profile_names
        self._sources_dir    = Path(sources_dir)
        self._out_path       = out_path
        self._client         = client
        self._top_evidence   = top_evidence
        self._top_chunks     = top_chunks
        self._max_iterations = max_iterations

        from research_agent.log import PROGRESS

        # Load execution profile (first in list)
        self._domain_profile: DomainProfile | None = None
        if profile_names:
            try:
                self._domain_profile = load_profile(profile_names[0])
                LOGGER.log(PROGRESS, "Execution profile loaded: %s", profile_names[0])
            except FileNotFoundError as exc:
                LOGGER.warning("Could not load profile %r: %s", profile_names[0], exc)

        # Verify supporting profiles (warn on missing; don't abort)
        for name in profile_names[1:]:
            try:
                load_profile(name)
                LOGGER.log(PROGRESS, "Supporting profile loaded: %s", name)
            except FileNotFoundError:
                LOGGER.warning("Supporting profile not found: %r", name)

    def run(self, question: str) -> AgentContext:
        """Build, validate, and execute the adaptive agent pipeline."""
        from .planner_agent  import PlannerAgent
        from .evidence_agent import EvidenceAgent
        from .qa_agent       import QAAgent
        from .report_agent   import ReportAgent

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
        def planner_factory() -> PlannerAgent:
            return PlannerAgent(client=self._client, domain_profiles=loaded_profiles)

        def evidence_factory() -> EvidenceAgent:
            return EvidenceAgent(
                sources_dir=self._sources_dir,
                client=self._client,
                top_evidence=self._top_evidence,
                top_chunks=self._top_chunks,
                domain_profile=self._domain_profile,
            )

        def qa_factory() -> QAAgent:
            return QAAgent()

        def report_factory() -> ReportAgent:
            return ReportAgent(
                out_path=self._out_path,
                domain_profile=self._domain_profile,
            )

        # Create Research Object before building context
        research_object = create_research_object(
            question=question,
            profile_name=execution_profile or None,
            profile_source="cli_argument",
            sources_dir=self._sources_dir,
            web_search=False,
            mock_mode=mock_mode,
        )

        # Build and validate context (J5.0b.1 / J5.0b.7)
        ctx = AgentContext(
            question=question,
            profiles=self._profile_names,
            execution_profile=execution_profile,
            research_object=research_object,
            run_id=uuid.uuid4().hex[:12],
        )
        try:
            ctx.validate()
        except ContextValidationError as exc:
            LOGGER.error("Context validation failed: %s", exc)
            raise

        from research_agent.log import PROGRESS
        LOGGER.log(
            PROGRESS,
            "AgentContext validated — profiles=%s execution=%s",
            ctx.profiles, ctx.execution_profile,
        )

        # Hand off to the adaptive orchestrator
        orchestrator = AgentOrchestrator(
            planner_factory=planner_factory,
            evidence_factory=evidence_factory,
            qa_factory=qa_factory,
            report_factory=report_factory,
            max_iterations=self._max_iterations,
        )
        return orchestrator.run(ctx)
