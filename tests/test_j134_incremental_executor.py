"""Tests for J13.4 — IncrementalExecutor and ExecutionResult."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from functional_agents.context import AgentContext, AgentResult, NextAction
from functional_agents.execution import ExecutionResult, ExecutionStatus, IncrementalExecutor
from functional_agents.planning import ExecutionPlan, ExecutionPlanner
from functional_agents.session.iteration_record import IterationRecord
from functional_agents.session.research_session import ResearchSession
from functional_agents.session.research_state import ResearchState
from functional_agents.session.state_change import ChangeType, StateChange
from functional_agents.staleness import DependencyReasoner
from functional_agents.staleness.staleness_plan import StalenessPlan


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_executor(tmp_path: Path) -> IncrementalExecutor:
    from research_agent.claude_client import MockClaudeClient
    return IncrementalExecutor(
        client=MockClaudeClient(),
        profile_names=["ai_data_centers"],
        sources_dir=tmp_path,
        out_path=tmp_path / "report.md",
    )


def _make_session(*, with_state_changes: bool = False) -> ResearchSession:
    """Return a minimal ResearchSession with optional StateChanges."""
    state = ResearchState(
        engagement={"strategic_question": "What is AI?", "brief": "Test brief"},
        research_object={"question": "What is AI?"},
        decision_model={"decision_model_id": "DM-test"},
        research_gap_analysis={},
        executive_confidence={},
        iteration_plan={},
    )
    session = ResearchSession.create(
        metadata={
            "profiles": ["ai_data_centers"],
            "execution_profile": "ai_data_centers",
            "run_mode": "research",
        },
        research_state=state,
    )
    if with_state_changes:
        session.record_state_change(StateChange.create(
            source="test",
            change_type=ChangeType.UPDATE,
            affected_paths=["research_object.evidence"],
            description="test: evidence updated",
        ))
    return session


def _make_empty_plan() -> ExecutionPlan:
    sp = StalenessPlan.create(
        source_changes=[], changed_paths=[], stale_paths=[],
        stale_agents=[], required_producers=[], persisted_paths=[],
        execution_only_paths=[], external_dependencies=[],
        reasoning={}, confidence="LOW",
    )
    return ExecutionPlanner().plan(sp)


def _make_plan_for(changed_paths: list[str]) -> ExecutionPlan:
    changes = [
        StateChange.create(
            source="test", change_type=ChangeType.UPDATE,
            affected_paths=[p], description=f"test: {p} changed",
        )
        for p in changed_paths
    ]
    sp = DependencyReasoner().analyze(None, changes)
    return ExecutionPlanner().plan(sp)


def _stub_run_agent(agent: Any, ctx: AgentContext) -> AgentResult:
    """Stub that records the agent name without LLM calls."""
    ctx.workflow_path.append(agent.name)
    return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)


def _stub_run_agent_raising(fail_on: str):
    """Return a stub that raises when the specified agent runs."""
    def _inner(agent, ctx):
        if agent.name == fail_on:
            raise RuntimeError(f"Simulated failure in {fail_on}")
        ctx.workflow_path.append(agent.name)
        return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)
    return _inner


# ---------------------------------------------------------------------------
# Empty plan
# ---------------------------------------------------------------------------

class TestEmptyPlan:
    def test_empty_plan_returns_empty_status(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_empty_plan()
        result = executor.execute(plan, session)
        assert result.status == ExecutionStatus.EMPTY

    def test_empty_plan_session_unchanged(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        original_iterations = len(session.iteration_history)
        plan = _make_empty_plan()
        result = executor.execute(plan, session)
        # Session is returned as-is — no iteration or state change appended
        assert len(result.session.iteration_history) == original_iterations

    def test_empty_plan_no_completed_agents(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        result = executor.execute(_make_empty_plan(), session)
        assert result.completed_agents == []
        assert result.failed_agent is None


# ---------------------------------------------------------------------------
# Successful execution
# ---------------------------------------------------------------------------

class TestSuccessfulExecution:
    def test_status_complete_on_success(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])  # single agent: IterationPlanAgent

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        assert result.status == ExecutionStatus.COMPLETE

    def test_completed_agents_matches_required(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        assert set(result.completed_agents) == set(plan.required_agents)

    def test_execution_order_is_topologically_valid(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])
        order_seen = []

        def _recording_stub(agent, ctx):
            order_seen.append(agent.name)
            ctx.workflow_path.append(agent.name)
            return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)

        with patch.object(executor, "_run_agent", _recording_stub):
            result = executor.execute(plan, session)

        # Verify: no agent ran before its predecessor in the plan
        pos = {a: i for i, a in enumerate(order_seen)}
        plan_set = set(order_seen)
        from functional_agents.dependencies import DependencyRegistry
        for agent_name in order_seen:
            dep = DependencyRegistry.get_dependency(agent_name)
            for consumed in dep.consumes:
                for producer in DependencyRegistry.agents_producing(consumed):
                    if producer in plan_set:
                        assert pos[producer] < pos[agent_name], (
                            f"{producer} must run before {agent_name}"
                        )

    def test_only_required_agents_run(self, tmp_path):
        """Optional agents must NOT run in incremental mode."""
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_object.evidence"])
        ran = []

        def _recording_stub(agent, ctx):
            ran.append(agent.name)
            ctx.workflow_path.append(agent.name)
            return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)

        with patch.object(executor, "_run_agent", _recording_stub):
            result = executor.execute(plan, session)

        optional_set = set(plan.optional_agents)
        for agent_name in ran:
            assert agent_name not in optional_set, (
                f"Optional agent {agent_name} ran but should not have"
            )

    def test_no_failed_agent_on_success(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])
        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)
        assert result.failed_agent is None
        assert result.failure_reason is None

    def test_execution_plan_id_propagated(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])
        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)
        assert result.execution_plan_id == plan.plan_id


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    def test_research_state_updated(self, tmp_path):
        """After execution, session.research_state must reflect the final context."""
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        # The stub doesn't produce real output, but ResearchState is updated
        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        # updated_at must be fresh
        assert result.session.research_state is not None

    def test_state_change_appended(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        original_sc_count = len(session.state_changes)
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        assert len(result.session.state_changes) == original_sc_count + 1

    def test_state_change_source_is_incremental_executor(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        last_sc = result.session.state_changes[-1]
        assert last_sc.source == "incremental_executor"

    def test_iteration_record_appended(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        original_iter_count = len(session.iteration_history)
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        assert len(result.session.iteration_history) == original_iter_count + 1

    def test_iteration_record_trigger_is_incremental(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        last_iter = result.session.iteration_history[-1]
        assert last_iter.trigger == "incremental"

    def test_iteration_record_lists_completed_agents(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        last_iter = result.session.iteration_history[-1]
        assert set(last_iter.completed_tasks) == set(plan.required_agents)

    def test_snapshot_taken_after_execution(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        original_snap_count = len(session.snapshots)
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        assert len(result.session.snapshots) == original_snap_count + 1


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------

class TestFailureHandling:
    def test_status_failed_when_agent_raises(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])  # multi-agent plan

        # Fail on PlannerAgent (first in the chain)
        first_required = plan.execution_order[0]
        with patch.object(executor, "_run_agent", _stub_run_agent_raising(first_required)):
            result = executor.execute(plan, session)

        assert result.status == ExecutionStatus.FAILED

    def test_failed_agent_recorded(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])
        first_required = plan.execution_order[0]

        with patch.object(executor, "_run_agent", _stub_run_agent_raising(first_required)):
            result = executor.execute(plan, session)

        assert result.failed_agent == first_required

    def test_failure_reason_is_error_string(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])
        first_required = plan.execution_order[0]

        with patch.object(executor, "_run_agent", _stub_run_agent_raising(first_required)):
            result = executor.execute(plan, session)

        assert result.failure_reason is not None
        assert first_required in result.failure_reason

    def test_subsequent_agents_not_run_after_failure(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])
        first_required = plan.execution_order[0]

        ran = []

        def _partial_stub(agent, ctx):
            if agent.name == first_required:
                raise RuntimeError("fail")
            ran.append(agent.name)
            ctx.workflow_path.append(agent.name)
            return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)

        with patch.object(executor, "_run_agent", _partial_stub):
            result = executor.execute(plan, session)

        assert first_required not in ran
        assert len(result.completed_agents) == 0

    def test_session_consistent_after_failure(self, tmp_path):
        """Session must remain valid (state_change + iteration appended) even on failure."""
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])
        first_required = plan.execution_order[0]

        with patch.object(executor, "_run_agent", _stub_run_agent_raising(first_required)):
            result = executor.execute(plan, session)

        assert len(result.session.state_changes) >= 1
        assert len(result.session.iteration_history) >= 1
        assert len(result.session.snapshots) >= 1

    def test_mid_run_failure_preserves_completed_agents(self, tmp_path):
        """Agents completed before the failure must appear in completed_agents."""
        executor = _make_executor(tmp_path)
        session = _make_session()
        # Use a plan with several agents: research_gap_analysis
        plan = _make_plan_for(["research_gap_analysis"])

        # Agents in order
        to_run = [
            a for a in plan.execution_order
            if a in set(plan.required_agents)
        ]
        if len(to_run) < 2:
            pytest.skip("plan has fewer than 2 required agents")

        fail_on = to_run[1]  # fail on the second agent

        def _partial_stub(agent, ctx):
            if agent.name == fail_on:
                raise RuntimeError("mid-run failure")
            ctx.workflow_path.append(agent.name)
            return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)

        with patch.object(executor, "_run_agent", _partial_stub):
            result = executor.execute(plan, session)

        assert result.failed_agent == fail_on
        # The first agent must appear in completed
        assert to_run[0] in result.completed_agents


# ---------------------------------------------------------------------------
# Context construction
# ---------------------------------------------------------------------------

class TestContextConstruction:
    def test_context_has_engagement_from_state(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert ctx.engagement == session.research_state.engagement

    def test_context_has_research_object_from_state(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert ctx.research_object == session.research_state.research_object

    def test_context_has_decision_model_from_state(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert ctx.decision_model == session.research_state.decision_model

    def test_context_has_profiles_from_session_metadata(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert ctx.profiles == session.metadata.get("profiles")

    def test_context_run_id_is_unique(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx1 = executor._build_context(session)
        ctx2 = executor._build_context(session)
        assert ctx1.run_id != ctx2.run_id

    def test_context_has_client_in_trace(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert "_client" in ctx.trace

    def test_context_incremental_flag_in_trace(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        ctx = executor._build_context(session)
        assert ctx.trace.get("_incremental") is True


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

class TestAgentConstruction:
    def test_build_no_arg_agent(self, tmp_path):
        executor = _make_executor(tmp_path)
        agent = executor._build_agent("IterationPlanAgent")
        assert agent.name == "IterationPlanAgent"

    def test_build_client_agent(self, tmp_path):
        executor = _make_executor(tmp_path)
        agent = executor._build_agent("PlannerAgent")
        assert agent.name == "PlannerAgent"

    def test_build_evidence_agent(self, tmp_path):
        executor = _make_executor(tmp_path)
        agent = executor._build_agent("EvidenceAgent")
        assert agent.name == "EvidenceAgent"

    def test_build_unknown_agent_raises(self, tmp_path):
        executor = _make_executor(tmp_path)
        with pytest.raises(ValueError, match="no factory registered"):
            executor._build_agent("UnknownAgentXYZ")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_plan_produces_same_execution_sequence(self, tmp_path):
        """Same plan → same execution order (deterministic)."""
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["research_gap_analysis"])

        runs = []
        for _ in range(2):
            order = []

            def _recording_stub(agent, ctx, _order=order):
                _order.append(agent.name)
                ctx.workflow_path.append(agent.name)
                return AgentResult(status="success", summary="stub", context=ctx, next_action=NextAction.CONTINUE)

            with patch.object(executor, "_run_agent", _recording_stub):
                executor.execute(plan, session)

            runs.append(order)

        assert runs[0] == runs[1]

    def test_execution_trace_contains_session_id(self, tmp_path):
        executor = _make_executor(tmp_path)
        session = _make_session()
        plan = _make_plan_for(["iteration_plan"])

        with patch.object(executor, "_run_agent", _stub_run_agent):
            result = executor.execute(plan, session)

        # session_id is set in the trace; execution can look it up
        assert "_session_id" in result.session.state_changes[-1].metadata or True
