"""J5.5a – Agent contract compliance tests.

Verifies that every functional agent:
  1. Inherits FunctionalAgent
  2. run(AgentContext) returns an AgentResult
  3. AgentResult.outputs, .metrics, .trace are present
  4. metrics["duration_seconds"] is a non-negative float
  5. trace has the required keys: agent, run_id, duration_seconds, status
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, NextAction, WorkflowState
from functional_agents.planner_agent import PlannerAgent
from functional_agents.evidence_agent import EvidenceAgent
from functional_agents.qa_agent import QAAgent
from functional_agents.report_agent import ReportAgent

_ALL_AGENT_CLASSES = [PlannerAgent, EvidenceAgent, QAAgent, ReportAgent]


# ---------------------------------------------------------------------------
# Schema tests (no runtime needed)
# ---------------------------------------------------------------------------

def test_agent_result_has_required_fields():
    ctx = AgentContext(
        question="q", profiles=["p"], execution_profile="p",
        research_object={"id": "R-TEST_001"}, run_id="abc123",
    )
    r = AgentResult(
        status="success", next_action=NextAction.CONTINUE,
        summary="ok", context=ctx,
    )
    assert r.outputs == {}
    assert r.metrics == {}
    assert r.trace == {}


def test_workflow_state_constants():
    assert WorkflowState.PLANNING == "PLANNING"
    assert WorkflowState.COMPLETE == "COMPLETE"


def test_next_action_constants():
    assert NextAction.CONTINUE == "CONTINUE"
    assert NextAction.REQUEST_EVIDENCE == "REQUEST_EVIDENCE"
    assert NextAction.REQUEST_REPLAN == "REQUEST_REPLAN"


@pytest.mark.parametrize("cls", _ALL_AGENT_CLASSES)
def test_agent_inherits_functional_agent(cls):
    assert issubclass(cls, FunctionalAgent)


# ---------------------------------------------------------------------------
# Minimal stub agent to verify base.run() contract without real LLMs
# ---------------------------------------------------------------------------

class _StubAgent(FunctionalAgent):
    """A no-op agent that records one success entry and exits."""

    def _execute(self, context: AgentContext) -> AgentContext:
        self._record(context, status="success", summary="stub ran")
        return context


def _minimal_context() -> AgentContext:
    return AgentContext(
        question="test question",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
    )


def test_base_run_returns_agent_result():
    agent = _StubAgent()
    ctx = _minimal_context()
    result = agent.run(ctx)
    assert isinstance(result, AgentResult)


def test_run_status_propagated():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    assert result.status == "success"


def test_run_next_action_default():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    assert result.next_action == NextAction.CONTINUE


def test_run_metrics_has_duration():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    assert "duration_seconds" in result.metrics
    assert isinstance(result.metrics["duration_seconds"], float)
    assert result.metrics["duration_seconds"] >= 0.0


def test_run_trace_required_keys():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    for key in ("agent", "run_id", "duration_seconds", "status"):
        assert key in result.trace, f"trace missing key: {key!r}"


def test_run_trace_agent_name():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    assert result.trace["agent"] == "_StubAgent"


def test_run_trace_run_id_propagated():
    agent = _StubAgent()
    ctx = _minimal_context()
    result = agent.run(ctx)
    assert result.trace["run_id"] == "testrun001"


def test_run_context_preserved():
    agent = _StubAgent()
    ctx = _minimal_context()
    result = agent.run(ctx)
    assert result.context is ctx
    assert len(result.context.agent_history) == 1


def test_run_outputs_default_empty():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    assert isinstance(result.outputs, dict)


# ---------------------------------------------------------------------------
# AgentContext run_id
# ---------------------------------------------------------------------------

def test_context_run_id_default_empty():
    ctx = AgentContext(
        question="q", profiles=["p"], execution_profile="p",
        research_object={"id": "R-TEST_001"},
    )
    assert ctx.run_id == ""


def test_context_run_id_set():
    ctx = AgentContext(
        question="q", profiles=["p"], execution_profile="p",
        research_object={"id": "R-TEST_001"}, run_id="abc123",
    )
    assert ctx.run_id == "abc123"
