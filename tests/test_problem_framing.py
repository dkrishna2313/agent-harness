"""Tests for ProblemFramingAgent and goal-driven workflow (J6.1)."""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.problem_framing_agent import ProblemFramingAgent
from research_agent.claude_client import DecisionModelPayload, MockClaudeClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goal_context(goal: str = "Develop a strategy for AI infrastructure investment.") -> AgentContext:
    return AgentContext(
        goal=goal,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-TEST_FRAMING"},
        run_id="framing001",
    )


def _question_context(question: str = "What is the TDP of the H100?") -> AgentContext:
    return AgentContext(
        question=question,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-TEST_Q"},
        run_id="q001",
    )


# ---------------------------------------------------------------------------
# J6.1.1 – WorkflowState.PROBLEM_FRAMING constant
# ---------------------------------------------------------------------------

def test_workflow_state_problem_framing_constant():
    assert WorkflowState.PROBLEM_FRAMING == "PROBLEM_FRAMING"


# ---------------------------------------------------------------------------
# J6.1.2 – ProblemFramingAgent inherits FunctionalAgent
# ---------------------------------------------------------------------------

def test_problem_framing_agent_inherits_functional_agent():
    assert issubclass(ProblemFramingAgent, FunctionalAgent)


# ---------------------------------------------------------------------------
# J6.1.3 – run() returns AgentResult with required contract fields
# ---------------------------------------------------------------------------

def test_problem_framing_run_returns_agent_result():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert "duration_seconds" in result.metrics
    assert result.metrics["duration_seconds"] >= 0.0
    assert result.trace["agent"] == "ProblemFramingAgent"


# ---------------------------------------------------------------------------
# J6.1.4 – Decision Model written to context.decision_model
# ---------------------------------------------------------------------------

def test_decision_model_written_to_context():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    dm = result.context.decision_model
    assert isinstance(dm, dict)
    assert "objective" in dm
    assert "decision_areas" in dm
    assert "critical_uncertainties" in dm
    assert "research_questions" in dm
    assert "evidence_requirements" in dm


def test_decision_model_non_empty_lists():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    dm = result.context.decision_model
    assert len(dm["decision_areas"]) >= 1
    assert len(dm["research_questions"]) >= 1


# ---------------------------------------------------------------------------
# J6.1.5 – Decision Model written to Research Object
# ---------------------------------------------------------------------------

def test_decision_model_written_to_research_object():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    ro = result.context.research_object
    assert "decision_model" in ro
    assert ro["decision_model"] == result.context.decision_model


def test_goal_written_to_research_object():
    goal = "Develop a strategy for AI infrastructure investment."
    agent = ProblemFramingAgent()
    ctx = _goal_context(goal)
    result = agent.run(ctx)
    assert result.context.research_object.get("goal") == goal


# ---------------------------------------------------------------------------
# J6.1.6 – Primary question populated from first research question
# ---------------------------------------------------------------------------

def test_question_populated_from_first_research_question():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    assert ctx.question == ""
    result = agent.run(ctx)
    assert result.context.question != ""
    assert result.context.question == result.context.decision_model["research_questions"][0]


def test_existing_question_not_overwritten():
    """If question is already set, ProblemFramingAgent must not overwrite it."""
    existing_q = "What GPU should we use for training?"
    agent = ProblemFramingAgent()
    ctx = AgentContext(
        question=existing_q,
        goal="Develop a strategy for AI infrastructure investment.",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-TEST_NOOVERWRITE"},
        run_id="nowrit001",
    )
    result = agent.run(ctx)
    assert result.context.question == existing_q


# ---------------------------------------------------------------------------
# J6.1.7 – AgentHistory entry recorded
# ---------------------------------------------------------------------------

def test_agent_history_entry_added():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    assert len(result.context.agent_history) == 1
    entry = result.context.agent_history[0]
    assert entry["agent"] == "ProblemFramingAgent"
    assert entry["status"] == "success"
    assert "summary" in entry


def test_agent_history_includes_counts():
    agent = ProblemFramingAgent()
    ctx = _goal_context()
    result = agent.run(ctx)
    entry = result.context.agent_history[0]
    assert "decision_areas_count" in entry
    assert "research_questions_count" in entry
    assert entry["decision_areas_count"] >= 1
    assert entry["research_questions_count"] >= 1


# ---------------------------------------------------------------------------
# J6.1.8 – MockClaudeClient.frame_problem()
# ---------------------------------------------------------------------------

def test_mock_client_frame_problem_returns_decision_model():
    client = MockClaudeClient()
    result = client.frame_problem("Develop AI infrastructure strategy.", [])
    assert isinstance(result, DecisionModelPayload)
    assert result.objective
    assert len(result.decision_areas) >= 1
    assert len(result.research_questions) >= 1


# ---------------------------------------------------------------------------
# J6.1.9 – AgentContext validation: goal OR question is sufficient
# ---------------------------------------------------------------------------

def test_context_validates_with_goal_only():
    ctx = AgentContext(
        goal="Develop AI strategy",
        profiles=["p1"],
        execution_profile="p1",
        research_object={"id": "R-X"},
    )
    ctx.validate()  # must not raise


def test_context_validates_with_question_only():
    ctx = AgentContext(
        question="What is the TDP of H100?",
        profiles=["p1"],
        execution_profile="p1",
        research_object={"id": "R-Y"},
    )
    ctx.validate()  # must not raise


def test_context_raises_without_question_or_goal():
    from functional_agents.context import ContextValidationError
    ctx = AgentContext(
        profiles=["p1"],
        execution_profile="p1",
        research_object={"id": "R-Z"},
    )
    with pytest.raises(ContextValidationError, match="question.*goal"):
        ctx.validate()


# ---------------------------------------------------------------------------
# J6.1.10 – Empty goal: graceful skip
# ---------------------------------------------------------------------------

def test_empty_goal_results_in_warning_status():
    agent = ProblemFramingAgent()
    ctx = AgentContext(
        goal="",
        question="What is the TDP of H100?",
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-EMPTY"},
        run_id="empty001",
    )
    result = agent.run(ctx)
    assert result.status == "warning"
    assert result.context.decision_model == {}


# ---------------------------------------------------------------------------
# J6.1.11 – AgentOrchestrator: PROBLEM_FRAMING prepended for goal-driven run
#
# The full orchestrator module imports research_agent.profile which requires
# yaml. We inject a stub into sys.modules so the import succeeds in
# environments where pyyaml is not installed.
# ---------------------------------------------------------------------------

@pytest.fixture()
def agent_orchestrator_class():
    """Return AgentOrchestrator with yaml stub in sys.modules."""
    import sys
    from unittest.mock import MagicMock
    existing = sys.modules.get("yaml")
    if existing is None:
        sys.modules["yaml"] = MagicMock()
    try:
        # Force fresh import if the module was not yet loaded
        import importlib
        import functional_agents.orchestrator as _m
        importlib.reload(_m)
        from functional_agents.orchestrator import AgentOrchestrator
        yield AgentOrchestrator
    finally:
        if existing is None:
            sys.modules.pop("yaml", None)


def test_agent_orchestrator_prepends_problem_framing_for_goal(agent_orchestrator_class):
    AgentOrchestrator = agent_orchestrator_class
    calls: list[str] = []

    class _PF(FunctionalAgent):
        def _execute(self, ctx: AgentContext) -> AgentContext:
            calls.append("framing")
            ctx.question = "derived question"
            self._record(ctx, status="success", summary="framing done")
            return ctx

    class _Stub(FunctionalAgent):
        def __init__(self, name_: str):
            self._name = name_
        @property
        def name(self) -> str:
            return self._name
        def _execute(self, ctx: AgentContext) -> AgentContext:
            calls.append(self._name)
            self._record(ctx, status="success", summary=f"{self._name} done")
            return ctx

    ctx = AgentContext(
        goal="Invest in AI",
        question="",
        profiles=["p"],
        execution_profile="p",
        research_object={"id": "R-ORCH"},
        run_id="orch001",
    )

    orch = AgentOrchestrator(
        problem_framing_factory=_PF,
        planner_factory=lambda: _Stub("Planner"),
        evidence_factory=lambda: _Stub("Evidence"),
        qa_factory=lambda: _Stub("QA"),
        report_factory=lambda: _Stub("Report"),
    )
    orch.run(ctx)
    assert calls[0] == "framing"
    assert "Planner" in calls
    assert "Evidence" in calls
    assert "QA" in calls
    assert "Report" in calls


def test_agent_orchestrator_skips_problem_framing_when_factory_none(agent_orchestrator_class):
    AgentOrchestrator = agent_orchestrator_class
    calls: list[str] = []

    class _Stub(FunctionalAgent):
        def __init__(self, name_: str):
            self._name = name_
        @property
        def name(self) -> str:
            return self._name
        def _execute(self, ctx: AgentContext) -> AgentContext:
            calls.append(self._name)
            self._record(ctx, status="success", summary=f"{self._name} done")
            return ctx

    ctx = AgentContext(
        question="What is GPU power?",
        profiles=["p"],
        execution_profile="p",
        research_object={"id": "R-ORCH2"},
        run_id="orch002",
    )

    orch = AgentOrchestrator(
        problem_framing_factory=None,
        planner_factory=lambda: _Stub("Planner"),
        evidence_factory=lambda: _Stub("Evidence"),
        qa_factory=lambda: _Stub("QA"),
        report_factory=lambda: _Stub("Report"),
    )
    orch.run(ctx)
    assert "framing" not in calls
    assert calls[0] == "Planner"


# ---------------------------------------------------------------------------
# J6.1.12 – Orchestrator class has run_from_goal() method
# ---------------------------------------------------------------------------

def test_orchestrator_class_has_run_from_goal():
    import sys
    from unittest.mock import MagicMock
    existing = sys.modules.get("yaml")
    if existing is None:
        sys.modules["yaml"] = MagicMock()
    try:
        import importlib
        import functional_agents.orchestrator as _m
        importlib.reload(_m)
        from functional_agents.orchestrator import Orchestrator
        assert hasattr(Orchestrator, "run_from_goal")
        assert callable(Orchestrator.run_from_goal)
    finally:
        if existing is None:
            sys.modules.pop("yaml", None)


# ---------------------------------------------------------------------------
# J9.1a – Strategic Framing Summary: bound derived framing so the raw brief
# does not propagate downstream and inflate prompts.
# ---------------------------------------------------------------------------

def _long_brief(n_chars: int = 2000) -> str:
    base = (
        "AI Data Center Power Infrastructure Strategy for Hyperscaler. "
        "Current situation: planning a large GB300 NVL72 deployment with rising "
        "rack density, multi-year grid interconnection queues, and SMR options. "
    )
    return (base * ((n_chars // len(base)) + 1))[:n_chars]


def test_objective_condensed_when_brief_is_long():
    agent = ProblemFramingAgent()  # no client → mock echoes goal into objective
    ctx = _goal_context(goal=_long_brief())
    result = agent.run(ctx)
    dm = result.context.decision_model
    # Raw brief is ~2000 chars; the propagated objective must be bounded.
    assert len(ctx.goal) > 1500
    assert len(dm["objective"]) <= 401, len(dm["objective"])


def test_research_questions_bounded():
    agent = ProblemFramingAgent()
    ctx = _goal_context(goal=_long_brief())
    result = agent.run(ctx)
    dm = result.context.decision_model
    for q in dm["research_questions"]:
        assert len(q) <= 401, len(q)


def test_propagated_question_is_bounded():
    """context.question feeds PlannerAgent/EvidenceAgent — it must not be the brief."""
    agent = ProblemFramingAgent()
    ctx = _goal_context(goal=_long_brief())
    result = agent.run(ctx)
    assert 0 < len(result.context.question) <= 401


def test_strategic_framing_summary_recorded():
    agent = ProblemFramingAgent()
    ctx = _goal_context(goal=_long_brief())
    result = agent.run(ctx)
    summary = result.context.trace.get("_strategic_framing_summary")
    assert summary is not None
    assert summary["raw_goal_chars"] > summary["objective_chars"]
    assert summary["decision_areas"] >= 1


def test_short_goal_objective_preserved():
    """Condensation must not truncate an already-short objective."""
    agent = ProblemFramingAgent()
    ctx = _goal_context(goal="Analyze AI data center power strategies.")
    result = agent.run(ctx)
    dm = result.context.decision_model
    assert dm["objective"].endswith("…") is False
