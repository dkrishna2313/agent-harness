"""J6.2 – ResearchStrategyAgent tests.

Verifies:
  1. Contract compliance (inherits FunctionalAgent, run() returns AgentResult)
  2. ResearchStrategyPayload structure and fields
  3. Strategy persisted to context.research_strategy, RO, and trace
  4. Orchestrator routes PROBLEM_FRAMING → RESEARCH_STRATEGY → PLANNING
  5. PlannerAgent receives research_strategy from context
  6. ReportAgent emits research_strategy trace block
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# yaml is not installed in the test environment — mock it before any
# functional_agents.orchestrator import chain triggers it.
@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield
    # Do not pop yaml — other tests in the same process may need the mock.


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.research_strategy_agent import ResearchStrategyAgent
from research_agent.claude_client import MockClaudeClient, ResearchStrategyPayload


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_context(*, goal: str = "Evaluate SMR investment opportunity") -> AgentContext:
    return AgentContext(
        goal=goal,
        profiles=["smr", "nvidia"],
        execution_profile="smr",
        research_object={"id": "R-TEST_001"},
        run_id="test001",
    )


def _context_with_decision_model(*, goal: str = "Evaluate SMR investment opportunity") -> AgentContext:
    ctx = _minimal_context(goal=goal)
    ctx.decision_model = {
        "objective": "Research SMR investment viability",
        "decision_areas": ["Technology Readiness", "Market Opportunity", "Regulatory Risk"],
        "critical_uncertainties": ["Licensing timeline", "Cost trajectory"],
        "research_questions": [
            "What is the current regulatory status of SMR technology?",
            "What is the market size for SMR deployment by 2035?",
            "What are the capital cost ranges for first-of-a-kind SMR plants?",
        ],
        "evidence_requirements": ["Regulatory filings", "Analyst cost estimates", "Deployment schedules"],
    }
    return ctx


# ---------------------------------------------------------------------------
# Contract compliance
# ---------------------------------------------------------------------------

def test_inherits_functional_agent():
    assert issubclass(ResearchStrategyAgent, FunctionalAgent)


def test_run_returns_agent_result():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    result = agent.run(ctx)
    assert isinstance(result, AgentResult)


def test_run_status_success():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    result = agent.run(ctx)
    assert result.status == "success"


def test_run_metrics_has_duration():
    agent = ResearchStrategyAgent()
    result = agent.run(_context_with_decision_model())
    assert "duration_seconds" in result.metrics
    assert result.metrics["duration_seconds"] >= 0.0


def test_run_trace_required_keys():
    agent = ResearchStrategyAgent()
    result = agent.run(_context_with_decision_model())
    for key in ("agent", "run_id", "duration_seconds", "status"):
        assert key in result.trace


# ---------------------------------------------------------------------------
# ResearchStrategyPayload structure
# ---------------------------------------------------------------------------

def test_payload_has_profile_priorities():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    rs = ctx.research_strategy
    assert isinstance(rs.get("profile_priorities"), dict)
    assert len(rs["profile_priorities"]) == 2  # smr, nvidia


def test_payload_has_question_priorities():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    rs = ctx.research_strategy
    qprios = rs.get("research_question_priorities", [])
    assert len(qprios) == 3
    assert all("question" in q and "priority" in q for q in qprios)


def test_payload_has_required_evidence():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert isinstance(ctx.research_strategy.get("required_evidence"), list)
    assert len(ctx.research_strategy["required_evidence"]) > 0


def test_payload_has_coverage_targets():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    targets = ctx.research_strategy.get("coverage_targets", {})
    assert isinstance(targets, dict)
    assert all(v in ("strong", "moderate", "light") for v in targets.values())


def test_payload_has_strategy_rationale():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert isinstance(ctx.research_strategy.get("strategy_rationale"), str)


# ---------------------------------------------------------------------------
# Persistence — context, RO, trace
# ---------------------------------------------------------------------------

def test_persisted_to_context_research_strategy():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert ctx.research_strategy
    assert "profile_priorities" in ctx.research_strategy


def test_persisted_to_research_object():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert "research_strategy" in ctx.research_object
    assert ctx.research_object["research_strategy"] == ctx.research_strategy


def test_stashed_in_trace():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert "_research_strategy" in ctx.trace
    assert ctx.trace["_research_strategy"] == ctx.research_strategy


def test_agent_history_recorded():
    agent = ResearchStrategyAgent()
    ctx = _context_with_decision_model()
    agent.run(ctx)
    assert len(ctx.agent_history) == 1
    entry = ctx.agent_history[0]
    assert entry["agent"] == "ResearchStrategyAgent"
    assert entry["status"] == "success"
    assert "profile_priorities_count" in entry
    assert "research_question_priorities_count" in entry


# ---------------------------------------------------------------------------
# Empty decision model — graceful skip
# ---------------------------------------------------------------------------

def test_empty_decision_model_returns_warning():
    agent = ResearchStrategyAgent()
    ctx = _minimal_context()
    # decision_model is empty by default
    result = agent.run(ctx)
    assert result.status == "warning"
    assert ctx.research_strategy == {}


# ---------------------------------------------------------------------------
# MockClaudeClient integration
# ---------------------------------------------------------------------------

def test_with_mock_client():
    agent = ResearchStrategyAgent(client=MockClaudeClient())
    ctx = _context_with_decision_model()
    result = agent.run(ctx)
    assert result.status == "success"
    assert ctx.research_strategy.get("profile_priorities") == {"smr": 1, "nvidia": 2}


# ---------------------------------------------------------------------------
# Orchestrator routes through RESEARCH_STRATEGY state (stub agents)
# ---------------------------------------------------------------------------

def _make_stub_agent(calls: list[str], name_: str, *, setup_fn=None):
    """Build a FunctionalAgent stub that records its name in calls."""
    class _Stub(FunctionalAgent):
        @property
        def name(self) -> str:
            return name_

        def _execute(self, ctx: AgentContext) -> AgentContext:
            calls.append(name_)
            if setup_fn:
                setup_fn(ctx)
            self._record(ctx, status="success", summary=f"{name_} done")
            return ctx

    return _Stub


def test_orchestrator_includes_research_strategy_in_workflow_path():
    """Goal-driven runs must traverse PROBLEM_FRAMING → RESEARCH_STRATEGY → PLANNING."""
    from functional_agents.orchestrator import AgentOrchestrator

    calls: list[str] = []

    def _pf_setup(ctx):
        ctx.question = "derived question from goal"
        ctx.decision_model = {"research_questions": ["Q1"], "decision_areas": ["A1"]}

    ctx = AgentContext(
        goal="Evaluate SMR investment",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_001"},
        run_id="orch001",
    )

    orch = AgentOrchestrator(
        problem_framing_factory=lambda: _make_stub_agent(calls, "ProblemFramingAgent", setup_fn=_pf_setup)(),
        research_strategy_factory=lambda: _make_stub_agent(calls, "ResearchStrategyAgent")(),
        planner_factory=lambda: _make_stub_agent(calls, "PlannerAgent")(),
        evidence_factory=lambda: _make_stub_agent(calls, "EvidenceAgent")(),
        qa_factory=lambda: _make_stub_agent(calls, "QAAgent")(),
        report_factory=lambda: _make_stub_agent(calls, "ReportAgent")(),
    )
    result_ctx = orch.run(ctx)

    assert "ProblemFramingAgent" in result_ctx.workflow_path
    assert "ResearchStrategyAgent" in result_ctx.workflow_path
    assert "PlannerAgent" in result_ctx.workflow_path
    path_indices = {name: result_ctx.workflow_path.index(name) for name in result_ctx.workflow_path}
    assert path_indices["ProblemFramingAgent"] < path_indices["ResearchStrategyAgent"]
    assert path_indices["ResearchStrategyAgent"] < path_indices["PlannerAgent"]


def test_orchestrator_no_research_strategy_without_goal():
    """Question-driven runs skip RESEARCH_STRATEGY."""
    from functional_agents.orchestrator import AgentOrchestrator

    calls: list[str] = []

    ctx = AgentContext(
        question="What is the power consumption of NVL72?",
        profiles=["nvidia"],
        execution_profile="nvidia",
        research_object={"id": "R-TEST_002"},
        run_id="orch002",
    )

    orch = AgentOrchestrator(
        planner_factory=lambda: _make_stub_agent(calls, "PlannerAgent")(),
        evidence_factory=lambda: _make_stub_agent(calls, "EvidenceAgent")(),
        qa_factory=lambda: _make_stub_agent(calls, "QAAgent")(),
        report_factory=lambda: _make_stub_agent(calls, "ReportAgent")(),
    )
    result_ctx = orch.run(ctx)

    assert "ResearchStrategyAgent" not in result_ctx.workflow_path
    assert "ProblemFramingAgent" not in result_ctx.workflow_path


# ---------------------------------------------------------------------------
# ReportAgent emits research_strategy trace block
# ---------------------------------------------------------------------------

def test_report_agent_emits_research_strategy_trace_key():
    """ReportAgent must emit research_strategy block when _research_strategy is in trace."""
    from functional_agents.report_agent import ReportAgent
    from pathlib import Path
    from research_agent.schemas import ResearchMemo
    import tempfile

    ctx = AgentContext(
        question="What are SMR deployment costs?",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_003", "research_id": "R-TEST_003"},
        run_id="rpt003",
    )
    ctx.plan = {
        "question": ctx.question,
        "research_type": "RESEARCH",
        "subquestions": [],
        "investigation_areas": [],
        "profiles_used": ["smr"],
        "reasoning": "",
    }
    ctx.qa = {
        "qa_summary": {"issues_found": 0, "coverage_issues": 0, "evidence_issues": 0, "contradiction_issues": 0},
        "confidence_assessment": {"overall_confidence": "MEDIUM"},
        "profiles_contributing": ["smr"],
        "profiles_missing": [],
        "coverage_status": "sufficient",
    }
    ctx.evidence_notes = [{
        "evidence_summary": {
            "total_evidence_items": 0,
            "subquestions_with_evidence": 0,
            "subquestions_without_evidence": 0,
            "investigation_areas_with_evidence": 0,
            "coverage_distribution": {},
        },
        "coverage_by_subquestion": {},
        "evidence_by_subquestion": {},
        "evidence_items": [],
    }]
    ctx.trace["_memo"] = ResearchMemo(
        title="Test memo",
        question=ctx.question,
        executive_summary="test",
        confirmed_facts=[],
        inferences=[],
        power_implications=[],
        cooling_implications=[],
        networking_implications=[],
        rack_architecture_implications=[],
        open_questions=[],
        source_notes=[],
        evidence=[],
    )
    ctx.trace["_research_strategy"] = {
        "profile_priorities": {"smr": 1},
        "research_question_priorities": [{"question": "Q1", "priority": 1}],
        "required_evidence": ["cost data"],
        "source_priorities": ["regulatory filings"],
        "coverage_targets": {"Technology Readiness": "strong"},
        "strategy_rationale": "Focus on cost and regulatory data.",
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "report.md"
        agent = ReportAgent(out_path=out_path)
        agent.run(ctx)

        import json
        trace_path = out_path.with_suffix(".trace.json")
        assert trace_path.exists()
        trace = json.loads(trace_path.read_text())

    assert "research_strategy" in trace
    rs_block = trace["research_strategy"]
    assert rs_block["profile_priorities"] == {"smr": 1}
    assert rs_block["coverage_targets"] == {"Technology Readiness": "strong"}
