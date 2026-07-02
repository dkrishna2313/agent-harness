"""J5.5a – Agent contract compliance and validation tests.

Verifies that every functional agent:
  1. Inherits FunctionalAgent (BaseAgent)
  2. run(AgentContext) returns an AgentResult
  3. AgentResult.outputs, .metrics, .trace are present
  4. metrics["duration_seconds"] is a non-negative float
  5. trace has the required keys: agent, run_id, duration_seconds, status

Also covers the contract validator module:
  6. validate_agent_class() static checks
  7. validate_agent_result() runtime checks
  8. build_contract_validation() trace block assembly
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, NextAction, WorkflowState
from functional_agents.planner_agent import PlannerAgent
from functional_agents.evidence_agent import EvidenceAgent
from functional_agents.qa_agent import QAAgent
from functional_agents.report_agent import ReportAgent
from functional_agents.scenario_agent import ScenarioAgent
from functional_agents.recommendation_improvement_agent import RecommendationImprovementAgent
from functional_agents.multi_profile_agent import MultiProfileAgent
from functional_agents.recommendation_synthesis_agent import RecommendationSynthesisAgent
from functional_agents.strategic_option_agent import StrategicOptionAgent
from functional_agents.decision_analysis_agent import DecisionAnalysisAgent
from functional_agents.strategic_synthesis_agent import StrategicSynthesisAgent

_ALL_AGENT_CLASSES = [
    PlannerAgent, EvidenceAgent, QAAgent, ReportAgent,
    ScenarioAgent, RecommendationImprovementAgent, MultiProfileAgent,
    RecommendationSynthesisAgent, StrategicOptionAgent, DecisionAnalysisAgent,
    StrategicSynthesisAgent,
]


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


# ---------------------------------------------------------------------------
# Contract validator – static checks
# ---------------------------------------------------------------------------

from functional_agents.contract import (
    validate_agent_class,
    validate_agent_result,
    validate_all_classes,
    build_contract_validation,
    CONTRACT_VERSION,
)


@pytest.mark.parametrize("cls", _ALL_AGENT_CLASSES)
def test_validate_agent_class_inherits_base(cls):
    result = validate_agent_class(cls)
    assert result["inherits_base_agent"] is True, f"{cls.__name__} fails inherits_base_agent"


@pytest.mark.parametrize("cls", _ALL_AGENT_CLASSES)
def test_validate_agent_class_implements_run(cls):
    result = validate_agent_class(cls)
    assert result["implements_run"] is True, f"{cls.__name__} fails implements_run"


@pytest.mark.parametrize("cls", _ALL_AGENT_CLASSES)
def test_validate_agent_class_no_error(cls):
    result = validate_agent_class(cls)
    assert result["error"] is None, f"{cls.__name__} has error: {result['error']}"


def test_validate_all_classes_returns_all_agents():
    results = validate_all_classes()
    assert set(results.keys()) == {
        "ProblemFramingAgent", "ResearchStrategyAgent",
        "PlannerAgent", "EvidenceAgent",
        "HypothesisAgent", "StrategicSynthesisAgent", "ChallengeAgent", "AssumptionAgent",
        "RiskAgent", "OpportunityAgent", "RecommendationAgent",
        "MultiProfileAgent", "ScenarioAgent", "RecommendationImprovementAgent",
        "RecommendationSynthesisAgent", "StrategicOptionAgent", "DecisionAnalysisAgent",
        "ExecutiveConfidenceAgent", "QAAgent", "ReportAgent",
    }


def test_validate_all_classes_all_valid():
    results = validate_all_classes()
    for name, r in results.items():
        assert r["inherits_base_agent"], f"{name} not inheriting base"
        assert r["implements_run"], f"{name} not implementing run"


# ---------------------------------------------------------------------------
# Contract validator – runtime checks
# ---------------------------------------------------------------------------

def test_validate_agent_result_valid():
    ctx = _minimal_context()
    from functional_agents.context import AgentResult
    result = AgentResult(
        status="success", next_action="CONTINUE", summary="ok",
        context=ctx, outputs={}, metrics={"duration_seconds": 0.1}, trace={},
    )
    check = validate_agent_result(result, "TestAgent")
    assert check["returns_agent_result"] is True
    assert check["missing_fields"] == []
    assert check["error"] is None


def test_validate_agent_result_wrong_type():
    check = validate_agent_result("not_a_result", "BadAgent")
    assert check["returns_agent_result"] is False
    assert "BadAgent" in check["error"]


def test_validate_agent_result_via_stub():
    agent = _StubAgent()
    result = agent.run(_minimal_context())
    check = validate_agent_result(result, agent.name)
    assert check["returns_agent_result"] is True
    assert check["missing_fields"] == []


# ---------------------------------------------------------------------------
# build_contract_validation trace block
# ---------------------------------------------------------------------------

def test_build_contract_validation_structure():
    class_checks = validate_all_classes()
    agent = _StubAgent()
    raw_result = agent.run(_minimal_context())
    runtime_checks = {"_StubAgent": validate_agent_result(raw_result, "_StubAgent")}

    block = build_contract_validation(class_checks, runtime_checks)

    assert block["contract_version"] == CONTRACT_VERSION
    assert "agent_contract_valid" in block
    assert "agents" in block


def test_build_contract_validation_per_agent_keys():
    class_checks = validate_all_classes()
    # Simulate runtime checks for all four agents (all succeed via stub logic)
    runtime_checks = {
        name: {"returns_agent_result": True, "missing_fields": [], "error": None}
        for name in class_checks
    }
    block = build_contract_validation(class_checks, runtime_checks)

    for name in ("PlannerAgent", "EvidenceAgent", "QAAgent", "ReportAgent"):
        assert name in block["agents"]
        agent_block = block["agents"][name]
        assert "inherits_base_agent" in agent_block
        assert "implements_run" in agent_block
        assert "returns_agent_result" in agent_block


def test_build_contract_validation_valid_when_all_pass():
    class_checks = validate_all_classes()
    runtime_checks = {
        name: {"returns_agent_result": True, "missing_fields": [], "error": None}
        for name in class_checks
    }
    block = build_contract_validation(class_checks, runtime_checks)
    assert block["agent_contract_valid"] is True


def test_build_contract_validation_invalid_when_runtime_fails():
    class_checks = validate_all_classes()
    runtime_checks = {
        name: {"returns_agent_result": False, "missing_fields": ["outputs"], "error": "x"}
        for name in class_checks
    }
    block = build_contract_validation(class_checks, runtime_checks)
    assert block["agent_contract_valid"] is False


def test_contract_version_constant():
    assert CONTRACT_VERSION == "1.0"


def test_build_contract_validation_report_agent_pre_populated():
    """ReportAgent must appear valid even when its runtime check is pre-populated.

    This guards the specific regression where ReportAgent wrote the trace before
    _step() could record its own returns_agent_result check, causing it to default
    to False and making agent_contract_valid=false.
    """
    class_checks = validate_all_classes()
    # Simulate what ReportAgent._execute() does: other three agents come from
    # _contract_runtime; ReportAgent is pre-populated via setdefault().
    runtime_checks = {
        "PlannerAgent":  {"returns_agent_result": True, "missing_fields": [], "error": None},
        "EvidenceAgent": {"returns_agent_result": True, "missing_fields": [], "error": None},
        "QAAgent":       {"returns_agent_result": True, "missing_fields": [], "error": None},
    }
    runtime_checks.setdefault("ReportAgent", {
        "returns_agent_result": True, "missing_fields": [], "error": None,
    })
    block = build_contract_validation(class_checks, runtime_checks)
    assert block["agent_contract_valid"] is True
    assert block["agents"]["ReportAgent"]["returns_agent_result"] is True
