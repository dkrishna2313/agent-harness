"""J6.3 – HypothesisAgent tests.

Verifies:
  1. Contract compliance (inherits FunctionalAgent, run() returns AgentResult)
  2. HypothesisItem / HypothesisPayload structure and fields
  3. At least 3 hypotheses generated
  4. Each hypothesis has evidence mapping, confidence, implications, disconfirming needs
  5. Hypotheses persisted in context.hypotheses, RO, and trace
  6. Orchestrator routes EVIDENCE → HYPOTHESIS → QA
  7. QAAgent validates hypothesis structure
  8. ReportAgent emits hypothesis_generation trace block
  9. ReportAgent markdown includes Strategic Hypotheses section
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

@pytest.fixture(autouse=True)
def _mock_yaml():
    sys.modules.setdefault("yaml", MagicMock())
    yield


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.hypothesis_agent import HypothesisAgent
from research_agent.claude_client import (
    MockClaudeClient,
    HypothesisPayload,
    HypothesisItem,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DECISION_MODEL = {
    "objective": "Develop a power supply strategy for AI data centres",
    "decision_areas": ["Grid Constraints", "SMR Viability", "Hybrid Portfolios"],
    "critical_uncertainties": ["Transmission queue timelines", "SMR licensing pace"],
    "research_questions": [
        "What are the current grid interconnection queue backlogs?",
        "When can SMRs be commercially deployed?",
        "What hybrid procurement strategies exist?",
    ],
    "evidence_requirements": ["Grid operator data", "SMR cost estimates", "Case studies"],
}

_EVIDENCE_ITEMS = [
    {"evidence_id": "E001", "claim": "Grid interconnection queues have grown 5x since 2020.", "source_document": "GridReport2024"},
    {"evidence_id": "E002", "claim": "AI campus power demand is projected at 50 GW by 2030.", "source_document": "AIGridStudy"},
    {"evidence_id": "E003", "claim": "SMR projects face 10-15 year licensing timelines.", "source_document": "NRCFiling"},
    {"evidence_id": "E004", "claim": "First SMR project in North America is targeting 2032.", "source_document": "TerraPowerUpdate"},
    {"evidence_id": "E005", "claim": "Hybrid power portfolios reduce cost variance by ~20%.", "source_document": "UtilityAnalysis"},
]


def _make_context_with_evidence() -> AgentContext:
    ctx = AgentContext(
        goal="Develop a strategy for supplying power to AI data centers",
        profiles=["smr", "ai_data_centers", "transmission"],
        execution_profile="smr",
        research_object={"id": "R-TEST_001", "contradictions": []},
        run_id="hyp001",
    )
    ctx.decision_model = _DECISION_MODEL
    ctx.research_strategy = {
        "profile_priorities": {"smr": 1, "ai_data_centers": 2, "transmission": 3},
        "research_question_priorities": [
            {"question": q, "priority": i + 1} for i, q in enumerate(_DECISION_MODEL["research_questions"])
        ],
        "coverage_targets": {"Grid Constraints": "strong", "SMR Viability": "strong"},
    }
    ctx.evidence_notes = [{
        "evidence_items": _EVIDENCE_ITEMS,
        "evidence_summary": {"total_evidence_items": 5},
        "coverage_by_subquestion": {},
        "evidence_by_subquestion": {},
        "profile_coverage_by_profile": {
            "smr": {"coverage_level": "MODERATE", "evidence_count": 2},
            "ai_data_centers": {"coverage_level": "STRONG", "evidence_count": 2},
            "transmission": {"coverage_level": "WEAK", "evidence_count": 1},
        },
    }]
    return ctx


# ---------------------------------------------------------------------------
# Contract compliance
# ---------------------------------------------------------------------------

def test_inherits_functional_agent():
    assert issubclass(HypothesisAgent, FunctionalAgent)


def test_run_returns_agent_result():
    agent = HypothesisAgent()
    result = agent.run(_make_context_with_evidence())
    assert isinstance(result, AgentResult)


def test_run_status_success():
    agent = HypothesisAgent()
    result = agent.run(_make_context_with_evidence())
    assert result.status == "success"


def test_run_metrics_has_duration():
    agent = HypothesisAgent()
    result = agent.run(_make_context_with_evidence())
    assert "duration_seconds" in result.metrics
    assert result.metrics["duration_seconds"] >= 0.0


def test_run_trace_required_keys():
    agent = HypothesisAgent()
    result = agent.run(_make_context_with_evidence())
    for key in ("agent", "run_id", "duration_seconds", "status"):
        assert key in result.trace


# ---------------------------------------------------------------------------
# Hypothesis count and structure
# ---------------------------------------------------------------------------

def test_at_least_three_hypotheses_generated():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    assert len(ctx.hypotheses) >= 3


def test_each_hypothesis_has_id():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        assert h.get("id"), f"Hypothesis missing id: {h}"


def test_each_hypothesis_has_title():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        assert h.get("title"), f"Hypothesis missing title: {h}"


def test_each_hypothesis_has_summary():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        assert h.get("summary"), f"Hypothesis missing summary: {h}"


def test_each_hypothesis_has_evidence_mapping():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        has_any = (
            isinstance(h.get("supporting_evidence"), list)
            or isinstance(h.get("contradicting_evidence"), list)
            or isinstance(h.get("evidence_gaps"), list)
        )
        assert has_any, f"Hypothesis {h.get('id')} missing evidence mapping"


def test_each_hypothesis_has_confidence():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        assert h.get("confidence") in ("high", "medium", "low"), (
            f"Hypothesis {h.get('id')} has invalid confidence: {h.get('confidence')}"
        )


def test_each_hypothesis_has_decision_implications():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        impls = h.get("decision_implications", [])
        assert isinstance(impls, list) and len(impls) > 0, (
            f"Hypothesis {h.get('id')} missing decision implications"
        )


def test_each_hypothesis_has_disconfirming_needs():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    for h in ctx.hypotheses:
        dis = h.get("disconfirming_evidence_needed", [])
        assert isinstance(dis, list) and len(dis) > 0, (
            f"Hypothesis {h.get('id')} missing disconfirming evidence needs"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_hypotheses_persisted_to_context():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    assert ctx.hypotheses
    assert isinstance(ctx.hypotheses, list)


def test_hypotheses_persisted_to_research_object():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    assert "hypotheses" in ctx.research_object
    assert ctx.research_object["hypotheses"] == ctx.hypotheses


def test_hypotheses_stashed_in_trace():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    assert "_hypotheses" in ctx.trace
    assert ctx.trace["_hypotheses"]["hypotheses"] == ctx.hypotheses


def test_agent_history_recorded():
    agent = HypothesisAgent()
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    assert len(ctx.agent_history) == 1
    entry = ctx.agent_history[0]
    assert entry["agent"] == "HypothesisAgent"
    assert entry["status"] == "success"
    assert "hypothesis_count" in entry


# ---------------------------------------------------------------------------
# MockClaudeClient integration
# ---------------------------------------------------------------------------

def test_with_mock_client():
    agent = HypothesisAgent(client=MockClaudeClient())
    ctx = _make_context_with_evidence()
    result = agent.run(ctx)
    assert result.status == "success"
    assert len(ctx.hypotheses) == 3
    ids = [h["id"] for h in ctx.hypotheses]
    assert ids == ["H1", "H2", "H3"]


def test_mock_hypotheses_include_evidence_ids_from_evidence():
    agent = HypothesisAgent(client=MockClaudeClient())
    ctx = _make_context_with_evidence()
    agent.run(ctx)
    # At least one hypothesis should reference one of our evidence IDs
    all_ev_refs = set()
    for h in ctx.hypotheses:
        all_ev_refs.update(h.get("supporting_evidence", []))
        all_ev_refs.update(h.get("contradicting_evidence", []))
    known_ids = {e["evidence_id"] for e in _EVIDENCE_ITEMS}
    assert all_ev_refs & known_ids, "No evidence IDs from context referenced in hypotheses"


# ---------------------------------------------------------------------------
# Orchestrator routing: EVIDENCE → HYPOTHESIS → QA
# ---------------------------------------------------------------------------

def test_orchestrator_routes_through_hypothesis():
    """EVIDENCE must transition to HYPOTHESIS when hypothesis_factory is set."""
    from functional_agents.orchestrator import AgentOrchestrator

    calls: list[str] = []

    def _make_stub(name_):
        class _Stub(FunctionalAgent):
            @property
            def name(self):
                return name_
            def _execute(self, ctx):
                calls.append(name_)
                self._record(ctx, status="success", summary=f"{name_} done")
                return ctx
        return _Stub

    ctx = AgentContext(
        question="How should we supply AI data centre power?",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_002"},
        run_id="orch002",
    )

    orch = AgentOrchestrator(
        planner_factory=lambda: _make_stub("PlannerAgent")(),
        evidence_factory=lambda: _make_stub("EvidenceAgent")(),
        hypothesis_factory=lambda: _make_stub("HypothesisAgent")(),
        qa_factory=lambda: _make_stub("QAAgent")(),
        report_factory=lambda: _make_stub("ReportAgent")(),
    )
    result_ctx = orch.run(ctx)

    path = result_ctx.workflow_path
    assert "HypothesisAgent" in path
    idx = {n: path.index(n) for n in path}
    assert idx["EvidenceAgent"] < idx["HypothesisAgent"] < idx["QAAgent"]


def test_orchestrator_skips_hypothesis_when_factory_none():
    """Without hypothesis_factory, EVIDENCE transitions directly to QA."""
    from functional_agents.orchestrator import AgentOrchestrator

    calls: list[str] = []

    def _make_stub(name_):
        class _Stub(FunctionalAgent):
            @property
            def name(self):
                return name_
            def _execute(self, ctx):
                calls.append(name_)
                self._record(ctx, status="success", summary=f"{name_} done")
                return ctx
        return _Stub

    ctx = AgentContext(
        question="How should we supply AI data centre power?",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_003"},
        run_id="orch003",
    )

    orch = AgentOrchestrator(
        planner_factory=lambda: _make_stub("PlannerAgent")(),
        evidence_factory=lambda: _make_stub("EvidenceAgent")(),
        qa_factory=lambda: _make_stub("QAAgent")(),
        report_factory=lambda: _make_stub("ReportAgent")(),
    )
    result_ctx = orch.run(ctx)

    assert "HypothesisAgent" not in result_ctx.workflow_path


# ---------------------------------------------------------------------------
# QA validates hypothesis structure
# ---------------------------------------------------------------------------

def test_qa_validates_hypotheses_present():
    from functional_agents.qa_agent import _validate_hypotheses

    hyps = [
        {
            "id": "H1", "title": "Grid constraints dominate", "summary": "...", "type": "constraint",
            "supporting_evidence": ["E001"], "contradicting_evidence": [], "evidence_gaps": ["data"],
            "confidence": "medium", "confidence_rationale": "...",
            "decision_implications": ["Act now"], "disconfirming_evidence_needed": ["X"],
        },
        {
            "id": "H2", "title": "SMR unlocks mid-term", "summary": "...", "type": "technology",
            "supporting_evidence": [], "contradicting_evidence": ["E003"], "evidence_gaps": ["cost"],
            "confidence": "low", "confidence_rationale": "...",
            "decision_implications": ["Wait and see"], "disconfirming_evidence_needed": ["Y"],
        },
        {
            "id": "H3", "title": "Hybrid portfolio wins", "summary": "...", "type": "portfolio",
            "supporting_evidence": ["E005"], "contradicting_evidence": [], "evidence_gaps": [],
            "confidence": "medium", "confidence_rationale": "...",
            "decision_implications": ["Diversify"], "disconfirming_evidence_needed": ["Z"],
        },
    ]
    result = _validate_hypotheses(hyps)
    assert result["hypotheses_present"] is True
    assert result["hypothesis_count"] == 3
    assert result["all_have_evidence_mapping"] is True
    assert result["all_have_confidence"] is True
    assert result["all_have_decision_implications"] is True
    assert result["all_have_disconfirming_evidence_needs"] is True
    assert result["issues"] == []


def test_qa_flags_missing_hypotheses():
    from functional_agents.qa_agent import _validate_hypotheses
    result = _validate_hypotheses([])
    assert result["hypotheses_present"] is False
    assert "No hypotheses generated" in result["issues"]


def test_qa_flags_insufficient_count():
    from functional_agents.qa_agent import _validate_hypotheses
    hyps = [
        {
            "id": "H1", "confidence": "medium",
            "decision_implications": ["act"], "disconfirming_evidence_needed": ["x"],
        },
    ]
    result = _validate_hypotheses(hyps)
    assert any("minimum 3" in issue for issue in result["issues"])


def test_qa_result_includes_hypothesis_validation():
    """Full QAAgent run must include hypothesis_validation in context.qa."""
    from functional_agents.qa_agent import QAAgent

    ctx = AgentContext(
        question="Test?",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_004"},
        run_id="qa004",
    )
    ctx.plan = {
        "subquestions": [],
        "investigation_areas": [],
        "research_type": "RESEARCH",
        "profiles_used": [],
        "reasoning": "",
    }
    ctx.evidence_notes = [{
        "evidence_summary": {"total_evidence_items": 0},
        "coverage_by_subquestion": {},
        "evidence_by_subquestion": {},
        "profile_coverage_by_profile": {},
    }]
    ctx.hypotheses = [
        {
            "id": "H1", "confidence": "medium", "title": "Test H1",
            "decision_implications": ["do x"], "disconfirming_evidence_needed": ["y"],
            "supporting_evidence": [], "contradicting_evidence": [], "evidence_gaps": [],
        },
        {
            "id": "H2", "confidence": "low", "title": "Test H2",
            "decision_implications": ["do y"], "disconfirming_evidence_needed": ["z"],
            "supporting_evidence": [], "contradicting_evidence": [], "evidence_gaps": [],
        },
        {
            "id": "H3", "confidence": "high", "title": "Test H3",
            "decision_implications": ["do z"], "disconfirming_evidence_needed": ["w"],
            "supporting_evidence": [], "contradicting_evidence": [], "evidence_gaps": [],
        },
    ]

    QAAgent().run(ctx)
    assert "hypothesis_validation" in ctx.qa
    hv = ctx.qa["hypothesis_validation"]
    assert hv["hypotheses_present"] is True
    assert hv["hypothesis_count"] == 3


# ---------------------------------------------------------------------------
# ReportAgent emits hypothesis_generation trace block and section
# ---------------------------------------------------------------------------

def test_report_agent_emits_hypothesis_trace_block():
    """ReportAgent must emit hypothesis_generation in trace when hypotheses are present."""
    from functional_agents.report_agent import ReportAgent
    from research_agent.schemas import ResearchMemo
    from pathlib import Path
    import json, tempfile

    ctx = AgentContext(
        question="AI power supply strategy?",
        profiles=["smr"],
        execution_profile="smr",
        research_object={"id": "R-TEST_005", "research_id": "R-TEST_005"},
        run_id="rpt005",
    )
    ctx.plan = {
        "question": ctx.question, "research_type": "RESEARCH",
        "subquestions": [], "investigation_areas": [], "profiles_used": ["smr"], "reasoning": "",
    }
    ctx.qa = {
        "qa_summary": {"issues_found": 0, "coverage_issues": 0, "evidence_issues": 0, "contradiction_issues": 0},
        "confidence_assessment": {"overall_confidence": "MEDIUM"},
        "profiles_contributing": ["smr"], "profiles_missing": [], "coverage_status": "sufficient",
    }
    ctx.evidence_notes = [{
        "evidence_summary": {
            "total_evidence_items": 3, "subquestions_with_evidence": 0,
            "subquestions_without_evidence": 0, "investigation_areas_with_evidence": 0,
            "coverage_distribution": {},
        },
        "coverage_by_subquestion": {}, "evidence_by_subquestion": {}, "evidence_items": [],
    }]
    ctx.trace["_memo"] = ResearchMemo(
        title="Test", question=ctx.question, executive_summary="test",
        confirmed_facts=[], inferences=[], power_implications=[], cooling_implications=[],
        networking_implications=[], rack_architecture_implications=[],
        open_questions=[], source_notes=[], evidence=[],
    )
    ctx.hypotheses = [
        {
            "id": "H1", "title": "Grid constraints dominate", "summary": "Grid is the bottleneck.",
            "type": "constraint_dominant", "confidence": "medium", "confidence_rationale": "Strong grid evidence.",
            "supporting_evidence": ["E001", "E002"], "contradicting_evidence": [],
            "evidence_gaps": ["Regional data"], "decision_implications": ["Build near grid"],
            "disconfirming_evidence_needed": ["Evidence queues are clearing"],
        },
        {
            "id": "H2", "title": "SMRs viable post-2030", "summary": "SMR timeline is 2030+.",
            "type": "technology_option", "confidence": "low", "confidence_rationale": "Sparse data.",
            "supporting_evidence": ["E003"], "contradicting_evidence": ["E004"],
            "evidence_gaps": ["Cost trajectory"], "decision_implications": ["Maintain SMR option"],
            "disconfirming_evidence_needed": ["Early commercial deployment evidence"],
        },
        {
            "id": "H3", "title": "Hybrid portfolio outperforms", "summary": "Diversify across options.",
            "type": "portfolio_strategy", "confidence": "medium", "confidence_rationale": "Decision theory.",
            "supporting_evidence": ["E005"], "contradicting_evidence": [],
            "evidence_gaps": ["Cross-strategy comparison"], "decision_implications": ["Set allocation thresholds"],
            "disconfirming_evidence_needed": ["Evidence one path dominates"],
        },
    ]
    ctx.trace["_hypotheses"] = {
        "hypotheses": ctx.hypotheses,
        "synthesis_note": "Three competing views of the energy strategy landscape.",
    }

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "report.md"
        ReportAgent(out_path=out_path).run(ctx)

        trace_path = out_path.with_suffix(".trace.json")
        assert trace_path.exists()
        trace = json.loads(trace_path.read_text())
        md = out_path.read_text()

    assert "hypothesis_generation" in trace
    hg = trace["hypothesis_generation"]
    assert hg["hypothesis_count"] == 3
    assert len(hg["hypotheses"]) == 3

    assert "## Strategic Hypotheses" in md
    assert "H1" in md and "H2" in md and "H3" in md
    assert "Grid constraints dominate" in md
    assert "candidate interpretations" in md
