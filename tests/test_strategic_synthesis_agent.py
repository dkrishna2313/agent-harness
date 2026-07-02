"""Tests for StrategicSynthesisAgent — cross-domain synthesis (J10.7)."""

from __future__ import annotations

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.strategic_synthesis_agent import StrategicSynthesisAgent


def _ctx(*, domains: int = 3, with_arch: bool = True) -> AgentContext:
    all_titles = ["Power Procurement", "Cooling Architecture", "Site Strategy"]
    titles = all_titles[:domains]
    domain_hypotheses = [
        {
            "decision_domain_id": f"domain-{i+1}",
            "decision_domain_title": t,
            "hypotheses": [{"id": "H1", "title": f"hyp for {t}"}],
            "synthesis_note": "",
            "diagnostics": {"hypothesis_count": 1},
        }
        for i, t in enumerate(titles)
    ]
    arch = {
        "decision_statement": "Determine the optimal power strategy.",
        "strategic_themes": titles,
        "executive_unknowns": ["Realized power draw"],
    } if with_arch else {}
    return AgentContext(
        question="q",
        engagement={"title": "E"} if with_arch else {},
        decision_architecture=arch,
        domain_plans=[{"decision_domain_title": t} for t in titles],
        domain_evidence=[{"decision_domain_title": t, "evidence": []} for t in titles],
        domain_hypotheses=domain_hypotheses,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-SS"},
        run_id="ss001",
    )


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------

def test_workflow_state_constant():
    assert WorkflowState.STRATEGIC_SYNTHESIS == "STRATEGIC_SYNTHESIS"


def test_inherits_functional_agent():
    assert issubclass(StrategicSynthesisAgent, FunctionalAgent)


def test_run_returns_agent_result():
    result = StrategicSynthesisAgent().run(_ctx())
    assert isinstance(result, AgentResult)
    assert result.status == "success"


# ---------------------------------------------------------------------------
# Synthesis output
# ---------------------------------------------------------------------------

def test_consumes_every_domain():
    ctx = _ctx(domains=3)
    StrategicSynthesisAgent().run(ctx)
    # A finding per domain proves each contributed.
    assert len(ctx.strategic_synthesis["cross_domain_findings"]) == 3


def test_all_output_fields_present():
    ctx = _ctx()
    StrategicSynthesisAgent().run(ctx)
    ss = ctx.strategic_synthesis
    for key in ("executive_summary", "cross_domain_findings", "cross_domain_dependencies",
                "cross_domain_conflicts", "strategic_levers", "dominant_constraints",
                "emerging_themes"):
        assert key in ss


def test_no_recommendations_in_output():
    ctx = _ctx()
    StrategicSynthesisAgent().run(ctx)
    ss = ctx.strategic_synthesis
    assert "recommendations" not in ss
    assert "recommendation_portfolio" not in ss


def test_dependencies_made_explicit():
    ctx = _ctx(domains=3)
    StrategicSynthesisAgent().run(ctx)
    deps = ctx.strategic_synthesis["cross_domain_dependencies"]
    assert deps and any("depends on" in d for d in deps)


def test_single_synthesis_object():
    ctx = _ctx()
    StrategicSynthesisAgent().run(ctx)
    assert isinstance(ctx.strategic_synthesis, dict)


# ---------------------------------------------------------------------------
# Persistence + trace
# ---------------------------------------------------------------------------

def test_persists_to_research_object():
    ctx = _ctx()
    StrategicSynthesisAgent().run(ctx)
    assert "strategic_synthesis" in ctx.research_object
    assert ctx.research_object["strategic_synthesis"]["cross_domain_findings"]


def test_trace_diagnostics_recorded():
    ctx = _ctx(domains=3)
    StrategicSynthesisAgent().run(ctx)
    diag = ctx.trace["_strategic_synthesis"]["diagnostics"]
    assert diag["domains_received"] == 3
    assert diag["dependencies_identified"] >= 1
    assert "conflicts_identified" in diag
    assert "strategic_themes" in diag


# ---------------------------------------------------------------------------
# No-op + fallback
# ---------------------------------------------------------------------------

def test_no_domains_skips_gracefully():
    ctx = AgentContext(
        question="q", profiles=["ai_data_centers"], execution_profile="ai_data_centers",
        research_object={"id": "R-EMPTY"}, run_id="e001",
    )
    result = StrategicSynthesisAgent().run(ctx)
    assert result.status in ("skipped", "success")
    assert ctx.trace["_strategic_synthesis"].get("skipped") is True


def test_goal_mode_single_domain_still_synthesizes():
    ctx = _ctx(domains=1)
    StrategicSynthesisAgent().run(ctx)
    assert ctx.strategic_synthesis["cross_domain_findings"]


def test_llm_failure_falls_back_to_deterministic():
    class _BoomClient:
        is_mock = False
        def generate_strategic_synthesis(self, *a, **k):
            raise RuntimeError("boom")
    ctx = _ctx()
    result = StrategicSynthesisAgent(client=_BoomClient()).run(ctx)
    assert result.status == "success"
    assert ctx.strategic_synthesis["cross_domain_findings"]
