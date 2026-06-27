"""J7.5c – OpportunityAgent persistence hardening tests.

Covers:
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - No-op when no assumptions
  - _execute(): writes opportunities, RO, trace
  - _opportunity_persistence trace block (J7.5c)
  - DM persistence: unknown category coerced to 'Other' (root-cause regression guard)
  - Linkage verification: no orphan IDs after persistence
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult
from functional_agents.opportunity_agent import OpportunityAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opportunity_item(o_id: str, category: str = "Market") -> dict:
    return {
        "opportunity_id": o_id,
        "statement": f"Opportunity {o_id} becomes possible",
        "category": category,
        "impact": "High",
        "likelihood": "Medium",
        "evidence_support": "Moderate",
        "confidence": "Medium",
        "rationale": "Strategically relevant.",
        "related_assumption_ids": ["A-001"],
        "enabled_recommendation_ids": ["REC-001"],
        "evidence_ids": ["EV-001"],
        "exploitation_notes": "",
        "status": "Active",
    }


def _ctx(*, with_assumptions=True) -> AgentContext:
    assumptions = [{"assumption_id": "A-001", "statement": "S", "supported_recommendation_ids": ["REC-001"]}] if with_assumptions else []
    return AgentContext(
        question="Should we invest?",
        profiles=["SMR"],
        execution_profile="SMR",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        assumptions=assumptions,
        recommendations=[{"recommendation_id": "REC-001", "title": "R1"}],
        risks=[],
        evidence_notes=[{"evidence_items": [{"evidence_id": "EV-001", "claim": "Fact."}]}],
        decision_model={"strategic_question": "Should we invest?"},
        research_strategy={},
    )


def _mock_client_with(items: list[dict]) -> MagicMock:
    from research_agent.claude_client import OpportunityItem, OpportunityPayload
    parsed = [OpportunityItem(**o) for o in items]
    payload = OpportunityPayload(opportunities=parsed)
    client = MagicMock()
    client.generate_opportunities.return_value = payload
    return client


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(OpportunityAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = OpportunityAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = OpportunityAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# No-op when assumptions absent
# ---------------------------------------------------------------------------

def test_agent_skips_without_assumptions():
    result = OpportunityAgent().run(_ctx(with_assumptions=False))
    assert result.status == "skipped"


def test_agent_skip_writes_trace():
    result = OpportunityAgent().run(_ctx(with_assumptions=False))
    trace = result.context.trace.get("_strategic_opportunities", {})
    assert trace.get("skipped") is True


def test_agent_skip_leaves_opportunities_empty():
    result = OpportunityAgent().run(_ctx(with_assumptions=False))
    assert result.context.opportunities == []


# ---------------------------------------------------------------------------
# Context writes
# ---------------------------------------------------------------------------

def test_agent_writes_opportunities_to_context():
    result = OpportunityAgent().run(_ctx())
    assert isinstance(result.context.opportunities, list)
    assert len(result.context.opportunities) > 0


def test_agent_writes_opportunities_to_ro():
    result = OpportunityAgent().run(_ctx())
    ro = result.context.research_object
    assert "strategic_opportunities" in ro
    assert len(ro["strategic_opportunities"]) > 0


def test_agent_writes_strategic_opportunities_trace():
    result = OpportunityAgent().run(_ctx())
    assert "_strategic_opportunities" in result.context.trace


def test_strategic_opportunities_trace_has_counts():
    result = OpportunityAgent().run(_ctx())
    trace = result.context.trace["_strategic_opportunities"]
    assert "opportunity_count" in trace
    assert "assumption_links" in trace
    assert "recommendation_links" in trace


# ---------------------------------------------------------------------------
# _opportunity_persistence trace block (J7.5c)
# ---------------------------------------------------------------------------

def test_opportunity_persistence_trace_present():
    result = OpportunityAgent().run(_ctx())
    assert "_opportunity_persistence" in result.context.trace


def test_opportunity_persistence_trace_has_required_fields():
    result = OpportunityAgent().run(_ctx())
    trace = result.context.trace["_opportunity_persistence"]
    for field in ("opportunity_count", "dm_persisted", "ro_persisted", "orphan_ids", "linkage_verified"):
        assert field in trace, f"Missing field: {field}"


def test_opportunity_persistence_trace_count_matches_context():
    result = OpportunityAgent().run(_ctx())
    trace = result.context.trace["_opportunity_persistence"]
    assert trace["opportunity_count"] == len(result.context.opportunities)


def test_opportunity_persistence_trace_orphan_ids_is_list():
    result = OpportunityAgent().run(_ctx())
    trace = result.context.trace["_opportunity_persistence"]
    assert isinstance(trace["orphan_ids"], list)


def test_opportunity_persistence_linkage_verified_when_no_orphans():
    """When orphan_ids is empty, linkage_verified must be True."""
    result = OpportunityAgent().run(_ctx())
    trace = result.context.trace["_opportunity_persistence"]
    if not trace["orphan_ids"]:
        assert trace["linkage_verified"] is True


# ---------------------------------------------------------------------------
# DM category coercion — root-cause regression guard (J7.5c)
# ---------------------------------------------------------------------------

def test_unknown_category_does_not_raise_on_model_validate():
    """StrategicOpportunity.model_validate must not raise for LLM-returned non-Literal categories."""
    from research_agent.decision_model import StrategicOpportunity
    opp = StrategicOpportunity.model_validate(_opportunity_item("OPP-TEST", category="Ecosystem"))
    assert opp.category == "Other"


def test_known_category_passes_through_unchanged():
    from research_agent.decision_model import StrategicOpportunity
    opp = StrategicOpportunity.model_validate(_opportunity_item("OPP-TEST", category="Infrastructure"))
    assert opp.category == "Infrastructure"


def test_agent_with_unknown_category_writes_all_to_context():
    """When the LLM returns an unknown category, ALL opportunities must still appear in context."""
    items = [
        _opportunity_item("OPP-001", category="Ecosystem"),   # unknown
        _opportunity_item("OPP-002", category="Market"),       # known
        _opportunity_item("OPP-003", category="ClimateTech"),  # unknown
    ]
    ctx = _ctx()
    result = OpportunityAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.opportunities) == 3
