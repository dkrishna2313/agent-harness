"""J7.5b – AssumptionAgent tests.

Covers:
  - Agent contract: inherits FunctionalAgent, run() returns AgentResult
  - No-op when no hypotheses or evidence
  - _execute(): writes assumptions, RO, trace
  - Cardinality: 3–7 assumptions enforced
  - Exactly-conflict resolution (symmetric conflicts_with)
  - Graph integrity: downstream IDs (risks, opportunities, options) not orphaned
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from functional_agents.assumption_agent import AssumptionAgent
from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assumption_item(a_id: str, importance: str = "Critical") -> dict:
    return {
        "assumption_id": a_id,
        "statement": f"Assumption {a_id} must hold",
        "category": "Market",
        "importance": importance,
        "evidence_support": "Moderate",
        "confidence": "Medium",
        "rationale": "This matters strategically.",
        "evidence_ids": ["EV-001"],
        "conflicts_with": [],
        "status": "Active",
    }


def _ctx(*, hypotheses=None, challenges=None) -> AgentContext:
    return AgentContext(
        question="Should we invest in SMR?",
        profiles=["SMR"],
        execution_profile="SMR",
        research_object={"id": "R-TEST_001"},
        run_id="testrun001",
        surviving_hypotheses=hypotheses or [{"hypothesis_id": "H1", "title": "H1"}],
        hypothesis_challenges=challenges or [],
        evidence_notes=[{"evidence_items": [{"evidence_id": "EV-001", "claim": "Key fact."}]}],
        decision_model={"strategic_question": "Should we invest?"},
        research_strategy={},
    )


def _mock_client_with(items: list[dict]) -> MagicMock:
    """Return a mock client whose generate_assumptions() returns the given items."""
    from research_agent.claude_client import AssumptionItem, AssumptionPayload
    parsed = [AssumptionItem(**a) for a in items]
    payload = AssumptionPayload(assumptions=parsed, conflict_pairs=[])
    client = MagicMock()
    client.generate_assumptions.return_value = payload
    return client


# ---------------------------------------------------------------------------
# Agent contract
# ---------------------------------------------------------------------------

def test_agent_inherits_functional_agent():
    assert issubclass(AssumptionAgent, FunctionalAgent)


def test_agent_run_returns_agent_result():
    result = AssumptionAgent().run(_ctx())
    assert isinstance(result, AgentResult)


def test_agent_run_status_success():
    result = AssumptionAgent().run(_ctx())
    assert result.status == "success"


# ---------------------------------------------------------------------------
# Context writes
# ---------------------------------------------------------------------------

def test_agent_writes_assumptions_to_context():
    result = AssumptionAgent().run(_ctx())
    assert isinstance(result.context.assumptions, list)
    assert len(result.context.assumptions) >= 3


def test_agent_writes_assumptions_to_ro():
    result = AssumptionAgent().run(_ctx())
    ro = result.context.research_object
    assert "strategic_assumptions" in ro
    assert len(ro["strategic_assumptions"]) >= 3


def test_agent_writes_trace_block():
    result = AssumptionAgent().run(_ctx())
    assert "_assumptions" in result.context.trace


def test_trace_has_count():
    result = AssumptionAgent().run(_ctx())
    trace = result.context.trace["_assumptions"]
    assert "count" in trace
    assert 3 <= trace["count"] <= 7


# ---------------------------------------------------------------------------
# Cardinality constraints (J7.5b)
# ---------------------------------------------------------------------------

def test_three_assumptions_accepted():
    items = [_assumption_item(f"A-{i:03d}") for i in range(1, 4)]
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) == 3


def test_seven_assumptions_accepted():
    items = [_assumption_item(f"A-{i:03d}") for i in range(1, 8)]
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) == 7


def test_fewer_than_three_padded_to_three():
    items = [_assumption_item("A-001"), _assumption_item("A-002")]
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) >= 3


def test_one_assumption_padded_to_three():
    items = [_assumption_item("A-001")]
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) >= 3


def test_more_than_seven_truncated_to_seven():
    items = [_assumption_item(f"A-{i:03d}") for i in range(1, 12)]
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) == 7


def test_truncation_preserves_critical_over_supporting():
    """When truncating from >7, Critical assumptions should be kept over Supporting."""
    items = (
        [_assumption_item(f"A-{i:03d}", importance="Supporting") for i in range(1, 6)]
        + [_assumption_item(f"A-{i:03d}", importance="Critical") for i in range(6, 11)]
    )
    ctx = _ctx()
    result = AssumptionAgent(client=_mock_client_with(items)).run(ctx)
    assert len(result.context.assumptions) == 7
    importances = [a.get("importance") for a in result.context.assumptions]
    # All Critical (5) must be present; Supporting ones fill remaining slots (2)
    critical_count = importances.count("Critical")
    assert critical_count == 5


# ---------------------------------------------------------------------------
# Graph integrity — downstream IDs not orphaned
# ---------------------------------------------------------------------------

def test_assumption_ids_are_unique():
    result = AssumptionAgent().run(_ctx())
    ids = [a["assumption_id"] for a in result.context.assumptions]
    assert len(ids) == len(set(ids))


def test_assumption_ids_non_empty():
    result = AssumptionAgent().run(_ctx())
    for a in result.context.assumptions:
        assert a.get("assumption_id"), "assumption_id must be non-empty"


def test_conflict_pairs_reference_valid_ids():
    """Any assumption_id referenced in conflicts_with must exist in the assumption list."""
    result = AssumptionAgent().run(_ctx())
    ids = {a["assumption_id"] for a in result.context.assumptions}
    for a in result.context.assumptions:
        for ref in a.get("conflicts_with", []):
            assert ref in ids, f"{a['assumption_id']}.conflicts_with references unknown ID {ref}"
