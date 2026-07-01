"""Tests for per-Decision-Domain evidence collection (J10.5).

The multi-domain wrapper is tested by stubbing _execute_single so the tests do
not require a Knowledge Store / retriever. The stub records the question it ran
on and writes an evidence collection derived from it — this exercises the
per-domain iteration, primary isolation, capture, and diagnostics.
"""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.evidence_agent import EvidenceAgent


class _StubEvidenceAgent(EvidenceAgent):
    """EvidenceAgent whose single-domain pass is a deterministic stub."""

    def _execute_single(self, context: AgentContext) -> AgentContext:
        q = context.question
        # Simulate the real agent's evidence_notes structure.
        context.evidence_notes = [{
            "evidence_items": [{"evidence_id": f"E-{q[:8]}", "claim": f"claim for {q}"}],
            "evidence_by_subquestion": {q: [f"E-{q[:8]}"]},
            "evidence_by_area": {"Area": [f"E-{q[:8]}"]},
            "coverage_by_subquestion": {q: {"coverage": "STRONG"}},
            "evidence_summary": {"total_evidence_items": 1},
            "profile_coverage_by_profile": {},
            "profiles_requested": context.profiles,
            "profiles_contributing": context.profiles,
            "profiles_missing": [],
        }]
        if context.research_object is not None:
            context.research_object["evidence_summary"] = {"total_evidence_items": 1}
        self._record(context, status="success", summary=f"stub evidence for {q}")
        return context


def _engagement_ctx() -> AgentContext:
    return AgentContext(
        question="PRIMARY question?",
        engagement={"title": "E"},
        plan={"question": "PRIMARY question?", "research_type": "RESEARCH",
              "subquestions": ["PRIMARY question?"], "investigation_areas": ["Area"],
              "profiles_used": ["ai_data_centers"], "reasoning": "r"},
        domain_plans=[
            {"question": "PRIMARY question?", "research_type": "RESEARCH",
             "subquestions": ["PRIMARY question?"], "investigation_areas": ["Area"],
             "profiles_used": ["ai_data_centers"], "reasoning": "r",
             "decision_domain_id": "domain-1", "decision_domain_title": "Power Procurement",
             "target_kind": "decision_domain", "is_primary": True},
            {"question": "Cooling question?", "research_type": "RESEARCH",
             "subquestions": ["Cooling question?"], "investigation_areas": ["Cooling"],
             "profiles_used": ["ai_data_centers"], "reasoning": "r",
             "decision_domain_id": "domain-2", "decision_domain_title": "Cooling Architecture",
             "target_kind": "decision_domain", "is_primary": False},
            {"question": "Site question?", "research_type": "RESEARCH",
             "subquestions": ["Site question?"], "investigation_areas": ["Site"],
             "profiles_used": ["ai_data_centers"], "reasoning": "r",
             "decision_domain_id": "domain-3", "decision_domain_title": "Site Strategy",
             "target_kind": "decision_domain", "is_primary": False},
        ],
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-DE"},
        run_id="de001",
    )


def _goal_ctx() -> AgentContext:
    return AgentContext(
        question="Goal question?",
        engagement={},
        plan={"question": "Goal question?", "research_type": "RESEARCH",
              "subquestions": ["Goal question?"], "investigation_areas": ["Area"],
              "profiles_used": ["ai_data_centers"], "reasoning": "r"},
        domain_plans=[
            {"question": "Goal question?", "research_type": "RESEARCH",
             "subquestions": ["Goal question?"], "investigation_areas": ["Area"],
             "profiles_used": ["ai_data_centers"], "reasoning": "r",
             "decision_domain_id": None, "decision_domain_title": None,
             "target_kind": "research_question", "is_primary": True},
        ],
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-GE"},
        run_id="ge001",
    )


# ---------------------------------------------------------------------------
# Multi-domain evidence
# ---------------------------------------------------------------------------

def test_one_evidence_collection_per_domain():
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    assert len(ctx.domain_evidence) == 3
    assert [d["decision_domain_title"] for d in ctx.domain_evidence] == [
        "Power Procurement", "Cooling Architecture", "Site Strategy",
    ]


def test_goal_mode_single_collection():
    ctx = _goal_ctx()
    _StubEvidenceAgent().run(ctx)
    assert len(ctx.domain_evidence) == 1


def test_primary_evidence_is_deterministic_and_first():
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    # domain_evidence[0] corresponds to the primary domain.
    assert ctx.domain_evidence[0]["decision_domain_id"] == "domain-1"
    # Primary context.evidence_notes reflects the PRIMARY question, not a secondary.
    assert ctx.evidence_notes[0]["evidence_items"][0]["evidence_id"].startswith("E-PRIMARY")
    assert "PRIMARY" in list(ctx.evidence_notes[0]["evidence_by_subquestion"].keys())[0]


def test_primary_evidence_notes_unchanged_downstream():
    """context.evidence_notes stays the PRIMARY collection (byte-identical path)."""
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    note = ctx.evidence_notes[0]
    # Exactly the primary domain's evidence; secondaries did not clobber it.
    assert list(note["evidence_by_subquestion"].keys()) == ["PRIMARY question?"]


def test_evidence_schema_unchanged():
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    note = ctx.evidence_notes[0]
    for key in ("evidence_items", "evidence_by_subquestion", "evidence_by_area",
                "coverage_by_subquestion", "evidence_summary"):
        assert key in note


def test_secondary_runs_do_not_pollute_research_object():
    ctx = _engagement_ctx()
    ctx.research_object["marker"] = "original"
    _StubEvidenceAgent().run(ctx)
    # Only the primary run touched the real RO; secondaries used scratch copies.
    assert ctx.research_object["marker"] == "original"
    assert ctx.research_object.get("evidence_summary") == {"total_evidence_items": 1}


def test_single_agent_history_entry():
    """Only the primary pass records into the real agent_history."""
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    # Only the primary pass records; secondary scratch contexts are isolated.
    assert len(ctx.agent_history) == 1


def test_domain_evidence_entry_shape():
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    entry = ctx.domain_evidence[1]
    assert set(entry.keys()) == {
        "decision_domain_id", "decision_domain_title", "evidence", "mapping", "coverage",
    }
    assert "evidence_by_subquestion" in entry["mapping"]
    assert "coverage_by_subquestion" in entry["coverage"]


def test_diagnostics_recorded():
    ctx = _engagement_ctx()
    _StubEvidenceAgent().run(ctx)
    diag = ctx.trace["_evidence_reasoning"]
    assert diag["plans_received"] == 3
    assert diag["evidence_sets_generated"] == 3
    assert diag["evidence_sets_executed"] == 1
    assert diag["primary_domain"] == "Power Procurement"


def test_goal_mode_diagnostics():
    ctx = _goal_ctx()
    _StubEvidenceAgent().run(ctx)
    diag = ctx.trace["_evidence_reasoning"]
    assert diag["evidence_sets_generated"] == 1
    assert diag["evidence_sets_executed"] == 1
