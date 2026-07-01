"""Tests for per-Decision-Domain hypothesis generation (J10.6).

The multi-domain wrapper is exercised by stubbing _execute_single so tests need
no LLM. The stub writes hypotheses derived from the current evidence set.
"""

from __future__ import annotations

from functional_agents.context import AgentContext
from functional_agents.hypothesis_agent import HypothesisAgent


class _StubHypothesisAgent(HypothesisAgent):
    def _execute_single(self, context: AgentContext) -> AgentContext:
        note = context.evidence_notes[0] if context.evidence_notes else {}
        items = note.get("evidence_items", [])
        tag = items[0]["evidence_id"] if items else "none"
        hyps = [{"id": "H1", "title": f"hypothesis from {tag}", "confidence": "medium"}]
        context.hypotheses = hyps
        if context.research_object is not None:
            context.research_object["hypotheses"] = hyps
        context.trace["_hypotheses"] = {"hypotheses": hyps, "synthesis_note": f"note-{tag}"}
        self._record(context, status="success", summary=f"stub hypotheses {tag}")
        return context


def _domain_evidence(domain_id, title, eid):
    return {
        "decision_domain_id": domain_id,
        "decision_domain_title": title,
        "evidence": [{"evidence_id": eid, "claim": f"claim {eid}"}],
        "mapping": {"evidence_by_subquestion": {}, "evidence_by_area": {}},
        "coverage": {"coverage_by_subquestion": {}, "evidence_summary": {}},
    }


def _engagement_ctx() -> AgentContext:
    primary_note = {
        "evidence_items": [{"evidence_id": "E-PRIMARY", "claim": "primary claim"}],
        "profile_coverage_by_profile": {},
    }
    return AgentContext(
        question="PRIMARY?",
        engagement={"title": "E"},
        decision_model={},
        evidence_notes=[primary_note],
        domain_evidence=[
            _domain_evidence("domain-1", "Power Procurement", "E-PRIMARY"),
            _domain_evidence("domain-2", "Cooling Architecture", "E-COOL"),
            _domain_evidence("domain-3", "Site Strategy", "E-SITE"),
        ],
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-DH", "contradictions": []},
        run_id="dh001",
    )


def _goal_ctx() -> AgentContext:
    note = {"evidence_items": [{"evidence_id": "E-G", "claim": "c"}], "profile_coverage_by_profile": {}}
    return AgentContext(
        question="Goal?",
        engagement={},
        decision_model={},
        evidence_notes=[note],
        domain_evidence=[_domain_evidence(None, None, "E-G")],
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        research_object={"id": "R-GH", "contradictions": []},
        run_id="gh001",
    )


# ---------------------------------------------------------------------------

def test_one_hypothesis_set_per_domain():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    assert len(ctx.domain_hypotheses) == 3
    assert [d["decision_domain_title"] for d in ctx.domain_hypotheses] == [
        "Power Procurement", "Cooling Architecture", "Site Strategy",
    ]


def test_goal_mode_single_hypothesis_set():
    ctx = _goal_ctx()
    _StubHypothesisAgent().run(ctx)
    assert len(ctx.domain_hypotheses) == 1


def test_primary_hypotheses_deterministic_and_first():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    assert ctx.domain_hypotheses[0]["decision_domain_id"] == "domain-1"
    # context.hypotheses reflects the PRIMARY evidence, not a secondary.
    assert "E-PRIMARY" in ctx.hypotheses[0]["title"]


def test_primary_hypotheses_unchanged_downstream():
    """context.hypotheses stays the primary set (ChallengeAgent unaffected)."""
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    assert ctx.hypotheses == ctx.domain_hypotheses[0]["hypotheses"]
    assert len(ctx.hypotheses) == 1


def test_hypothesis_schema_unchanged():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    h = ctx.hypotheses[0]
    assert set(("id", "title", "confidence")).issubset(h.keys())


def test_secondary_runs_do_not_pollute_research_object():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    # RO hypotheses reflect the PRIMARY run only.
    assert "E-PRIMARY" in ctx.research_object["hypotheses"][0]["title"]


def test_single_agent_history_entry():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    assert len(ctx.agent_history) == 1


def test_domain_hypotheses_entry_shape():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    entry = ctx.domain_hypotheses[1]
    assert set(entry.keys()) == {
        "decision_domain_id", "decision_domain_title",
        "hypotheses", "synthesis_note", "diagnostics",
    }
    assert entry["diagnostics"]["hypothesis_count"] == 1


def test_diagnostics_recorded():
    ctx = _engagement_ctx()
    _StubHypothesisAgent().run(ctx)
    diag = ctx.trace["_hypothesis_reasoning"]
    assert diag["evidence_sets_received"] == 3
    assert diag["hypothesis_sets_generated"] == 3
    assert diag["hypothesis_sets_executed"] == 1
    assert diag["primary_domain"] == "Power Procurement"


def test_goal_mode_diagnostics():
    ctx = _goal_ctx()
    _StubHypothesisAgent().run(ctx)
    diag = ctx.trace["_hypothesis_reasoning"]
    assert diag["hypothesis_sets_generated"] == 1
    assert diag["hypothesis_sets_executed"] == 1
