"""PH5.5e — Evidence Grounding and Traceability tests.

Verifies:
  1. _grounding_strength() maps hybrid_score to correct GroundingStrength value.
  2. _coverage_contribution() maps subquestion evidence count to correct strength.
  3. ground_evidence() produces a new Evidence object with grounding fields set.
  4. ground_evidence() does not mutate the original Evidence object.
  5. Subquestion assignment: populated only from evidence_by_subquestion mapping.
  6. Area assignment: populated only from evidence_by_area mapping.
  7. coverage_contribution uses the least-covered assigned subquestion.
  8. Unmapped evidence gets empty assignments and WEAK contribution.
  9. _execute_kb() stores _grounded_evidence in context.trace.
 10. Grounded evidence has all four grounding fields populated.
 11. Grounded evidence count matches candidate count.
 12. Serialization: model_dump() output is JSON-serializable.
 13. Determinism: identical inputs produce identical outputs.
 14. No fabrication: assignments are never invented.
 15. PH5.5c and PH5.5d outputs are preserved alongside grounding.
"""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from knowledge.models import Evidence, GroundingStrength, KnowledgeMetadata
from knowledge.grounding import (
    _grounding_strength,
    _coverage_contribution,
    _STRONG_SCORE,
    _MODERATE_SCORE,
    _COMPLETE_THRESHOLD,
    ground_evidence,
)
from knowledge.retriever import (
    EvidenceRetriever,
    RetrievalResult,
    RetrievedEvidence,
    RETRIEVAL_MODE_LEXICAL,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SQ_RISK = "What are the key deployment risks for SMRs?"
_SQ_FUEL = "What are the fuel supply constraints for HALEU?"
_AREA_SAFETY = "nuclear safety"
_AREA_COST = "capital cost"


def _make_evidence(**kwargs) -> Evidence:
    defaults = dict(
        statement="SMR test claim about deployment.",
        supporting_source_ids=["src-001"],
        extraction_run_id="run-001",
        category="reactor design",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


def _make_metadata(evidence_id: str) -> KnowledgeMetadata:
    return KnowledgeMetadata(
        evidence_id=evidence_id,
        overall_score=3.5,
        retrieval_priority=3,
        strategic_value=0.7,
    )


def _make_candidate(rank: int = 1, score: float = 0.75) -> RetrievedEvidence:
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    return RetrievedEvidence(
        evidence=ev,
        metadata=meta,
        score=score,
        rank=rank,
        metadata_factor=EvidenceRetriever._metadata_factor(meta),
    )


class _MockRetriever:
    provider = None

    def __init__(self, items: list[RetrievedEvidence], matched: int | None = None) -> None:
        self._items = items
        self._matched = matched if matched is not None else len(items)

    def retrieve(self, query: str, *, mode=RETRIEVAL_MODE_LEXICAL, top_k=20, **_kw) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            items=list(self._items[:top_k]),
            domains_searched=["test"],
            total_candidates=self._matched,
            matched_candidates=self._matched,
            retrieval_method="lexical-v1",
            latency_ms=1.0,
            mode=mode,
        )


def _make_context(
    question: str = "Should we invest in SMRs?",
    subquestions: list[str] | None = None,
    investigation_areas: list[str] | None = None,
) -> types.SimpleNamespace:
    ctx = types.SimpleNamespace()
    ctx.plan = {
        "subquestions": subquestions or [],
        "investigation_areas": investigation_areas or [],
    }
    ctx.question = question
    ctx.trace = {}
    ctx.execution_profile = "test_profile"
    ctx.profiles = ["test_profile"]
    ctx.evidence_notes = []
    ctx.research_object = None
    ctx.domain_plans = []
    ctx.domain_evidence = []
    ctx.agent_history = []
    ctx.append_history = lambda entry: ctx.agent_history.append(entry)
    return ctx


def _noop_set_evidence_note(self: Any, context: Any, note: dict) -> None:
    context.evidence_notes = [note]


# ---------------------------------------------------------------------------
# 1. _grounding_strength() helper
# ---------------------------------------------------------------------------


def test_grounding_strength_above_strong_threshold():
    assert _grounding_strength(_STRONG_SCORE) == "STRONG"


def test_grounding_strength_above_strong_threshold_high():
    assert _grounding_strength(0.95) == "STRONG"


def test_grounding_strength_moderate_range():
    assert _grounding_strength(_MODERATE_SCORE) == "MODERATE"


def test_grounding_strength_moderate_below_strong():
    assert _grounding_strength(_STRONG_SCORE - 0.01) == "MODERATE"


def test_grounding_strength_below_moderate_threshold():
    assert _grounding_strength(0.0) == "WEAK"


def test_grounding_strength_just_below_moderate():
    assert _grounding_strength(_MODERATE_SCORE - 0.01) == "WEAK"


def test_grounding_strength_returns_valid_literal():
    for score in (0.0, 0.1, 0.3, 0.5, 0.6, 0.8, 1.0):
        result = _grounding_strength(score)
        assert result in ("STRONG", "MODERATE", "WEAK")


# ---------------------------------------------------------------------------
# 2. _coverage_contribution() helper
# ---------------------------------------------------------------------------


def test_coverage_contribution_sole_item_is_strong():
    assert _coverage_contribution(1) == "STRONG"


def test_coverage_contribution_zero_items_is_strong():
    # Guard edge case: subquestion with count=0 should not crash
    assert _coverage_contribution(0) == "STRONG"


def test_coverage_contribution_two_items_is_moderate():
    assert _coverage_contribution(2) == "MODERATE"


def test_coverage_contribution_three_items_is_moderate():
    assert _coverage_contribution(3) == "MODERATE"


def test_coverage_contribution_at_complete_threshold_is_weak():
    assert _coverage_contribution(_COMPLETE_THRESHOLD) == "WEAK"


def test_coverage_contribution_above_threshold_is_weak():
    assert _coverage_contribution(_COMPLETE_THRESHOLD + 2) == "WEAK"


def test_coverage_contribution_returns_valid_literal():
    for count in range(0, 8):
        result = _coverage_contribution(count)
        assert result in ("STRONG", "MODERATE", "WEAK")


# ---------------------------------------------------------------------------
# 3. ground_evidence() unit tests
# ---------------------------------------------------------------------------


def test_ground_evidence_creates_new_object():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.8,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={_SQ_RISK: 2},
    )
    assert grounded is not ev


def test_ground_evidence_does_not_mutate_original():
    ev = _make_evidence()
    original_sqs = list(ev.subquestion_assignments)
    original_strength = ev.grounding_strength
    ground_evidence(
        ev,
        hybrid_score=0.8,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={_SQ_RISK: 2},
    )
    assert ev.subquestion_assignments == original_sqs
    assert ev.grounding_strength == original_strength


def test_ground_evidence_subquestion_assignments_set():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.7,
        subquestion_assignments=[_SQ_RISK, _SQ_FUEL],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1, _SQ_FUEL: 2},
    )
    assert _SQ_RISK in grounded.subquestion_assignments
    assert _SQ_FUEL in grounded.subquestion_assignments


def test_ground_evidence_area_assignments_set():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.5,
        subquestion_assignments=[],
        area_assignments=[_AREA_SAFETY, _AREA_COST],
        evidence_counts={},
    )
    assert _AREA_SAFETY in grounded.investigation_area_assignments
    assert _AREA_COST in grounded.investigation_area_assignments


def test_ground_evidence_grounding_strength_from_score():
    ev = _make_evidence()
    grounded_strong = ground_evidence(
        ev,
        hybrid_score=0.9,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1},
    )
    grounded_weak = ground_evidence(
        ev,
        hybrid_score=0.1,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1},
    )
    assert grounded_strong.grounding_strength == "STRONG"
    assert grounded_weak.grounding_strength == "WEAK"


def test_ground_evidence_coverage_contribution_from_count():
    ev = _make_evidence()
    sole_item = ground_evidence(
        ev,
        hybrid_score=0.5,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1},
    )
    saturated = ground_evidence(
        ev,
        hybrid_score=0.5,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: _COMPLETE_THRESHOLD},
    )
    assert sole_item.coverage_contribution == "STRONG"
    assert saturated.coverage_contribution == "WEAK"


def test_ground_evidence_no_subquestion_gives_weak_contribution():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.8,
        subquestion_assignments=[],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={},
    )
    assert grounded.coverage_contribution == "WEAK"


def test_ground_evidence_uses_least_covered_subquestion():
    """coverage_contribution uses the subquestion with fewest items (most critical)."""
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.5,
        subquestion_assignments=[_SQ_RISK, _SQ_FUEL],
        area_assignments=[],
        evidence_counts={_SQ_RISK: _COMPLETE_THRESHOLD, _SQ_FUEL: 1},
    )
    # min(4, 1) = 1 → STRONG (sole item for HALEU subquestion)
    assert grounded.coverage_contribution == "STRONG"


def test_ground_evidence_evidence_id_preserved():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.7,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1},
    )
    assert grounded.evidence_id == ev.evidence_id


def test_ground_evidence_statement_preserved():
    ev = _make_evidence(statement="Specific deployment claim about SMRs.")
    grounded = ground_evidence(
        ev,
        hybrid_score=0.7,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],
        evidence_counts={_SQ_RISK: 1},
    )
    assert grounded.statement == ev.statement


# ---------------------------------------------------------------------------
# 4. _execute_kb() integration
# ---------------------------------------------------------------------------


def test_execute_kb_stores_grounded_evidence(monkeypatch):
    """_execute_kb() writes _grounded_evidence to context.trace."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    assert "_grounded_evidence" in ctx.trace


def test_execute_kb_grounded_evidence_is_list(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    assert isinstance(ctx.trace["_grounded_evidence"], list)


def test_execute_kb_grounded_evidence_count_matches_candidates(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    n = 3
    items = [_make_candidate(rank=i, score=0.9 - 0.1 * i) for i in range(1, n + 1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items, matched=n))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    assert len(ctx.trace["_grounded_evidence"]) == n


def test_execute_kb_grounded_evidence_has_grounding_fields(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    ge = ctx.trace["_grounded_evidence"][0]
    assert "subquestion_assignments" in ge
    assert "investigation_area_assignments" in ge
    assert "grounding_strength" in ge
    assert "coverage_contribution" in ge


def test_execute_kb_grounding_strength_is_valid_literal(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    for ge in ctx.trace["_grounded_evidence"]:
        assert ge["grounding_strength"] in ("STRONG", "MODERATE", "WEAK", None)


def test_execute_kb_coverage_contribution_is_valid_literal(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.5)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    for ge in ctx.trace["_grounded_evidence"]:
        assert ge["coverage_contribution"] in ("STRONG", "MODERATE", "WEAK", None)


def test_execute_kb_evidence_id_round_trips(monkeypatch):
    """Grounded evidence IDs match the original candidate IDs."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    original_id = items[0].evidence.evidence_id
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    grounded_ids = [ge["evidence_id"] for ge in ctx.trace["_grounded_evidence"]]
    assert original_id in grounded_ids


def test_execute_kb_prior_ph5_outputs_preserved(monkeypatch):
    """PH5.5c provenance and PH5.5d completeness are not disturbed by grounding."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    assert "_retrieval_provenance" in ctx.trace
    assert "_assembly_completeness" in ctx.trace
    assert "_grounded_evidence" in ctx.trace


def test_execute_kb_no_subquestions_still_grounds(monkeypatch):
    """Grounding still runs when no subquestions are defined."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.7)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[])
    agent._execute_kb(ctx)

    # Should produce grounded evidence even with no subquestions (assignments empty)
    ge_list = ctx.trace["_grounded_evidence"]
    assert isinstance(ge_list, list)
    assert len(ge_list) >= 1
    for ge in ge_list:
        assert ge["subquestion_assignments"] == []


# ---------------------------------------------------------------------------
# 5. Serialization
# ---------------------------------------------------------------------------


def test_ground_evidence_model_dump_includes_grounding_fields():
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.5,  # between _MODERATE_SCORE and _STRONG_SCORE → MODERATE
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={_SQ_RISK: 2},
    )
    d = grounded.model_dump()
    assert d["subquestion_assignments"] == [_SQ_RISK]
    assert d["investigation_area_assignments"] == [_AREA_SAFETY]
    assert d["grounding_strength"] == "MODERATE"
    assert "coverage_contribution" in d


def test_grounded_evidence_json_serializable(monkeypatch):
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    try:
        json.dumps(ctx.trace["_grounded_evidence"])
    except TypeError as exc:
        pytest.fail(f"_grounded_evidence is not JSON-serializable: {exc}")


def test_ground_evidence_roundtrip():
    """model_dump → model_validate produces identical grounded Evidence."""
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.65,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={_SQ_RISK: 1},
    )
    d = grounded.model_dump()
    ev2 = Evidence.model_validate(d)
    assert ev2.grounding_strength == grounded.grounding_strength
    assert ev2.coverage_contribution == grounded.coverage_contribution
    assert ev2.subquestion_assignments == grounded.subquestion_assignments
    assert ev2.investigation_area_assignments == grounded.investigation_area_assignments


# ---------------------------------------------------------------------------
# 6. Determinism
# ---------------------------------------------------------------------------


def test_grounding_strength_is_deterministic():
    for score in (0.0, 0.29, 0.30, 0.59, 0.60, 0.80, 1.0):
        first = _grounding_strength(score)
        second = _grounding_strength(score)
        assert first == second


def test_coverage_contribution_is_deterministic():
    for count in range(0, _COMPLETE_THRESHOLD + 3):
        first = _coverage_contribution(count)
        second = _coverage_contribution(count)
        assert first == second


def test_ground_evidence_is_deterministic():
    ev = _make_evidence()
    kwargs = dict(
        hybrid_score=0.7,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[_AREA_SAFETY],
        evidence_counts={_SQ_RISK: 2},
    )
    g1 = ground_evidence(ev, **kwargs)
    g2 = ground_evidence(ev, **kwargs)
    assert g1.grounding_strength == g2.grounding_strength
    assert g1.coverage_contribution == g2.coverage_contribution
    assert g1.subquestion_assignments == g2.subquestion_assignments


def test_execute_kb_grounding_is_deterministic(monkeypatch):
    """Two identical _execute_kb() runs produce identical _grounded_evidence."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx1 = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx1)

    # Re-create the agent (same retriever data) to avoid any caching effects
    agent2 = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)
    ctx2 = _make_context(subquestions=[_SQ_RISK])
    agent2._execute_kb(ctx2)

    ge1 = ctx1.trace["_grounded_evidence"]
    ge2 = ctx2.trace["_grounded_evidence"]
    assert len(ge1) == len(ge2)
    for d1, d2 in zip(ge1, ge2):
        assert d1["grounding_strength"] == d2["grounding_strength"]
        assert d1["coverage_contribution"] == d2["coverage_contribution"]


# ---------------------------------------------------------------------------
# 7. No-fabrication guard
# ---------------------------------------------------------------------------


def test_no_assignment_when_not_in_subquestion_map():
    """Evidence not in evidence_by_subquestion gets no subquestion assignment."""
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.9,
        subquestion_assignments=[],  # explicitly empty — not in any mapping
        area_assignments=[],
        evidence_counts={_SQ_RISK: 5},
    )
    assert grounded.subquestion_assignments == []


def test_no_area_assignment_when_not_in_area_map():
    """Evidence not in evidence_by_area gets no investigation area assignment."""
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.8,
        subquestion_assignments=[_SQ_RISK],
        area_assignments=[],  # explicitly empty
        evidence_counts={_SQ_RISK: 1},
    )
    assert grounded.investigation_area_assignments == []


def test_unmapped_evidence_gets_weak_contribution(monkeypatch):
    """Evidence that maps to no subquestion gets WEAK coverage_contribution."""
    ev = _make_evidence()
    grounded = ground_evidence(
        ev,
        hybrid_score=0.95,  # high score — but not assigned to any subquestion
        subquestion_assignments=[],
        area_assignments=[],
        evidence_counts={},
    )
    assert grounded.coverage_contribution == "WEAK"


def test_grounding_does_not_alter_retrieval_provenance(monkeypatch):
    """_grounded_evidence in trace is independent of _retrieval_provenance."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_candidate(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    prov = ctx.trace["_retrieval_provenance"]
    grounded = ctx.trace["_grounded_evidence"]

    # Provenance records have retrieval fields; grounded records have grounding fields
    assert all("retrieval_rank" in p for p in prov)
    assert all("grounding_strength" in g for g in grounded)

    # Modifying the grounded list does not touch provenance
    grounded.clear()
    assert len(ctx.trace["_retrieval_provenance"]) > 0
