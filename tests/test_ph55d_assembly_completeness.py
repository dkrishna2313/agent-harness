"""PH5.5d — Evidence Assembly Completeness tests.

Verifies:
  1. SubquestionCompleteness constructs and serializes correctly.
  2. assess_subquestion() produces correct counts and status.
  3. assess_assembly_completeness() aggregates per-subquestion assessments.
  4. CompletenessStatus thresholds (COMPLETE / PARTIAL / INCOMPLETE / UNKNOWN).
  5. Gap notes are generated correctly and deterministically.
  6. Coverage fraction and completeness score calculations.
  7. No-subquestions edge case → UNKNOWN status.
  8. Contradiction handling.
  9. Missing investigation area detection.
 10. EvidenceAgent._execute_kb() stores _assembly_completeness in trace.
 11. Serialization: model_dump() / model_validate() round-trip.
 12. Regression: no existing PH5.5a–PH5.5c behaviors affected.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from knowledge.models import (
    AssemblyCompleteness,
    CompletenessStatus,
    Evidence,
    KnowledgeMetadata,
    SubquestionCompleteness,
)
from knowledge.assembly import (
    assess_subquestion,
    assess_assembly_completeness,
    _coverage_fraction,
    _score,
    _status,
    _gap_notes,
    _overall_status,
    _COMPLETE_THRESHOLD,
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
_SQ_COST = "What is the cost trajectory for BWRX-300 units?"


def _ev_ids(n: int, prefix: str = "ev") -> list[str]:
    return [f"{prefix}-{i:03d}" for i in range(1, n + 1)]


def _make_evidence(**kwargs) -> Evidence:
    defaults = dict(
        statement="Test claim about SMR deployment.",
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


def _make_retrieved_evidence(rank: int = 1, score: float = 0.8) -> RetrievedEvidence:
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
    question: str = "What are the SMR deployment risks?",
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
# 1. SubquestionCompleteness model
# ---------------------------------------------------------------------------


def test_subquestion_completeness_constructs():
    sc = SubquestionCompleteness(subquestion_text=_SQ_RISK)
    assert sc.subquestion_text == _SQ_RISK


def test_subquestion_completeness_defaults():
    sc = SubquestionCompleteness(subquestion_text=_SQ_RISK)
    assert sc.evidence_count == 0
    assert sc.supporting_evidence_count == 0
    assert sc.contradicting_evidence_count == 0
    assert sc.missing_area_count == 0
    assert sc.coverage_fraction == 0.0
    assert sc.completeness_score == 0.0
    assert sc.completeness_status == "UNKNOWN"
    assert sc.gap_notes == []
    assert sc.evidence_ids == []
    assert sc.research_question_id is None
    assert sc.subquestion_id is None


def test_subquestion_completeness_roundtrip():
    sc = SubquestionCompleteness(
        subquestion_text=_SQ_RISK,
        evidence_count=3,
        supporting_evidence_count=3,
        contradicting_evidence_count=0,
        coverage_fraction=0.75,
        completeness_score=0.75,
        completeness_status="PARTIAL",
        gap_notes=["Partial coverage (3 items); 1 more would reach COMPLETE."],
        evidence_ids=["ev-001", "ev-002", "ev-003"],
    )
    d = sc.model_dump()
    sc2 = SubquestionCompleteness.model_validate(d)
    assert sc2.evidence_count == 3
    assert sc2.completeness_status == "PARTIAL"
    assert sc2.coverage_fraction == 0.75
    assert sc2.evidence_ids == ["ev-001", "ev-002", "ev-003"]


# ---------------------------------------------------------------------------
# 2. AssemblyCompleteness model
# ---------------------------------------------------------------------------


def test_assembly_completeness_constructs():
    ac = AssemblyCompleteness(question=_SQ_RISK)
    assert ac.question == _SQ_RISK
    assert ac.total_subquestions == 0
    assert ac.overall_completeness_status == "UNKNOWN"


def test_assembly_completeness_roundtrip():
    ac = AssemblyCompleteness(
        question="What is the risk?",
        total_subquestions=3,
        covered_subquestions=2,
        total_evidence_count=15,
        overall_completeness_score=0.6,
        overall_completeness_status="PARTIAL",
        gap_summary=["1 subquestion(s) have no evidence coverage."],
    )
    d = ac.model_dump()
    ac2 = AssemblyCompleteness.model_validate(d)
    assert ac2.total_subquestions == 3
    assert ac2.covered_subquestions == 2
    assert ac2.overall_completeness_status == "PARTIAL"


# ---------------------------------------------------------------------------
# 3. Internal helper functions
# ---------------------------------------------------------------------------


def test_status_complete():
    assert _status(_COMPLETE_THRESHOLD) == "COMPLETE"
    assert _status(_COMPLETE_THRESHOLD + 5) == "COMPLETE"


def test_status_partial_single():
    assert _status(1) == "PARTIAL"


def test_status_partial_moderate():
    assert _status(_COMPLETE_THRESHOLD - 1) == "PARTIAL"


def test_status_incomplete():
    assert _status(0) == "INCOMPLETE"


def test_coverage_fraction_zero():
    assert _coverage_fraction(0) == 0.0


def test_coverage_fraction_partial():
    assert _coverage_fraction(2) == pytest.approx(2 / _COMPLETE_THRESHOLD)


def test_coverage_fraction_saturates():
    assert _coverage_fraction(_COMPLETE_THRESHOLD) == 1.0
    assert _coverage_fraction(_COMPLETE_THRESHOLD * 3) == 1.0


def test_score_no_contradictions():
    s = _score(4, 0)
    assert s == pytest.approx(1.0)


def test_score_with_contradictions():
    # 4 items, 2 contradicting → penalty = 0.5 → score = 0.5 * (1 - 0.5) = 0.5
    s = _score(4, 2)
    assert s == pytest.approx(0.5)


def test_score_zero_evidence():
    assert _score(0, 0) == 0.0


def test_score_contradiction_saturates():
    # More contradictions than evidence items: penalty capped at 1.0 → score = 0.0
    s = _score(2, 10)
    assert s == 0.0


def test_gap_notes_no_evidence():
    notes = _gap_notes(0, 0)
    assert len(notes) == 1
    assert "No evidence" in notes[0]


def test_gap_notes_single_item():
    notes = _gap_notes(1, 0)
    assert len(notes) == 1
    assert "Single evidence" in notes[0] or "corroboration" in notes[0]


def test_gap_notes_partial_coverage():
    notes = _gap_notes(2, 0)
    assert len(notes) == 1
    assert "Partial" in notes[0] or "partial" in notes[0]


def test_gap_notes_complete():
    notes = _gap_notes(_COMPLETE_THRESHOLD, 0)
    assert notes == []


def test_gap_notes_with_contradiction():
    notes = _gap_notes(4, 1)
    assert any("contradicting" in n or "contradiction" in n for n in notes)


def test_overall_status_all_complete():
    assert _overall_status(["COMPLETE", "COMPLETE", "COMPLETE"]) == "COMPLETE"


def test_overall_status_any_partial():
    assert _overall_status(["COMPLETE", "PARTIAL", "INCOMPLETE"]) == "PARTIAL"


def test_overall_status_all_incomplete():
    assert _overall_status(["INCOMPLETE", "INCOMPLETE"]) == "INCOMPLETE"


def test_overall_status_empty():
    assert _overall_status([]) == "UNKNOWN"


def test_overall_status_mixed_partial():
    assert _overall_status(["PARTIAL", "PARTIAL"]) == "PARTIAL"


# ---------------------------------------------------------------------------
# 4. assess_subquestion() factory
# ---------------------------------------------------------------------------


def test_assess_subquestion_no_evidence():
    sc = assess_subquestion(_SQ_RISK, [])
    assert sc.evidence_count == 0
    assert sc.completeness_status == "INCOMPLETE"
    assert sc.coverage_fraction == 0.0
    assert sc.completeness_score == 0.0
    assert sc.evidence_ids == []
    assert len(sc.gap_notes) == 1


def test_assess_subquestion_single_item():
    sc = assess_subquestion(_SQ_RISK, ["ev-001"])
    assert sc.evidence_count == 1
    assert sc.completeness_status == "PARTIAL"
    assert sc.coverage_fraction == pytest.approx(1 / _COMPLETE_THRESHOLD)
    assert sc.evidence_ids == ["ev-001"]


def test_assess_subquestion_complete():
    ids = _ev_ids(_COMPLETE_THRESHOLD)
    sc = assess_subquestion(_SQ_RISK, ids)
    assert sc.evidence_count == _COMPLETE_THRESHOLD
    assert sc.completeness_status == "COMPLETE"
    assert sc.coverage_fraction == pytest.approx(1.0)
    assert sc.completeness_score == pytest.approx(1.0)
    assert sc.gap_notes == []


def test_assess_subquestion_supporting_equals_total_without_contradictions():
    sc = assess_subquestion(_SQ_RISK, _ev_ids(3))
    assert sc.supporting_evidence_count == 3
    assert sc.contradicting_evidence_count == 0


def test_assess_subquestion_with_contradictions():
    sc = assess_subquestion(_SQ_RISK, _ev_ids(4), contradicting_ids=["ev-001"])
    assert sc.contradicting_evidence_count == 1
    assert sc.completeness_score < 1.0


def test_assess_subquestion_stores_evidence_ids():
    ids = ["ev-x", "ev-y", "ev-z"]
    sc = assess_subquestion(_SQ_RISK, ids)
    assert set(sc.evidence_ids) == set(ids)


def test_assess_subquestion_research_question_id_propagated():
    sc = assess_subquestion(_SQ_RISK, [], research_question_id="rq-001")
    assert sc.research_question_id == "rq-001"


def test_assess_subquestion_missing_area_count_zero():
    sc = assess_subquestion(_SQ_RISK, _ev_ids(3))
    assert sc.missing_area_count == 0


# ---------------------------------------------------------------------------
# 5. assess_assembly_completeness() — basic flow
# ---------------------------------------------------------------------------


def test_no_subquestions_returns_unknown():
    ac = assess_assembly_completeness(
        question="General risk question?",
        subquestions=[],
        evidence_by_subquestion={},
    )
    assert ac.overall_completeness_status == "UNKNOWN"
    assert ac.total_subquestions == 0
    assert "No subquestions" in ac.gap_summary[0]


def test_all_subquestions_with_complete_evidence():
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {
        _SQ_RISK: _ev_ids(_COMPLETE_THRESHOLD, "risk"),
        _SQ_FUEL: _ev_ids(_COMPLETE_THRESHOLD, "fuel"),
    }
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac.overall_completeness_status == "COMPLETE"
    assert ac.covered_subquestions == 2
    assert ac.total_subquestions == 2
    assert "adequate" in ac.gap_summary[0].lower()


def test_all_subquestions_empty_returns_incomplete():
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {_SQ_RISK: [], _SQ_FUEL: []}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac.overall_completeness_status == "INCOMPLETE"
    assert ac.covered_subquestions == 0


def test_mixed_coverage_returns_partial():
    sqs = [_SQ_RISK, _SQ_FUEL, _SQ_COST]
    ebsq = {
        _SQ_RISK: _ev_ids(_COMPLETE_THRESHOLD, "risk"),
        _SQ_FUEL: _ev_ids(1, "fuel"),
        _SQ_COST: [],
    }
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac.overall_completeness_status == "PARTIAL"
    assert ac.covered_subquestions == 2  # RISK (COMPLETE) + FUEL (PARTIAL)


def test_assessment_count_matches_subquestions():
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {_SQ_RISK: _ev_ids(2), _SQ_FUEL: _ev_ids(1)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert len(ac.subquestion_assessments) == 2


def test_subquestion_order_preserved():
    sqs = [_SQ_RISK, _SQ_FUEL, _SQ_COST]
    ebsq = {sq: [] for sq in sqs}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    texts = [a.subquestion_text for a in ac.subquestion_assessments]
    assert texts == sqs


def test_total_evidence_count_from_total_retrieved():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(2)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq, total_retrieved=50)
    assert ac.total_evidence_count == 50


def test_total_evidence_count_fallback():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: ["ev-001", "ev-002"]}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac.total_evidence_count == 2


def test_overall_score_is_average_of_subquestion_scores():
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {
        _SQ_RISK: _ev_ids(_COMPLETE_THRESHOLD),  # score = 1.0
        _SQ_FUEL: _ev_ids(0),                    # score = 0.0
    }
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac.overall_completeness_score == pytest.approx(0.5, abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Missing area detection
# ---------------------------------------------------------------------------


def test_missing_areas_counted():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(4)}
    areas = ["SMR Technology", "Licensing", "Grid Integration"]
    ebarea = {"SMR Technology": ["ev-001"], "Licensing": [], "Grid Integration": []}
    ac = assess_assembly_completeness("Q?", sqs, ebsq, investigation_areas=areas, evidence_by_area=ebarea)
    assert ac.missing_area_count == 2


def test_no_missing_areas():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(4)}
    areas = ["SMR Technology"]
    ebarea = {"SMR Technology": ["ev-001"]}
    ac = assess_assembly_completeness("Q?", sqs, ebsq, investigation_areas=areas, evidence_by_area=ebarea)
    assert ac.missing_area_count == 0


def test_gap_summary_includes_missing_areas():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(4)}
    areas = ["Licensing", "Grid Integration"]
    ebarea = {"Licensing": [], "Grid Integration": []}
    ac = assess_assembly_completeness("Q?", sqs, ebsq, investigation_areas=areas, evidence_by_area=ebarea)
    assert any("uncovered" in s or "investigation area" in s.lower() for s in ac.gap_summary)


# ---------------------------------------------------------------------------
# 7. Contradiction handling
# ---------------------------------------------------------------------------


def test_contradictions_flow_into_subquestion():
    sqs = [_SQ_RISK]
    ev_ids = ["ev-a", "ev-b", "ev-c", "ev-d"]
    ebsq = {_SQ_RISK: ev_ids}
    contradictions = [
        {"evidence_id_a": "ev-a", "evidence_id_b": "ev-z"},
    ]
    ac = assess_assembly_completeness("Q?", sqs, ebsq, validated_contradictions=contradictions)
    sq_a = ac.subquestion_assessments[0]
    assert sq_a.contradicting_evidence_count == 1
    assert sq_a.completeness_score < 1.0


def test_empty_contradictions_no_effect():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(4)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq, validated_contradictions=[])
    assert ac.subquestion_assessments[0].contradicting_evidence_count == 0
    assert ac.subquestion_assessments[0].completeness_score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. Gap summary generation
# ---------------------------------------------------------------------------


def test_gap_summary_incomplete_subquestions():
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {_SQ_RISK: [], _SQ_FUEL: []}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert any("no evidence" in s.lower() for s in ac.gap_summary)


def test_gap_summary_partial_subquestions():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(2)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert any("partial" in s.lower() for s in ac.gap_summary)


def test_gap_summary_all_complete_positive():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(_COMPLETE_THRESHOLD)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    assert any("adequate" in s.lower() for s in ac.gap_summary)


# ---------------------------------------------------------------------------
# 9. Serialization
# ---------------------------------------------------------------------------


def test_assembly_completeness_model_dump_includes_all_fields():
    sqs = [_SQ_RISK]
    ebsq = {_SQ_RISK: _ev_ids(3)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    d = ac.model_dump()
    for key in (
        "question", "total_subquestions", "covered_subquestions",
        "total_evidence_count", "missing_area_count",
        "overall_completeness_score", "overall_completeness_status",
        "subquestion_assessments", "gap_summary",
    ):
        assert key in d, f"model_dump missing key: {key}"


def test_subquestion_completeness_model_dump_includes_all_fields():
    sc = assess_subquestion(_SQ_RISK, _ev_ids(2))
    d = sc.model_dump()
    for key in (
        "subquestion_text", "evidence_count", "supporting_evidence_count",
        "contradicting_evidence_count", "missing_area_count",
        "coverage_fraction", "completeness_score", "completeness_status",
        "gap_notes", "evidence_ids",
    ):
        assert key in d, f"model_dump missing key: {key}"


def test_json_roundtrip():
    import json
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {_SQ_RISK: _ev_ids(4), _SQ_FUEL: _ev_ids(1)}
    ac = assess_assembly_completeness("Q?", sqs, ebsq)
    j = ac.model_dump_json()
    ac2 = AssemblyCompleteness.model_validate_json(j)
    assert ac2.overall_completeness_status == ac.overall_completeness_status
    assert len(ac2.subquestion_assessments) == len(ac.subquestion_assessments)


# ---------------------------------------------------------------------------
# 10. EvidenceAgent._execute_kb() integration
# ---------------------------------------------------------------------------


def test_execute_kb_stores_assembly_completeness(monkeypatch):
    """_execute_kb() writes _assembly_completeness to context.trace."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    assert "_assembly_completeness" in ctx.trace


def test_execute_kb_completeness_is_dict(monkeypatch):
    """_assembly_completeness is serialized as a dict in the trace."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    ac_dict = ctx.trace["_assembly_completeness"]
    assert isinstance(ac_dict, dict)
    assert "overall_completeness_status" in ac_dict


def test_execute_kb_completeness_status_is_valid_literal(monkeypatch):
    """overall_completeness_status is always a valid CompletenessStatus literal."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    status = ctx.trace["_assembly_completeness"]["overall_completeness_status"]
    assert status in ("COMPLETE", "PARTIAL", "INCOMPLETE", "UNKNOWN")


def test_execute_kb_no_subquestions_gives_unknown(monkeypatch):
    """When plan has no subquestions, completeness status is UNKNOWN."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[])
    agent._execute_kb(ctx)

    status = ctx.trace["_assembly_completeness"]["overall_completeness_status"]
    assert status == "UNKNOWN"


def test_execute_kb_total_evidence_count_reflects_candidates(monkeypatch):
    """total_evidence_count matches the number of candidates passed to assessment."""
    from functional_agents.evidence_agent import EvidenceAgent

    n_items = 3
    items = [_make_retrieved_evidence(rank=i, score=0.8 - 0.1 * i) for i in range(1, n_items + 1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    ac = ctx.trace["_assembly_completeness"]
    assert ac["total_evidence_count"] == n_items


def test_execute_kb_gap_summary_present(monkeypatch):
    """gap_summary is always a non-empty list."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    gap_summary = ctx.trace["_assembly_completeness"]["gap_summary"]
    assert isinstance(gap_summary, list)
    assert len(gap_summary) >= 1


def test_execute_kb_completeness_json_serializable(monkeypatch):
    """_assembly_completeness can be JSON-serialized without error."""
    import json
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(subquestions=[_SQ_RISK])
    agent._execute_kb(ctx)

    serialized = json.dumps(ctx.trace["_assembly_completeness"])
    parsed = json.loads(serialized)
    assert "overall_completeness_status" in parsed


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_produce_identical_output():
    """assess_assembly_completeness is deterministic for identical inputs."""
    sqs = [_SQ_RISK, _SQ_FUEL]
    ebsq = {_SQ_RISK: _ev_ids(3), _SQ_FUEL: _ev_ids(1)}
    ac1 = assess_assembly_completeness("Q?", sqs, ebsq)
    ac2 = assess_assembly_completeness("Q?", sqs, ebsq)
    assert ac1.model_dump() == ac2.model_dump()


def test_score_is_deterministic():
    for _ in range(5):
        assert _score(3, 0) == _coverage_fraction(3)


# ---------------------------------------------------------------------------
# 12. No fabrication guard
# ---------------------------------------------------------------------------


def test_no_evidence_gives_no_fake_coverage():
    sc = assess_subquestion(_SQ_RISK, [])
    assert sc.coverage_fraction == 0.0
    assert sc.completeness_score == 0.0
    assert sc.supporting_evidence_count == 0


def test_unknown_when_no_subquestions():
    ac = assess_assembly_completeness("Q?", [], {})
    assert ac.overall_completeness_status == "UNKNOWN"
    assert ac.overall_completeness_score == 0.0
