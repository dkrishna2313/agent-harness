"""PH5.5c — Retrieval Provenance tests.

Verifies:
  1. RetrievedEvidence.metadata_factor field exists and is populated.
  2. build_retrieval_provenance() factory builds a correct RetrievalProvenance.
  3. RetrievalProvenance new PH5.5c fields (retrieval_timestamp,
     retrieved_candidate_count, reranked, reranker_model).
  4. EvidenceAgent._execute_kb() stores provenance in context.trace.
  5. Query attribution: primary query for direct hits, subquestion for expansion.
  6. Passthrough reranker path: reranked=False, reranker_model=None.
  7. LLM reranker model string parsed from RerankResult.reranker.
  8. Provenance is fully serializable via model_dump().
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field as dc_field
from typing import Any
from uuid import uuid4

import pytest

from knowledge.models import (
    Evidence,
    KnowledgeMetadata,
    RetrievalProvenance,
)
from knowledge.retriever import (
    RetrievalResult,
    RetrievedEvidence,
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    EvidenceRetriever,
    build_retrieval_provenance,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_evidence(**kwargs) -> Evidence:
    defaults = dict(
        statement="The BWRX-300 has a thermal output of 300 MWt.",
        supporting_source_ids=["src-abc"],
        extraction_run_id="run-001",
        category="reactor design",
    )
    defaults.update(kwargs)
    return Evidence(**defaults)


def _make_metadata(evidence_id: str, **kwargs) -> KnowledgeMetadata:
    defaults = dict(
        evidence_id=evidence_id,
        overall_score=3.5,
        retrieval_priority=3,
        strategic_value=0.7,
    )
    defaults.update(kwargs)
    return KnowledgeMetadata(**defaults)


def _make_retrieved_evidence(rank: int = 1, score: float = 0.8, **kwargs) -> RetrievedEvidence:
    ev = _make_evidence(**{k: v for k, v in kwargs.items() if k in ("statement", "evidence_id")})
    meta = _make_metadata(ev.evidence_id)
    mf = EvidenceRetriever._metadata_factor(meta)
    return RetrievedEvidence(
        evidence=ev,
        metadata=meta,
        score=score,
        rank=rank,
        lexical_score=0.6,
        semantic_score=0.9,
        source_domain="smr",
        metadata_factor=mf,
    )


def _make_result(query: str, items: list[RetrievedEvidence], **kwargs) -> RetrievalResult:
    defaults: dict[str, Any] = dict(
        domains_searched=["smr"],
        total_candidates=100,
        matched_candidates=len(items),
        retrieval_method="lexical-v1",
        latency_ms=5.0,
        mode=RETRIEVAL_MODE_LEXICAL,
    )
    defaults.update(kwargs)  # callers can override any default, including matched_candidates
    return RetrievalResult(query=query, items=items, **defaults)


def _make_provenance(**kwargs) -> RetrievalProvenance:
    defaults = dict(
        evidence_id="ev-001",
        retrieval_query="What is the thermal output?",
        retrieval_mode="hybrid",
        retrieval_rank=1,
        hybrid_score=0.842,
    )
    defaults.update(kwargs)
    return RetrievalProvenance(**defaults)


# ---------------------------------------------------------------------------
# Minimal mock retriever for integration tests
# ---------------------------------------------------------------------------

class _MockRetriever:
    """Returns a fixed set of RetrievedEvidence for any query."""

    provider = None  # forces RETRIEVAL_MODE_LEXICAL in _execute_kb

    def __init__(self, items: list[RetrievedEvidence], matched: int | None = None) -> None:
        self._items = items
        self._matched = matched if matched is not None else len(items)

    def retrieve(
        self,
        query: str,
        *,
        mode: str = RETRIEVAL_MODE_LEXICAL,
        top_k: int = 20,
        **_kw: Any,
    ) -> RetrievalResult:
        chosen = self._items[:top_k]
        return RetrievalResult(
            query=query,
            items=list(chosen),
            domains_searched=["test"],
            total_candidates=self._matched,
            matched_candidates=self._matched,
            retrieval_method="lexical-v1",
            latency_ms=1.0,
            mode=mode,
        )


def _make_context(
    question: str = "What is the risk?",
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
    # AgentContext.append_history() method used by _record()
    ctx.append_history = lambda entry: ctx.agent_history.append(entry)
    return ctx


def _noop_set_evidence_note(self: Any, context: Any, note: dict) -> None:
    """Bypass the LLM evidence boundary; store note directly."""
    context.evidence_notes = [note]


# ---------------------------------------------------------------------------
# 1. RetrievedEvidence.metadata_factor field
# ---------------------------------------------------------------------------


def test_retrieved_evidence_has_metadata_factor_field():
    """metadata_factor field exists on RetrievedEvidence."""
    ev = _make_retrieved_evidence()
    assert hasattr(ev, "metadata_factor")


def test_retrieved_evidence_metadata_factor_defaults_none():
    """metadata_factor defaults to None when not supplied."""
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    item = RetrievedEvidence(evidence=ev, metadata=meta, score=0.5, rank=1)
    assert item.metadata_factor is None


def test_retrieved_evidence_metadata_factor_set_explicitly():
    """metadata_factor is stored when set at construction time."""
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    item = RetrievedEvidence(evidence=ev, metadata=meta, score=0.5, rank=1, metadata_factor=1.23)
    assert item.metadata_factor == pytest.approx(1.23)


def test_retrieved_evidence_metadata_factor_matches_static_method():
    """metadata_factor matches what EvidenceRetriever._metadata_factor() computes."""
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id, overall_score=4.0, retrieval_priority=4, strategic_value=0.8)
    expected = EvidenceRetriever._metadata_factor(meta)
    item = RetrievedEvidence(evidence=ev, metadata=meta, score=0.5, rank=1, metadata_factor=expected)
    assert item.metadata_factor == pytest.approx(expected)


def test_retrieved_evidence_metadata_factor_is_float():
    """metadata_factor is a float value when populated."""
    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    mf = EvidenceRetriever._metadata_factor(meta)
    assert isinstance(mf, float)
    # Range check: quality [0.8,1.2] × priority [0.9,1.1] × strategic [0.9,1.1]
    assert 0.64 <= mf <= 1.46


# ---------------------------------------------------------------------------
# 2. build_retrieval_provenance() factory unit tests
# ---------------------------------------------------------------------------


def test_factory_basic_fields():
    """Factory populates core retrieval fields from candidate and result."""
    item = _make_retrieved_evidence(rank=3, score=0.75)
    result = _make_result("What is risk?", [item], matched_candidates=50)
    rp = build_retrieval_provenance(item, result)

    assert rp.evidence_id == item.evidence.evidence_id
    assert rp.retrieval_query == "What is risk?"
    assert rp.retrieval_mode == RETRIEVAL_MODE_LEXICAL
    assert rp.retrieval_rank == item.rank
    assert rp.hybrid_score == item.score
    assert rp.lexical_score == item.lexical_score
    assert rp.semantic_score == item.semantic_score
    assert rp.metadata_factor == item.metadata_factor


def test_factory_retrieval_query_override():
    """retrieval_query override is used instead of result.query."""
    item = _make_retrieved_evidence(rank=1)
    result = _make_result("primary query", [item])
    rp = build_retrieval_provenance(item, result, retrieval_query="subquestion query")
    assert rp.retrieval_query == "subquestion query"


def test_factory_retrieval_query_falls_back_to_result():
    """retrieval_query falls back to result.query when not overridden."""
    item = _make_retrieved_evidence(rank=1)
    result = _make_result("primary query", [item])
    rp = build_retrieval_provenance(item, result)
    assert rp.retrieval_query == "primary query"


def test_factory_passthrough_reranker_defaults():
    """Default path: passthrough, no reranking, model=None."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])
    rp = build_retrieval_provenance(item, result)

    assert rp.reranker == "passthrough"
    assert rp.reranked is False
    assert rp.reranker_model is None
    assert rp.rerank_score is None
    assert rp.rerank_rationale is None


def test_factory_llm_reranker_path():
    """LLM reranker path sets reranked=True, type, model, score, rationale."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])
    rp = build_retrieval_provenance(
        item, result,
        reranked=True,
        reranker_type="llm",
        reranker_model="claude-haiku-4-5-20251001",
        rerank_score=0.91,
        rerank_rationale="High semantic match for risk query.",
    )

    assert rp.reranked is True
    assert rp.reranker == "llm"
    assert rp.reranker_model == "claude-haiku-4-5-20251001"
    assert rp.rerank_score == pytest.approx(0.91)
    assert rp.rerank_rationale == "High semantic match for risk query."


def test_factory_retrieved_candidate_count_from_result():
    """retrieved_candidate_count defaults to result.matched_candidates."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item], matched_candidates=73)
    rp = build_retrieval_provenance(item, result)
    assert rp.retrieved_candidate_count == 73


def test_factory_retrieved_candidate_count_override():
    """retrieved_candidate_count override takes precedence over result."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item], matched_candidates=10)
    rp = build_retrieval_provenance(item, result, retrieved_candidate_count=99)
    assert rp.retrieved_candidate_count == 99


def test_factory_retrieval_timestamp_stored():
    """retrieval_timestamp is stored as supplied."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])
    rp = build_retrieval_provenance(item, result, retrieval_timestamp="2026-07-16T12:34:56Z")
    assert rp.retrieval_timestamp == "2026-07-16T12:34:56Z"


def test_factory_retrieval_timestamp_none_by_default():
    """retrieval_timestamp is None when not supplied."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])
    rp = build_retrieval_provenance(item, result)
    assert rp.retrieval_timestamp is None


def test_factory_semantic_model_from_result():
    """retrieval_model_version comes from result.semantic_model."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item], semantic_model="all-MiniLM-L6-v2")
    rp = build_retrieval_provenance(item, result)
    assert rp.retrieval_model_version == "all-MiniLM-L6-v2"


def test_factory_result_mode_propagated():
    """retrieval_mode mirrors result.mode."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item], mode=RETRIEVAL_MODE_HYBRID)
    rp = build_retrieval_provenance(item, result)
    assert rp.retrieval_mode == "hybrid"


def test_factory_returns_retrieval_provenance_instance():
    """build_retrieval_provenance() returns a RetrievalProvenance object."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])
    rp = build_retrieval_provenance(item, result)
    assert isinstance(rp, RetrievalProvenance)


# ---------------------------------------------------------------------------
# 3. RetrievalProvenance new PH5.5c fields
# ---------------------------------------------------------------------------


def test_new_fields_default_safely():
    """All four PH5.5c fields default to safe values."""
    rp = _make_provenance()
    assert rp.retrieval_timestamp is None
    assert rp.retrieved_candidate_count == 0
    assert rp.reranked is False
    assert rp.reranker_model is None


def test_retrieval_timestamp_stores_iso8601():
    rp = _make_provenance(retrieval_timestamp="2026-07-16T10:00:00Z")
    assert rp.retrieval_timestamp == "2026-07-16T10:00:00Z"


def test_retrieved_candidate_count_stores_int():
    rp = _make_provenance(retrieved_candidate_count=42)
    assert rp.retrieved_candidate_count == 42


def test_reranked_stores_bool_true():
    rp = _make_provenance(reranked=True)
    assert rp.reranked is True


def test_reranked_stores_bool_false():
    rp = _make_provenance(reranked=False)
    assert rp.reranked is False


def test_reranker_model_stores_string():
    rp = _make_provenance(reranker_model="claude-haiku-4-5-20251001")
    assert rp.reranker_model == "claude-haiku-4-5-20251001"


def test_new_fields_in_model_dump():
    """model_dump() includes all four new PH5.5c fields."""
    rp = _make_provenance(
        retrieval_timestamp="2026-07-16T00:00:00Z",
        retrieved_candidate_count=15,
        reranked=True,
        reranker_model="claude-haiku-4-5-20251001",
    )
    d = rp.model_dump()
    assert d["retrieval_timestamp"] == "2026-07-16T00:00:00Z"
    assert d["retrieved_candidate_count"] == 15
    assert d["reranked"] is True
    assert d["reranker_model"] == "claude-haiku-4-5-20251001"


def test_new_fields_roundtrip():
    """New fields survive model_dump / model_validate round-trip."""
    rp = _make_provenance(
        retrieval_timestamp="2026-07-16T09:15:00Z",
        retrieved_candidate_count=88,
        reranked=True,
        reranker_model="claude-sonnet-5",
    )
    rp2 = RetrievalProvenance.model_validate(rp.model_dump())
    assert rp2.retrieval_timestamp == rp.retrieval_timestamp
    assert rp2.retrieved_candidate_count == rp.retrieved_candidate_count
    assert rp2.reranked == rp.reranked
    assert rp2.reranker_model == rp.reranker_model


# ---------------------------------------------------------------------------
# 4. EvidenceAgent._execute_kb() integration
# ---------------------------------------------------------------------------


def test_execute_kb_stores_retrieval_provenance(monkeypatch):
    """_execute_kb() writes _retrieval_provenance to context.trace."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1, score=0.8)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(question="What is the risk?")
    agent._execute_kb(ctx)

    assert "_retrieval_provenance" in ctx.trace


def test_execute_kb_provenance_count_equals_candidates(monkeypatch):
    """Number of provenance records equals number of final candidates."""
    from functional_agents.evidence_agent import EvidenceAgent

    ev1 = _make_evidence(statement="Claim one about risk.")
    ev2 = _make_evidence(statement="Claim two about safety.")
    meta1 = _make_metadata(ev1.evidence_id)
    meta2 = _make_metadata(ev2.evidence_id)
    items = [
        RetrievedEvidence(evidence=ev1, metadata=meta1, score=0.9, rank=1,
                          metadata_factor=EvidenceRetriever._metadata_factor(meta1)),
        RetrievedEvidence(evidence=ev2, metadata=meta2, score=0.7, rank=2,
                          metadata_factor=EvidenceRetriever._metadata_factor(meta2)),
    ]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    provenance = ctx.trace["_retrieval_provenance"]
    assert len(provenance) == 2


def test_execute_kb_provenance_is_list_of_dicts(monkeypatch):
    """Provenance records are serialized as list[dict] in trace."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    provenance = ctx.trace["_retrieval_provenance"]
    assert isinstance(provenance, list)
    assert isinstance(provenance[0], dict)


def test_execute_kb_provenance_has_required_fields(monkeypatch):
    """Each provenance record has the core required fields."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1, score=0.73)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(question="BWRX-300 deployment risks?")
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    for key in ("evidence_id", "retrieval_query", "retrieval_mode", "retrieval_rank",
                "hybrid_score", "lexical_score", "semantic_score", "reranker",
                "reranked", "retrieved_candidate_count", "retrieval_timestamp"):
        assert key in rp, f"provenance record missing key: {key}"


def test_execute_kb_provenance_evidence_id_matches_candidate(monkeypatch):
    """evidence_id in provenance matches the candidate's evidence_id."""
    from functional_agents.evidence_agent import EvidenceAgent

    item = _make_retrieved_evidence(rank=1)
    agent = EvidenceAgent(retriever=_MockRetriever([item]))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    assert rp["evidence_id"] == item.evidence.evidence_id


def test_execute_kb_provenance_scores_match_candidate(monkeypatch):
    """hybrid_score, lexical_score, semantic_score match the candidate."""
    from functional_agents.evidence_agent import EvidenceAgent

    ev = _make_evidence()
    meta = _make_metadata(ev.evidence_id)
    item = RetrievedEvidence(
        evidence=ev, metadata=meta, score=0.654, rank=1,
        lexical_score=0.5, semantic_score=0.8,
        metadata_factor=EvidenceRetriever._metadata_factor(meta),
    )
    agent = EvidenceAgent(retriever=_MockRetriever([item]))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    assert rp["hybrid_score"] == pytest.approx(0.654)
    assert rp["lexical_score"] == pytest.approx(0.5)
    assert rp["semantic_score"] == pytest.approx(0.8)


def test_execute_kb_provenance_timestamp_is_populated(monkeypatch):
    """retrieval_timestamp is a non-empty string in UTC format."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    ts = rp["retrieval_timestamp"]
    assert ts is not None
    assert len(ts) >= 10  # at least "YYYY-MM-DD"
    assert "T" in ts


def test_execute_kb_provenance_reranked_false_by_default(monkeypatch):
    """Without a reranker, reranked=False in every provenance record."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items), use_reranker=False)
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    assert rp["reranked"] is False


def test_execute_kb_provenance_retrieved_candidate_count(monkeypatch):
    """retrieved_candidate_count reflects the primary retrieval matched count."""
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    # matched=55 via _MockRetriever — reflects what the store matched
    agent = EvidenceAgent(retriever=_MockRetriever(items, matched=55))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    assert rp["retrieved_candidate_count"] == 55


def test_execute_kb_provenance_serializable(monkeypatch):
    """Provenance records can be JSON-serialised without error."""
    import json
    from functional_agents.evidence_agent import EvidenceAgent

    items = [_make_retrieved_evidence(rank=1)]
    agent = EvidenceAgent(retriever=_MockRetriever(items))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context()
    agent._execute_kb(ctx)

    serialized = json.dumps(ctx.trace["_retrieval_provenance"])
    assert isinstance(serialized, str)
    parsed = json.loads(serialized)
    assert len(parsed) == 1


# ---------------------------------------------------------------------------
# 5. Query attribution
# ---------------------------------------------------------------------------


def test_primary_query_attributed_to_direct_hits(monkeypatch):
    """Items from primary retrieval carry the primary question as retrieval_query."""
    from functional_agents.evidence_agent import EvidenceAgent

    item = _make_retrieved_evidence(rank=1)
    agent = EvidenceAgent(retriever=_MockRetriever([item]))
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(question="SMR deployment timeline?")
    agent._execute_kb(ctx)

    rp = ctx.trace["_retrieval_provenance"][0]
    assert rp["retrieval_query"] == "SMR deployment timeline?"


def test_subquestion_query_attributed_to_expansion_hits(monkeypatch):
    """Items added during subquestion expansion carry the subquestion as retrieval_query."""
    from functional_agents.evidence_agent import EvidenceAgent

    primary_ev = _make_evidence(statement="Primary result about licensing.")
    primary_meta = _make_metadata(primary_ev.evidence_id)
    primary_item = RetrievedEvidence(
        evidence=primary_ev, metadata=primary_meta, score=0.9, rank=1,
        metadata_factor=EvidenceRetriever._metadata_factor(primary_meta),
    )

    sq_ev = _make_evidence(statement="Expansion result about fuel supply.")
    sq_meta = _make_metadata(sq_ev.evidence_id)
    sq_item = RetrievedEvidence(
        evidence=sq_ev, metadata=sq_meta, score=0.7, rank=1,
        metadata_factor=EvidenceRetriever._metadata_factor(sq_meta),
    )

    subquestion = "What are the fuel supply constraints?"

    class _SqRetriever:
        """Returns primary_item for any query, plus sq_item only for the subquestion."""
        provider = None

        def retrieve(self, query, *, mode=RETRIEVAL_MODE_LEXICAL, top_k=20, **_kw):
            if query == subquestion:
                return RetrievalResult(
                    query=query,
                    items=[sq_item],
                    domains_searched=["test"],
                    total_candidates=5,
                    matched_candidates=1,
                    retrieval_method="lexical-v1",
                    latency_ms=1.0,
                    mode=mode,
                )
            return RetrievalResult(
                query=query,
                items=[primary_item],
                domains_searched=["test"],
                total_candidates=10,
                matched_candidates=1,
                retrieval_method="lexical-v1",
                latency_ms=1.0,
                mode=mode,
            )

    agent = EvidenceAgent(retriever=_SqRetriever())
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(
        question="SMR risks?",
        subquestions=[subquestion],
    )
    agent._execute_kb(ctx)

    provenance_by_id = {rp["evidence_id"]: rp for rp in ctx.trace["_retrieval_provenance"]}

    assert provenance_by_id[primary_ev.evidence_id]["retrieval_query"] == "SMR risks?"
    assert provenance_by_id[sq_ev.evidence_id]["retrieval_query"] == subquestion


def test_primary_and_subquestion_both_have_provenance(monkeypatch):
    """Both primary and expansion items appear in the provenance list."""
    from functional_agents.evidence_agent import EvidenceAgent

    primary_ev = _make_evidence(statement="Primary evidence claim.")
    sq_ev = _make_evidence(statement="Subquestion evidence claim.")

    primary_meta = _make_metadata(primary_ev.evidence_id)
    sq_meta = _make_metadata(sq_ev.evidence_id)

    primary_item = RetrievedEvidence(
        evidence=primary_ev, metadata=primary_meta, score=0.9, rank=1,
        metadata_factor=EvidenceRetriever._metadata_factor(primary_meta),
    )
    sq_item = RetrievedEvidence(
        evidence=sq_ev, metadata=sq_meta, score=0.6, rank=1,
        metadata_factor=EvidenceRetriever._metadata_factor(sq_meta),
    )

    subquestion = "Subquestion about deployment?"

    class _TwoItemRetriever:
        provider = None

        def retrieve(self, query, *, mode=RETRIEVAL_MODE_LEXICAL, top_k=20, **_kw):
            if query == subquestion:
                return RetrievalResult(
                    query=query, items=[sq_item], domains_searched=["test"],
                    total_candidates=5, matched_candidates=1,
                    retrieval_method="lexical-v1", latency_ms=1.0, mode=mode,
                )
            return RetrievalResult(
                query=query, items=[primary_item], domains_searched=["test"],
                total_candidates=10, matched_candidates=1,
                retrieval_method="lexical-v1", latency_ms=1.0, mode=mode,
            )

    agent = EvidenceAgent(retriever=_TwoItemRetriever())
    monkeypatch.setattr(EvidenceAgent, "_set_evidence_note", _noop_set_evidence_note)

    ctx = _make_context(question="Main question?", subquestions=[subquestion])
    agent._execute_kb(ctx)

    evidence_ids_in_prov = {rp["evidence_id"] for rp in ctx.trace["_retrieval_provenance"]}
    assert primary_ev.evidence_id in evidence_ids_in_prov
    assert sq_ev.evidence_id in evidence_ids_in_prov


# ---------------------------------------------------------------------------
# 6. Reranker model string parsing
# ---------------------------------------------------------------------------


def test_reranker_model_parsed_from_llm_prefix():
    """'llm-<model>' reranker string parses model correctly."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])

    # Simulate what the agent does when parsing RerankResult.reranker
    rr_str = "llm-claude-haiku-4-5-20251001"
    reranker_type = "llm" if rr_str.startswith("llm-") else "passthrough"
    reranker_model = rr_str[4:] if rr_str.startswith("llm-") else None

    rp = build_retrieval_provenance(
        item, result,
        reranked=True,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
    )
    assert rp.reranker == "llm"
    assert rp.reranker_model == "claude-haiku-4-5-20251001"


def test_passthrough_reranker_string_parses_correctly():
    """'passthrough' reranker string maps to no model."""
    item = _make_retrieved_evidence()
    result = _make_result("q", [item])

    rr_str = "passthrough"
    reranker_type = "llm" if rr_str.startswith("llm-") else "passthrough"
    reranker_model = rr_str[4:] if rr_str.startswith("llm-") else None

    rp = build_retrieval_provenance(
        item, result,
        reranker_type=reranker_type,
        reranker_model=reranker_model,
    )
    assert rp.reranker == "passthrough"
    assert rp.reranker_model is None


def test_reranker_model_stored_in_model_dump():
    """reranker_model appears in model_dump() for LLM path."""
    rp = _make_provenance(
        reranker="llm",
        reranked=True,
        reranker_model="claude-opus-4-8",
    )
    d = rp.model_dump()
    assert d["reranker"] == "llm"
    assert d["reranker_model"] == "claude-opus-4-8"
    assert d["reranked"] is True
