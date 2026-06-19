"""Tests for research_agent.retrieval_planner and select_top_chunks_multi (J3.0/J3.0a)."""

from __future__ import annotations

import pytest

from research_agent.retrieval_planner import (
    RetrievalPlanner,
    RetrievalPlan,
    QueryMode,
    classify_question_mode,
    detect_entity_lock,
    detect_metric_lock,
    _content_words,
)
from research_agent.retrieval import select_top_chunks_multi
from research_agent.schemas import Chunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(chunk_id: str, doc: str, text: str, num: int = 0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_name=doc,
        text=text,
        chunk_number=num,
        page_number=1,
        start_offset=0,
        end_offset=len(text),
    )


# ---------------------------------------------------------------------------
# _content_words
# ---------------------------------------------------------------------------

def test_content_words_removes_stopwords():
    words = _content_words("What is the power of a rack?")
    assert "what" not in words
    assert "the" not in words
    assert "power" in words
    assert "rack" in words


def test_content_words_filters_short_tokens():
    words = _content_words("a b c power")
    assert "a" not in words
    assert "b" not in words
    assert "power" in words


# ---------------------------------------------------------------------------
# J3.0a: classify_question_mode
# ---------------------------------------------------------------------------

class TestClassifyMode:
    def test_how_many_is_fact_lookup(self):
        assert classify_question_mode("How many GPUs are in GB200 NVL72?") == QueryMode.FACT_LOOKUP

    def test_how_much_is_fact_lookup(self):
        assert classify_question_mode("How much power does the NVL72 consume?") == QueryMode.FACT_LOOKUP

    def test_what_is_the_is_fact_lookup(self):
        assert classify_question_mode("What is the rack power requirement of GB200 NVL72?") == QueryMode.FACT_LOOKUP

    def test_factors_is_exploratory(self):
        assert classify_question_mode("What factors drive SMR LCOE?") == QueryMode.EXPLORATORY_RESEARCH

    def test_barriers_is_exploratory(self):
        assert classify_question_mode("What barriers exist to large-scale SMR deployment?") == QueryMode.EXPLORATORY_RESEARCH

    def test_challenges_is_exploratory(self):
        assert classify_question_mode("What are the main challenges for SMR licensing?") == QueryMode.EXPLORATORY_RESEARCH

    def test_compare_is_comparison(self):
        assert classify_question_mode("Compare GB200 NVL72 and DGX H100.") == QueryMode.COMPARISON

    def test_versus_is_comparison(self):
        assert classify_question_mode("GB200 NVL72 vs DGX H100 power comparison") == QueryMode.COMPARISON

    def test_why_is_explanation(self):
        assert classify_question_mode("Why does GB200 require liquid cooling?") == QueryMode.EXPLANATION

    def test_how_does_is_explanation(self):
        assert classify_question_mode("How does the NVL72 cooling system work?") == QueryMode.EXPLANATION

    def test_exploratory_beats_what_is(self):
        # "What are the main challenges" should be EXPLORATORY, not FACT_LOOKUP
        assert classify_question_mode("What are the main challenges for SMR deployment?") == QueryMode.EXPLORATORY_RESEARCH

    def test_short_question_defaults_fact_lookup(self):
        # Very short specific questions without other signals → FACT_LOOKUP
        assert classify_question_mode("NVL72 rack weight?") == QueryMode.FACT_LOOKUP


# ---------------------------------------------------------------------------
# J3.0a: detect_entity_lock
# ---------------------------------------------------------------------------

class TestEntityLock:
    def test_nvl72_detected(self):
        assert detect_entity_lock("How many GPUs in GB200 NVL72?") == "GB200 NVL72"

    def test_nvl36_detected(self):
        assert detect_entity_lock("How many GPUs in NVL36?") == "GB200 NVL36"

    def test_dgx_h100_detected(self):
        assert detect_entity_lock("What is the DGX H100 rack power?") == "DGX H100"

    def test_gb200_detected_without_nvl72(self):
        # GB200 without NVL72 still locks to GB200
        assert detect_entity_lock("What is GB200 memory capacity?") == "GB200"

    def test_bwrx300_detected(self):
        result = detect_entity_lock("What is the BWRX-300 electrical output?")
        assert result == "BWRX-300"

    def test_no_entity_returns_none(self):
        assert detect_entity_lock("What factors drive LCOE?") is None

    def test_nvl72_wins_over_gb200(self):
        # "GB200 NVL72" is more specific than just "GB200"
        result = detect_entity_lock("GB200 NVL72 GPU count question")
        assert result == "GB200 NVL72"


# ---------------------------------------------------------------------------
# J3.0a: detect_metric_lock
# ---------------------------------------------------------------------------

class TestMetricLock:
    def test_gpu_count_detected(self):
        assert detect_metric_lock("How many GPUs are integrated into NVL72?") == "gpu_count"

    def test_rack_power_detected(self):
        assert detect_metric_lock("What is the rack power requirement?") == "rack_power"

    def test_how_much_power_detected(self):
        assert detect_metric_lock("How much power does NVL72 consume?") == "rack_power"

    def test_lcoe_detected(self):
        assert detect_metric_lock("What is the LCOE for BWRX-300?") == "lcoe"

    def test_memory_detected(self):
        assert detect_metric_lock("What is the GPU memory capacity?") == "memory_capacity"

    def test_no_metric_returns_none(self):
        assert detect_metric_lock("What factors drive SMR deployment?") is None


# ---------------------------------------------------------------------------
# RetrievalPlanner.plan — FACT_LOOKUP mode
# ---------------------------------------------------------------------------

class TestFactLookupPlan:
    def test_nvl72_gpu_count_entity_and_metric_locked(self):
        plan = RetrievalPlanner().plan("How many GPUs are integrated into a single GB200 NVL72 rack?")
        assert plan.planner_mode == QueryMode.FACT_LOOKUP
        assert plan.entity_lock == "GB200 NVL72"
        assert plan.metric_lock == "gpu_count"

    def test_nvl72_gpu_count_stays_below_max(self):
        plan = RetrievalPlanner().plan("How many GPUs are integrated into a single GB200 NVL72 rack?")
        assert plan.query_count <= 3

    def test_nvl72_gpu_count_no_adjacent_products(self):
        """FACT_LOOKUP must not pull in NVL36, DGX, GB300, etc."""
        plan = RetrievalPlanner().plan("How many GPUs are integrated into a single GB200 NVL72 rack?")
        all_text = " ".join(plan.queries).lower()
        forbidden = ["nvl36", "dgx", "gb300", "h100", "hopper", "grace cpu hopper"]
        for term in forbidden:
            assert term not in all_text, (
                f"Adjacent product term '{term}' found in FACT_LOOKUP queries: {plan.queries}"
            )

    def test_nvl72_gpu_count_all_queries_anchored(self):
        """Every query in a FACT_LOOKUP with entity lock must mention the entity."""
        plan = RetrievalPlanner().plan("How many GPUs are integrated into a single GB200 NVL72 rack?")
        for q in plan.queries:
            # Primary question is always included (contains entity); rest must too
            assert "NVL72" in q or "GB200" in q, (
                f"Query not anchored to entity: {q!r}"
            )

    def test_rack_power_fact_lookup(self):
        plan = RetrievalPlanner().plan("What is the rack power requirement of GB200 NVL72?")
        assert plan.planner_mode == QueryMode.FACT_LOOKUP
        assert plan.entity_lock == "GB200 NVL72"
        assert plan.metric_lock == "rack_power"
        # Should not include cooling or GPU-count queries
        all_text = " ".join(plan.queries).lower()
        assert "cooling" not in all_text or "power" in all_text  # power queries are OK

    def test_fact_lookup_without_entity_still_works(self):
        # No named entity — planner falls back to content-word queries
        plan = RetrievalPlanner().plan("What is the total rack power consumption?")
        assert plan.planner_mode == QueryMode.FACT_LOOKUP
        assert plan.entity_lock is None
        assert plan.query_count >= 1  # at minimum the primary question

    def test_fact_lookup_query_count_is_bounded(self):
        plan = RetrievalPlanner(max_queries=3).plan("How many GPUs are in NVL72?")
        assert plan.query_count <= 3


# ---------------------------------------------------------------------------
# RetrievalPlanner.plan — EXPLORATORY mode
# ---------------------------------------------------------------------------

class TestExploratoryPlan:
    def test_smr_lcoe_factors_is_exploratory(self):
        plan = RetrievalPlanner().plan("What factors drive SMR LCOE?")
        assert plan.planner_mode == QueryMode.EXPLORATORY_RESEARCH

    def test_exploratory_generates_more_queries(self):
        plan = RetrievalPlanner().plan("What factors drive SMR LCOE?")
        assert plan.query_count >= 3  # exploratory produces more than FACT_LOOKUP

    def test_exploratory_no_entity_lock(self):
        plan = RetrievalPlanner().plan("What barriers exist to large-scale SMR deployment?")
        assert plan.entity_lock is None

    def test_exploratory_no_metric_lock(self):
        plan = RetrievalPlanner().plan("What factors drive SMR LCOE?")
        assert plan.metric_lock is None


# ---------------------------------------------------------------------------
# RetrievalPlanner.plan — COMPARISON / EXPLANATION modes
# ---------------------------------------------------------------------------

class TestOtherModes:
    def test_comparison_detected(self):
        plan = RetrievalPlanner().plan("Compare GB200 NVL72 and DGX H100.")
        assert plan.planner_mode == QueryMode.COMPARISON

    def test_explanation_detected(self):
        plan = RetrievalPlanner().plan("Why does GB200 require liquid cooling?")
        assert plan.planner_mode == QueryMode.EXPLANATION

    def test_comparison_query_count(self):
        plan = RetrievalPlanner().plan("Compare GB200 NVL72 and DGX H100.")
        assert 1 <= plan.query_count <= 5

    def test_explanation_query_count(self):
        plan = RetrievalPlanner().plan("Why does GB200 require liquid cooling?")
        assert 1 <= plan.query_count <= 5


# ---------------------------------------------------------------------------
# RetrievalPlan.to_dict — trace contract
# ---------------------------------------------------------------------------

class TestPlanToDict:
    def test_to_dict_has_planner_mode(self):
        plan = RetrievalPlanner().plan("How many GPUs are in NVL72?")
        d = plan.to_dict()
        assert "planner_mode" in d
        assert d["planner_mode"] == "FACT_LOOKUP"

    def test_to_dict_has_entity_lock(self):
        plan = RetrievalPlanner().plan("How many GPUs are in NVL72?")
        d = plan.to_dict()
        assert "entity_lock" in d
        assert d["entity_locked"] is True

    def test_to_dict_has_metric_lock(self):
        plan = RetrievalPlanner().plan("How many GPUs are in NVL72?")
        d = plan.to_dict()
        assert "metric_lock" in d
        assert d["metric_locked"] is True

    def test_to_dict_exploratory_no_locks(self):
        plan = RetrievalPlanner().plan("What factors drive SMR LCOE?")
        d = plan.to_dict()
        assert d["entity_locked"] is False
        assert d["metric_locked"] is False

    def test_to_dict_required_keys(self):
        plan = RetrievalPlanner().plan("What is the rack power?")
        d = plan.to_dict()
        required = {
            "primary_question", "planner_mode", "entity_lock", "metric_lock",
            "entity_locked", "metric_locked", "queries", "detected_topics",
            "expansion_source", "query_count",
        }
        assert required <= set(d.keys())


# ---------------------------------------------------------------------------
# Profile-aware expansion
# ---------------------------------------------------------------------------

class TestProfileAwarePlan:
    def test_smr_profile_no_ai_dc_terms(self):
        from research_agent.profile import load_profile
        profile = load_profile("smr")
        plan = RetrievalPlanner(profile=profile).plan("What factors drive SMR LCOE?")
        ai_dc_terms = ["NVL72", "GB200", "PDU", "CDU"]
        all_text = " ".join(plan.queries)
        assert not any(t in all_text for t in ai_dc_terms)

    def test_ai_dc_profile_cool_terms(self):
        from research_agent.profile import load_profile
        profile = load_profile("ai_data_centers")
        plan = RetrievalPlanner(profile=profile).plan("Why does NVL72 need liquid cooling?")
        all_text = " ".join(plan.queries).lower()
        assert "cool" in all_text or "liquid" in all_text or "thermal" in all_text


# ---------------------------------------------------------------------------
# select_top_chunks_multi
# ---------------------------------------------------------------------------

def _make_chunks() -> list[Chunk]:
    return [
        _make_chunk("A-1", "docA.pdf", "rack power consumption 100 kW total load", 0),
        _make_chunk("A-2", "docA.pdf", "cooling system liquid water loop CDU unit", 1),
        _make_chunk("B-1", "docB.pdf", "NVLink switch fabric bandwidth 400 GB/s", 0),
        _make_chunk("B-2", "docB.pdf", "GPU thermal design power TDP 700W per chip", 1),
        _make_chunk("C-1", "docC.pdf", "SMR economics LCOE levelized cost electricity", 0),
    ]


def test_multi_returns_chunks_and_scores_and_stats():
    chunks = _make_chunks()
    selected, scores, stats = select_top_chunks_multi(
        chunks, ["rack power", "cooling system"], top_n=3
    )
    assert isinstance(selected, list)
    assert isinstance(scores, list)
    assert isinstance(stats, dict)


def test_multi_deduplicates_chunks():
    chunks = _make_chunks()
    selected, _, _ = select_top_chunks_multi(
        chunks, ["rack power consumption", "power total load"], top_n=5
    )
    ids = [c.chunk_id for c in selected]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in multi-query selection"


def test_multi_stats_keys():
    chunks = _make_chunks()
    _, _, stats = select_top_chunks_multi(chunks, ["power"], top_n=3)
    assert "queries_generated" in stats
    assert "queries_executed" in stats
    assert "chunks_retrieved" in stats
    assert "unique_sources" in stats


def test_multi_stats_queries_count():
    chunks = _make_chunks()
    queries = ["power rack", "cooling liquid", "GPU TDP"]
    _, _, stats = select_top_chunks_multi(chunks, queries, top_n=3)
    assert stats["queries_generated"] == 3
    assert stats["queries_executed"] == 3


def test_multi_unique_sources_correct():
    chunks = _make_chunks()
    selected, _, stats = select_top_chunks_multi(chunks, ["rack power cooling GPU"], top_n=5)
    expected = len({c.document_name for c in selected})
    assert stats["unique_sources"] == expected


def test_multi_empty_queries():
    chunks = _make_chunks()
    selected, scores, stats = select_top_chunks_multi(chunks, [], top_n=5)
    assert selected == []
    assert scores == []
    assert stats["chunks_retrieved"] == 0


def test_multi_single_query_matches_select_top_chunks():
    from research_agent.retrieval import select_top_chunks
    chunks = _make_chunks()
    query = "rack power consumption"
    single, single_scores = select_top_chunks(chunks, query, top_n=3)
    multi, multi_scores, _ = select_top_chunks_multi(chunks, [query], top_n=3)
    assert {c.chunk_id for c in single} == {c.chunk_id for c in multi}


def test_multi_output_is_sorted_by_doc_chunk_order():
    chunks = _make_chunks()
    selected, _, _ = select_top_chunks_multi(
        chunks, ["power cooling networking"], top_n=5
    )
    order = [(c.document_name, c.chunk_number) for c in selected]
    assert order == sorted(order)


# ---------------------------------------------------------------------------
# Agent integration: retrieval_plan in trace metadata
# ---------------------------------------------------------------------------

def test_agent_mock_trace_has_retrieval_plan(tmp_path):
    """ResearchMemo.metadata should include retrieval_plan when using MockClaudeClient."""
    from research_agent.agent import DcPowerAgent
    from research_agent.claude_client import MockClaudeClient
    from research_agent.schemas import SourceDocument

    doc_text = "The NVL72 rack consumes 120 kW total rack power. Liquid cooling is required."
    source_file = tmp_path / "test_doc.txt"
    source_file.write_text(doc_text)

    agent = DcPowerAgent(client=MockClaudeClient())
    docs = [SourceDocument(path=source_file, title="Test Doc", extension=".txt", text=doc_text)]
    memo = agent.analyze("How many GPUs are in the GB200 NVL72 rack?", docs)

    assert "retrieval_plan" in memo.metadata, "retrieval_plan missing from memo.metadata"
    plan_data = memo.metadata["retrieval_plan"]
    assert "planner_mode" in plan_data
    assert "entity_lock" in plan_data
    assert "metric_lock" in plan_data
    assert plan_data["planner_mode"] == "FACT_LOOKUP"
    assert plan_data["entity_locked"] is True

    assert "retrieval_stats" in memo.metadata
    stats = memo.metadata["retrieval_stats"]
    assert "queries_generated" in stats
    assert "unique_sources" in stats


def test_agent_mock_trace_exploratory_plan(tmp_path):
    """EXPLORATORY_RESEARCH question should produce an unlocked plan."""
    from research_agent.agent import DcPowerAgent
    from research_agent.claude_client import MockClaudeClient
    from research_agent.schemas import SourceDocument

    doc_text = "SMR economics depend on construction cost, LCOE, and financing."
    source_file = tmp_path / "smr_doc.txt"
    source_file.write_text(doc_text)

    agent = DcPowerAgent(client=MockClaudeClient())
    docs = [SourceDocument(path=source_file, title="SMR Doc", extension=".txt", text=doc_text)]
    memo = agent.analyze("What factors drive SMR LCOE?", docs)

    plan_data = memo.metadata["retrieval_plan"]
    assert plan_data["planner_mode"] == "EXPLORATORY_RESEARCH"
    assert plan_data["entity_locked"] is False
    # Exploratory plans have more queries
    assert plan_data["query_count"] >= 2
