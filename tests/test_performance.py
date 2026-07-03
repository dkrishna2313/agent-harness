"""Tests for PH3.1 performance instrumentation + report generator.

Covers the universal stage-timing mechanism, boundary-stage recording, the
enriched tracker summary, and the performance_report module.  All behaviour is
measurement-only; these tests assert the *shape* of the captured data.
"""

from __future__ import annotations

import time
import types

import pytest

from functional_agents.performance import (
    AgentPerfRecord,
    LLMCallRecord,
    PerformanceTracker,
    SubPhaseRecord,
    STAGE_RETRIEVAL,
    STAGE_NORMALIZATION,
    STAGE_VALIDATION,
    STAGE_BUSINESS_LOGIC,
    STAGE_OTHER,
    STAGE_CATEGORIES,
    stage_timer,
    record_stage,
    record_boundary_stages,
)
from functional_agents import performance_report as pr


def _ctx_with_tracker():
    """Minimal stand-in for an AgentContext carrying a perf tracker."""
    tracker = PerformanceTracker()
    ctx = types.SimpleNamespace(trace={"_perf_tracker": tracker})
    return ctx, tracker


# ---------------------------------------------------------------------------
# AgentPerfRecord — stage breakdown & derived timings
# ---------------------------------------------------------------------------

def test_stage_breakdown_groups_by_category():
    rec = AgentPerfRecord(
        agent_name="EvidenceAgent",
        wall_ms=1000.0,
        llm_calls=[LLMCallRecord("op", "m", 200.0, 1, 1, 2, True)],
        sub_phases=[
            SubPhaseRecord("retrieval_query", 100.0, STAGE_RETRIEVAL),
            SubPhaseRecord("reranking", 300.0, STAGE_RETRIEVAL),
            SubPhaseRecord("mapping", 50.0, STAGE_BUSINESS_LOGIC),
        ],
    )
    bd = rec.stage_breakdown()
    assert bd[STAGE_RETRIEVAL] == 400.0
    assert bd[STAGE_BUSINESS_LOGIC] == 50.0
    assert rec.measured_stage_ms == 450.0
    # unattributed = wall - llm - measured = 1000 - 200 - 450 = 350
    assert rec.unattributed_ms == 350.0


def test_unattributed_floored_at_zero():
    rec = AgentPerfRecord(
        agent_name="X", wall_ms=100.0,
        llm_calls=[LLMCallRecord("op", "m", 90.0, 0, 0, 0, True)],
        sub_phases=[SubPhaseRecord("s", 50.0, STAGE_OTHER)],
    )
    # 100 - 90 - 50 = -40 → floored to 0
    assert rec.unattributed_ms == 0.0


def test_to_dict_includes_stage_fields():
    rec = AgentPerfRecord(
        agent_name="X", wall_ms=500.0,
        sub_phases=[SubPhaseRecord("q", 120.0, STAGE_RETRIEVAL)],
    )
    d = rec.to_dict()
    assert d["stage_breakdown"] == {STAGE_RETRIEVAL: 120.0}
    assert d["non_llm_ms"] == 500.0
    assert d["sub_phases"][0]["category"] == STAGE_RETRIEVAL


# ---------------------------------------------------------------------------
# Tracker — record_stage / add_sub_phase / summary
# ---------------------------------------------------------------------------

def test_record_stage_and_flush():
    tracker = PerformanceTracker()
    tracker.record_stage("q", STAGE_RETRIEVAL, 10.0, scanned=5)
    phases = tracker.flush_sub_phases()
    assert len(phases) == 1
    assert phases[0].category == STAGE_RETRIEVAL
    assert phases[0].metadata == {"scanned": 5}
    # flush drains
    assert tracker.flush_sub_phases() == []


def test_add_sub_phase_alias_defaults_to_other():
    tracker = PerformanceTracker()
    tracker.add_sub_phase("legacy", 5.0)
    phases = tracker.flush_sub_phases()
    assert phases[0].category == STAGE_OTHER


def test_unknown_category_coerced_to_other():
    tracker = PerformanceTracker()
    tracker.record_stage("weird", "not_a_category", 5.0)
    assert tracker.flush_sub_phases()[0].category == STAGE_OTHER


def test_summary_has_llm_pct_and_bottlenecks():
    tracker = PerformanceTracker()
    tracker.record(AgentPerfRecord(
        agent_name="A", wall_ms=1000.0,
        llm_calls=[LLMCallRecord("op", "m", 800.0, 10, 5, 15, True)],
        sub_phases=[SubPhaseRecord("r", 100.0, STAGE_RETRIEVAL)],
    ))
    tracker.record(AgentPerfRecord(
        agent_name="B", wall_ms=500.0,
        llm_calls=[LLMCallRecord("op", "m", 400.0, 2, 2, 4, True)],
    ))
    s = tracker.summary()
    assert s["totals"]["pipeline_wall_ms"] == 1500.0
    assert s["totals"]["llm_pct"] == round(100 * 1200 / 1500, 1)
    assert s["totals"]["non_llm_pct"] == round(100 - s["totals"]["llm_pct"], 1)
    assert s["stage_totals"][STAGE_RETRIEVAL] == 100.0
    # bottleneck agents ranked by wall time, A first
    assert s["bottlenecks"]["agents"][0]["agent"] == "A"
    assert s["bottlenecks"]["stages"][0]["stage"] == STAGE_RETRIEVAL


# ---------------------------------------------------------------------------
# Module-level helpers: stage_timer / record_stage / record_boundary_stages
# ---------------------------------------------------------------------------

def test_stage_timer_records_into_tracker():
    ctx, tracker = _ctx_with_tracker()
    with stage_timer(ctx, "work", STAGE_BUSINESS_LOGIC):
        time.sleep(0.001)
    phases = tracker.flush_sub_phases()
    assert len(phases) == 1
    assert phases[0].name == "work"
    assert phases[0].category == STAGE_BUSINESS_LOGIC
    assert phases[0].duration_ms >= 0.0


def test_stage_timer_noop_without_tracker():
    ctx = types.SimpleNamespace(trace={})
    ran = []
    with stage_timer(ctx, "work", STAGE_OTHER):
        ran.append(True)
    assert ran == [True]  # block still executes


def test_stage_timer_noop_with_none_context():
    with stage_timer(None, "work", STAGE_OTHER):
        pass  # must not raise


def test_record_stage_helper():
    ctx, tracker = _ctx_with_tracker()
    record_stage(ctx, "s", STAGE_VALIDATION, 42.0)
    phases = tracker.flush_sub_phases()
    assert phases[0].duration_ms == 42.0
    assert phases[0].category == STAGE_VALIDATION


def test_record_boundary_stages_reads_durations():
    ctx, tracker = _ctx_with_tracker()
    diagnostics = {
        "stages": {"generation": "ok", "normalization": "ok", "validation": "ok"},
        "failed_stage": None,
        "normalization": {"repairs": [], "duration_ms": 1.5},
        "validation": {"passed": True, "duration_ms": 2.5},
    }
    record_boundary_stages(ctx, diagnostics)
    phases = {p.category: p for p in tracker.flush_sub_phases()}
    assert phases[STAGE_NORMALIZATION].duration_ms == 1.5
    assert phases[STAGE_VALIDATION].duration_ms == 2.5


def test_record_boundary_stages_handles_missing_durations():
    ctx, tracker = _ctx_with_tracker()
    # No duration_ms → nothing recorded, no error
    record_boundary_stages(ctx, {"normalization": {"repairs": []}, "validation": {}})
    assert tracker.flush_sub_phases() == []


def test_record_boundary_stages_noop_without_tracker():
    ctx = types.SimpleNamespace(trace={})
    record_boundary_stages(ctx, {"normalization": {"duration_ms": 1.0}})  # must not raise


# ---------------------------------------------------------------------------
# Boundary framework now emits per-stage duration_ms
# ---------------------------------------------------------------------------

def test_run_boundary_adds_duration_ms():
    from functional_agents.boundary_framework import run_boundary, BoundaryError

    out, diag = run_boundary(
        {"x": 1},
        normalize=lambda r: (r, {"repairs": []}),
        validate=lambda n: ({"typed": n}, {"passed": True}),
        error_base=BoundaryError,
    )
    assert "duration_ms" in diag["normalization"]
    assert "duration_ms" in diag["validation"]
    # top-level schema unchanged (PH2.5 contract preserved)
    assert set(diag) == {"stages", "failed_stage", "normalization", "validation"}


# ---------------------------------------------------------------------------
# performance_report module
# ---------------------------------------------------------------------------

def _sample_summary():
    tracker = PerformanceTracker()
    tracker.record(AgentPerfRecord(
        agent_name="EvidenceAgent", wall_ms=1000.0,
        sub_phases=[SubPhaseRecord("reranking", 800.0, STAGE_RETRIEVAL)],
    ))
    tracker.record(AgentPerfRecord(
        agent_name="PlannerAgent", wall_ms=2000.0,
        llm_calls=[LLMCallRecord("op", "m", 1900.0, 100, 50, 150, True)],
    ))
    return tracker.summary()


def test_extract_performance_from_serialized_key():
    perf = _sample_summary()
    assert pr.extract_performance({"performance": perf}) is perf
    assert pr.extract_performance({"_performance": perf}) is perf
    assert pr.extract_performance({"trace": {"performance": perf}}) is perf
    assert pr.extract_performance(perf) is perf  # raw summary
    assert pr.extract_performance({"nothing": 1}) is None


def test_build_report_shape():
    report = pr.build_report(_sample_summary())
    assert report["agent_count"] == 2
    # agents sorted by wall time, Planner (2000) first
    assert report["agents"][0]["agent"] == "PlannerAgent"
    assert report["totals"]["llm_pct"] > 0
    assert report["stage_totals"][STAGE_RETRIEVAL] == 800.0
    assert report["bottlenecks"]["agents"][0]["agent"] == "PlannerAgent"


def test_build_report_backfills_old_trace_totals():
    # Simulate a pre-PH3.1 totals block (no non_llm_ms / llm_pct)
    old = {
        "totals": {"pipeline_wall_ms": 1000.0, "llm_total_ms": 900.0,
                   "llm_call_count": 1, "prompt_tokens": 0,
                   "completion_tokens": 0, "total_tokens": 0},
        "agents": [{"agent": "A", "wall_ms": 1000.0, "llm_total_ms": 900.0,
                    "llm_call_count": 1, "total_tokens": 0}],
    }
    report = pr.build_report(old)
    assert report["totals"]["non_llm_ms"] == 100.0
    assert report["totals"]["llm_pct"] == 90.0


def test_render_markdown_contains_sections():
    md = pr.render_markdown(pr.build_report(_sample_summary()), source="x.json")
    assert "# Platform Performance Report (PH3.1)" in md
    assert "## Pipeline totals" in md
    assert "## Per-agent timing" in md
    assert "## Non-LLM time by stage" in md
    assert "## Largest bottlenecks" in md


def test_all_stage_categories_are_unique():
    assert len(set(STAGE_CATEGORIES)) == len(STAGE_CATEGORIES)
