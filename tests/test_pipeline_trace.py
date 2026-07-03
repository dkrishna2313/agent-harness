"""Tests for the PH3.4 Canonical Pipeline Trace.

Verifies: canonical trace generation from a real AgentContext, schema
validation (is_canonical_trace / require_canonical_trace), correct grouping
of boundaries/prompt-slices/performance/agents, deterministic agent-key
naming, and that malformed/legacy trace shapes are rejected with a clear error.
"""

from __future__ import annotations

import json
import types

import pytest

from functional_agents.pipeline_trace import (
    SCHEMA_VERSION,
    CanonicalTraceError,
    _canonical_agent_key,
    build_canonical_trace,
    is_canonical_trace,
    require_canonical_trace,
    write_canonical_trace,
)


# ---------------------------------------------------------------------------
# Canonical agent key derivation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls_name,expected", [
    ("PlannerAgent", "planner"),
    ("EvidenceAgent", "evidence"),
    ("HypothesisAgent", "hypothesis"),
    ("RecommendationAgent", "recommendation"),
    ("ReportAgent", "report"),
    ("StrategicSynthesisAgent", "strategic_synthesis"),
    ("RecommendationImprovementAgent", "recommendation_improvement"),
    ("QAAgent", "qa"),
    ("MultiProfileAgent", "multi_profile"),
    ("DecisionAnalysisAgent", "decision_analysis"),
    ("ExecutiveConfidenceAgent", "executive_confidence"),
])
def test_canonical_agent_key_matches_existing_trace_key_convention(cls_name, expected):
    # These must match the literal "_<key>_boundary" / "_<key>_prompt_slice"
    # strings already hardcoded by each agent (PH2.x / PH3.3) — this function
    # doesn't invent a new convention, it reproduces the existing one.
    assert _canonical_agent_key(cls_name) == expected


def test_canonical_agent_key_handles_bare_agent_name():
    assert _canonical_agent_key("Agent") == "agent"


# ---------------------------------------------------------------------------
# build_canonical_trace
# ---------------------------------------------------------------------------

def _make_context(**overrides):
    base = dict(
        run_id="run123",
        question="What are the power constraints?",
        goal="",
        engagement=None,
        profiles=["ai_data_centers"],
        execution_profile="ai_data_centers",
        agent_history=[
            {"agent": "PlannerAgent", "status": "success", "summary": "plan ok"},
            {"agent": "EvidenceAgent", "status": "success", "summary": "evidence ok"},
            {"agent": "HypothesisAgent", "status": "success", "summary": "3 hypotheses"},
            {"agent": "RecommendationAgent", "status": "warning", "summary": "no hyps"},
        ],
        trace={
            "_planner_boundary": {"stages": {}, "failed_stage": None},
            "_evidence_boundary": {"stages": {}, "failed_stage": None},
            "_hypothesis_boundary": {"stages": {}, "failed_stage": None},
            "_recommendation_boundary": {"stages": {}, "failed_stage": "generation"},
            "_planner_prompt_slice": {"original_bytes": 100, "sliced_bytes": 100, "bytes_saved": 0},
            "_hypothesis_prompt_slice": {"original_bytes": 500, "sliced_bytes": 200, "bytes_saved": 300},
            "_performance": {"totals": {"pipeline_wall_ms": 1000.0}, "agents": []},
            "_client": object(),  # scratch — must be ignored entirely
            "_perf_tracker": object(),  # scratch — must be ignored entirely
        },
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_build_canonical_trace_has_all_required_sections():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    assert is_canonical_trace(trace)
    assert trace["schema_version"] == SCHEMA_VERSION


def test_build_canonical_trace_pipeline_section():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    p = trace["pipeline"]
    assert p["run_id"] == "run123"
    assert p["question"] == "What are the power constraints?"
    assert p["run_mode"] == "question"
    assert p["profiles"] == ["ai_data_centers"]


def test_build_canonical_trace_infers_goal_mode():
    ctx = _make_context(question="", goal="Develop a strategy")
    trace = build_canonical_trace(ctx)
    assert trace["pipeline"]["run_mode"] == "goal"


def test_build_canonical_trace_infers_engagement_mode():
    ctx = _make_context(engagement={"decision_statement": "x"})
    trace = build_canonical_trace(ctx)
    assert trace["pipeline"]["run_mode"] == "engagement"


def test_build_canonical_trace_agents_grouped_by_canonical_key():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    assert set(trace["agents"]) == {"planner", "evidence", "hypothesis", "recommendation"}
    assert trace["agents"]["planner"]["status"] == "success"
    assert trace["agents"]["planner"]["agent_class"] == "PlannerAgent"
    assert "agent" not in trace["agents"]["planner"]  # renamed to agent_class, not duplicated


def test_build_canonical_trace_last_entry_wins_for_looping_agents():
    ctx = _make_context(agent_history=[
        {"agent": "PlannerAgent", "status": "warning", "summary": "first pass"},
        {"agent": "PlannerAgent", "status": "success", "summary": "final pass"},
    ])
    trace = build_canonical_trace(ctx)
    assert trace["agents"]["planner"]["status"] == "success"
    assert trace["agents"]["planner"]["summary"] == "final pass"


def test_build_canonical_trace_boundaries_grouped_and_unmodified():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    b = trace["boundaries"]
    assert set(b) == {"planner", "evidence", "hypothesis", "recommendation"}
    assert b["recommendation"]["failed_stage"] == "generation"
    assert b["planner"] == ctx.trace["_planner_boundary"]  # verbatim, not rewritten


def test_build_canonical_trace_prompt_slices_grouped():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    ps = trace["prompt_slices"]
    assert set(ps) == {"planner", "hypothesis"}
    assert ps["hypothesis"]["bytes_saved"] == 300


def test_build_canonical_trace_performance_passthrough():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    assert trace["performance"]["totals"]["pipeline_wall_ms"] == 1000.0


def test_build_canonical_trace_performance_none_when_absent():
    ctx = _make_context(trace={})
    trace = build_canonical_trace(ctx)
    assert trace["performance"] is None


def test_build_canonical_trace_contracts_section():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    c = trace["contracts"]
    assert "run(context: AgentContext) -> AgentResult" in c["functional_agent_contract"]
    assert c["agents_conforming"] == sorted(["planner", "evidence", "hypothesis", "recommendation"])


def test_build_canonical_trace_summary_counts():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    s = trace["summary"]
    assert s["agents_run"] == 4
    assert s["agents_succeeded"] == 3
    assert s["agents_warning"] == 1
    assert s["agents_failed"] == 0
    assert s["boundaries_failed"] == 1
    assert s["boundaries_passed"] == 3
    assert s["prompt_slices_applied"] == 1  # only hypothesis had bytes_saved > 0
    assert s["pipeline_status"] == "partial"


def test_build_canonical_trace_status_failed_when_any_agent_errors():
    ctx = _make_context(agent_history=[
        {"agent": "PlannerAgent", "status": "error", "summary": "boom"},
    ])
    trace = build_canonical_trace(ctx)
    assert trace["summary"]["pipeline_status"] == "failed"


def test_build_canonical_trace_never_mutates_context():
    ctx = _make_context()
    before_trace = dict(ctx.trace)
    before_history = list(ctx.agent_history)
    build_canonical_trace(ctx)
    assert ctx.trace == before_trace
    assert ctx.agent_history == before_history


def test_build_canonical_trace_json_safe():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    json.dumps(trace, default=str)  # must not raise (scratch objects excluded)


def test_build_canonical_trace_degrades_gracefully_on_empty_context():
    ctx = types.SimpleNamespace()  # nothing set at all
    trace = build_canonical_trace(ctx)
    assert is_canonical_trace(trace)
    assert trace["agents"] == {}
    assert trace["summary"]["agents_run"] == 0


# ---------------------------------------------------------------------------
# is_canonical_trace / require_canonical_trace
# ---------------------------------------------------------------------------

def test_is_canonical_trace_true_for_well_formed():
    ctx = _make_context()
    assert is_canonical_trace(build_canonical_trace(ctx)) is True


@pytest.mark.parametrize("bad", [
    None,
    "a string",
    [],
    {},
    {"schema_version": "wrong-version", "pipeline": {}, "agents": {}, "boundaries": {},
     "performance": None, "prompt_slices": {}, "contracts": {}, "summary": {}},
    {"schema_version": SCHEMA_VERSION, "pipeline": {}},  # missing most keys
])
def test_is_canonical_trace_false_for_malformed(bad):
    assert is_canonical_trace(bad) is False


def test_require_canonical_trace_returns_valid_trace():
    ctx = _make_context()
    trace = build_canonical_trace(ctx)
    assert require_canonical_trace(trace) is trace


def test_require_canonical_trace_raises_with_actionable_message():
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace({"agent": "planner", "status": "success"}, source="mini.trace.json")
    msg = str(ei.value)
    assert "mini.trace.json" in msg
    assert "canonical" in msg.lower()


def test_require_canonical_trace_raises_for_non_dict():
    with pytest.raises(CanonicalTraceError):
        require_canonical_trace("not a dict", source="x.json")


def test_require_canonical_trace_reports_missing_keys():
    with pytest.raises(CanonicalTraceError, match="missing keys"):
        require_canonical_trace({"schema_version": SCHEMA_VERSION}, source="x.json")


# ---------------------------------------------------------------------------
# PH3.4a — improved diagnostic messages
# ---------------------------------------------------------------------------

def test_require_canonical_trace_identifies_mini_trace():
    mini_trace = {
        "agent": "planner", "agent_class": "PlannerAgent", "status": "success",
        "execution_time_ms": 5.0, "llm_call_count": 0, "llm_mode": "mock",
        "boundary": {}, "prompt_slice": {}, "validation": {}, "objects_produced": [],
    }
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace(mini_trace, source="planner.trace.json")
    msg = str(ei.value)
    assert "planner.trace.json" in msg
    assert "run_agent.py MINI TRACE" in msg
    assert "python3 -m functional_agents.cli run" in msg


def test_require_canonical_trace_identifies_legacy_report_trace():
    legacy = {"timestamp": "2026-01-01", "question": "q", "documents_loaded": 3,
              "question_topics_detected": ["power"]}
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace(legacy, source="harness_report.trace.json")
    assert "legacy ReportAgent" in str(ei.value)


def test_require_canonical_trace_identifies_bare_performance_summary():
    bare = {"totals": {"pipeline_wall_ms": 100.0}, "agents": []}
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace(bare, source="perf.json")
    assert "PerformanceTracker.summary()" in str(ei.value)


def test_require_canonical_trace_reports_wrong_schema_version():
    wrong = {
        "schema_version": "some-old-version",
        "pipeline": {}, "agents": {}, "boundaries": {},
        "performance": None, "prompt_slices": {}, "contracts": {}, "summary": {},
    }
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace(wrong, source="x.json")
    msg = str(ei.value)
    assert "some-old-version" in msg
    assert SCHEMA_VERSION in msg


def test_require_canonical_trace_generic_unrecognized_shape():
    with pytest.raises(CanonicalTraceError) as ei:
        require_canonical_trace({"totally": "unrelated", "shape": 1}, source="x.json")
    assert "unrecognized trace format" in str(ei.value)


# ---------------------------------------------------------------------------
# write_canonical_trace
# ---------------------------------------------------------------------------

def test_write_canonical_trace_creates_file(tmp_path):
    ctx = _make_context()
    out_path = write_canonical_trace(ctx, tmp_path)
    assert out_path.name == "pipeline.trace.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert is_canonical_trace(written)


def test_write_canonical_trace_creates_missing_dirs(tmp_path):
    ctx = _make_context()
    nested = tmp_path / "a" / "b" / "c"
    out_path = write_canonical_trace(ctx, nested)
    assert out_path.exists()
