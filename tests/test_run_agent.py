"""Tests for the agent-level execution harness (PH2.0)."""

from __future__ import annotations

import json

import pytest

from functional_agents.run_agent import (
    run_agent,
    load_context,
    context_diff,
    HarnessError,
    AGENT_REGISTRY,
)

_FIXTURES = "fixtures"


# ---------------------------------------------------------------------------
# Each agent runs independently (mock / no-llm)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent,produced", [
    ("planner", "plan"),
    ("evidence", "evidence_notes"),
    ("hypothesis", "hypotheses"),
    ("recommendation", "recommendations"),
    ("report", "artifacts"),
])
def test_agent_runs_independently(agent, produced):
    res = run_agent(agent, f"{_FIXTURES}/{agent}_start.json", no_llm=True)
    assert res["mini_trace"]["status"] == "success"
    # The declared product appears in the diff (added or modified).
    changed = set(res["diff"]["added"]) | set(res["diff"]["modified"])
    assert produced in changed


def test_planner_produces_plan_content():
    res = run_agent("planner", f"{_FIXTURES}/planner_start.json", no_llm=True)
    assert res["context"]["plan"]["research_type"]
    assert res["context"]["plan"]["subquestions"]


def test_evidence_produces_evidence_and_preserves_citation_fields():
    res = run_agent("evidence", f"{_FIXTURES}/evidence_start.json", no_llm=True)
    notes = res["context"]["evidence_notes"]
    assert notes and "evidence_items" in notes[0]
    # Evidence items retain source attribution (citation basis).
    for item in notes[0]["evidence_items"]:
        assert "evidence_id" in item and "source_document" in item


def test_recommendation_produced():
    res = run_agent("recommendation", f"{_FIXTURES}/recommendation_start.json", no_llm=True)
    assert res["context"]["recommendations"]


def test_report_produces_executive_report(tmp_path):
    out = tmp_path / "report.md"
    res = run_agent("report", f"{_FIXTURES}/report_start.json", no_llm=True, out_path=str(out))
    assert res["context"]["artifacts"]["report_path"]
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# Mini trace + context diff
# ---------------------------------------------------------------------------

def test_mini_trace_shape():
    res = run_agent("planner", f"{_FIXTURES}/planner_start.json", no_llm=True)
    mt = res["mini_trace"]
    for key in ("agent", "status", "execution_time_ms", "llm_call_count",
                "llm_mode", "normalization", "validation", "objects_produced", "warnings"):
        assert key in mt
    assert mt["llm_mode"] == "mock"
    assert mt["validation"]["preconditions_passed"]
    assert mt["validation"]["postconditions_passed"]


def test_context_diff_categories():
    res = run_agent("planner", f"{_FIXTURES}/planner_start.json", no_llm=True)
    d = res["diff"]
    assert "plan" in d["added"]
    assert set(d) == {"added", "modified", "unchanged"}
    # question was present in the fixture and untouched by planner.
    assert "question" in d["unchanged"]


def test_output_and_trace_written(tmp_path):
    out = tmp_path / "ctx.json"
    trace = tmp_path / "mini.trace.json"
    res = run_agent("planner", f"{_FIXTURES}/planner_start.json", no_llm=True)
    out.write_text(json.dumps(res["context"]), encoding="utf-8")
    trace.write_text(json.dumps(res["mini_trace"]), encoding="utf-8")
    assert json.loads(out.read_text())["plan"]["research_type"]
    assert json.loads(trace.read_text())["agent"] == "planner"


# ---------------------------------------------------------------------------
# Negative tests — deterministic, actionable errors
# ---------------------------------------------------------------------------

def test_unknown_agent_errors():
    with pytest.raises(HarnessError, match="Unknown agent"):
        run_agent("bogus", f"{_FIXTURES}/planner_start.json", no_llm=True)


def test_missing_fixture_errors():
    with pytest.raises(HarnessError, match="not found"):
        run_agent("planner", f"{_FIXTURES}/does_not_exist.json", no_llm=True)


def test_invalid_json_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid", encoding="utf-8")
    with pytest.raises(HarnessError, match="Invalid JSON"):
        run_agent("planner", str(bad), no_llm=True)


def test_precondition_failure_missing_upstream():
    # hypothesis needs evidence_notes; planner fixture has none.
    with pytest.raises(HarnessError, match="Precondition failed"):
        run_agent("hypothesis", f"{_FIXTURES}/planner_start.json", no_llm=True)


def test_non_object_fixture_errors(tmp_path):
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(HarnessError, match="must be a JSON object"):
        run_agent("planner", str(bad), no_llm=True)


def test_unknown_fixture_keys_ignored(tmp_path):
    fx = tmp_path / "fx.json"
    fx.write_text(json.dumps({
        "question": "Q?", "profiles": ["ai_data_centers"],
        "execution_profile": "ai_data_centers", "run_id": "x",
        "research_object": {"research_id": "R"},
        "totally_unknown_key": 123,
    }), encoding="utf-8")
    ctx = load_context(str(fx))  # must not raise
    assert ctx.question == "Q?"
    assert not hasattr(ctx, "totally_unknown_key")


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

def test_registry_covers_required_agents():
    for a in ("planner", "evidence", "hypothesis", "recommendation", "report"):
        assert a in AGENT_REGISTRY
        assert "pre" in AGENT_REGISTRY[a] and "post" in AGENT_REGISTRY[a]


# ---------------------------------------------------------------------------
# PH3.4a — CLI guard: refuse reserved canonical-trace filenames before execution
# ---------------------------------------------------------------------------

from functional_agents.run_agent import main as run_agent_main


def test_cli_refuses_exact_reserved_trace_filename(tmp_path, capsys):
    trace_path = tmp_path / "pipeline.trace.json"
    exit_code = run_agent_main([
        "--agent", "planner",
        "--fixture", f"{_FIXTURES}/planner_start.json",
        "--trace", str(trace_path),
    ])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "CanonicalTraceReservedError" in err
    assert not trace_path.exists()  # never written — failed before execution


def test_cli_refuses_suffix_reserved_trace_filename(tmp_path, capsys):
    trace_path = tmp_path / "ph34_pipeline.trace.json"
    exit_code = run_agent_main([
        "--agent", "planner",
        "--fixture", f"{_FIXTURES}/planner_start.json",
        "--trace", str(trace_path),
    ])
    assert exit_code == 2
    assert "CanonicalTraceReservedError" in capsys.readouterr().err
    assert not trace_path.exists()


def test_cli_never_overwrites_existing_canonical_trace(tmp_path, capsys):
    canonical = tmp_path / "pipeline.trace.json"
    canonical.write_text('{"schema_version": "ph3.4-canonical-v1", "sentinel": true}')
    run_agent_main([
        "--agent", "planner",
        "--fixture", f"{_FIXTURES}/planner_start.json",
        "--trace", str(canonical),
    ])
    # Untouched — still exactly the sentinel content, not a mini-trace.
    assert json.loads(canonical.read_text())["sentinel"] is True


def test_cli_allows_normal_trace_filename(tmp_path, capsys):
    trace_path = tmp_path / "planner.trace.json"
    exit_code = run_agent_main([
        "--agent", "planner",
        "--fixture", f"{_FIXTURES}/planner_start.json",
        "--trace", str(trace_path),
    ])
    assert exit_code == 0
    assert trace_path.exists()
    assert "status         : success" in capsys.readouterr().out


def test_cli_allows_no_trace_flag_at_all(capsys):
    exit_code = run_agent_main([
        "--agent", "planner",
        "--fixture", f"{_FIXTURES}/planner_start.json",
    ])
    assert exit_code == 0
