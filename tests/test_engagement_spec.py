"""Tests for the Strategic Engagement input model and loader (J9.1)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from functional_agents.cli import app
from functional_agents.engagement_spec import (
    EngagementError,
    EngagementSpec,
    load_engagement_spec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_ENGAGEMENT = {
    "title": "AI Data Center Power Strategy",
    "client": "Hyperscaler",
    "industry": "AI Infrastructure",
    "current_situation": "Planning a large GB300 NVL72 deployment.",
    "objectives": ["Identify power strategies", "Determine cooling architecture"],
    "constraints": ["24 month window", "Net-zero targets"],
    "stakeholders": ["CIO", "Energy Procurement"],
    "assumptions": ["Power draw keeps rising"],
    "success_criteria": ["Ranked strategies with trade-offs"],
    "decision_horizon": "24 months",
    "priorities": ["Speed to power", "Cost"],
    "risks": ["Grid interconnection delays"],
    "known_unknowns": ["Realized GB300 power draw"],
}


def _write_yaml(path, payload, *, wrap=True):
    import yaml
    data = {"engagement": payload} if wrap else payload
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _write_json(path, payload, *, wrap=True):
    data = {"engagement": payload} if wrap else payload
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_spec_parses_all_fields():
    spec = EngagementSpec.model_validate(_FULL_ENGAGEMENT)
    assert spec.title == "AI Data Center Power Strategy"
    assert spec.client == "Hyperscaler"
    assert len(spec.objectives) == 2
    assert spec.decision_horizon == "24 months"
    assert spec.known_unknowns == ["Realized GB300 power draw"]


def test_spec_fields_optional():
    spec = EngagementSpec.model_validate({"title": "Minimal"})
    assert spec.title == "Minimal"
    assert spec.objectives == []
    assert spec.constraints == []


def test_missing_important_fields_reported_not_invented():
    spec = EngagementSpec.model_validate({"client": "Acme"})
    missing = spec.missing_important_fields()
    assert "title" in missing
    assert "current_situation" in missing
    assert "objectives" in missing


def test_unknown_field_rejected():
    with pytest.raises(Exception):
        EngagementSpec.model_validate({"title": "X", "bogus_field": 1})


def test_to_framing_brief_includes_sections():
    spec = EngagementSpec.model_validate(_FULL_ENGAGEMENT)
    brief = spec.to_framing_brief()
    assert "AI Data Center Power Strategy" in brief
    assert "Hyperscaler" in brief
    assert "Current situation:" in brief
    assert "Objectives:" in brief
    assert "Constraints:" in brief
    assert "Known unknowns:" in brief
    # A rich brief must be more than a one-line goal.
    assert brief.count("\n") >= 5


def test_to_framing_brief_notes_missing_fields():
    spec = EngagementSpec.model_validate({"client": "Acme"})
    brief = spec.to_framing_brief()
    assert "did not specify" in brief
    assert "Do not invent" in brief


def test_trace_metadata_counts():
    spec = EngagementSpec.model_validate(_FULL_ENGAGEMENT)
    meta = spec.to_trace_metadata()
    assert meta["objective_count"] == 2
    assert meta["constraint_count"] == 2
    assert meta["risk_count"] == 1
    assert meta["missing_important_fields"] == []


# ---------------------------------------------------------------------------
# Loader — YAML
# ---------------------------------------------------------------------------

def test_load_yaml_wrapped(tmp_path):
    p = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT, wrap=True)
    spec = load_engagement_spec(p)
    assert spec.title == "AI Data Center Power Strategy"
    assert len(spec.objectives) == 2


def test_load_yaml_bare_mapping(tmp_path):
    p = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT, wrap=False)
    spec = load_engagement_spec(p)
    assert spec.client == "Hyperscaler"


def test_load_yml_extension(tmp_path):
    p = _write_yaml(tmp_path / "eng.yml", _FULL_ENGAGEMENT)
    spec = load_engagement_spec(p)
    assert spec.title == "AI Data Center Power Strategy"


# ---------------------------------------------------------------------------
# Loader — JSON
# ---------------------------------------------------------------------------

def test_load_json_wrapped(tmp_path):
    p = _write_json(tmp_path / "eng.json", _FULL_ENGAGEMENT, wrap=True)
    spec = load_engagement_spec(p)
    assert spec.title == "AI Data Center Power Strategy"
    assert len(spec.constraints) == 2


def test_load_json_bare_mapping(tmp_path):
    p = _write_json(tmp_path / "eng.json", _FULL_ENGAGEMENT, wrap=False)
    spec = load_engagement_spec(p)
    assert spec.industry == "AI Infrastructure"


def test_yaml_and_json_equivalent(tmp_path):
    yp = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT)
    jp = _write_json(tmp_path / "eng.json", _FULL_ENGAGEMENT)
    assert load_engagement_spec(yp).model_dump() == load_engagement_spec(jp).model_dump()


# ---------------------------------------------------------------------------
# Loader — error handling
# ---------------------------------------------------------------------------

def test_missing_file_errors():
    with pytest.raises(EngagementError, match="not found"):
        load_engagement_spec("does/not/exist.yaml")


def test_empty_file_errors(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(EngagementError):
        load_engagement_spec(p)


def test_non_mapping_errors(tmp_path):
    p = tmp_path / "list.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(EngagementError, match="mapping"):
        load_engagement_spec(p)


def test_invalid_json_errors(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(EngagementError, match="Invalid JSON"):
        load_engagement_spec(p)


def test_unknown_field_in_file_errors(tmp_path):
    p = _write_json(tmp_path / "eng.json", {"title": "X", "nope": 1})
    with pytest.raises(EngagementError, match="validation failed"):
        load_engagement_spec(p)


def test_effectively_empty_engagement_errors(tmp_path):
    p = _write_yaml(tmp_path / "eng.yaml", {}, wrap=False)
    with pytest.raises(EngagementError):
        load_engagement_spec(p)


def test_sample_engagement_file_loads():
    """The committed sample engagement must be valid."""
    spec = load_engagement_spec("engagements/hyperscaler_ai_strategy.yaml")
    assert spec.title
    assert spec.objectives
    assert spec.missing_important_fields() == []


# ---------------------------------------------------------------------------
# CLI — mutual exclusivity & backwards compatibility
# ---------------------------------------------------------------------------

def _mk_sources(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    (sources / "doc.md").write_text(
        "AI data centers require significant power and liquid cooling at high rack density.",
        encoding="utf-8",
    )
    return sources


def test_cli_goal_and_engagement_mutually_exclusive(tmp_path):
    p = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT)
    result = CliRunner().invoke(
        app,
        ["run", "--goal", "Test goal", "--engagement", str(p), "--mock",
         "--out", str(tmp_path / "out.md")],
    )
    assert result.exit_code == 1
    assert "exactly one of" in result.output


def test_cli_question_and_engagement_mutually_exclusive(tmp_path):
    p = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT)
    result = CliRunner().invoke(
        app,
        ["run", "Some question?", "--engagement", str(p), "--mock",
         "--out", str(tmp_path / "out.md")],
    )
    assert result.exit_code == 1
    assert "exactly one of" in result.output


def test_cli_no_input_errors(tmp_path):
    result = CliRunner().invoke(app, ["run", "--mock", "--out", str(tmp_path / "out.md")])
    assert result.exit_code == 1
    assert "provide a QUESTION" in result.output


def test_cli_bad_engagement_file_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("engagement:\n  bogus: 1\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["run", "--engagement", str(bad), "--mock", "--out", str(tmp_path / "out.md")],
    )
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_cli_goal_still_works(tmp_path):
    """Backwards compatibility: --goal runs unchanged."""
    sources = _mk_sources(tmp_path)
    out = tmp_path / "goal.md"
    result = CliRunner().invoke(
        app,
        ["run", "--goal", "Analyze AI data center power strategies.",
         "--sources", str(sources), "--mock", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Research (goal)" in result.output


def test_cli_engagement_mode_runs(tmp_path):
    """Strategic Engagement Mode runs end-to-end and records run mode in trace."""
    sources = _mk_sources(tmp_path)
    p = _write_yaml(tmp_path / "eng.yaml", _FULL_ENGAGEMENT)
    out = tmp_path / "eng_run.md"
    result = CliRunner().invoke(
        app,
        ["run", "--engagement", str(p), "--sources", str(sources),
         "--mock", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    assert "Strategic Engagement" in result.output

    trace = json.loads(out.with_suffix(".trace.json").read_text(encoding="utf-8"))
    assert trace["run_mode"] == "strategic_engagement"
    assert trace["engagement"]["title"] == "AI Data Center Power Strategy"
    assert trace["engagement"]["objective_count"] == 2


def test_cli_question_mode_trace_records_research_mode(tmp_path):
    sources = _mk_sources(tmp_path)
    out = tmp_path / "q_run.md"
    result = CliRunner().invoke(
        app,
        ["run", "What power do AI data centers need?", "--sources", str(sources),
         "--mock", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    trace = json.loads(out.with_suffix(".trace.json").read_text(encoding="utf-8"))
    assert trace["run_mode"] == "research"
    assert "engagement" not in trace
