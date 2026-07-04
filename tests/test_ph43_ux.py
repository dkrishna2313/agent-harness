"""PH4.3 — Production UX & Operational Polish tests.

Covers:
- Run directory creation (RUN-YYYYMMDD-HHMMSS naming)
- Artifact naming (report.md, pipeline.trace.json, research_object.json, engagement.json)
- CLI completion summary (run ID, status, profiles, output dir)
- Backwards compatibility: --out still writes to specified path
- _make_run_dir determinism within the same second (collision handling)
- _write_run_artifacts writes expected files
- _print_run_summary emits expected fields
- Default log level is PROGRESS
"""

from __future__ import annotations

import json
import re
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from functional_agents.cli import (
    _make_run_dir,
    _print_run_summary,
    _write_run_artifacts,
    app,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_ctx(*, run_id="abc123", profiles=None, deliverables=None):
    """Return a minimal AgentContext-like mock."""
    ctx = MagicMock()
    ctx.run_id = run_id
    ctx.profiles = profiles or ["ai_data_centers"]
    ctx.workflow_state = "COMPLETE"
    ctx.agent_history = [{"agent": "PlannerAgent"}, {"agent": "ReportAgent"}]
    ctx.deliverables = deliverables or []
    ctx.artifacts = {"report_path": "/tmp/report.md", "trace_path": "/tmp/report.trace.json"}
    ctx.trace = {
        "_engagement_id": "ENG-001",
        "_engagement": {"title": "Test Engagement"},
        "_performance": {
            "totals": {"total_tokens": 1000, "llm_call_count": 5, "pipeline_wall_ms": 1234}
        },
    }
    ctx.research_object = {"research_id": "R-001", "question": "Test question"}
    return ctx


# ---------------------------------------------------------------------------
# R1 — Run directory naming
# ---------------------------------------------------------------------------

class TestRunDirNaming:
    def test_r1_run_dir_created_under_outputs_runs(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = _make_run_dir()
        assert run_dir.exists()
        assert run_dir.is_dir()
        # Must be under outputs/runs/ (resolve relative path against cwd)
        assert run_dir.resolve().parent == (tmp_path / "outputs" / "runs").resolve()

    def test_r1_run_dir_name_format(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_dir = _make_run_dir()
        name = run_dir.name
        assert re.fullmatch(r"RUN-\d{8}-\d{6}", name), f"unexpected name: {name!r}"

    def test_r1_run_dir_mkdir_parents(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # outputs/runs/ does not exist yet — _make_run_dir must create it
        assert not (tmp_path / "outputs").exists()
        _make_run_dir()
        assert (tmp_path / "outputs" / "runs").exists()

    def test_r1_collision_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Two calls in quick succession must both succeed (mkdir exist_ok=True)
        d1 = _make_run_dir()
        d2 = _make_run_dir()
        assert d1.exists()
        assert d2.exists()


# ---------------------------------------------------------------------------
# R2 — Artifact writing
# ---------------------------------------------------------------------------

class TestRunArtifacts:
    def test_r2_research_object_written(self, tmp_path):
        ctx = _fake_ctx()
        _write_run_artifacts(ctx, tmp_path)
        ro_path = tmp_path / "research_object.json"
        assert ro_path.exists()
        data = json.loads(ro_path.read_text())
        assert data["research_id"] == "R-001"

    def test_r2_engagement_written(self, tmp_path):
        ctx = _fake_ctx()
        _write_run_artifacts(ctx, tmp_path)
        eng_path = tmp_path / "engagement.json"
        assert eng_path.exists()
        data = json.loads(eng_path.read_text())
        assert data["engagement_id"] == "ENG-001"
        assert data["title"] == "Test Engagement"

    def test_r2_no_crash_when_research_object_missing(self, tmp_path):
        ctx = _fake_ctx()
        ctx.research_object = None
        _write_run_artifacts(ctx, tmp_path)  # must not raise
        assert not (tmp_path / "research_object.json").exists()

    def test_r2_no_crash_when_engagement_missing(self, tmp_path):
        ctx = _fake_ctx()
        ctx.trace = {}
        _write_run_artifacts(ctx, tmp_path)  # must not raise
        assert not (tmp_path / "engagement.json").exists()

    def test_r2_research_object_valid_json(self, tmp_path):
        ctx = _fake_ctx()
        ctx.research_object = {"a": 1, "b": [1, 2, 3]}
        _write_run_artifacts(ctx, tmp_path)
        data = json.loads((tmp_path / "research_object.json").read_text())
        assert data["a"] == 1


# ---------------------------------------------------------------------------
# R3 — CLI completion summary
# ---------------------------------------------------------------------------

class TestRunSummary:
    def _capture_summary(self, tmp_path, **kwargs):
        ctx = _fake_ctx(**kwargs)
        buf = StringIO()
        with patch("builtins.print"):
            with patch("typer.echo", side_effect=lambda msg="", **kw: buf.write(str(msg) + "\n")):
                _print_run_summary(ctx, "Strategic Engagement", tmp_path, 1.4, is_run_dir=True)
        return buf.getvalue()

    def test_r3_run_id_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path, run_id="deadbeef1234")
        assert "deadbeef1234" in out

    def test_r3_status_success_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path)
        assert "SUCCESS" in out

    def test_r3_profiles_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path, profiles=["ai_data_centers", "transmission"])
        assert "ai_data_centers" in out
        assert "transmission" in out

    def test_r3_output_dir_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path)
        assert str(tmp_path) in out

    def test_r3_elapsed_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path)
        assert "1.4s" in out

    def test_r3_tokens_in_summary(self, tmp_path):
        out = self._capture_summary(tmp_path)
        assert "1,000" in out  # formatted token count

    def test_r3_deliverables_section_present(self, tmp_path):
        out = self._capture_summary(tmp_path)
        assert "Deliverables" in out

    def test_r3_no_crash_when_perf_missing(self, tmp_path):
        ctx = _fake_ctx()
        ctx.trace = {}
        buf = StringIO()
        with patch("typer.echo", side_effect=lambda msg="", **kw: buf.write(str(msg) + "\n")):
            _print_run_summary(ctx, "Research (question)", tmp_path, 0.5, is_run_dir=False)
        out = buf.getvalue()
        assert "Status" in out  # must still print


# ---------------------------------------------------------------------------
# R4 — CLI integration: --out backwards compatibility
# ---------------------------------------------------------------------------

class TestCLIBackwardsCompat:
    """Ensure --out <path> still works — report written to specified path."""

    def test_r4_explicit_out_respects_path(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "test.md").write_text("AI data center power requirements.", encoding="utf-8")
        out = tmp_path / "my_report.md"
        engagement_yaml = tmp_path / "eng.yaml"
        engagement_yaml.write_text(
            "engagement:\n  title: Test\n  objectives:\n    - Assess power\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            [
                "run",
                "--engagement", str(engagement_yaml),
                "--profiles", "ai_data_centers",
                "--sources", str(sources),
                "--out", str(out),
                "--mock",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists(), f"report not found at {out}"

    def test_r4_no_out_creates_run_dir(self, tmp_path, monkeypatch):
        """When --out is omitted, a RUN-* directory is created under outputs/runs/."""
        monkeypatch.chdir(tmp_path)
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "test.md").write_text("AI data center power requirements.", encoding="utf-8")
        engagement_yaml = tmp_path / "eng.yaml"
        engagement_yaml.write_text(
            "engagement:\n  title: Test\n  objectives:\n    - Assess power\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            [
                "run",
                "--engagement", str(engagement_yaml),
                "--profiles", "ai_data_centers",
                "--sources", str(sources),
                "--mock",
            ],
        )
        assert result.exit_code == 0, result.output
        runs_dir = tmp_path / "outputs" / "runs"
        assert runs_dir.exists()
        run_dirs = list(runs_dir.iterdir())
        assert len(run_dirs) == 1
        assert re.fullmatch(r"RUN-\d{8}-\d{6}", run_dirs[0].name)
        assert (run_dirs[0] / "report.md").exists()


# ---------------------------------------------------------------------------
# R5 — Default log level is PROGRESS
# ---------------------------------------------------------------------------

class TestDefaultLogLevel:
    def test_r5_default_log_level_is_progress(self):
        """The CLI passes 'PROGRESS' when --log-level is not specified."""
        calls = []
        with patch("functional_agents.cli._configure_logging", side_effect=lambda **kw: calls.append(kw)):
            with patch("functional_agents.cli._make_run_dir", return_value=Path("/tmp/RUN-TEST")):
                with patch("functional_agents.cli._build_client"):
                    # We can't easily run the full CLI here, so test the internal default
                    pass
        # Confirmed via source inspection: `log_level or "PROGRESS"` is the pattern.
        # Test that the module uses the PROGRESS default string.
        import inspect
        import functional_agents.cli as cli_mod
        src = inspect.getsource(cli_mod.main)
        assert 'log_level or "PROGRESS"' in src


# ---------------------------------------------------------------------------
# R6 — Artifact naming determinism
# ---------------------------------------------------------------------------

class TestArtifactNaming:
    def test_r6_report_named_report_md_in_run_dir(self, tmp_path, monkeypatch):
        """When --out is omitted, the report is always named report.md."""
        monkeypatch.chdir(tmp_path)
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "test.md").write_text("Power.", encoding="utf-8")
        engagement_yaml = tmp_path / "eng.yaml"
        engagement_yaml.write_text(
            "engagement:\n  title: Test\n  objectives:\n    - Assess power\n",
            encoding="utf-8",
        )
        result = CliRunner().invoke(
            app,
            ["run", "--engagement", str(engagement_yaml), "--profiles", "ai_data_centers",
             "--sources", str(sources), "--mock"],
        )
        assert result.exit_code == 0, result.output
        run_dirs = list((tmp_path / "outputs" / "runs").iterdir())
        assert len(run_dirs) == 1
        report = run_dirs[0] / "report.md"
        assert report.exists(), f"report.md not found in run dir {run_dirs[0]}"
