"""Tests for DX1 — --save-context flag on `functional_agents.cli run`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from functional_agents.cli import app
from functional_agents.context import AgentContext
from functional_agents.context_snapshot import context_to_jsonable, load_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

runner = CliRunner()


def _minimal_ctx(question: str = "What is AI?") -> AgentContext:
    ctx = AgentContext(question=question)
    ctx.plan = {"research_type": "RESEARCH", "subquestions": [question]}
    return ctx


def _mock_orchestrator(ctx: AgentContext):
    """Return a mock Orchestrator whose run() methods return the given context."""
    orch = MagicMock()
    orch.run.return_value = ctx
    orch.run_from_goal.return_value = ctx
    orch.run_from_engagement.return_value = ctx
    return orch


# ---------------------------------------------------------------------------
# 1. --save-context appears in CLI help
# ---------------------------------------------------------------------------

class TestCLIHelp:
    def test_save_context_in_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--save-context" in result.output

    def test_save_context_help_text(self):
        result = runner.invoke(app, ["run", "--help"])
        assert "AgentContext" in result.output or "fixture" in result.output


# ---------------------------------------------------------------------------
# 2. Context file created when requested
# ---------------------------------------------------------------------------

class TestContextFileCreated:
    def test_context_file_created_on_success(self, tmp_path):
        ctx = _minimal_ctx()
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            result = runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        assert ctx_path.exists(), f"context file not created; CLI output:\n{result.output}"

    def test_context_file_message_emitted(self, tmp_path):
        ctx = _minimal_ctx()
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            result = runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        assert str(ctx_path) in result.output or "Context saved" in result.output

    def test_context_file_is_valid_json(self, tmp_path):
        ctx = _minimal_ctx()
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        raw = json.loads(ctx_path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict)

    def test_context_file_parent_dirs_created(self, tmp_path):
        ctx = _minimal_ctx()
        ctx_path = tmp_path / "deep" / "nested" / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        assert ctx_path.exists()


# ---------------------------------------------------------------------------
# 3. No file created when option omitted
# ---------------------------------------------------------------------------

class TestNoFileWhenOmitted:
    def test_no_context_file_when_flag_absent(self, tmp_path):
        ctx = _minimal_ctx()
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
            ])

        json_files = list(tmp_path.glob("*.json"))
        assert not json_files, f"Unexpected JSON files: {json_files}"


# ---------------------------------------------------------------------------
# 4. Generated JSON loads via load_context
# ---------------------------------------------------------------------------

class TestLoadContextRoundtrip:
    def test_saved_context_loads_via_load_context(self, tmp_path):
        ctx = _minimal_ctx("Test question for roundtrip")
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "Test question for roundtrip",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        loaded = load_context(ctx_path)
        assert isinstance(loaded, AgentContext)

    def test_loaded_context_preserves_question(self, tmp_path):
        question = "What is the optimal AI infrastructure strategy?"
        ctx = _minimal_ctx(question)
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", question,
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        loaded = load_context(ctx_path)
        assert loaded.question == question

    def test_loaded_context_preserves_plan(self, tmp_path):
        ctx = _minimal_ctx()
        ctx.plan = {
            "research_type": "RESEARCH",
            "subquestions": ["sub-q-1", "sub-q-2"],
            "investigation_areas": [],
        }
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        loaded = load_context(ctx_path)
        assert loaded.plan.get("research_type") == "RESEARCH"
        assert loaded.plan.get("subquestions") == ["sub-q-1", "sub-q-2"]

    def test_serialization_uses_context_to_jsonable(self, tmp_path):
        """Generated JSON keys must match CONTEXT_FIELDS (no extra/missing keys)."""
        from functional_agents.context_snapshot import CONTEXT_FIELDS
        ctx = _minimal_ctx()
        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        raw = json.loads(ctx_path.read_text(encoding="utf-8"))
        assert set(raw.keys()) == CONTEXT_FIELDS


# ---------------------------------------------------------------------------
# 5. Loaded context executes ReportAgent successfully
# ---------------------------------------------------------------------------

class TestReportAgentReplay:
    def test_report_agent_runs_from_saved_context(self, tmp_path):
        """End-to-end: save context from pipeline → replay with ReportAgent."""
        from functional_agents.run_agent import run_agent

        ctx = _minimal_ctx("What is AI infrastructure?")
        ctx.plan = {
            "research_type": "RESEARCH",
            "subquestions": ["What is AI?"],
            "investigation_areas": [],
        }
        ctx.evidence_notes = [{
            "evidence_items": [{"claim": "AI requires GPUs.", "evidence_id": "E001"}],
            "evidence_summary": {"total_evidence_items": 1},
            "coverage_by_subquestion": {},
            "evidence_by_subquestion": {},
        }]
        ctx.qa = {
            "qa_summary": {"issues_found": 0},
            "confidence_assessment": {"overall_confidence": "HIGH"},
        }

        ctx_path = tmp_path / "context.json"
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI infrastructure?",
                "--mock",
                "--out", str(report_path),
                "--save-context", str(ctx_path),
            ])

        assert ctx_path.exists()

        result = run_agent(
            "report", ctx_path,
            no_llm=True,
            out_path=tmp_path / "replay_report.md",
        )
        assert result["mini_trace"]["status"] == "success"
        assert result["mini_trace"]["error"] is None

    def test_report_artifact_produced_from_saved_context(self, tmp_path):
        from functional_agents.run_agent import run_agent

        ctx = _minimal_ctx("What is AI infrastructure?")
        ctx.plan = {"research_type": "RESEARCH", "subquestions": ["What is AI?"], "investigation_areas": []}
        ctx.evidence_notes = [{
            "evidence_items": [],
            "evidence_summary": {"total_evidence_items": 0},
            "coverage_by_subquestion": {},
            "evidence_by_subquestion": {},
        }]
        ctx.qa = {
            "qa_summary": {"issues_found": 0},
            "confidence_assessment": {"overall_confidence": "MEDIUM"},
        }

        ctx_path = tmp_path / "context.json"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            runner.invoke(app, [
                "run", "What is AI infrastructure?",
                "--mock",
                "--out", str(tmp_path / "r.md"),
                "--save-context", str(ctx_path),
            ])

        result = run_agent(
            "report", ctx_path,
            no_llm=True,
            out_path=tmp_path / "replay_report.md",
        )
        assert "report_path" in result["mini_trace"].get("objects_produced", []) \
            or result["mini_trace"]["status"] == "success"


# ---------------------------------------------------------------------------
# 6. Existing CLI behaviour unchanged
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_run_without_save_context_still_works(self, tmp_path):
        ctx = _minimal_ctx()
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            result = runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
            ])

        assert result.exit_code == 0

    def test_incremental_mode_ignores_save_context(self, tmp_path):
        """--incremental with no --session should still exit 1 (pre-existing guard)."""
        ctx_path = tmp_path / "context.json"
        result = runner.invoke(app, [
            "run", "--incremental",
            "--save-context", str(ctx_path),
        ])
        assert result.exit_code == 1
        assert not ctx_path.exists()

    def test_existing_flags_still_accepted(self, tmp_path):
        ctx = _minimal_ctx()
        report_path = tmp_path / "report.md"

        with patch("functional_agents.orchestrator.Orchestrator", return_value=_mock_orchestrator(ctx)):
            result = runner.invoke(app, [
                "run", "What is AI?",
                "--mock",
                "--out", str(report_path),
                "--profiles", "ai_data_centers",
                "--top-evidence", "10",
            ])

        assert result.exit_code == 0
