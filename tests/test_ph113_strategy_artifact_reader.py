"""PH11.3 — StrategyArtifactReader and CLI strategy commands.

Covers:
- StrategyArtifactReader.load_trace: success, missing file, invalid JSON, invalid payload
- StrategyArtifactReader.load_index: success, missing file, invalid JSON
- StrategyArtifactReader.summarize: all fields present and correct
- StrategyArtifactReader.find_theory: success, blank ID, not found
- StrategyArtifactReader.find_evaluation: success, blank ID, not found
- CLI strategy inspect: text, json, missing, invalid
- CLI strategy theory: text, json, missing, blank ID, not found
- CLI strategy evaluation: text, json, missing, blank ID, not found
- CLI strategy artifacts: text, json, missing, malformed
- CLI strategy --format validation
- Read-only contract: no artifact is written by any CLI command
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from functional_agents.cli import app
from functional_agents.strategy.strategy_artifact_reader import StrategyArtifactReader
from functional_agents.strategy.strategic_position import (
    StrategicExecution,
    StrategicJustification,
    StrategicPosition,
    StrategicRecommendation,
    TheoryOfWinning,
)
from functional_agents.strategy.strategic_choice import StrategicChoice
from functional_agents.strategy.strategic_choice_set import StrategicChoiceSet
from functional_agents.strategy.strategy_lineage import build_strategy_lineage
from functional_agents.strategy.strategy_plan import StrategyPlan
from functional_agents.strategy.strategy_selector import StrategySelection
from functional_agents.strategy.strategy_trace import (
    StrategyTrace,
    write_artifact_index,
    write_strategy_trace,
)
from functional_agents.strategy.theory_evaluation import CriterionScore, TheoryEvaluation


# ---------------------------------------------------------------------------
# Shared trace-building helpers
# ---------------------------------------------------------------------------

def _plan(pid: str = "P-TEST") -> StrategyPlan:
    return StrategyPlan(plan_id=pid, framework="executive", active_dimensions=[])


def _choice_set(sid: str) -> StrategicChoiceSet:
    ch = StrategicChoice(
        id=f"SC-{sid}", dimension="market", selected_value="OPT-A",
        rationale="r", confidence="High", supporting_assumptions=[], requiredness="optional",
    )
    return StrategicChoiceSet(
        id=sid, choices=[ch], overall_confidence="High",
        internal_conflicts=[], completeness=1.0, rationale="r",
    )


def _theory(tid: str, scid: str, oid: str = "OPT-A") -> TheoryOfWinning:
    return TheoryOfWinning(
        theory_id=tid,
        source_choice_set_id=scid,
        recommended_option_id=oid,
        recommended_option_title=f"Option {oid}",
        winning_position="We win by going first.",
        winning_mechanism="Speed-to-market advantage.",
        confidence="High",
        success_conditions=["Market grows > 10%."],
        failure_modes=[{"description": "Competitor enters early."}],
        assumptions=[{"description": "Capital available."}],
        evidence=["E001", "E002"],
    )


def _eval(tid: str, score: float = 0.8) -> TheoryEvaluation:
    return TheoryEvaluation(
        theory_id=tid,
        criteria_scores={
            "feasibility": CriterionScore(score=score, rationale="Looks good.", weight=1.0),
            "impact": CriterionScore(score=score * 0.9, rationale="High impact.", weight=1.5),
        },
        strengths=["Fast.", "Cheap."],
        weaknesses=["Risky."],
        residual_risks=[{"description": "Regulatory headwind."}],
        overall_score=score,
        confidence="High",
        metadata={},
    )


def _selection(winner: str, runner: str | None = None) -> StrategySelection:
    return StrategySelection(
        winner_theory_id=winner,
        winner_score=0.85,
        runner_up_theory_id=runner,
        runner_up_score=0.70 if runner else None,
        score_margin=0.15 if runner else None,
    )


def _position(theory: TheoryOfWinning, pid: str = "SP-TEST") -> StrategicPosition:
    return StrategicPosition(
        position_id=pid,
        created_at="2026-07-26T00:00:00+00:00",
        theory_of_winning=theory,
        recommendation=StrategicRecommendation(
            recommended_option_id=theory.recommended_option_id,
            recommended_option_title=theory.recommended_option_title,
            board_recommendation="Go",
            decision_readiness="Ready",
            overall_confidence="High",
        ),
        justification=StrategicJustification(
            decision_analysis={}, strategic_options=[],
            assumptions=[], risks=[], opportunities=[],
        ),
        execution=StrategicExecution(recommendations=[], validation_priorities=[]),
    )


def _make_trace(
    n: int = 2,
    plan_id: str = "P-TEST",
    research_id: str = "R-TEST",
) -> StrategyTrace:
    plan = _plan(plan_id)
    choice_sets = [_choice_set(f"SCS-{i}") for i in range(n)]
    theories = [_theory(f"TH-SCS-{i}", f"SCS-{i}") for i in range(n)]
    evaluations = [_eval(f"TH-SCS-{i}", 0.9 - i * 0.1) for i in range(n)]
    winner = theories[0]
    runner_id = theories[1].theory_id if n > 1 else None
    sel = _selection(winner.theory_id, runner_id)
    pos = _position(winner)
    trace_id = f"STRAT-{plan.plan_id}"
    lineage = build_strategy_lineage(
        research_id=research_id,
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        trace_id=trace_id,
    )
    return StrategyTrace(
        trace_id=trace_id,
        created_at="2026-07-26T00:00:00+00:00",
        plan=plan,
        choice_sets=choice_sets,
        theories=theories,
        evaluations=evaluations,
        selection=sel,
        strategic_position=pos,
        lineage=lineage,
        metadata={
            "research_id": research_id,
            "plan_id": plan.plan_id,
            "framework": "executive",
            "selected_theory_id": winner.theory_id,
        },
    )


# ---------------------------------------------------------------------------
# TestStrategyArtifactReaderLoadTrace
# ---------------------------------------------------------------------------

class TestStrategyArtifactReaderLoadTrace:
    def test_load_trace_success(self, tmp_path):
        trace = _make_trace()
        trace_path = write_strategy_trace(trace, tmp_path)
        reader = StrategyArtifactReader()
        loaded = reader.load_trace(trace_path)
        assert loaded.trace_id == trace.trace_id
        assert len(loaded.theories) == 2

    def test_load_trace_missing_file_raises(self, tmp_path):
        reader = StrategyArtifactReader()
        with pytest.raises(FileNotFoundError, match="strategy.trace.json"):
            reader.load_trace(tmp_path / "strategy.trace.json")

    def test_load_trace_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "strategy.trace.json"
        bad.write_text("{ not: valid json }", encoding="utf-8")
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.load_trace(bad)

    def test_load_trace_invalid_payload_raises(self, tmp_path):
        bad = tmp_path / "strategy.trace.json"
        bad.write_text(json.dumps({"trace_id": "X"}), encoding="utf-8")
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="Invalid StrategyTrace"):
            reader.load_trace(bad)

    def test_load_trace_returns_canonical_model(self, tmp_path):
        trace = _make_trace()
        trace_path = write_strategy_trace(trace, tmp_path)
        reader = StrategyArtifactReader()
        loaded = reader.load_trace(trace_path)
        assert isinstance(loaded, StrategyTrace)


# ---------------------------------------------------------------------------
# TestStrategyArtifactReaderLoadIndex
# ---------------------------------------------------------------------------

class TestStrategyArtifactReaderLoadIndex:
    def test_load_index_success(self, tmp_path):
        trace = _make_trace()
        trace_path = write_strategy_trace(trace, tmp_path)
        idx_path = write_artifact_index(trace, trace_path, tmp_path)
        reader = StrategyArtifactReader()
        idx = reader.load_index(idx_path)
        assert "entries" in idx
        assert len(idx["entries"]) == 1

    def test_load_index_missing_file_raises(self, tmp_path):
        reader = StrategyArtifactReader()
        with pytest.raises(FileNotFoundError, match="artifact.index.json"):
            reader.load_index(tmp_path / "artifact.index.json")

    def test_load_index_invalid_json_raises(self, tmp_path):
        bad = tmp_path / "artifact.index.json"
        bad.write_text("not json at all", encoding="utf-8")
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            reader.load_index(bad)

    def test_load_index_returns_raw_dict(self, tmp_path):
        trace = _make_trace()
        trace_path = write_strategy_trace(trace, tmp_path)
        idx_path = write_artifact_index(trace, trace_path, tmp_path)
        reader = StrategyArtifactReader()
        idx = reader.load_index(idx_path)
        assert isinstance(idx, dict)

    def test_load_index_does_not_invent_entries(self, tmp_path):
        # An index with no entries should return empty list, not something invented.
        idx_path = tmp_path / "artifact.index.json"
        idx_path.write_text(json.dumps({"schema_version": "v1", "entries": []}), encoding="utf-8")
        reader = StrategyArtifactReader()
        idx = reader.load_index(idx_path)
        assert idx["entries"] == []


# ---------------------------------------------------------------------------
# TestStrategyArtifactReaderSummarize
# ---------------------------------------------------------------------------

class TestStrategyArtifactReaderSummarize:
    def test_summarize_all_keys_present(self, tmp_path):
        trace = _make_trace()
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        expected_keys = {
            "trace_id", "created_at", "framework", "research_id", "plan_id",
            "choice_set_count", "theory_count", "evaluation_count",
            "winner_theory_id", "winner_option_id", "runner_up_theory_id",
            "winner_score", "runner_up_score", "score_margin",
            "tie_breaker_used", "strategic_position_id",
        }
        assert expected_keys == set(summary.keys())

    def test_summarize_trace_id(self):
        trace = _make_trace(plan_id="PLAN-X")
        reader = StrategyArtifactReader()
        assert reader.summarize(trace)["trace_id"] == "STRAT-PLAN-X"

    def test_summarize_research_id(self):
        trace = _make_trace(research_id="R-VERIFY")
        reader = StrategyArtifactReader()
        assert reader.summarize(trace)["research_id"] == "R-VERIFY"

    def test_summarize_counts(self):
        trace = _make_trace(n=3)
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        assert summary["choice_set_count"] == 3
        assert summary["theory_count"] == 3
        assert summary["evaluation_count"] == 3

    def test_summarize_winner_fields(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        assert summary["winner_theory_id"] == "TH-SCS-0"
        assert summary["winner_option_id"] == "OPT-A"
        assert summary["runner_up_theory_id"] == "TH-SCS-1"

    def test_summarize_scores(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        assert summary["winner_score"] == pytest.approx(0.85)
        assert summary["runner_up_score"] == pytest.approx(0.70)
        assert summary["score_margin"] == pytest.approx(0.15)

    def test_summarize_strategic_position_id(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        assert summary["strategic_position_id"] == "SP-TEST"

    def test_summarize_framework_from_metadata(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        summary = reader.summarize(trace)
        assert summary["framework"] == "executive"


# ---------------------------------------------------------------------------
# TestStrategyArtifactReaderFindTheory
# ---------------------------------------------------------------------------

class TestStrategyArtifactReaderFindTheory:
    def test_find_theory_success(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        theory = reader.find_theory(trace, "TH-SCS-0")
        assert theory.theory_id == "TH-SCS-0"
        assert theory.source_choice_set_id == "SCS-0"

    def test_find_theory_second_theory(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        theory = reader.find_theory(trace, "TH-SCS-1")
        assert theory.theory_id == "TH-SCS-1"

    def test_find_theory_blank_id_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="non-empty"):
            reader.find_theory(trace, "")

    def test_find_theory_whitespace_id_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="non-empty"):
            reader.find_theory(trace, "   ")

    def test_find_theory_not_found_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="TH-NONEXISTENT"):
            reader.find_theory(trace, "TH-NONEXISTENT")

    def test_find_theory_not_found_lists_available(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="Available"):
            reader.find_theory(trace, "TH-GHOST")


# ---------------------------------------------------------------------------
# TestStrategyArtifactReaderFindEvaluation
# ---------------------------------------------------------------------------

class TestStrategyArtifactReaderFindEvaluation:
    def test_find_evaluation_success(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        ev = reader.find_evaluation(trace, "TH-SCS-0")
        assert ev.theory_id == "TH-SCS-0"
        assert "feasibility" in ev.criteria_scores

    def test_find_evaluation_blank_id_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="non-empty"):
            reader.find_evaluation(trace, "")

    def test_find_evaluation_whitespace_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="non-empty"):
            reader.find_evaluation(trace, "   ")

    def test_find_evaluation_not_found_raises(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="TH-NONE"):
            reader.find_evaluation(trace, "TH-NONE")

    def test_find_evaluation_not_found_lists_available(self):
        trace = _make_trace(n=2)
        reader = StrategyArtifactReader()
        with pytest.raises(ValueError, match="Available"):
            reader.find_evaluation(trace, "TH-GHOST")


# ---------------------------------------------------------------------------
# CLI helper: runner + trace fixture
# ---------------------------------------------------------------------------

runner = CliRunner()


def _write_trace_fixture(tmp_path: Path, n: int = 2) -> tuple[Path, StrategyTrace]:
    trace = _make_trace(n=n, research_id="R-CLI")
    trace_path = write_strategy_trace(trace, tmp_path)
    return trace_path, trace


def _write_index_fixture(tmp_path: Path) -> tuple[Path, Path]:
    trace_path, trace = _write_trace_fixture(tmp_path)
    idx_path = write_artifact_index(trace, trace_path, tmp_path)
    return trace_path, idx_path


# ---------------------------------------------------------------------------
# CLI: strategy inspect
# ---------------------------------------------------------------------------

class TestCLIStrategyInspect:
    def test_inspect_text_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "inspect", "--trace", str(trace_path)])
        assert result.exit_code == 0, result.output

    def test_inspect_text_contains_trace_id(self, tmp_path):
        trace_path, trace = _write_trace_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "inspect", "--trace", str(trace_path)])
        assert trace.trace_id in result.output

    def test_inspect_text_contains_research_id(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "inspect", "--trace", str(trace_path)])
        assert "R-CLI" in result.output

    def test_inspect_text_contains_winner(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "inspect", "--trace", str(trace_path)])
        assert "TH-SCS-0" in result.output

    def test_inspect_json_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(trace_path), "--format", "json"]
        )
        assert result.exit_code == 0, result.output

    def test_inspect_json_is_valid_json(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(trace_path), "--format", "json"]
        )
        parsed = json.loads(result.output)
        assert "trace_id" in parsed

    def test_inspect_json_no_ansi(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(trace_path), "--format", "json"]
        )
        assert "\x1b[" not in result.output

    def test_inspect_missing_file_exit_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(tmp_path / "strategy.trace.json")]
        )
        assert result.exit_code != 0

    def test_inspect_missing_file_error_message(self, tmp_path):
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(tmp_path / "strategy.trace.json")]
        )
        assert "Error" in result.output or "Error" in (result.stderr or "")

    def test_inspect_invalid_json_exit_nonzero(self, tmp_path):
        bad = tmp_path / "strategy.trace.json"
        bad.write_text("{bad", encoding="utf-8")
        result = runner.invoke(app, ["strategy", "inspect", "--trace", str(bad)])
        assert result.exit_code != 0

    def test_inspect_unsupported_format_exit_nonzero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(trace_path), "--format", "yaml"]
        )
        assert result.exit_code != 0

    def test_inspect_read_only_no_new_files(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        before = set(tmp_path.iterdir())
        runner.invoke(app, ["strategy", "inspect", "--trace", str(trace_path)])
        after = set(tmp_path.iterdir())
        assert after == before


# ---------------------------------------------------------------------------
# CLI: strategy theory
# ---------------------------------------------------------------------------

class TestCLIStrategyTheory:
    def test_theory_text_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-SCS-0"]
        )
        assert result.exit_code == 0, result.output

    def test_theory_text_contains_theory_id(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-SCS-0"]
        )
        assert "TH-SCS-0" in result.output

    def test_theory_text_contains_choice_set_id(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-SCS-0"]
        )
        assert "SCS-0" in result.output

    def test_theory_json_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        assert result.exit_code == 0, result.output

    def test_theory_json_is_valid_json(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        parsed = json.loads(result.output)
        assert parsed["theory_id"] == "TH-SCS-0"

    def test_theory_json_no_ansi(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        assert "\x1b[" not in result.output

    def test_theory_missing_file_exit_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(tmp_path / "strategy.trace.json"),
                  "--theory-id", "TH-SCS-0"]
        )
        assert result.exit_code != 0

    def test_theory_not_found_exit_nonzero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-GHOST"]
        )
        assert result.exit_code != 0

    def test_theory_not_found_names_the_id(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-GHOST"]
        )
        combined = result.output + (result.stderr or "")
        assert "TH-GHOST" in combined

    def test_theory_read_only_no_new_files(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        before = set(tmp_path.iterdir())
        runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path), "--theory-id", "TH-SCS-0"]
        )
        after = set(tmp_path.iterdir())
        assert after == before


# ---------------------------------------------------------------------------
# CLI: strategy evaluation
# ---------------------------------------------------------------------------

class TestCLIStrategyEvaluation:
    def test_evaluation_text_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0"]
        )
        assert result.exit_code == 0, result.output

    def test_evaluation_text_contains_score(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0"]
        )
        assert "0.900" in result.output

    def test_evaluation_text_contains_criteria(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0"]
        )
        assert "feasibility" in result.output

    def test_evaluation_json_exit_zero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        assert result.exit_code == 0, result.output

    def test_evaluation_json_is_valid(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        parsed = json.loads(result.output)
        assert parsed["theory_id"] == "TH-SCS-0"
        assert parsed["overall_score"] == pytest.approx(0.9)

    def test_evaluation_json_no_ansi(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "json"]
        )
        assert "\x1b[" not in result.output

    def test_evaluation_missing_file_exit_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(tmp_path / "strategy.trace.json"),
                  "--theory-id", "TH-SCS-0"]
        )
        assert result.exit_code != 0

    def test_evaluation_not_found_exit_nonzero(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-GHOST"]
        )
        assert result.exit_code != 0

    def test_evaluation_read_only_no_new_files(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        before = set(tmp_path.iterdir())
        runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0"]
        )
        after = set(tmp_path.iterdir())
        assert after == before


# ---------------------------------------------------------------------------
# CLI: strategy artifacts
# ---------------------------------------------------------------------------

class TestCLIStrategyArtifacts:
    def test_artifacts_text_exit_zero(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(idx_path)])
        assert result.exit_code == 0, result.output

    def test_artifacts_text_contains_artifact_type(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(idx_path)])
        assert "strategy_trace" in result.output

    def test_artifacts_text_contains_trace_id(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(idx_path)])
        assert "STRAT-P-TEST" in result.output

    def test_artifacts_text_contains_research_id(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(idx_path)])
        assert "R-CLI" in result.output

    def test_artifacts_json_exit_zero(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "artifacts", "--index", str(idx_path), "--format", "json"]
        )
        assert result.exit_code == 0, result.output

    def test_artifacts_json_is_valid(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "artifacts", "--index", str(idx_path), "--format", "json"]
        )
        parsed = json.loads(result.output)
        assert "entries" in parsed
        assert len(parsed["entries"]) == 1

    def test_artifacts_json_no_ansi(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "artifacts", "--index", str(idx_path), "--format", "json"]
        )
        assert "\x1b[" not in result.output

    def test_artifacts_missing_file_exit_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["strategy", "artifacts", "--index", str(tmp_path / "artifact.index.json")]
        )
        assert result.exit_code != 0

    def test_artifacts_malformed_no_entries_exit_nonzero(self, tmp_path):
        bad = tmp_path / "artifact.index.json"
        bad.write_text(json.dumps({"schema_version": "v1"}), encoding="utf-8")
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(bad)])
        assert result.exit_code != 0

    def test_artifacts_malformed_identifies_path(self, tmp_path):
        bad = tmp_path / "artifact.index.json"
        bad.write_text(json.dumps({"schema_version": "v1", "entries": "not-a-list"}), encoding="utf-8")
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(bad)])
        combined = result.output + (result.stderr or "")
        assert "Error" in combined

    def test_artifacts_read_only_no_new_files(self, tmp_path):
        trace_path, idx_path = _write_index_fixture(tmp_path)
        before = set(tmp_path.iterdir())
        runner.invoke(app, ["strategy", "artifacts", "--index", str(idx_path)])
        after = set(tmp_path.iterdir())
        assert after == before

    def test_artifacts_empty_entries_exit_zero(self, tmp_path):
        empty_idx = tmp_path / "artifact.index.json"
        empty_idx.write_text(
            json.dumps({"schema_version": "v1", "updated_at": "2026-07-26", "entries": []}),
            encoding="utf-8",
        )
        result = runner.invoke(app, ["strategy", "artifacts", "--index", str(empty_idx)])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# CLI: unsupported format handled consistently
# ---------------------------------------------------------------------------

class TestCLIStrategyFormatValidation:
    def test_inspect_yaml_format_rejected(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "inspect", "--trace", str(trace_path), "--format", "yaml"]
        )
        assert result.exit_code != 0

    def test_theory_csv_format_rejected(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "theory", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "csv"]
        )
        assert result.exit_code != 0

    def test_evaluation_xml_format_rejected(self, tmp_path):
        trace_path, _ = _write_trace_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "evaluation", "--trace", str(trace_path),
                  "--theory-id", "TH-SCS-0", "--format", "xml"]
        )
        assert result.exit_code != 0

    def test_artifacts_text_format_rejected(self, tmp_path):
        _, idx_path = _write_index_fixture(tmp_path)
        result = runner.invoke(
            app, ["strategy", "artifacts", "--index", str(idx_path), "--format", "html"]
        )
        assert result.exit_code != 0
