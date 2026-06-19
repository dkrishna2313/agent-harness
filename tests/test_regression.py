"""Tests for dc_power_agent.evaluation.regression (J2.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dc_power_agent.evaluation.regression import (
    compare_reports,
    build_regression_json,
    build_regression_md,
    build_regression_trace,
    write_regression_json,
    write_regression_md,
    load_report,
    RegressionResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_report(
    *,
    overall: float = 0.90,
    coverage: float = 0.85,
    citation: float = 1.0,
    hallucination: float = 0.05,
    contra: float = 0.90,
    qa_passed: int = 20,
    qa_total: int = 23,
    contra_passed: int = 9,
    contra_total: int = 11,
    qa_results: list | None = None,
    contra_results: list | None = None,
) -> dict:
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "summary": {
            "overall_score": overall,
            "fact_coverage_score": coverage,
            "citation_score": citation,
            "hallucination_rate": hallucination,
            "contradiction_accuracy": contra,
            "qa_questions_passed": qa_passed,
            "qa_questions_total": qa_total,
            "contradiction_tests_passed": contra_passed,
            "contradiction_tests_total": contra_total,
        },
        "qa_results": qa_results or [],
        "contradiction_results": contra_results or [],
    }


def _qa_result(qid: str, passed: bool, domain: str = "nvidia", difficulty: str = "easy", coverage: float = 1.0) -> dict:
    return {
        "question_id": qid,
        "domain": domain,
        "difficulty": difficulty,
        "passed": passed,
        "fact_coverage_score": coverage,
    }


def _contra_result(cid: str, correct: bool, domain: str = "smr", expected: str = "contradiction") -> dict:
    return {
        "contradiction_id": cid,
        "domain": domain,
        "expected_result": expected,
        "correct": correct,
    }


# ---------------------------------------------------------------------------
# compare_reports – metric deltas
# ---------------------------------------------------------------------------

def test_pass_when_no_change():
    baseline = _make_report()
    current = _make_report()
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is True
    assert result.fail_reasons == []


def test_fail_when_overall_drops_beyond_threshold():
    baseline = _make_report(overall=0.90)
    current = _make_report(overall=0.85)   # -0.05, threshold 0.03
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is False
    assert any("Overall" in r for r in result.fail_reasons)


def test_pass_when_overall_drops_within_threshold():
    baseline = _make_report(overall=0.90)
    current = _make_report(overall=0.88)   # -0.02, threshold 0.03 → within tolerance
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is True


def test_fail_when_hallucination_increases_beyond_threshold():
    baseline = _make_report(hallucination=0.05)
    current = _make_report(hallucination=0.10)   # +0.05, threshold 0.03
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is False
    assert any("Hallucination" in r for r in result.fail_reasons)


def test_pass_when_hallucination_increases_within_threshold():
    baseline = _make_report(hallucination=0.05)
    current = _make_report(hallucination=0.07)   # +0.02, within 0.03
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is True


def test_metric_improvement_noted():
    baseline = _make_report(overall=0.85)
    current = _make_report(overall=0.92)   # +0.07 → improved
    result = compare_reports(current, baseline, fail_threshold=0.03)
    assert result.passed is True
    overall = next(m for m in result.metric_deltas if m.metric == "overall_score")
    assert overall.status == "improved"
    assert result.improvements


def test_all_metrics_present():
    result = compare_reports(_make_report(), _make_report())
    keys = {m.metric for m in result.metric_deltas}
    expected = {
        "overall_score", "fact_coverage_score", "citation_score",
        "hallucination_rate", "contradiction_accuracy",
        "qa_questions_passed", "contradiction_tests_passed",
    }
    assert expected <= keys


def test_metric_delta_fields():
    baseline = _make_report(overall=0.80)
    current = _make_report(overall=0.90)
    result = compare_reports(current, baseline)
    m = next(x for x in result.metric_deltas if x.metric == "overall_score")
    assert m.baseline == pytest.approx(0.80)
    assert m.current == pytest.approx(0.90)
    assert m.delta == pytest.approx(0.10)
    assert m.higher_is_better is True


# ---------------------------------------------------------------------------
# compare_reports – question diffs
# ---------------------------------------------------------------------------

def test_question_regression_detected():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    result = compare_reports(current, baseline)
    assert result.regressed_questions
    assert result.regressed_questions[0].question_id == "Q001"
    assert result.regressed_questions[0].status == "regressed"


def test_question_improvement_detected():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    result = compare_reports(current, baseline)
    assert result.improved_questions
    assert result.improved_questions[0].status == "improved"


def test_question_unchanged_pass():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    result = compare_reports(current, baseline)
    q = result.question_diffs[0]
    assert q.status == "unchanged_pass"


def test_question_unchanged_fail():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    result = compare_reports(current, baseline)
    q = result.question_diffs[0]
    assert q.status == "unchanged_fail"


def test_new_question_detected():
    baseline = _make_report(qa_results=[])
    current = _make_report(qa_results=[_qa_result("Q_NEW", passed=True)])
    result = compare_reports(current, baseline)
    assert result.question_diffs[0].status == "new"


def test_removed_question_detected():
    baseline = _make_report(qa_results=[_qa_result("Q_OLD", passed=True)])
    current = _make_report(qa_results=[])
    result = compare_reports(current, baseline)
    assert result.question_diffs[0].status == "removed"


def test_question_regression_adds_fail_reason():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    result = compare_reports(current, baseline)
    assert result.passed is False
    assert any("Q001" in r for r in result.fail_reasons)


# ---------------------------------------------------------------------------
# compare_reports – contradiction diffs
# ---------------------------------------------------------------------------

def test_contradiction_regression_detected():
    baseline = _make_report(contra_results=[_contra_result("C001", correct=True)])
    current = _make_report(contra_results=[_contra_result("C001", correct=False)])
    result = compare_reports(current, baseline)
    assert result.regressed_contradictions
    assert result.regressed_contradictions[0].contradiction_id == "C001"
    assert result.regressed_contradictions[0].status == "regressed"


def test_contradiction_improvement_detected():
    baseline = _make_report(contra_results=[_contra_result("C001", correct=False)])
    current = _make_report(contra_results=[_contra_result("C001", correct=True)])
    result = compare_reports(current, baseline)
    assert result.improved_contradictions
    assert result.improved_contradictions[0].status == "improved"


def test_contradiction_regression_adds_fail_reason():
    baseline = _make_report(contra_results=[_contra_result("C001", correct=True)])
    current = _make_report(contra_results=[_contra_result("C001", correct=False)])
    result = compare_reports(current, baseline)
    assert result.passed is False
    assert any("C001" in r for r in result.fail_reasons)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def test_regression_json_structure():
    result = compare_reports(_make_report(), _make_report())
    report = build_regression_json(result)
    assert "status" in report
    assert "metric_deltas" in report
    assert "question_diffs" in report
    assert "contradiction_diffs" in report
    assert "regressions" in report
    assert "improvements" in report
    assert report["passed"] is True
    assert report["status"] == "pass"


def test_regression_json_fail_status():
    baseline = _make_report(overall=0.90)
    current = _make_report(overall=0.80)
    result = compare_reports(current, baseline, fail_threshold=0.03)
    report = build_regression_json(result)
    assert report["status"] == "fail"
    assert report["passed"] is False


def test_regression_md_contains_key_sections():
    result = compare_reports(_make_report(), _make_report())
    md = build_regression_md(result)
    assert "# Regression Report" in md
    assert "## Status:" in md
    assert "## Metric Deltas" in md
    assert "PASS" in md


def test_regression_md_shows_fail():
    baseline = _make_report(overall=0.90)
    current = _make_report(overall=0.80)
    result = compare_reports(current, baseline, fail_threshold=0.03)
    md = build_regression_md(result)
    assert "FAIL" in md


def test_regression_md_shows_question_changes():
    baseline = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    result = compare_reports(current, baseline)
    md = build_regression_md(result)
    assert "Q001" in md
    assert "regressed" in md


def test_write_regression_json(tmp_path):
    result = compare_reports(_make_report(), _make_report())
    out = write_regression_json(result, tmp_path / "regression_report.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert "metric_deltas" in data


def test_write_regression_md(tmp_path):
    result = compare_reports(_make_report(), _make_report())
    out = write_regression_md(result, tmp_path / "regression_report.md")
    assert out.exists()
    assert "# Regression Report" in out.read_text()


def test_load_report_roundtrip(tmp_path):
    report = _make_report(overall=0.91)
    p = tmp_path / "report.json"
    p.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_report(p)
    assert loaded["summary"]["overall_score"] == pytest.approx(0.91)


def test_load_report_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def _write_report(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_make_report(**kwargs)), encoding="utf-8")


def test_save_baseline_cli(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    report_path = tmp_path / "outputs" / "evaluation_report.json"
    _write_report(report_path, overall=0.91)
    baseline_dir = tmp_path / "baseline"

    result = CliRunner().invoke(
        app,
        ["save-baseline", "--report", str(report_path), "--baseline-dir", str(baseline_dir)],
    )
    assert result.exit_code == 0, result.output
    assert (baseline_dir / "evaluation_report.json").exists()
    meta = json.loads((baseline_dir / "baseline_metadata.json").read_text())
    assert meta["overall_score"] == pytest.approx(0.91)
    assert "created_at" in meta
    assert "source_report" in meta


def test_regress_cli_pass(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    current_path = tmp_path / "outputs" / "evaluation_report.json"
    baseline_path = tmp_path / "baseline" / "evaluation_report.json"
    _write_report(current_path, overall=0.91)
    _write_report(baseline_path, overall=0.90)

    result = CliRunner().invoke(
        app,
        [
            "regress",
            "--current", str(current_path),
            "--baseline", str(baseline_path),
            "--out-dir", str(tmp_path / "outputs"),
            "--fail-threshold", "0.03",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output
    assert (tmp_path / "outputs" / "regression_report.json").exists()
    assert (tmp_path / "outputs" / "regression_report.md").exists()
    assert (tmp_path / "outputs" / "regression.trace.json").exists()


def test_regress_cli_fail_exits_1(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    current_path = tmp_path / "outputs" / "evaluation_report.json"
    baseline_path = tmp_path / "baseline" / "evaluation_report.json"
    _write_report(current_path, overall=0.80)   # big drop
    _write_report(baseline_path, overall=0.91)

    result = CliRunner().invoke(
        app,
        [
            "regress",
            "--current", str(current_path),
            "--baseline", str(baseline_path),
            "--out-dir", str(tmp_path / "outputs"),
            "--fail-threshold", "0.03",
        ],
    )
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_regress_cli_question_regression_exits_1(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    current_path = tmp_path / "outputs" / "evaluation_report.json"
    baseline_path = tmp_path / "baseline" / "evaluation_report.json"

    baseline = _make_report(qa_results=[_qa_result("Q001", passed=True)])
    current = _make_report(qa_results=[_qa_result("Q001", passed=False)])
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(current))
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    baseline_path.write_text(json.dumps(baseline))

    result = CliRunner().invoke(
        app,
        [
            "regress",
            "--current", str(current_path),
            "--baseline", str(baseline_path),
            "--out-dir", str(tmp_path / "outputs"),
        ],
    )
    assert result.exit_code == 1
    assert "Q001" in result.output


def test_regress_cli_custom_threshold(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    current_path = tmp_path / "current.json"
    baseline_path = tmp_path / "baseline.json"
    # -0.05 drop; passes with threshold=0.10 but fails with threshold=0.03
    _write_report(current_path, overall=0.85)
    _write_report(baseline_path, overall=0.90)

    result = CliRunner().invoke(
        app,
        [
            "regress",
            "--current", str(current_path),
            "--baseline", str(baseline_path),
            "--out-dir", str(tmp_path / "out"),
            "--fail-threshold", "0.10",
        ],
    )
    assert result.exit_code == 0, result.output  # 0.05 drop < 0.10 threshold
