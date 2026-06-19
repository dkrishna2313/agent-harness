"""Tests for dc_power_agent.evaluation (J2.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dc_power_agent.evaluation.loader import (
    QAQuestion,
    ContradictionCase,
    load_qa_questions,
    load_contradiction_cases,
    _str_list,
)
from dc_power_agent.evaluation.scorer import (
    score_qa_response,
    score_contradiction_result,
    _collect_answer_text,
)
from dc_power_agent.evaluation.runner import EvaluationRunner, _compute_aggregates, EvaluationRun
from dc_power_agent.evaluation.report import build_json_report, build_md_report
from dc_power_agent.schemas import (
    ResearchMemo,
    EvidenceItem,
    Contradiction,
    SuppressedComparison,
)


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------

def _write_qa_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_contra_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_qa_questions_basic(tmp_path):
    _write_qa_yaml(
        tmp_path / "nvidia" / "Q001.yaml",
        "question_id: Q001\ndomain: nvidia\ndifficulty: easy\n"
        "question: What is power?\nmust_include:\n  - 120 kW\nmust_not_include:\n  - 30 kW\n",
    )
    questions = load_qa_questions(tmp_path)
    assert len(questions) == 1
    q = questions[0]
    assert q.question_id == "Q001"
    assert q.domain == "nvidia"
    assert q.must_include == ["120 kW"]
    assert q.must_not_include == ["30 kW"]


def test_load_qa_questions_strips_inline_comments(tmp_path):
    _write_qa_yaml(
        tmp_path / "smr" / "S001.yaml",
        "question_id: S001\ndomain: smr\ndifficulty: medium\n"
        "question: Why SMR?\n"
        "must_include:\n  - 300 MWe  # from datasheet\n",
    )
    questions = load_qa_questions(tmp_path)
    assert questions[0].must_include == ["300 MWe"]


def test_load_contradiction_cases_basic(tmp_path):
    _write_contra_yaml(
        tmp_path / "contradictions" / "C001.yaml",
        "contradiction_id: C001\ndomain: smr\n"
        "expected_result: contradiction\ncategory: numeric_conflict\n"
        "claim_a: Alpha is 300 MWe.\nclaim_b: Alpha is 500 MW.\n",
    )
    cases = load_contradiction_cases(tmp_path)
    assert len(cases) == 1
    c = cases[0]
    assert c.contradiction_id == "C001"
    assert c.expected_result == "contradiction"


def test_load_qa_missing_dir_returns_empty(tmp_path):
    questions = load_qa_questions(tmp_path)
    assert questions == []


def test_str_list_none():
    assert _str_list(None) == []


def test_str_list_scalar():
    assert _str_list("hello") == ["hello"]


# ---------------------------------------------------------------------------
# scorer – Q&A
# ---------------------------------------------------------------------------

def _make_memo(text: str, citations: bool = True) -> ResearchMemo:
    citation = " [Source: test.pdf, Evidence: E001]" if citations else ""
    return ResearchMemo(
        title="Test",
        question="test question",
        executive_summary=text + citation,
    )


def _make_question(**kwargs) -> QAQuestion:
    defaults = dict(
        question_id="TEST_001",
        domain="nvidia",
        difficulty="easy",
        question="What is the power?",
        must_include=["120 kW"],
        must_not_include=["30 kW"],
    )
    defaults.update(kwargs)
    return QAQuestion(**defaults)


def test_score_qa_must_include_hit():
    q = _make_question(must_include=["120 kW"])
    memo = _make_memo("The rack requires 120 kW.")
    s = score_qa_response(q, memo)
    assert s.must_include_hits == 1
    assert s.fact_coverage_score == 1.0


def test_score_qa_must_include_miss():
    q = _make_question(must_include=["120 kW", "liquid cooling"])
    memo = _make_memo("The rack has some power.")
    s = score_qa_response(q, memo)
    assert s.must_include_hits == 0
    assert s.fact_coverage_score == 0.0


def test_score_qa_acceptable_alternative_used_when_primary_misses():
    q = _make_question(
        must_include=["120 kW"],
        acceptable_alternatives=["132 kW"],
    )
    memo = _make_memo("Total power is 132 kW.")
    s = score_qa_response(q, memo)
    assert s.must_include_hits == 1


def test_score_qa_must_not_include_violation():
    q = _make_question(must_not_include=["30 kW"])
    memo = _make_memo("Power is 30 kW per shelf.")
    s = score_qa_response(q, memo)
    assert "30 kW" in s.must_not_include_violations
    assert s.hallucination_penalty == 1.0
    assert s.passed is False


def test_score_qa_no_violations_passes():
    q = _make_question(must_include=["120 kW"], must_not_include=["30 kW"])
    memo = _make_memo("The rack requires 120 kW of DC power.")
    s = score_qa_response(q, memo)
    assert s.hallucination_penalty == 0.0
    assert s.passed is True


def test_score_qa_citation_present():
    q = _make_question()
    memo = _make_memo("Power is 120 kW.", citations=True)
    s = score_qa_response(q, memo)
    assert s.citation_count >= 1
    assert s.citation_score == 1.0


def test_score_qa_no_citation():
    q = _make_question()
    memo = _make_memo("Power is 120 kW.", citations=False)
    s = score_qa_response(q, memo)
    assert s.citation_score == 0.0


def test_score_qa_no_must_include_gives_full_coverage():
    q = _make_question(must_include=[])
    memo = _make_memo("Some answer.")
    s = score_qa_response(q, memo)
    assert s.fact_coverage_score == 1.0


# ---------------------------------------------------------------------------
# scorer – contradictions
# ---------------------------------------------------------------------------

def _make_case(**kwargs) -> ContradictionCase:
    defaults = dict(
        contradiction_id="C001",
        domain="smr",
        expected_result="contradiction",
        category="duration_conflict",
        claim_a="Alpha is 24 months.",
        claim_b="Alpha is 144 months.",
    )
    defaults.update(kwargs)
    return ContradictionCase(**defaults)


def _make_contradiction() -> Contradiction:
    return Contradiction(
        contradiction_id="C-1",
        topic="duration",
        evidence_a_id="E001",
        evidence_b_id="E002",
        evidence_a_claim="Alpha is 24 months.",
        evidence_b_claim="Alpha is 144 months.",
        evidence_a_source="s",
        evidence_b_source="s",
        severity="high",
        explanation="Ranges do not overlap.",
    )


def test_contradiction_score_true_positive():
    case = _make_case(expected_result="contradiction")
    s = score_contradiction_result(case, [_make_contradiction()], [])
    assert s.correct is True
    assert s.actual_result == "contradiction"


def test_contradiction_score_true_negative():
    case = _make_case(expected_result="no_contradiction")
    s = score_contradiction_result(case, [], [])
    assert s.correct is True
    assert s.actual_result == "no_contradiction"


def test_contradiction_score_false_positive():
    case = _make_case(expected_result="no_contradiction")
    s = score_contradiction_result(case, [_make_contradiction()], [])
    assert s.correct is False


def test_contradiction_score_suppression_should_fire_but_didnt():
    case = _make_case(
        expected_result="no_contradiction",
        suppression_should_fire=True,
        expected_suppression_reason="scope_mismatch",
    )
    s = score_contradiction_result(case, [], [])
    assert s.suppression_correct is False


def test_contradiction_score_suppression_fired_correctly():
    case = _make_case(
        expected_result="no_contradiction",
        suppression_should_fire=True,
        expected_suppression_reason="scope_mismatch",
    )
    sup = SuppressedComparison(
        evidence_a_id="CA",
        evidence_b_id="CB",
        evidence_a_claim="a",
        evidence_b_claim="b",
        reason="scope_mismatch",
    )
    s = score_contradiction_result(case, [], [sup])
    assert s.suppression_fired is True
    assert s.suppression_correct is True


# ---------------------------------------------------------------------------
# runner – aggregation
# ---------------------------------------------------------------------------

def _qa_score(passed: bool, domain: str = "nvidia", coverage: float = 1.0, citation: float = 1.0):
    from dc_power_agent.evaluation.scorer import QAScore
    s = QAScore(
        question_id="Q",
        domain=domain,
        difficulty="easy",
        question="q",
        fact_coverage_score=coverage,
        citation_score=citation,
        must_include_total=1,
        must_include_hits=1 if passed else 0,
        overall_score=coverage,
        passed=passed,
    )
    return s


def _contra_score(correct: bool, domain: str = "smr"):
    from dc_power_agent.evaluation.scorer import ContradictionScore
    return ContradictionScore(
        contradiction_id="C",
        domain=domain,
        expected_result="contradiction",
        actual_result="contradiction" if correct else "no_contradiction",
        correct=correct,
    )


def test_compute_aggregates_basic():
    run = EvaluationRun(
        qa_scores=[_qa_score(True), _qa_score(False, coverage=0.0)],
        contradiction_scores=[_contra_score(True), _contra_score(False)],
    )
    _compute_aggregates(run)
    assert run.fact_coverage_score == pytest.approx(0.5, abs=0.01)
    assert run.contradiction_accuracy == pytest.approx(0.5, abs=0.01)
    assert run.failed_qa == [run.qa_scores[1]]
    assert run.failed_contradictions == [run.contradiction_scores[1]]


def test_compute_aggregates_no_qa():
    run = EvaluationRun(
        qa_scores=[],
        contradiction_scores=[_contra_score(True)],
    )
    _compute_aggregates(run)
    assert run.contradiction_accuracy == 1.0
    assert run.overall_score > 0


# ---------------------------------------------------------------------------
# runner – end-to-end (mock LLM, no sources dir needed)
# ---------------------------------------------------------------------------

def test_evaluation_runner_end_to_end(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()

    questions = [
        QAQuestion(
            question_id="Q001",
            domain="nvidia",
            difficulty="easy",
            question="What is the rack power?",
            must_include=["120 kW"],
            must_not_include=["30 kW"],
        )
    ]
    cases = [
        ContradictionCase(
            contradiction_id="C001",
            domain="smr",
            expected_result="contradiction",
            category="duration_conflict",
            claim_a="Reactor Alpha construction duration is 24 to 36 months.",
            claim_b="Reactor Alpha construction schedule is estimated at 7 to 12 years.",
            entity="Reactor Alpha",
            scope_a="unit",
            scope_b="unit",
        )
    ]

    runner = EvaluationRunner(sources_dir=sources)
    run = runner.run(questions, cases)

    assert len(run.qa_scores) == 1
    assert len(run.contradiction_scores) == 1
    assert run.contradiction_scores[0].correct is True  # known regression case
    assert 0.0 <= run.overall_score <= 1.0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def test_build_json_report_structure():
    run = EvaluationRun(
        qa_scores=[_qa_score(True), _qa_score(False, coverage=0.0)],
        contradiction_scores=[_contra_score(True)],
    )
    _compute_aggregates(run)
    report = build_json_report(run)

    assert "summary" in report
    assert "domain_scores" in report
    assert "qa_results" in report
    assert "contradiction_results" in report
    assert "failed_tests" in report
    assert report["summary"]["qa_questions_total"] == 2
    assert report["summary"]["contradiction_tests_total"] == 1


def test_build_md_report_contains_sections():
    run = EvaluationRun(
        qa_scores=[_qa_score(True), _qa_score(False, coverage=0.0)],
        contradiction_scores=[_contra_score(True)],
    )
    _compute_aggregates(run)
    md = build_md_report(run)

    assert "# Evaluation Report" in md
    assert "## Summary" in md
    assert "## Domain Scores" in md
    assert "## Q&A Results" in md
    assert "## Contradiction Test Results" in md
    assert "Overall score" in md


def test_build_md_report_failed_section_present_when_failures():
    run = EvaluationRun(
        qa_scores=[_qa_score(False, coverage=0.0)],
        contradiction_scores=[],
    )
    _compute_aggregates(run)
    md = build_md_report(run)
    assert "## Failed Tests" in md


def test_build_md_report_no_failed_section_when_all_pass():
    run = EvaluationRun(
        qa_scores=[_qa_score(True)],
        contradiction_scores=[_contra_score(True)],
    )
    _compute_aggregates(run)
    md = build_md_report(run)
    assert "## Failed Tests" not in md


# ---------------------------------------------------------------------------
# CLI – benchmark subcommand
# ---------------------------------------------------------------------------

def test_benchmark_cli_mock_mode(tmp_path):
    from typer.testing import CliRunner
    from dc_power_agent.eval_runner import app

    # Minimal eval structure
    nvidia_dir = tmp_path / "eval" / "nvidia"
    nvidia_dir.mkdir(parents=True)
    (nvidia_dir / "Q001.yaml").write_text(
        "question_id: Q001\ndomain: nvidia\ndifficulty: easy\n"
        "question: What is the rack power?\nmust_include:\n  - power\n",
        encoding="utf-8",
    )
    contra_dir = tmp_path / "eval" / "contradictions"
    contra_dir.mkdir()
    (contra_dir / "C001.yaml").write_text(
        "contradiction_id: C001\ndomain: smr\nexpected_result: contradiction\n"
        "category: duration_conflict\n"
        "claim_a: Reactor Alpha takes 24 months.\n"
        "claim_b: Reactor Alpha takes 144 months.\n",
        encoding="utf-8",
    )

    sources = tmp_path / "sources"
    sources.mkdir()
    out_dir = tmp_path / "outputs"

    result = CliRunner().invoke(
        app,
        [
            "benchmark",
            "--eval-dir", str(tmp_path / "eval"),
            "--sources", str(sources),
            "--out-dir", str(out_dir),
            "--mock",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (out_dir / "evaluation_report.json").exists()
    assert (out_dir / "evaluation_report.md").exists()
    assert "Evaluation Complete" in result.output
    assert "Overall score:" in result.output
