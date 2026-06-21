"""J5.7 – Per-agent evaluation scoring tests.

Covers:
  1. score_agents() returns AgentScores with expected fields
  2. Planner score uses coverage_matrix correctly
  3. Evidence score uses item count + quality
  4. QA score uses research_gaps count
  5. Report score uses citation_count + confirmed_facts
  6. aggregate_agent_scores() returns correct means
  7. EvaluationRun has agent_scores + aggregate fields
  8. build_json_report() includes agent_evaluation block with aggregate + per_question
  9. Regression _COMPARABLE_METRICS includes four agent metrics
 10. Agent scores propagate into regression summary comparison
"""

from __future__ import annotations

import pytest

from research_agent.evaluation.agent_scorer import (
    AgentScores,
    aggregate_agent_scores,
    score_agents,
)
from research_agent.evaluation.regression import _COMPARABLE_METRICS
from research_agent.evaluation.runner import EvaluationRun
from research_agent.evaluation.report import build_json_report


# ---------------------------------------------------------------------------
# Minimal stub helpers
# ---------------------------------------------------------------------------

def _make_coverage_area(topic: str, level: str, evidence_count: int = 1):
    from research_agent.schemas import CoverageArea
    return CoverageArea(
        topic=topic,
        evidence_count=evidence_count,
        source_count=1,
        coverage_level=level,
        rationale="test",
    )


def _make_evidence_item(source_doc: str, overall_score: float = 4.0):
    from research_agent.schemas import EvidenceItem
    return EvidenceItem(
        evidence_id="E001",
        claim="test claim",
        source_document=source_doc,
        evidence_snippet="snippet",
        category="power",
        relevance="direct",
        confidence="high",
        entity="TestCo",
        scope="global",
        overall_score=overall_score,
    )


def _make_research_gap(gap_id: str):
    from research_agent.schemas import ResearchGap
    return ResearchGap(
        gap_id=gap_id,
        topic="test",
        priority="high",
        description="gap",
        rationale="missing data",
    )


def _make_memo(
    coverage_areas=None,
    evidence_items=None,
    research_gaps=None,
    confirmed_facts=None,
):
    from research_agent.schemas import ResearchMemo
    return ResearchMemo(
        title="Test Memo",
        question="What is the test question?",
        executive_summary="summary",
        source_notes=evidence_items or [],
        coverage_matrix=coverage_areas or [],
        research_gaps=research_gaps or [],
        confirmed_facts=confirmed_facts or [],
    )


def _make_qa_score(citation_count: int = 2):
    from research_agent.evaluation.scorer import QAScore
    s = QAScore(
        question_id="Q1",
        domain="test",
        difficulty="medium",
        question="?",
        citation_count=citation_count,
        citation_score=1.0 if citation_count > 0 else 0.0,
    )
    return s


# ---------------------------------------------------------------------------
# 1. score_agents returns AgentScores with expected fields
# ---------------------------------------------------------------------------

def test_score_agents_returns_agent_scores():
    memo = _make_memo()
    qa_score = _make_qa_score()
    result = score_agents("Q1", "test", memo, qa_score)
    assert isinstance(result, AgentScores)
    assert result.question_id == "Q1"
    assert result.domain == "test"
    for field in ("planner_score", "evidence_score", "qa_score", "report_score"):
        assert 0.0 <= getattr(result, field) <= 1.0, f"{field} out of range"


# ---------------------------------------------------------------------------
# 2. Planner score — coverage_matrix
# ---------------------------------------------------------------------------

def test_planner_score_full_coverage():
    areas = [
        _make_coverage_area("Power", "strong"),
        _make_coverage_area("Cooling", "moderate"),
        _make_coverage_area("Grid", "strong"),
    ]
    memo = _make_memo(coverage_areas=areas)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.investigation_area_count == 3
    assert result.investigation_areas_covered == 3
    assert result.planner_score > 0.8


def test_planner_score_partial_coverage():
    areas = [
        _make_coverage_area("Power", "strong"),
        _make_coverage_area("Cooling", "none"),
        _make_coverage_area("Grid", "none"),
    ]
    memo = _make_memo(coverage_areas=areas)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.investigation_areas_covered == 1
    assert result.planner_score < 0.7


def test_planner_score_neutral_when_no_matrix():
    memo = _make_memo(coverage_areas=[])
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.planner_score == 0.5


# ---------------------------------------------------------------------------
# 3. Evidence score — count + quality
# ---------------------------------------------------------------------------

def test_evidence_score_high_quality():
    items = [_make_evidence_item(f"doc{i}.pdf", overall_score=4.5) for i in range(20)]
    memo = _make_memo(evidence_items=items)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.evidence_count == 20
    assert result.high_quality_evidence == 20
    assert result.source_diversity == 20
    assert result.evidence_score > 0.9


def test_evidence_score_low_count():
    items = [_make_evidence_item("doc.pdf", overall_score=2.0)]
    memo = _make_memo(evidence_items=items)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.evidence_count == 1
    assert result.evidence_score < 0.15


def test_evidence_score_zero_items():
    memo = _make_memo(evidence_items=[])
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.evidence_count == 0
    assert result.evidence_score == 0.0


# ---------------------------------------------------------------------------
# 4. QA score — gap detection
# ---------------------------------------------------------------------------

def test_qa_score_many_gaps():
    gaps = [_make_research_gap(f"G{i:03d}") for i in range(5)]
    memo = _make_memo(research_gaps=gaps)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.gaps_identified == 5
    assert result.qa_score == 1.0


def test_qa_score_no_gaps():
    memo = _make_memo(research_gaps=[])
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.gaps_identified == 0
    assert result.qa_score == 0.0


def test_qa_score_partial_gaps():
    gaps = [_make_research_gap(f"G{i}") for i in range(2)]
    memo = _make_memo(research_gaps=gaps)
    result = score_agents("Q1", "test", memo, _make_qa_score())
    assert result.qa_score == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# 5. Report score — citations + confirmed facts
# ---------------------------------------------------------------------------

def test_report_score_full():
    facts = [f"Fact {i}" for i in range(10)]
    memo = _make_memo(confirmed_facts=facts)
    qa_score = _make_qa_score(citation_count=3)
    result = score_agents("Q1", "test", memo, qa_score)
    assert result.confirmed_facts == 10
    assert result.citation_count == 3
    assert result.report_score == 1.0


def test_report_score_no_citations():
    facts = ["Fact A"] * 5
    memo = _make_memo(confirmed_facts=facts)
    qa_score = _make_qa_score(citation_count=0)
    result = score_agents("Q1", "test", memo, qa_score)
    assert result.report_citation_score == 0.0
    assert result.report_score == pytest.approx(0.25)  # 0.5*0.0 + 0.5*0.5


# ---------------------------------------------------------------------------
# 6. aggregate_agent_scores
# ---------------------------------------------------------------------------

def test_aggregate_empty():
    agg = aggregate_agent_scores([])
    assert agg["planner_score"] == 0.0
    assert agg["evidence_score"] == 0.0
    assert agg["qa_score"] == 0.0
    assert agg["report_score"] == 0.0


def test_aggregate_means():
    s1 = AgentScores(question_id="Q1", domain="x", planner_score=0.8, evidence_score=0.6, qa_score=0.4, report_score=1.0)
    s2 = AgentScores(question_id="Q2", domain="x", planner_score=0.4, evidence_score=0.2, qa_score=0.8, report_score=0.6)
    agg = aggregate_agent_scores([s1, s2])
    assert agg["planner_score"] == pytest.approx(0.6)
    assert agg["evidence_score"] == pytest.approx(0.4)
    assert agg["qa_score"] == pytest.approx(0.6)
    assert agg["report_score"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 7. EvaluationRun has agent_scores + aggregate fields
# ---------------------------------------------------------------------------

def test_evaluation_run_has_agent_fields():
    run = EvaluationRun()
    assert hasattr(run, "agent_scores")
    assert isinstance(run.agent_scores, list)
    for field in ("planner_score", "evidence_score", "qa_agent_score", "report_score"):
        assert hasattr(run, field), f"EvaluationRun missing {field}"
        assert getattr(run, field) == 0.0


# ---------------------------------------------------------------------------
# 8. build_json_report includes agent_evaluation block
# ---------------------------------------------------------------------------

def test_json_report_agent_evaluation_block():
    run = EvaluationRun()
    run.agent_scores = [
        AgentScores(question_id="Q1", domain="d", planner_score=0.9, evidence_score=0.8, qa_score=0.7, report_score=0.85),
    ]
    run.planner_score = 0.9
    run.evidence_score = 0.8
    run.qa_agent_score = 0.7
    run.report_score = 0.85

    report = build_json_report(run)
    assert "agent_evaluation" in report
    ae = report["agent_evaluation"]
    assert "aggregate" in ae
    assert "per_question" in ae

    agg = ae["aggregate"]
    assert agg["planner_score"] == 0.9
    assert agg["evidence_score"] == 0.8
    assert agg["qa_score"] == 0.7
    assert agg["report_score"] == 0.85

    pq = ae["per_question"]
    assert len(pq) == 1
    assert pq[0]["question_id"] == "Q1"
    assert pq[0]["planner_score"] == 0.9
    assert "detail" in pq[0]
    assert "planner" in pq[0]["detail"]
    assert "evidence" in pq[0]["detail"]
    assert "qa" in pq[0]["detail"]
    assert "report" in pq[0]["detail"]


def test_json_report_summary_includes_agent_scores():
    run = EvaluationRun()
    run.planner_score = 0.75
    run.evidence_score = 0.65
    run.qa_agent_score = 0.55
    run.report_score = 0.85

    report = build_json_report(run)
    summary = report["summary"]
    assert summary["planner_score"] == 0.75
    assert summary["evidence_score"] == 0.65
    assert summary["qa_agent_score"] == 0.55
    assert summary["report_score"] == 0.85


# ---------------------------------------------------------------------------
# 9. Regression _COMPARABLE_METRICS includes agent metrics
# ---------------------------------------------------------------------------

def test_regression_metrics_include_agent_scores():
    metric_keys = {m[0] for m in _COMPARABLE_METRICS}
    for key in ("planner_score", "evidence_score", "qa_agent_score", "report_score"):
        assert key in metric_keys, f"regression missing metric: {key}"


def test_regression_agent_metrics_higher_is_better():
    agent_metrics = {m[0]: m[2] for m in _COMPARABLE_METRICS}
    for key in ("planner_score", "evidence_score", "qa_agent_score", "report_score"):
        assert agent_metrics[key] is True, f"{key} should be higher_is_better=True"


# ---------------------------------------------------------------------------
# 10. Regression reads agent scores from summary
# ---------------------------------------------------------------------------

def test_regression_compares_agent_scores():
    from research_agent.evaluation.regression import compare_reports

    baseline = {
        "summary": {
            "overall_score": 0.8,
            "fact_coverage_score": 0.75,
            "citation_score": 0.9,
            "hallucination_rate": 0.0,
            "contradiction_accuracy": 1.0,
            "qa_questions_passed": 5,
            "contradiction_tests_passed": 3,
            "planner_score": 0.70,
            "evidence_score": 0.65,
            "qa_agent_score": 0.60,
            "report_score": 0.80,
        },
        "qa_results": [],
        "contradiction_results": [],
    }
    current = {
        "summary": {
            **baseline["summary"],
            "planner_score": 0.85,   # improved
            "evidence_score": 0.50,  # regressed (delta = -0.15 > threshold 0.03)
        },
        "qa_results": [],
        "contradiction_results": [],
    }
    result = compare_reports(current, baseline, fail_threshold=0.03)

    # evidence_score regressed — should appear in regressions
    regression_keys = {m.metric for m in result.regressions}
    assert "evidence_score" in regression_keys

    # planner_score improved — should appear in improvements
    improvement_keys = {m.metric for m in result.improvements}
    assert "planner_score" in improvement_keys
