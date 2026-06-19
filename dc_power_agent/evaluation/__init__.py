"""Automated evaluation framework for the research harness (J2.2)."""

from .loader import load_qa_questions, load_contradiction_cases
from .scorer import score_qa_response, score_contradiction_result
from .runner import EvaluationRunner
from .report import build_json_report, build_md_report, build_trace, write_trace
from .validator import validate_benchmark, ValidationReport, ValidationError
from .regression import (
    compare_reports,
    build_regression_json,
    build_regression_md,
    build_regression_trace,
    write_regression_json,
    write_regression_md,
    write_regression_trace,
    load_report,
    RegressionResult,
    MetricDelta,
    QuestionDiff,
    ContradictionDiff,
)

__all__ = [
    "load_qa_questions",
    "load_contradiction_cases",
    "score_qa_response",
    "score_contradiction_result",
    "EvaluationRunner",
    "build_json_report",
    "build_md_report",
    "build_trace",
    "write_trace",
    "validate_benchmark",
    "ValidationReport",
    "ValidationError",
    "compare_reports",
    "build_regression_json",
    "build_regression_md",
    "build_regression_trace",
    "write_regression_json",
    "write_regression_md",
    "write_regression_trace",
    "load_report",
    "RegressionResult",
    "MetricDelta",
    "QuestionDiff",
    "ContradictionDiff",
]
