"""Regression comparison between two evaluation reports (J2.3).

Compares a current ``evaluation_report.json`` against a baseline to detect
score regressions and question-level changes.  Pure data comparison — does
not touch retrieval, contradiction detection, or scoring logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

MetricStatus = Literal["pass", "fail", "improved"]
ItemStatus = Literal[
    "unchanged_pass",
    "unchanged_fail",
    "improved",
    "regressed",
    "new",
    "removed",
]

_COMPARABLE_METRICS: list[tuple[str, str, bool]] = [
    # (key_in_summary, display_name, higher_is_better)
    ("overall_score",            "Overall score",           True),
    ("fact_coverage_score",      "Fact coverage",           True),
    ("citation_score",           "Citation coverage",       True),
    ("hallucination_rate",       "Hallucination rate",      False),
    ("contradiction_accuracy",   "Contradiction accuracy",  True),
    ("qa_questions_passed",      "Q&A passed (count)",      True),
    ("contradiction_tests_passed", "Contradiction passed (count)", True),
    # J5.7 — per-agent scores (sourced from agent_evaluation.aggregate in report)
    ("planner_score",   "Planner score",  True),
    ("evidence_score",  "Evidence score", True),
    ("qa_agent_score",  "QA agent score", True),
    ("report_score",    "Report score",   True),
]


@dataclass(frozen=True)
class MetricDelta:
    """Comparison result for one summary metric."""

    metric: str
    display_name: str
    baseline: float
    current: float
    delta: float
    pct_delta: float          # relative change (current - baseline) / baseline
    higher_is_better: bool
    status: MetricStatus      # "pass" | "fail" | "improved"
    threshold: float          # absolute threshold used to decide pass/fail


@dataclass(frozen=True)
class QuestionDiff:
    """Per-question pass/fail change."""

    question_id: str
    domain: str
    difficulty: str
    baseline_passed: bool | None
    current_passed: bool | None
    baseline_coverage: float
    current_coverage: float
    status: ItemStatus


@dataclass(frozen=True)
class ContradictionDiff:
    """Per-contradiction-test correct/incorrect change."""

    contradiction_id: str
    domain: str
    expected_result: str
    baseline_correct: bool | None
    current_correct: bool | None
    status: ItemStatus


@dataclass
class RegressionResult:
    """Full comparison between a current and a baseline evaluation report."""

    # Metadata
    current_generated_at: str = ""
    baseline_generated_at: str = ""
    fail_threshold: float = 0.03

    # Metric-level comparisons
    metric_deltas: list[MetricDelta] = field(default_factory=list)

    # Item-level diffs
    question_diffs: list[QuestionDiff] = field(default_factory=list)
    contradiction_diffs: list[ContradictionDiff] = field(default_factory=list)

    # Aggregated change lists
    regressions: list[MetricDelta] = field(default_factory=list)
    improvements: list[MetricDelta] = field(default_factory=list)
    regressed_questions: list[QuestionDiff] = field(default_factory=list)
    improved_questions: list[QuestionDiff] = field(default_factory=list)
    regressed_contradictions: list[ContradictionDiff] = field(default_factory=list)
    improved_contradictions: list[ContradictionDiff] = field(default_factory=list)

    # Overall verdict
    passed: bool = True
    fail_reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def compare_reports(
    current: dict,
    baseline: dict,
    *,
    fail_threshold: float = 0.03,
) -> RegressionResult:
    """Compare two loaded evaluation-report dicts and return a RegressionResult."""

    result = RegressionResult(
        current_generated_at=current.get("generated_at", ""),
        baseline_generated_at=baseline.get("generated_at", ""),
        fail_threshold=fail_threshold,
    )

    cur_summary = current.get("summary", {})
    base_summary = baseline.get("summary", {})

    # Metric deltas
    for key, display, higher_is_better in _COMPARABLE_METRICS:
        cur_val = float(cur_summary.get(key, 0))
        base_val = float(base_summary.get(key, 0))
        delta = cur_val - base_val
        pct_delta = (delta / base_val) if base_val != 0 else 0.0

        # Determine status
        degraded = (
            (higher_is_better and delta < -fail_threshold)
            or (not higher_is_better and delta > fail_threshold)
        )
        improved = (
            (higher_is_better and delta > fail_threshold)
            or (not higher_is_better and delta < -fail_threshold)
        )
        status: MetricStatus = "fail" if degraded else ("improved" if improved else "pass")

        md = MetricDelta(
            metric=key,
            display_name=display,
            baseline=base_val,
            current=cur_val,
            delta=delta,
            pct_delta=pct_delta,
            higher_is_better=higher_is_better,
            status=status,
            threshold=fail_threshold,
        )
        result.metric_deltas.append(md)
        if status == "fail":
            result.regressions.append(md)
            result.fail_reasons.append(
                f"{display}: {base_val:.4f} → {cur_val:.4f} "
                f"(Δ{delta:+.4f}, threshold ±{fail_threshold:.3f})"
            )
        elif status == "improved":
            result.improvements.append(md)

    # Question-level diffs
    base_qa = {r["question_id"]: r for r in baseline.get("qa_results", [])}
    cur_qa = {r["question_id"]: r for r in current.get("qa_results", [])}

    all_qids = sorted(set(base_qa) | set(cur_qa))
    for qid in all_qids:
        b = base_qa.get(qid)
        c = cur_qa.get(qid)
        if b is None:
            diff = QuestionDiff(
                question_id=qid,
                domain=c.get("domain", ""),
                difficulty=c.get("difficulty", ""),
                baseline_passed=None,
                current_passed=c.get("passed"),
                baseline_coverage=0.0,
                current_coverage=float(c.get("fact_coverage_score", 0)),
                status="new",
            )
        elif c is None:
            diff = QuestionDiff(
                question_id=qid,
                domain=b.get("domain", ""),
                difficulty=b.get("difficulty", ""),
                baseline_passed=b.get("passed"),
                current_passed=None,
                baseline_coverage=float(b.get("fact_coverage_score", 0)),
                current_coverage=0.0,
                status="removed",
            )
        else:
            bp, cp = b.get("passed", False), c.get("passed", False)
            if not bp and cp:
                item_status: ItemStatus = "improved"
            elif bp and not cp:
                item_status = "regressed"
            elif bp and cp:
                item_status = "unchanged_pass"
            else:
                item_status = "unchanged_fail"
            diff = QuestionDiff(
                question_id=qid,
                domain=c.get("domain", b.get("domain", "")),
                difficulty=c.get("difficulty", b.get("difficulty", "")),
                baseline_passed=bp,
                current_passed=cp,
                baseline_coverage=float(b.get("fact_coverage_score", 0)),
                current_coverage=float(c.get("fact_coverage_score", 0)),
                status=item_status,
            )
        result.question_diffs.append(diff)
        if diff.status == "regressed":
            result.regressed_questions.append(diff)
        elif diff.status == "improved":
            result.improved_questions.append(diff)

    # Contradiction-level diffs
    base_contra = {r["contradiction_id"]: r for r in baseline.get("contradiction_results", [])}
    cur_contra = {r["contradiction_id"]: r for r in current.get("contradiction_results", [])}

    all_cids = sorted(set(base_contra) | set(cur_contra))
    for cid in all_cids:
        b = base_contra.get(cid)
        c = cur_contra.get(cid)
        if b is None:
            cdiff = ContradictionDiff(
                contradiction_id=cid,
                domain=c.get("domain", ""),
                expected_result=c.get("expected_result", ""),
                baseline_correct=None,
                current_correct=c.get("correct"),
                status="new",
            )
        elif c is None:
            cdiff = ContradictionDiff(
                contradiction_id=cid,
                domain=b.get("domain", ""),
                expected_result=b.get("expected_result", ""),
                baseline_correct=b.get("correct"),
                current_correct=None,
                status="removed",
            )
        else:
            bc, cc = b.get("correct", False), c.get("correct", False)
            if not bc and cc:
                c_status: ItemStatus = "improved"
            elif bc and not cc:
                c_status = "regressed"
            elif bc and cc:
                c_status = "unchanged_pass"
            else:
                c_status = "unchanged_fail"
            cdiff = ContradictionDiff(
                contradiction_id=cid,
                domain=c.get("domain", b.get("domain", "")),
                expected_result=c.get("expected_result", b.get("expected_result", "")),
                baseline_correct=bc,
                current_correct=cc,
                status=c_status,
            )
        result.contradiction_diffs.append(cdiff)
        if cdiff.status == "regressed":
            result.regressed_contradictions.append(cdiff)
        elif cdiff.status == "improved":
            result.improved_contradictions.append(cdiff)

    # Also fail if any questions or contradictions regressed
    if result.regressed_questions:
        for q in result.regressed_questions:
            result.fail_reasons.append(
                f"Q&A regressed: {q.question_id} (was passing, now failing)"
            )
    if result.regressed_contradictions:
        for c in result.regressed_contradictions:
            result.fail_reasons.append(
                f"Contradiction regressed: {c.contradiction_id} (was correct, now incorrect)"
            )

    result.passed = len(result.fail_reasons) == 0
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_regression_json(result: RegressionResult, *, run_meta: dict | None = None) -> dict:
    ts = datetime.now(timezone.utc).isoformat()
    return {
        "generated_at": ts,
        "regression_type": "evaluation_regression",
        **(run_meta or {}),
        "status": "pass" if result.passed else "fail",
        "passed": result.passed,
        "fail_reasons": result.fail_reasons,
        "fail_threshold": result.fail_threshold,
        "baseline_generated_at": result.baseline_generated_at,
        "current_generated_at": result.current_generated_at,
        "metric_deltas": [
            {
                "metric": m.metric,
                "display_name": m.display_name,
                "baseline": m.baseline,
                "current": m.current,
                "delta": round(m.delta, 6),
                "pct_delta": round(m.pct_delta, 6),
                "higher_is_better": m.higher_is_better,
                "status": m.status,
                "threshold": m.threshold,
            }
            for m in result.metric_deltas
        ],
        "question_diffs": [
            {
                "question_id": q.question_id,
                "domain": q.domain,
                "difficulty": q.difficulty,
                "baseline_passed": q.baseline_passed,
                "current_passed": q.current_passed,
                "baseline_coverage": q.baseline_coverage,
                "current_coverage": q.current_coverage,
                "status": q.status,
            }
            for q in result.question_diffs
        ],
        "contradiction_diffs": [
            {
                "contradiction_id": c.contradiction_id,
                "domain": c.domain,
                "expected_result": c.expected_result,
                "baseline_correct": c.baseline_correct,
                "current_correct": c.current_correct,
                "status": c.status,
            }
            for c in result.contradiction_diffs
        ],
        "regressions": {
            "metric_regressions": [m.metric for m in result.regressions],
            "question_regressions": [q.question_id for q in result.regressed_questions],
            "contradiction_regressions": [c.contradiction_id for c in result.regressed_contradictions],
        },
        "improvements": {
            "metric_improvements": [m.metric for m in result.improvements],
            "question_improvements": [q.question_id for q in result.improved_questions],
            "contradiction_improvements": [c.contradiction_id for c in result.improved_contradictions],
        },
    }


def build_regression_md(result: RegressionResult, *, run_meta: dict | None = None) -> str:
    meta = run_meta or {}
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    verdict = "PASS" if result.passed else "FAIL"
    verdict_badge = f"✓ {verdict}" if result.passed else f"✗ {verdict}"

    lines: list[str] = [
        "# Regression Report",
        "",
        f"Generated: {ts}",
    ]
    if meta.get("current"):
        lines.append(f"Current: `{meta['current']}`")
    if meta.get("baseline"):
        lines.append(f"Baseline: `{meta['baseline']}`")
    lines += [
        f"Threshold: ±{result.fail_threshold:.3f}",
        "",
        f"## Status: {verdict_badge}",
        "",
    ]

    if result.fail_reasons:
        lines += ["**Failures:**", ""]
        for reason in result.fail_reasons:
            lines.append(f"- {reason}")
        lines.append("")

    # Metric delta table
    lines += [
        "---",
        "",
        "## Metric Deltas",
        "",
        "| Metric | Baseline | Current | Δ | Status |",
        "|---|---|---|---|---|",
    ]
    for m in result.metric_deltas:
        arrow = "↑" if m.delta > 0 else ("↓" if m.delta < 0 else "→")
        delta_str = f"{arrow} {m.delta:+.4f}"
        status_icon = {"pass": "✓", "fail": "✗", "improved": "↑"}[m.status]
        lines.append(
            f"| {m.display_name} | {m.baseline:.4f} | {m.current:.4f} | {delta_str} | {status_icon} {m.status} |"
        )
    lines.append("")

    # Question-level changes
    changed_qs = [q for q in result.question_diffs if q.status not in ("unchanged_pass", "unchanged_fail")]
    if changed_qs:
        lines += [
            "---",
            "",
            "## Question Changes",
            "",
            "| ID | Domain | Status | Baseline | Current |",
            "|---|---|---|---|---|",
        ]
        for q in sorted(changed_qs, key=lambda x: x.status):
            icon = {"improved": "↑", "regressed": "↓", "new": "+", "removed": "−"}[q.status]
            bp = "pass" if q.baseline_passed else ("fail" if q.baseline_passed is not None else "—")
            cp = "pass" if q.current_passed else ("fail" if q.current_passed is not None else "—")
            lines.append(f"| {q.question_id} | {q.domain} | {icon} {q.status} | {bp} | {cp} |")
        lines.append("")

    # Unchanged failures (always worth surfacing)
    unchanged_fail = [q for q in result.question_diffs if q.status == "unchanged_fail"]
    if unchanged_fail:
        lines += [
            "---",
            "",
            "## Persistent Failures",
            "",
            "These Q&A questions failed in both baseline and current run:",
            "",
        ]
        for q in unchanged_fail:
            lines.append(f"- **{q.question_id}** ({q.domain}/{q.difficulty})")
        lines.append("")

    # Contradiction changes
    changed_cs = [c for c in result.contradiction_diffs if c.status not in ("unchanged_pass", "unchanged_fail")]
    if changed_cs:
        lines += [
            "---",
            "",
            "## Contradiction Test Changes",
            "",
            "| ID | Domain | Status | Baseline | Current |",
            "|---|---|---|---|---|",
        ]
        for c in sorted(changed_cs, key=lambda x: x.status):
            icon = {"improved": "↑", "regressed": "↓", "new": "+", "removed": "−"}[c.status]
            bc = "correct" if c.baseline_correct else ("incorrect" if c.baseline_correct is not None else "—")
            cc = "correct" if c.current_correct else ("incorrect" if c.current_correct is not None else "—")
            lines.append(f"| {c.contradiction_id} | {c.domain} | {icon} {c.status} | {bc} | {cc} |")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_regression_trace(result: RegressionResult, *, run_meta: dict | None = None) -> dict:
    """Detailed trace for debugging regressions."""
    base = build_regression_json(result, run_meta=run_meta)
    base["trace_type"] = "regression_trace"
    # Add full per-question details for regressed items
    base["regressed_question_details"] = [
        {
            "question_id": q.question_id,
            "domain": q.domain,
            "difficulty": q.difficulty,
            "baseline_coverage": q.baseline_coverage,
            "current_coverage": q.current_coverage,
        }
        for q in result.regressed_questions
    ]
    base["regressed_contradiction_details"] = [
        {
            "contradiction_id": c.contradiction_id,
            "domain": c.domain,
            "expected_result": c.expected_result,
        }
        for c in result.regressed_contradictions
    ]
    return base


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def write_regression_json(result: RegressionResult, path: str | Path, *, run_meta: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_regression_json(result, run_meta=run_meta), indent=2), encoding="utf-8")
    return out


def write_regression_md(result: RegressionResult, path: str | Path, *, run_meta: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_regression_md(result, run_meta=run_meta), encoding="utf-8")
    return out


def write_regression_trace(result: RegressionResult, path: str | Path, *, run_meta: dict | None = None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_regression_trace(result, run_meta=run_meta), indent=2), encoding="utf-8")
    return out


def load_report(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Report not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))
