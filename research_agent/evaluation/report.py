"""Report generation for evaluation runs (JSON + Markdown)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .runner import EvaluationRun
from .scorer import QAScore, ContradictionScore


def build_json_report(run: EvaluationRun, *, run_meta: dict | None = None) -> dict:
    """Serialise an EvaluationRun to a JSON-compatible dict."""

    known_limitation = [s for s in run.contradiction_scores if s.known_limitation]
    benchmark_errors = [s for s in run.qa_scores if s.benchmark_error]

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        **(run_meta or {}),
        "summary": {
            "overall_score": run.overall_score,
            "fact_coverage_score": run.fact_coverage_score,
            "hallucination_rate": run.hallucination_rate,
            "citation_score": run.citation_score,
            "contradiction_accuracy": run.contradiction_accuracy,
            "qa_questions_total": len(run.qa_scores),
            "qa_questions_passed": sum(1 for s in run.qa_scores if s.passed and not s.benchmark_error),
            "contradiction_tests_total": len([s for s in run.contradiction_scores if not s.known_limitation]),
            "contradiction_tests_passed": sum(
                1 for s in run.contradiction_scores if s.correct and not s.known_limitation
            ),
            "known_limitation_cases": len(known_limitation),
            "benchmark_errors": len(benchmark_errors) + len(run.validation_errors),
            # J5.7 — aggregate agent scores (also in agent_evaluation block)
            "planner_score": run.planner_score,
            "evidence_score": run.evidence_score,
            "qa_agent_score": run.qa_agent_score,
            "report_score": run.report_score,
        },
        "domain_scores": run.domain_scores,
        "qa_results": [_qa_score_dict(s) for s in run.qa_scores],
        "contradiction_results": [_contra_score_dict(s) for s in run.contradiction_scores],
        "agent_evaluation": _agent_evaluation_dict(run),
        "failed_tests": {
            "qa": [_qa_score_dict(s) for s in run.failed_qa],
            "contradictions": [_contra_score_dict(s) for s in run.failed_contradictions],
        },
        "known_limitation_cases": [_contra_score_dict(s) for s in known_limitation],
        "benchmark_validation_errors": [
            {"item_id": e.item_id, "code": e.code, "message": e.message, "source_file": e.source_file}
            for e in run.validation_errors
        ],
    }
    return report


def write_json_report(run: EvaluationRun, path: str | Path, *, run_meta: dict | None = None) -> Path:
    """Write evaluation report JSON to *path*."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_json_report(run, run_meta=run_meta)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def build_md_report(run: EvaluationRun, *, run_meta: dict | None = None) -> str:
    """Render an EvaluationRun as a Markdown document."""

    meta = run_meta or {}
    lines: list[str] = []
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# Evaluation Report",
        "",
        f"Generated: {ts}",
    ]
    if meta.get("eval_dir"):
        lines.append(f"Eval dir: `{meta['eval_dir']}`")
    if meta.get("profile"):
        lines.append(f"Profile: `{meta['profile']}`")
    if meta.get("web_search"):
        lines.append("Web search: enabled")

    lines += [
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Score |",
        f"|---|---|",
        f"| **Overall score** | {_pct(run.overall_score)} |",
        f"| Fact coverage | {_pct(run.fact_coverage_score)} |",
        f"| Citation coverage | {_pct(run.citation_score)} |",
        f"| Hallucination rate | {_pct(run.hallucination_rate)} |",
        f"| Contradiction accuracy | {_pct(run.contradiction_accuracy)} |",
        f"| Q&A passed | {sum(1 for s in run.qa_scores if s.passed)}/{len(run.qa_scores)} |",
        f"| Contradiction tests passed | {sum(1 for s in run.contradiction_scores if s.correct)}/{len(run.contradiction_scores)} |",
        "",
        "---",
        "",
        "## Domain Scores",
        "",
    ]

    for domain, entry in sorted(run.domain_scores.items()):
        lines.append(f"### {domain.upper()}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Questions | {entry.get('questions', 0)} |")
        if "fact_coverage_score" in entry:
            lines.append(f"| Fact coverage | {_pct(entry['fact_coverage_score'])} |")
        if "citation_score" in entry:
            lines.append(f"| Citation score | {_pct(entry['citation_score'])} |")
        if "pass_rate" in entry:
            lines.append(f"| Pass rate | {_pct(entry['pass_rate'])} |")
        if "contradiction_accuracy" in entry:
            lines.append(f"| Contradiction tests | {entry.get('contradiction_tests', 0)} |")
            lines.append(f"| Contradiction accuracy | {_pct(entry['contradiction_accuracy'])} |")
        lines.append("")

    # Q&A results table
    lines += [
        "---",
        "",
        "## Q&A Results",
        "",
        "| ID | Domain | Diff | Coverage | Citations | Pass | Failures |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in run.qa_scores:
        status = "✓" if s.passed else "✗"
        failures = "; ".join(s.fail_reasons[:2]) if s.fail_reasons else ""
        lines.append(
            f"| {s.question_id} | {s.domain} | {s.difficulty} "
            f"| {_pct(s.fact_coverage_score)} ({s.must_include_hits}/{s.must_include_total}) "
            f"| {s.citation_count} "
            f"| {status} "
            f"| {failures} |"
        )

    # Contradiction results table
    lines += [
        "",
        "---",
        "",
        "## Contradiction Test Results",
        "",
        "| ID | Domain | Expected | Actual | Pass | Suppression | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in run.contradiction_scores:
        status = "✓" if s.correct else "✗"
        sup = ", ".join(s.actual_suppression_reasons) if s.actual_suppression_reasons else "—"
        notes = s.notes[:80] if s.notes else ""
        lines.append(
            f"| {s.contradiction_id} | {s.domain} | {s.expected_result} "
            f"| {s.actual_result} | {status} | {sup} | {notes} |"
        )

    # Failed tests detail
    if run.failed_qa or run.failed_contradictions:
        lines += [
            "",
            "---",
            "",
            "## Failed Tests",
            "",
        ]
        if run.failed_qa:
            lines.append("### Failed Q&A")
            lines.append("")
            for s in run.failed_qa:
                lines.append(f"**{s.question_id}** — {s.question[:80]}")
                for reason in s.fail_reasons:
                    lines.append(f"- {reason}")
                if s.missing_facts:
                    lines.append(f"- Missing facts: {', '.join(repr(f) for f in s.missing_facts)}")
                if s.unexpected_facts:
                    lines.append(f"- Unexpected facts: {', '.join(repr(f) for f in s.unexpected_facts)}")
                if s.actual_answer:
                    lines.append(f"- Answer preview: `{s.actual_answer[:200].replace(chr(10), ' ')}`")
                lines.append("")

        if run.failed_contradictions:
            lines.append("### Failed Contradiction Tests")
            lines.append("")
            for s in run.failed_contradictions:
                lines.append(f"**{s.contradiction_id}** — expected `{s.expected_result}`, got `{s.actual_result}`")
                lines.append("")
                lines.append(f"- Evidence A: {s.evidence_a[:100]}")
                lines.append(f"- Evidence B: {s.evidence_b[:100]}")
                if s.entity_a or s.entity_b:
                    lines.append(f"- Entity: A={s.entity_a!r}  B={s.entity_b!r}")
                if s.scope_a or s.scope_b:
                    lines.append(f"- Scope: A={s.scope_a!r}  B={s.scope_b!r}")
                if s.metric_a or s.metric_b:
                    lines.append(f"- Metric: A={s.metric_a!r}  B={s.metric_b!r}")
                if s.actual_suppression_reasons:
                    lines.append(f"- Suppression: {', '.join(s.actual_suppression_reasons)}")
                else:
                    lines.append("- Suppression: none fired")
                if s.notes:
                    lines.append(f"- Notes: {s.notes[:200]}")
                lines.append("")

    # Known limitation section
    known_lim = [s for s in run.contradiction_scores if s.known_limitation]
    if known_lim:
        lines += [
            "---",
            "",
            "## Known Limitations (excluded from score)",
            "",
        ]
        for s in known_lim:
            status = "✓ correct" if s.correct else "✗ incorrect"
            lines.append(f"**{s.contradiction_id}** — {status} (expected `{s.expected_result}`, got `{s.actual_result}`)")
            lines.append(f"- Entity: A={s.entity_a!r}  B={s.entity_b!r}")
            lines.append(f"- Scope: A={s.scope_a!r}  B={s.scope_b!r}")
            lines.append(f"- Metric: A={s.metric_a!r}  B={s.metric_b!r}")
            lines.append("")

    if run.validation_errors:
        lines += [
            "---",
            "",
            "## Benchmark Validation Errors",
            "",
        ]
        for e in run.validation_errors:
            lines.append(f"**{e.item_id}** `{e.code}`: {e.message}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_md_report(run: EvaluationRun, path: str | Path, *, run_meta: dict | None = None) -> Path:
    """Write evaluation report Markdown to *path*."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_md_report(run, run_meta=run_meta), encoding="utf-8")
    return out


def build_trace(run: EvaluationRun, *, run_meta: dict | None = None) -> dict:
    """Build a detailed evaluation trace for debugging (J2.2a.4)."""

    known_limitation = [s for s in run.contradiction_scores if s.known_limitation]
    ts = datetime.now(timezone.utc).isoformat()

    return {
        "generated_at": ts,
        "trace_type": "evaluation",
        **(run_meta or {}),
        "run_summary": {
            "overall_score": run.overall_score,
            "fact_coverage_score": run.fact_coverage_score,
            "hallucination_rate": run.hallucination_rate,
            "citation_score": run.citation_score,
            "contradiction_accuracy": run.contradiction_accuracy,
            "qa_total": len(run.qa_scores),
            "qa_passed": sum(1 for s in run.qa_scores if s.passed and not s.benchmark_error),
            "contradiction_total": len([s for s in run.contradiction_scores if not s.known_limitation]),
            "contradiction_passed": sum(
                1 for s in run.contradiction_scores if s.correct and not s.known_limitation
            ),
            "known_limitation_excluded": len(known_limitation),
            "benchmark_errors": len(run.validation_errors),
        },
        "agent_evaluation": _agent_evaluation_dict(run),
        "qa_results": [_qa_trace_dict(s) for s in run.qa_scores],
        "contradiction_results": [_contra_trace_dict(s) for s in run.contradiction_scores],
        "failed_tests": {
            "qa": [_qa_trace_dict(s) for s in run.failed_qa],
            "contradictions": [_contra_trace_dict(s) for s in run.failed_contradictions],
        },
        "known_limitation_cases": [_contra_trace_dict(s) for s in known_limitation],
        "benchmark_validation_errors": [
            {
                "item_id": e.item_id,
                "code": e.code,
                "message": e.message,
                "source_file": e.source_file,
                "benchmark_error": e.benchmark_error,
            }
            for e in run.validation_errors
        ],
    }


def write_trace(run: EvaluationRun, path: str | Path, *, run_meta: dict | None = None) -> Path:
    """Write evaluation trace JSON to *path* (J2.2a.4)."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(build_trace(run, run_meta=run_meta), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value:.1%}"


def _qa_score_dict(s: QAScore) -> dict:
    return {
        "question_id": s.question_id,
        "domain": s.domain,
        "difficulty": s.difficulty,
        "fact_coverage_score": s.fact_coverage_score,
        "must_include_hits": s.must_include_hits,
        "must_include_total": s.must_include_total,
        "exact_matches_found": s.exact_matches_found,
        "semantic_matches_found": s.semantic_matches_found,
        "hallucination_penalty": s.hallucination_penalty,
        "must_not_include_violations": s.must_not_include_violations,
        "context_allowed_count": s.context_allowed_count,
        "citation_count": s.citation_count,
        "citation_score": s.citation_score,
        "evidence_count": s.evidence_count,
        "overall_score": s.overall_score,
        "passed": s.passed,
        "fail_reasons": s.fail_reasons,
        "benchmark_error": s.benchmark_error,
        "benchmark_error_reason": s.benchmark_error_reason,
    }


def _qa_trace_dict(s: QAScore) -> dict:
    """Extended Q&A dict including diagnostics (J2.2a.2), semantic matches (J3.1a), prohibited audit (J3.1c)."""
    d = _qa_score_dict(s)
    d.update({
        "expected": {
            "must_include": [],   # populated below if diagnostic context available
            "must_not_include": [],
        },
        "missing_facts": s.missing_facts,
        "unexpected_facts": s.unexpected_facts,
        "actual_answer": s.actual_answer,
        "semantic_matches": s.semantic_matches,
        "prohibited_term_audit": s.prohibited_term_audit,
        "retrieval_diversity": s.retrieval_diversity,
    })
    return d


def _contra_score_dict(s: ContradictionScore) -> dict:
    return {
        "contradiction_id": s.contradiction_id,
        "domain": s.domain,
        "expected_result": s.expected_result,
        "actual_result": s.actual_result,
        "correct": s.correct,
        "suppression_fired": s.suppression_fired,
        "expected_suppression_reason": s.expected_suppression_reason,
        "actual_suppression_reasons": s.actual_suppression_reasons,
        "suppression_correct": s.suppression_correct,
        "known_limitation": s.known_limitation,
        "notes": s.notes,
    }


def _agent_evaluation_dict(run: "EvaluationRun") -> dict:
    """Serialise J5.7 agent evaluation block."""
    per_question = []
    for s in run.agent_scores:
        per_question.append({
            "question_id": s.question_id,
            "domain": s.domain,
            "planner_score": s.planner_score,
            "evidence_score": s.evidence_score,
            "qa_score": s.qa_score,
            "report_score": s.report_score,
            "recommendation_score": s.recommendation_score,
            "detail": {
                "planner": {
                    "investigation_area_count": s.investigation_area_count,
                    "investigation_areas_covered": s.investigation_areas_covered,
                },
                "evidence": {
                    "evidence_count": s.evidence_count,
                    "high_quality_evidence": s.high_quality_evidence,
                    "source_diversity": s.source_diversity,
                },
                "qa": {
                    "gaps_identified": s.gaps_identified,
                    "contradictions_found": s.contradictions_found,
                },
                "report": {
                    "citation_count": s.citation_count,
                    "confirmed_facts": s.confirmed_facts,
                    "citation_score": s.report_citation_score,
                },
            },
        })
    return {
        "aggregate": {
            "planner_score": run.planner_score,
            "evidence_score": run.evidence_score,
            "qa_score": run.qa_agent_score,
            "report_score": run.report_score,
            "recommendation_score": run.recommendation_score,
        },
        "per_question": per_question,
    }


def _contra_trace_dict(s: ContradictionScore) -> dict:
    """Extended contradiction dict including entity/scope/metric diagnostics (J2.2a.3)."""
    d = _contra_score_dict(s)
    d.update({
        "evidence_a": s.evidence_a,
        "evidence_b": s.evidence_b,
        "entity_a": s.entity_a,
        "entity_b": s.entity_b,
        "scope_a": s.scope_a,
        "scope_b": s.scope_b,
        "metric_a": s.metric_a,
        "metric_b": s.metric_b,
        "suppression_reasons": s.actual_suppression_reasons,
        "suppression_details": s.suppression_details,
    })
    return d
