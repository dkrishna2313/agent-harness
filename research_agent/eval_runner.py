"""Evaluation runners: J2.1-format benchmark suite and legacy question list."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from .agent import DEFAULT_TOP_EVIDENCE, DcPowerAgent
from .cli import _build_client, _configure_logging
from .evaluator import classify_question_topics
from .loaders import load_sources
from .markdown import memo_to_markdown, write_markdown
from .profile import load_profile
from .schemas import ResearchMemo
from .trace import build_trace, write_trace

app = typer.Typer(add_completion=False, no_args_is_help=True)

_CITATION_RE = re.compile(r"\[Source:\s*[^,\]]+,\s*Evidence:\s*E\d{3}\]")


@dataclass(frozen=True)
class EvalResult:
    question: str
    warnings: list[str]
    detected_topics: list[str]
    evidence_count: int
    citation_count: int
    memo_path: Path
    trace_path: Path


@app.command("run")
def main(
    sources: Annotated[
        Path,
        typer.Option(
            "--sources",
            "-s",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory containing .pdf, .md, and .txt sources.",
        ),
    ] = Path("sources"),
    evals: Annotated[
        Path,
        typer.Option("--evals", "-e", exists=True, dir_okay=False, readable=True),
    ] = Path("evals/questions.yaml"),
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Markdown eval report output path."),
    ] = Path("outputs/eval_report.md"),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name for Claude runs."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic local mock mode instead of Claude."),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose diagnostics (shorthand for --log-level DEBUG)."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."),
    ] = None,
    top_evidence: Annotated[
        int,
        typer.Option(
            "--top-evidence",
            help="Maximum ranked evidence items to pass into memo synthesis.",
        ),
    ] = DEFAULT_TOP_EVIDENCE,
) -> None:
    """Run a small question regression suite and write a Markdown report."""

    _configure_logging(verbose, log_level)

    collection = load_sources(sources)
    questions = load_eval_questions(evals)
    results: list[EvalResult] = []
    eval_output_dir = out.parent / "evals"

    for index, question in enumerate(questions, start=1):
        client, startup_warnings = _build_client(mock=mock, live_llm=False, model=model)
        mock_mode = getattr(client, "is_mock", False)
        memo = DcPowerAgent(client=client, top_evidence=top_evidence).analyze(
            question, collection.documents
        )
        if startup_warnings:
            memo = memo.model_copy(
                update={"evaluation_warnings": startup_warnings + memo.evaluation_warnings}
            )
        if collection.errors:
            load_warnings = [
                f"Source load warning for {error.path.name}: {error.message}"
                for error in collection.errors
            ]
            memo = memo.model_copy(
                update={"evaluation_warnings": memo.evaluation_warnings + load_warnings}
            )

        memo_path = eval_output_dir / f"eval_{index:03d}.md"
        trace_path = write_eval_artifacts(
            memo=memo,
            memo_path=memo_path,
            question=question,
            source_directory=sources,
            documents=collection.documents,
            mock_mode=mock_mode,
        )
        results.append(eval_result_from_memo(memo, memo_path=memo_path, trace_path=trace_path))

    output_path = write_markdown(
        eval_report_to_markdown(
            results,
            source_directory=sources,
            eval_path=evals,
            document_count=len(collection.documents),
        ),
        out,
    )
    typer.echo(f"Wrote {output_path}")


def load_eval_questions(path: str | Path) -> list[str]:
    """Load a minimal YAML question list without requiring PyYAML."""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    questions: list[str] = []
    in_questions = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "questions:":
            in_questions = True
            continue
        if line.startswith("- "):
            questions.append(_strip_yaml_scalar(line[2:].strip()))
            continue
        if in_questions and line.startswith("-"):
            questions.append(_strip_yaml_scalar(line[1:].strip()))

    if not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def write_eval_artifacts(
    *,
    memo: ResearchMemo,
    memo_path: Path,
    question: str,
    source_directory: Path,
    documents,
    mock_mode: bool,
) -> Path:
    write_markdown(memo_to_markdown(memo), memo_path)
    trace_payload = build_trace(
        question=question,
        source_directory=source_directory,
        output_path=memo_path,
        documents=documents,
        memo=memo,
        mock_mode=mock_mode,
    )
    return write_trace(trace_payload, memo_path)


def eval_result_from_memo(memo: ResearchMemo, *, memo_path: Path, trace_path: Path) -> EvalResult:
    evidence = memo.source_notes or memo.evidence
    return EvalResult(
        question=memo.question,
        warnings=list(memo.evaluation_warnings),
        detected_topics=sorted(classify_question_topics(memo.question)),
        evidence_count=len(evidence),
        citation_count=count_memo_citations(memo),
        memo_path=memo_path,
        trace_path=trace_path,
    )


def count_memo_citations(memo: ResearchMemo) -> int:
    cited_sections = [
        memo.confirmed_facts,
        memo.power_implications,
        memo.cooling_implications,
        memo.networking_implications,
        memo.rack_architecture_implications,
    ]
    return sum(len(_CITATION_RE.findall(item)) for section in cited_sections for item in section)


def eval_report_to_markdown(
    results: list[EvalResult],
    *,
    source_directory: Path,
    eval_path: Path,
    document_count: int,
) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"**Source directory:** {source_directory}",
        f"**Eval file:** {eval_path}",
        f"**Documents loaded:** {document_count}",
        f"**Questions evaluated:** {len(results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        topics = ", ".join(result.detected_topics) if result.detected_topics else "none"
        lines.extend(
            [
                f"## {index}. {result.question}",
                "",
                f"- Detected topics: {topics}",
                f"- Evidence count: {result.evidence_count}",
                f"- Citation count: {result.citation_count}",
                f"- Memo: {result.memo_path}",
                f"- Trace: {result.trace_path}",
                f"- Warning count: {len(result.warnings)}",
                "- Warnings:",
            ]
        )
        if result.warnings:
            lines.extend(f"  - {warning}" for warning in result.warnings)
        else:
            lines.append("  - None")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _strip_yaml_scalar(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


@app.command("benchmark")
def benchmark(
    eval_dir: Annotated[
        Path,
        typer.Option(
            "--eval-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Root directory of J2.1 evaluation dataset (contains nvidia/, smr/, contradictions/).",
        ),
    ] = Path("eval"),
    sources: Annotated[
        Path,
        typer.Option(
            "--sources",
            "-s",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Directory of source documents for Q&A questions.",
        ),
    ] = Path("sources"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "--out", "-o", help="Directory for report outputs."),
    ] = Path("outputs"),
    profile: Annotated[
        str | None,
        typer.Option("--profile", "-p", help="Domain profile name or path (e.g. smr, ai_data_centers)."),
    ] = None,
    knowledge_store: Annotated[
        Path | None,
        typer.Option("--knowledge-store", help="Knowledge Store directory for Knowledge Layer evidence retrieval."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use mock (no LLM) mode for Q&A runs."),
    ] = False,
    web_search: Annotated[
        bool,
        typer.Option("--web-search", help="Enable web search retrieval (K1.0)."),
    ] = False,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            help="Comma-separated question IDs to run (e.g. NVIDIA_003,SMR_010). Skips all others.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging (shorthand for --log-level DEBUG)."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            help=(
                "Parallel workers for Q&A and contradiction runs. "
                "1 = sequential (default). "
                "Recommended: 3-5 for live LLM runs (respect API rate limits)."
            ),
        ),
    ] = 1,
) -> None:
    """Run the J2.1 gold evaluation dataset and produce scored reports.

    Example:

        python3 -m research_agent.eval_runner benchmark --eval-dir ./eval --web-search
        python3 -m research_agent.eval_runner benchmark --workers 5
    """
    import logging

    _configure_logging(verbose, log_level)

    # Lazy import to keep startup fast
    from .evaluation.loader import load_qa_questions, load_contradiction_cases
    from .evaluation.runner import EvaluationRunner
    from .evaluation.report import write_json_report, write_md_report

    # Load profile
    domain_profile = None
    if profile:
        try:
            domain_profile = load_profile(profile)
            typer.echo(f"Profile: {profile}")
        except Exception as exc:
            typer.echo(f"Warning: could not load profile {profile!r}: {exc}", err=True)

    # Load benchmark data
    qa_questions = load_qa_questions(eval_dir)
    contradiction_cases = load_contradiction_cases(eval_dir)

    # --only filter: restrict to named Q&A questions only
    only_ids: set[str] | None = None
    if only:
        only_ids = {qid.strip() for qid in only.split(",") if qid.strip()}
        unknown = only_ids - {q.question_id for q in qa_questions}
        if unknown:
            typer.echo(f"Warning: unknown question ID(s): {', '.join(sorted(unknown))}", err=True)
        qa_questions = [q for q in qa_questions if q.question_id in only_ids]
        contradiction_cases = []   # skip contradiction tests for targeted runs
        typer.echo(f"--only filter: running {len(qa_questions)} Q&A question(s): {', '.join(q.question_id for q in qa_questions)}")
    else:
        typer.echo(
            f"Loaded {len(qa_questions)} Q&A questions and "
            f"{len(contradiction_cases)} contradiction cases from {eval_dir}"
        )

    # Build agent
    client, startup_warnings = _build_client(mock=mock, live_llm=not mock, model=model)
    if startup_warnings:
        for w in startup_warnings:
            typer.echo(f"  Warning: {w}", err=True)

    if web_search and domain_profile is not None:
        from .profile import WebSearchConfig
        domain_profile = domain_profile.model_copy(
            update={"web_search": WebSearchConfig(enabled=True)}
        )
    elif web_search:
        from .profile import get_default_profile, WebSearchConfig
        domain_profile = get_default_profile().model_copy(
            update={"web_search": WebSearchConfig(enabled=True)}
        )

    agent = DcPowerAgent(client=client, profile=domain_profile)
    runner = EvaluationRunner(
        agent=agent,
        sources_dir=sources,
        profile=domain_profile,
        ro_out_dir=out_dir,
        workers=workers,
    )

    typer.echo("Running evaluation suite…")
    run = runner.run(qa_questions, contradiction_cases)

    run_meta = {
        "eval_dir": str(eval_dir),
        "profile": profile,
        "web_search": web_search,
        "mock_mode": mock,
    }

    # Write reports
    json_path = write_json_report(run, out_dir / "evaluation_report.json", run_meta=run_meta)
    md_path = write_md_report(run, out_dir / "evaluation_report.md", run_meta=run_meta)

    from .evaluation.report import write_trace, print_benchmark_perf_summary
    trace_path = write_trace(run, out_dir / "evaluation.trace.json", run_meta=run_meta)

    # J8.9a — print benchmark performance summary
    print_benchmark_perf_summary(run)

    # Print evaluation summary
    typer.echo("")
    typer.echo("=== Evaluation Complete ===")
    typer.echo(f"  Overall score:            {run.overall_score:.1%}")
    typer.echo(f"  Fact coverage:            {run.fact_coverage_score:.1%}")
    typer.echo(f"  Citation coverage:        {run.citation_score:.1%}")
    typer.echo(f"  Hallucination rate:       {run.hallucination_rate:.1%}")
    typer.echo(f"  Contradiction accuracy:   {run.contradiction_accuracy:.1%}")
    typer.echo(f"  Q&A passed:               {sum(1 for s in run.qa_scores if s.passed)}/{len(run.qa_scores)}")
    typer.echo(f"  Contradiction passed:     {sum(1 for s in run.contradiction_scores if s.correct)}/{len(run.contradiction_scores)}")
    typer.echo("")
    for domain, entry in sorted(run.domain_scores.items()):
        typer.echo(f"  [{domain.upper()}] coverage={entry.get('fact_coverage_score', 0):.1%}  citations={entry.get('citation_score', 0):.1%}  pass_rate={entry.get('pass_rate', 0):.1%}")
    typer.echo("")
    if run.failed_qa:
        typer.echo(f"  Failed Q&A ({len(run.failed_qa)}):")
        for s in run.failed_qa:
            typer.echo(f"    ✗ {s.question_id}: {'; '.join(s.fail_reasons[:1])}")
    if run.failed_contradictions:
        typer.echo(f"  Failed contradiction tests ({len(run.failed_contradictions)}):")
        for s in run.failed_contradictions:
            typer.echo(f"    ✗ {s.contradiction_id}: expected={s.expected_result} got={s.actual_result}")
    typer.echo("")
    typer.echo(f"Reports written to:")
    typer.echo(f"  {json_path}")
    typer.echo(f"  {md_path}")
    typer.echo(f"  {trace_path}")


@app.command("save-baseline")
def save_baseline(
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            "-r",
            exists=True,
            dir_okay=False,
            readable=True,
            help="evaluation_report.json to promote to baseline.",
        ),
    ] = Path("outputs/evaluation_report.json"),
    baseline_dir: Annotated[
        Path,
        typer.Option("--baseline-dir", "-b", help="Directory to write baseline files into."),
    ] = Path("baseline"),
) -> None:
    """Promote an evaluation report to the baseline (J2.3.2).

    Example:

        python3 -m research_agent.eval_runner save-baseline \\
          --report outputs/evaluation_report.json \\
          --baseline-dir baseline
    """
    import shutil
    from datetime import datetime, timezone

    baseline_dir.mkdir(parents=True, exist_ok=True)

    from .evaluation.regression import load_report
    data = load_report(report)

    dest = baseline_dir / "evaluation_report.json"
    shutil.copy2(report, dest)

    summary = data.get("summary", {})
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report),
        "overall_score": summary.get("overall_score", 0),
        "fact_coverage_score": summary.get("fact_coverage_score", 0),
        "citation_score": summary.get("citation_score", 0),
        "contradiction_accuracy": summary.get("contradiction_accuracy", 0),
        "hallucination_rate": summary.get("hallucination_rate", 0),
        "qa_questions_passed": summary.get("qa_questions_passed", 0),
        "qa_questions_total": summary.get("qa_questions_total", 0),
        "contradiction_tests_passed": summary.get("contradiction_tests_passed", 0),
        "contradiction_tests_total": summary.get("contradiction_tests_total", 0),
        "report_generated_at": data.get("generated_at", ""),
    }
    import json
    meta_path = baseline_dir / "baseline_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    typer.echo(f"Baseline saved to {dest}")
    typer.echo(f"Metadata written to {meta_path}")
    typer.echo(f"  overall_score:          {metadata['overall_score']:.4f}")
    typer.echo(f"  fact_coverage_score:    {metadata['fact_coverage_score']:.4f}")
    typer.echo(f"  citation_score:         {metadata['citation_score']:.4f}")
    typer.echo(f"  contradiction_accuracy: {metadata['contradiction_accuracy']:.4f}")


@app.command("regress")
def regress(
    current: Annotated[
        Path,
        typer.Option(
            "--current",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Current evaluation_report.json to compare.",
        ),
    ] = Path("outputs/evaluation_report.json"),
    baseline: Annotated[
        Path,
        typer.Option(
            "--baseline",
            "-b",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Baseline evaluation_report.json to compare against.",
        ),
    ] = Path("baseline/evaluation_report.json"),
    out_dir: Annotated[
        Path,
        typer.Option("--out-dir", "--out", "-o", help="Directory for regression report outputs."),
    ] = Path("outputs"),
    knowledge_store: Annotated[
        Path | None,
        typer.Option("--knowledge-store", help="Knowledge Store directory (accepted for forward compat; not used by regression comparison)."),
    ] = None,
    fail_threshold: Annotated[
        float,
        typer.Option(
            "--fail-threshold",
            help="Absolute score drop (or hallucination rise) that triggers a failure.",
        ),
    ] = 0.03,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging (shorthand for --log-level DEBUG)."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL, PROGRESS."),
    ] = None,
) -> None:
    """Compare a current evaluation report against the baseline (J2.3).

    Returns exit code 0 on pass, 1 on regression.

    Example:

        python3 -m research_agent.eval_runner regress \\
          --current outputs/evaluation_report.json \\
          --baseline baseline/evaluation_report.json \\
          --fail-threshold 0.03
    """
    _configure_logging(verbose, log_level)
    from .evaluation.regression import (
        load_report,
        compare_reports,
        write_regression_json,
        write_regression_md,
        write_regression_trace,
    )

    cur_data = load_report(current)
    base_data = load_report(baseline)

    result = compare_reports(cur_data, base_data, fail_threshold=fail_threshold)

    run_meta = {
        "current": str(current),
        "baseline": str(baseline),
        "fail_threshold": fail_threshold,
    }
    json_path = write_regression_json(result, out_dir / "regression_report.json", run_meta=run_meta)
    md_path = write_regression_md(result, out_dir / "regression_report.md", run_meta=run_meta)
    trace_path = write_regression_trace(result, out_dir / "regression.trace.json", run_meta=run_meta)

    # Print summary
    verdict = "PASS" if result.passed else "FAIL"
    typer.echo(f"\n=== Regression Check: {verdict} ===\n")

    for m in result.metric_deltas:
        icon = {"pass": " ", "fail": "✗", "improved": "↑"}[m.status]
        typer.echo(
            f"  {icon} {m.display_name:<28} "
            f"{m.baseline:.4f} → {m.current:.4f}  "
            f"(Δ{m.delta:+.4f})"
        )

    if result.improved_questions or result.regressed_questions:
        typer.echo("")
    for q in result.improved_questions:
        typer.echo(f"  ↑ Q&A improved:    {q.question_id}")
    for q in result.regressed_questions:
        typer.echo(f"  ↓ Q&A regressed:   {q.question_id}")

    if result.improved_contradictions or result.regressed_contradictions:
        typer.echo("")
    for c in result.improved_contradictions:
        typer.echo(f"  ↑ Contradiction improved:  {c.contradiction_id}")
    for c in result.regressed_contradictions:
        typer.echo(f"  ↓ Contradiction regressed: {c.contradiction_id}")

    if result.fail_reasons:
        typer.echo("")
        typer.echo("Failures:")
        for reason in result.fail_reasons:
            typer.echo(f"  ✗ {reason}")

    typer.echo(f"\nReports written to:")
    typer.echo(f"  {json_path}")
    typer.echo(f"  {md_path}")
    typer.echo(f"  {trace_path}")

    if not result.passed:
        raise typer.Exit(1)


@app.command("citation-audit")
def citation_audit(
    report: Annotated[
        Path,
        typer.Option(
            "--report",
            "-r",
            exists=True,
            dir_okay=False,
            readable=True,
            help="evaluation_report.json to audit.",
        ),
    ] = Path("outputs/evaluation_report.json"),
    trace: Annotated[
        Path | None,
        typer.Option(
            "--trace",
            "-t",
            dir_okay=False,
            readable=True,
            help="evaluation.trace.json — provides actual_answer text per question.",
        ),
    ] = Path("outputs/evaluation.trace.json"),
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            help="Flag questions with citation_score strictly below this value.",
        ),
    ] = 1.0,
) -> None:
    """Identify benchmark questions with imperfect citation coverage (diagnostic only).

    Reads evaluation_report.json and evaluation.trace.json, flags every question
    whose citation_score is below --threshold (default 1.0), prints a summary
    table sorted worst-first, then prints the actual answer text for each flagged
    question so the gap can be inspected.

    This command is read-only: it does not modify any output files, scores,
    or benchmark/regression logic.

    Example:

        python3 -m research_agent.eval_runner citation-audit \\
          --report outputs/evaluation_report.json \\
          --trace outputs/evaluation.trace.json

        python3 -m research_agent.eval_runner citation-audit \\
          --report outputs/evaluation_report.json \\
          --trace outputs/evaluation.trace.json \\
          --threshold 0.5
    """
    from .evaluation.citation_audit import run_citation_audit

    # trace path may not exist (optional) — pass None if missing
    trace_path = trace if (trace is not None and trace.exists()) else None
    exit_code = run_citation_audit(report, trace_path, threshold)
    raise typer.Exit(exit_code)


@app.command("audit")
def audit(
    eval_dir: Annotated[
        Path,
        typer.Option(
            "--eval-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help="Root directory of J2.1 evaluation dataset.",
        ),
    ] = Path("eval"),
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Treat warnings as errors."),
    ] = False,
) -> None:
    """Validate benchmark files without running the harness (J2.2a.5).

    Checks for: YAML validity, duplicate IDs, missing required fields,
    must_include/must_not_include conflicts, invalid contradiction definitions.

    Example:

        python3 -m research_agent.eval_runner audit --eval-dir ./eval
    """
    from .evaluation.loader import load_qa_questions, load_contradiction_cases
    from .evaluation.validator import validate_benchmark

    try:
        qa_questions = load_qa_questions(eval_dir)
        contradiction_cases = load_contradiction_cases(eval_dir)
    except ValueError as exc:
        typer.echo(f"YAML parse error: {exc}", err=True)
        raise typer.Exit(1) from exc

    report = validate_benchmark(qa_questions, contradiction_cases)

    typer.echo(f"Benchmark audit: {len(qa_questions)} Q&A, {len(contradiction_cases)} contradiction cases")
    typer.echo("")

    if report.errors:
        typer.echo(f"ERRORS ({len(report.errors)}):")
        for e in report.errors:
            typer.echo(f"  ✗ [{e.item_id}] {e.code}: {e.message}", err=True)

    if report.warnings:
        typer.echo(f"WARNINGS ({len(report.warnings)}):")
        for w in report.warnings:
            symbol = "✗" if strict else "⚠"
            typer.echo(f"  {symbol} [{w.item_id}] {w.code}: {w.message}")

    typer.echo("")

    fail = report.errors or (strict and report.warnings)
    if fail:
        typer.echo("Benchmark audit FAILED", err=True)
        raise typer.Exit(1)
    else:
        typer.echo("Benchmark audit passed")
        if report.warnings:
            typer.echo(f"  ({len(report.warnings)} warning(s) — run with --strict to treat as errors)")


if __name__ == "__main__":
    app()
