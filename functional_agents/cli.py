"""CLI for the functional agent pipeline (J5.0a.7)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from research_agent.cli import _configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("run")
def main(
    question: Annotated[str, typer.Argument(help="Research question to answer. Omit if using --goal.")] = "",
    goal: Annotated[
        str | None,
        typer.Option("--goal", "-g", help="High-level business goal (goal-driven mode). Mutually exclusive with QUESTION."),
    ] = None,
    sources: Annotated[
        Path,
        typer.Option("--sources", "-s", exists=True, file_okay=False, dir_okay=True,
                     help="Directory containing source documents."),
    ] = Path("sources"),
    profiles: Annotated[
        str,
        typer.Option("--profiles", help="Comma-separated profile names. First is the execution profile."),
    ] = "ai_data_centers",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Markdown output path."),
    ] = Path("outputs/j50a_functional_agents.md"),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock client instead of Claude."),
    ] = False,
    use_extraction_cache: Annotated[
        bool,
        typer.Option("--use-extraction-cache", help="Cache chunk extractions to disk (.cache/extraction/). Speeds up repeat runs on the same sources."),
    ] = False,
    web_search: Annotated[
        bool,
        typer.Option("--web-search", help="Enable web search retrieval."),
    ] = False,
    top_evidence: Annotated[
        int,
        typer.Option("--top-evidence", help="Maximum evidence items passed to synthesis."),
    ] = 50,
    top_chunks: Annotated[
        int,
        typer.Option("--top-chunks", help="Maximum chunks sent to evidence extraction."),
    ] = 20,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL."),
    ] = None,
) -> None:
    """Run the functional agent pipeline and write a Markdown research memo.

    Either pass a QUESTION as a positional argument (question-driven mode) or
    use --goal for goal-driven mode where ProblemFramingAgent derives the
    research questions automatically.
    """

    if goal and question:
        typer.echo("Error: provide either QUESTION or --goal, not both.", err=True)
        raise typer.Exit(code=1)
    if not goal and not question:
        typer.echo("Error: provide a QUESTION or --goal.", err=True)
        raise typer.Exit(code=1)

    _configure_logging(verbose=False, log_level=log_level or "INFO")

    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]

    # Build client
    client = _build_client(mock=mock, model=model, use_extraction_cache=use_extraction_cache)

    from .orchestrator import Orchestrator

    orchestrator = Orchestrator(
        profile_names=profile_names,
        sources_dir=sources,
        out_path=out,
        client=client,
        top_evidence=top_evidence,
        top_chunks=top_chunks,
        web_search=web_search,
    )

    try:
        ctx = orchestrator.run_from_goal(goal) if goal else orchestrator.run(question)
    except Exception as exc:
        logging.error("Functional agent pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc

    agents_run = [h["agent"] for h in ctx.agent_history]
    typer.echo(f"Agents run: {', '.join(agents_run)}")
    typer.echo(f"Profiles:   {', '.join(ctx.profiles)}")
    typer.echo(f"Report:     {ctx.artifacts.get('report_path', ctx.artifacts)}")


def _build_client(*, mock: bool, model: str | None, use_extraction_cache: bool = False):
    from research_agent.claude_client import ClaudeClient, MockClaudeClient

    if mock:
        return MockClaudeClient()
    if not os.getenv("ANTHROPIC_API_KEY"):
        logging.warning("ANTHROPIC_API_KEY missing — using mock client.")
        return MockClaudeClient()
    try:
        return ClaudeClient(model=model, use_extraction_cache=use_extraction_cache)
    except Exception as exc:
        logging.error("Claude client setup failed: %s — using mock.", exc)
        return MockClaudeClient()


@app.command("stress-test")
def stress_test_cmd(
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory / base path for stress-test artefacts."),
    ] = Path("outputs/j67a_stress_test"),
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run the J6.7a recommendation improvement stress test.

    Generates four synthetic weak recommendations (one isolated weakness each),
    evaluates them, runs the improvement agent, re-evaluates, and writes a
    before/after proof report.  Exits with code 1 if no recommendation improves.
    """
    _configure_logging(verbose=False, log_level=log_level or "INFO")

    from .recommendation_stress_test import run_stress_test, build_report_section

    results = run_stress_test(out_path=out)

    qa = results["qa_validation"]
    metrics = results["improvement_metrics"]

    typer.echo(f"Recommendations tested : {len(results['synthetic_recommendations'])}")
    typer.echo(f"Recommendations improved: {metrics['recommendations_improved']}")
    typer.echo(f"Average score before   : {metrics['average_score_before']:.3f}")
    typer.echo(f"Average score after    : {metrics['average_score_after']:.3f}")
    typer.echo(f"Average delta          : +{metrics['average_delta']:.3f}")
    typer.echo(f"Loop validated         : {'YES' if qa['improvement_loop_validated'] else 'NO'}")

    # Print markdown table to stdout for quick inspection
    typer.echo("")
    typer.echo(build_report_section(results))

    if not qa["improvement_loop_validated"]:
        typer.echo("FAIL: improvement loop not validated — no recommendation improved.", err=True)
        raise typer.Exit(code=1)


@app.command("scenario-validate")
def scenario_validate_cmd(
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for trace and research-object artefacts."),
    ] = Path("outputs"),
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run the J6.8a scenario analysis validation harness.

    Exercises ScenarioAgent end-to-end with synthetic AI infrastructure
    recommendations (power / cooling / capital / grid types), writes
    j68a_scenario_validation.trace.json, and updates
    latest_research_object.json to prove Base / Upside / Downside
    scenarios are generated and used.
    """
    _configure_logging(verbose=False, log_level=log_level or "INFO")

    from .scenario_validation import run_scenario_validation, build_validation_report

    results = run_scenario_validation(out_path=out)

    report = build_validation_report(results)
    typer.echo(report)

    qa = results["qa_validation"]
    summary = results["scenario_analysis_summary"]

    typer.echo(f"\nScenarios generated        : {summary.get('scenario_count', 0)}")
    typer.echo(f"Recommendations stress-tested: {summary.get('recommendations_stress_tested', 0)}")
    typer.echo(f"Average robustness score    : {summary.get('average_robustness_score', 0):.3f}")
    typer.echo(f"QA validated               : {'YES' if qa.get('scenarios_present') else 'NO'}")
    typer.echo(f"Trace written to           : {out}/j68a_scenario_validation.trace.json")

    if not qa.get("scenarios_present"):
        typer.echo("FAIL: no scenarios generated.", err=True)
        raise typer.Exit(code=1)


@app.command("profile-compare")
def profile_compare_cmd(
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for comparison report and JSON."),
    ] = Path("outputs"),
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Evidence items retrieved per run."),
    ] = 18,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run the J5.6b profile-driven retrieval validation.

    Executes the same goal three times with different profile selections
    (Run A: ai_data_centers, Run B: transmission, Run C: both) using a
    50-item synthetic evidence corpus scored against real profile term sets.
    Reports evidence, finding, and recommendation overlap with Jaccard
    similarity metrics.  Exits with code 1 if profiles do not produce
    measurably different outputs.
    """
    _configure_logging(verbose=False, log_level=log_level or "INFO")

    from .profile_comparison import run_all, build_comparison_report, write_artifacts

    results = run_all(n=top_n)
    bv = results["behavioral_validation"]
    sims = results["similarity_matrix"]
    runs = results["runs"]

    write_artifacts(results, Path(out))

    report = build_comparison_report(results)
    typer.echo(report)

    # Key metrics summary
    typer.echo(f"Run A evidence: {runs['run_a']['evidence_count']} items  keywords={runs['run_a']['finding_keywords']}")
    typer.echo(f"Run B evidence: {runs['run_b']['evidence_count']} items  keywords={runs['run_b']['finding_keywords']}")
    typer.echo(f"Run C evidence: {runs['run_c']['evidence_count']} items  keywords={runs['run_c']['finding_keywords']}")
    typer.echo("")
    typer.echo(f"A vs B evidence similarity : {sims['a_vs_b']['evidence_similarity']:.3f}")
    typer.echo(f"A vs B finding similarity  : {sims['a_vs_b']['finding_similarity']:.3f}")
    typer.echo(f"A vs B rec similarity      : {sims['a_vs_b']['recommendation_similarity']:.3f}")
    typer.echo("")
    typer.echo(f"Retrieval changed    : {'YES' if bv['retrieval_changed'] else 'NO'}")
    typer.echo(f"Evidence changed     : {'YES' if bv['evidence_changed'] else 'NO'}")
    typer.echo(f"Findings changed     : {'YES' if bv['findings_changed'] else 'NO'}")
    typer.echo(f"Recommendations changed: {'YES' if bv['recommendations_changed'] else 'NO'}")
    typer.echo(f"Report written to    : {out}/j56b_profile_comparison_report.md")

    if not all(bv.values()):
        failed = [k for k, v in bv.items() if not v]
        typer.echo(f"FAIL: behavioral validation criteria not met: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


@app.command("corpus-validate")
def corpus_validate_cmd(
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for corpus validation report and JSON."),
    ] = Path("outputs"),
    top_n: Annotated[
        int,
        typer.Option("--top-n", help="Evidence items retrieved per run."),
    ] = 18,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run the J5.6c profile corpus validation.

    Executes three runs against a 60-item source-attributed corpus (32
    ai_data_centers items from NVIDIA/ASHRAE/hyperscalers + 28 transmission
    items from PJM/MISO/ERCOT/FERC/NERC) and proves that profile selection
    produces profile-specific source pools.  Writes
    j56c_profile_corpus_report.md and j56c_profile_corpus.json.  Exits with
    code 1 if source pools are not measurably different.
    """
    _configure_logging(verbose=False, log_level=log_level or "INFO")

    from .profile_corpus_validator import (
        run_corpus_validation,
        build_corpus_report,
        write_corpus_artifacts,
    )

    results = run_corpus_validation(n=top_n)
    bv = results["behavioral_validation"]
    sims = results["similarity_matrix"]
    runs = results["runs"]

    write_corpus_artifacts(results, Path(out))

    report = build_corpus_report(results)
    typer.echo(report)

    for rid, run in runs.items():
        prs = run.get("profile_retrieval_summary", {})
        for profile, summary in prs.items():
            sources = summary.get("evidence_sources", [])
            typer.echo(f"{rid.upper()} / {profile}: {len(sources)} sources — {', '.join(sources[:5])}{'…' if len(sources) > 5 else ''}")
    typer.echo("")
    typer.echo(f"A vs B source Jaccard   : {sims['a_vs_b']['source_similarity']:.3f}")
    typer.echo(f"A vs B evidence Jaccard : {sims['a_vs_b']['evidence_similarity']:.3f}")
    typer.echo("")
    for key, val in bv.items():
        label = key.replace("_", " ").title()
        typer.echo(f"{label:<42}: {'YES' if val else 'NO'}")
    typer.echo(f"\nReport: {out}/j56c_profile_corpus_report.md")

    if not all(bv.values()):
        failed = [k for k, v in bv.items() if not v]
        typer.echo(f"FAIL: behavioral validation criteria not met: {', '.join(failed)}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
