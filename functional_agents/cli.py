"""CLI for the functional agent pipeline (J5.0a.7 / J13.1)."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

from research_agent.cli import _configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ---------------------------------------------------------------------------
# session sub-app (J13.1)
# ---------------------------------------------------------------------------

session_app = typer.Typer(no_args_is_help=True, help="Manage research sessions.")
app.add_typer(session_app, name="session")


def _make_run_dir() -> Path:
    """Create and return a timestamped run directory under outputs/runs/."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("outputs/runs") / f"RUN-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_run_artifacts(ctx: object, run_dir: Path) -> None:
    """Write research_object.json and engagement.json into the run directory."""
    ro = getattr(ctx, "research_object", None)
    if ro:
        try:
            (run_dir / "research_object.json").write_text(
                json.dumps(ro, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logging.warning("[RunArtifacts] research_object.json write failed: %s", exc)

    engagement_meta = {}
    _trace = getattr(ctx, "trace", {}) or {}
    _engagement_id = _trace.get("_engagement_id")
    if _engagement_id:
        engagement_meta["engagement_id"] = _engagement_id
    _engagement = _trace.get("_engagement")
    if _engagement:
        engagement_meta.update(_engagement)
    if engagement_meta:
        try:
            (run_dir / "engagement.json").write_text(
                json.dumps(engagement_meta, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logging.warning("[RunArtifacts] engagement.json write failed: %s", exc)


def _print_run_summary(
    ctx: object,
    mode: str,
    out_dir: Path,
    elapsed_s: float,
    *,
    is_run_dir: bool,
    session_file: Path | None = None,
) -> None:
    """Print a structured post-run summary to stdout."""
    _trace = getattr(ctx, "trace", {}) or {}
    perf = _trace.get("_performance", {})
    totals = perf.get("totals", {}) if perf else {}
    tokens = totals.get("total_tokens", 0)
    llm_calls = totals.get("llm_call_count", 0)
    workflow_state = getattr(ctx, "workflow_state", "COMPLETE")

    run_id = getattr(ctx, "run_id", "")
    agents_run = [h["agent"] for h in (getattr(ctx, "agent_history", None) or [])]
    profiles = list(getattr(ctx, "profiles", None) or [])

    status = str(workflow_state).upper().replace("WORKFLOWSTATE.", "")
    if status == "COMPLETE":
        status_label = "SUCCESS"
    elif status == "ERROR":
        status_label = "ERROR"
    else:
        status_label = "PARTIAL"

    deliverables = getattr(ctx, "deliverables", None) or []

    typer.echo("")
    typer.echo(f"Run:      {run_id}")
    typer.echo(f"Mode:     {mode}")
    typer.echo(f"Profiles: {', '.join(profiles)}")
    typer.echo(f"Status:   {status_label}")
    typer.echo(f"Agents:   {len(agents_run)} run")
    typer.echo(f"Elapsed:  {elapsed_s:.1f}s")
    if tokens:
        typer.echo(f"Tokens:   {tokens:,}  ({llm_calls} LLM calls)")
    if session_file is not None:
        typer.echo(f"Session:  {session_file}")

    report_path = (getattr(ctx, "artifacts", None) or {}).get("report_path")
    typer.echo("")
    typer.echo("Deliverables:")
    if report_path:
        typer.echo(f"  [OK] Report             {report_path}")
    trace_path = (getattr(ctx, "artifacts", None) or {}).get("trace_path")
    if trace_path:
        typer.echo(f"  [OK] Agent trace        {trace_path}")
    canonical_trace = out_dir / "pipeline.trace.json"
    if canonical_trace.exists():
        typer.echo(f"  [OK] Pipeline trace     {canonical_trace}")
    if is_run_dir:
        ro_path = out_dir / "research_object.json"
        if ro_path.exists():
            typer.echo(f"  [OK] Research object    {ro_path}")
        eng_path = out_dir / "engagement.json"
        if eng_path.exists():
            typer.echo(f"  [OK] Engagement         {eng_path}")
    for d in deliverables:
        dpath = d.get("path", "")
        dtype = d.get("type", "")
        if dpath and dpath != report_path and Path(dpath).exists():
            typer.echo(f"  [OK] {dtype:<19} {dpath}")

    typer.echo("")
    typer.echo(f"Output:   {out_dir}/")


@app.command("run")
def main(
    question: Annotated[str, typer.Argument(help="Research question to answer. Omit if using --goal.")] = "",
    goal: Annotated[
        str | None,
        typer.Option("--goal", "-g", help="High-level business goal (goal-driven mode). Mutually exclusive with QUESTION and --engagement."),
    ] = None,
    engagement: Annotated[
        Path | None,
        typer.Option(
            "--engagement",
            help="Path to a Strategic Engagement file (.yaml/.yml/.json). Strategic Engagement Mode. Mutually exclusive with QUESTION and --goal.",
        ),
    ] = None,
    session_file: Annotated[
        Path | None,
        typer.Option(
            "--session",
            help=(
                "Path to a session file (.json). Session-driven mode: loads the engagement from the "
                "session and updates the session after the run. Mutually exclusive with QUESTION, --goal, --engagement."
            ),
        ),
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
        Path | None,
        typer.Option("--out", "-o", help="Markdown report path. Default: outputs/runs/RUN-<timestamp>/report.md"),
    ] = None,
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
    knowledge_store: Annotated[
        Path | None,
        typer.Option(
            "--knowledge-store",
            help=(
                "Path to the Knowledge Store directory. Enables Knowledge Layer evidence retrieval "
                "(hybrid semantic + lexical + LLM reranking) instead of legacy document extraction. "
                "Defaults to 'knowledge_store/' if that directory exists."
            ),
        ),
    ] = None,
    rerank: Annotated[
        bool,
        typer.Option("--rerank/--no-rerank", help="Apply LLM reranking to retrieved evidence (requires ANTHROPIC_API_KEY)."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level: DEBUG, INFO, PROGRESS, WARNING, ERROR, CRITICAL. Default: PROGRESS."),
    ] = None,
    incremental: Annotated[
        bool,
        typer.Option(
            "--incremental",
            help=(
                "Incremental execution mode (J13.4). Requires --session. "
                "Runs only the agents needed to restore stale PERSISTED state instead of "
                "the full pipeline. The session must have StateChanges to analyze."
            ),
        ),
    ] = False,
    save_context: Annotated[
        Path | None,
        typer.Option(
            "--save-context",
            help=(
                "Persist the completed AgentContext as JSON suitable for replay with "
                "`functional_agents.run_agent --fixture`. Only written on successful completion."
            ),
        ),
    ] = None,
) -> None:
    """Run the functional agent pipeline and write a Markdown research memo.

    Either pass a QUESTION as a positional argument (question-driven mode) or
    use --goal for goal-driven mode where ProblemFramingAgent derives the
    research questions automatically.

    When a knowledge store is available (default: knowledge_store/), evidence is
    retrieved via the Knowledge Layer (hybrid retrieval + optional LLM reranking)
    instead of the legacy document extraction pipeline.

    With --incremental (requires --session): only the required agents run based on
    the session's recorded StateChanges. The full pipeline is NOT run.
    """

    # J13.4 — incremental mode requires --session
    if incremental and session_file is None:
        typer.echo(
            "Error: --incremental requires --session (a session file with prior StateChanges).",
            err=True,
        )
        raise typer.Exit(code=1)

    # J9.1 / J13.1 – four mutually exclusive entry points.
    provided = [
        name for name, val in (
            ("QUESTION", bool(question)),
            ("--goal", bool(goal)),
            ("--engagement", engagement is not None),
            ("--session", session_file is not None),
        ) if val
    ]
    if len(provided) > 1:
        typer.echo(
            f"Error: provide exactly one of QUESTION, --goal, --engagement, or --session "
            f"(got: {', '.join(provided)}).",
            err=True,
        )
        raise typer.Exit(code=1)
    if not provided:
        typer.echo("Error: provide a QUESTION, --goal, --engagement, or --session.", err=True)
        raise typer.Exit(code=1)

    # Load and validate the engagement up front so errors are clear and early.
    engagement_spec = None
    _cli_session = None  # J13.1 — loaded session when --session is used

    if session_file is not None:
        # J13.1 — session-driven mode: load engagement spec from the session file.
        from .session import load_session_file, SessionNotFoundError
        try:
            _cli_session = load_session_file(session_file)
        except SessionNotFoundError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        # J13.4 — incremental mode does not require engagement_spec; it reads
        # ResearchState and StateChanges directly from the session.
        if not incremental:
            spec_dict = _cli_session.metadata.get("engagement_spec") or {}
            if not spec_dict:
                typer.echo(
                    f"Error: session {session_file} has no engagement_spec in metadata. "
                    "Create the session with 'session create --engagement <file>'.",
                    err=True,
                )
                raise typer.Exit(code=1)
            from .engagement_spec import EngagementSpec, EngagementError
            try:
                engagement_spec = EngagementSpec.model_validate(spec_dict)
            except Exception as exc:
                typer.echo(f"Error: could not reconstruct engagement spec from session: {exc}", err=True)
                raise typer.Exit(code=1) from exc

    elif engagement is not None:
        from .engagement_spec import load_engagement_spec, EngagementError
        try:
            engagement_spec = load_engagement_spec(engagement)
        except EngagementError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc

    _configure_logging(verbose=False, log_level=log_level or "PROGRESS")

    # Resolve output path — auto-create a timestamped run directory when --out is not set.
    is_run_dir = out is None
    if is_run_dir:
        run_dir = _make_run_dir()
        out = run_dir / "report.md"
    else:
        run_dir = Path(out).parent

    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]

    # Auto-detect knowledge store when not explicitly provided
    resolved_ks: Path | None = knowledge_store
    if resolved_ks is None:
        default_ks = Path("knowledge_store")
        if default_ks.exists():
            resolved_ks = default_ks

    # Build client
    client = _build_client(mock=mock, model=model, use_extraction_cache=use_extraction_cache)

    # J13.4 — incremental execution mode: use IncrementalExecutor, then return.
    if incremental:
        _configure_logging(verbose=False, log_level=log_level or "PROGRESS")
        assert _cli_session is not None  # guaranteed by the earlier mutual-exclusivity check
        if not _cli_session.state_changes:
            typer.echo(
                f"Session {session_file} has no StateChanges — nothing to run incrementally.\n"
                "Use 'run --session' (without --incremental) to do a full pipeline run.",
                err=True,
            )
            raise typer.Exit(code=1)
        from .staleness import DependencyReasoner
        from .planning import ExecutionPlanner
        from .execution import IncrementalExecutor, ExecutionStatus
        from .session import save_session_file

        staleness_plan = DependencyReasoner().analyze(
            _cli_session.research_state, _cli_session.state_changes
        )
        if not staleness_plan.stale_agents:
            typer.echo("Nothing is stale — no incremental execution needed.")
            raise typer.Exit(code=0)

        execution_plan = ExecutionPlanner().plan(staleness_plan)
        if not execution_plan.required_agents:
            typer.echo("Execution plan is empty — no agents to run.")
            raise typer.Exit(code=0)

        _inc_out = out if out is not None else Path("outputs") / "report.md"
        executor = IncrementalExecutor(
            client=client,
            profile_names=profile_names,
            sources_dir=sources,
            out_path=_inc_out,
            top_evidence=top_evidence,
            top_chunks=top_chunks,
            web_search=web_search,
            knowledge_store=resolved_ks,
            use_reranker=rerank,
        )

        _inc_start = time.monotonic()
        result = executor.execute(execution_plan, _cli_session)
        _inc_elapsed = time.monotonic() - _inc_start

        try:
            save_session_file(result.session, session_file)
        except Exception as exc:
            logging.warning("[Incremental] session save failed: %s", exc)

        typer.echo("")
        typer.echo(f"Incremental Execution")
        typer.echo(f"Status:          {result.status}")
        typer.echo(f"Elapsed:         {_inc_elapsed:.1f}s")
        typer.echo(f"Agents run:      {len(result.completed_agents)}/{len(execution_plan.required_agents)}")
        typer.echo(f"Plan steps:      {execution_plan.estimated_steps}")
        if result.failed_agent:
            typer.echo(f"Failed at:       {result.failed_agent}")
            typer.echo(f"Reason:          {result.failure_reason}")
        typer.echo(f"Session:         {session_file}")
        typer.echo(f"Plan ID:         {execution_plan.plan_id}")
        typer.echo("")
        if result.completed_agents:
            typer.echo(f"Completed agents ({len(result.completed_agents)}):")
            for a in result.completed_agents:
                typer.echo(f"  {a}")
            typer.echo("")
        if result.status == ExecutionStatus.FAILED:
            raise typer.Exit(code=1)
        return

    from .orchestrator import Orchestrator

    orchestrator = Orchestrator(
        profile_names=profile_names,
        sources_dir=sources,
        out_path=out,
        client=client,
        top_evidence=top_evidence,
        top_chunks=top_chunks,
        web_search=web_search,
        knowledge_store=resolved_ks,
        use_reranker=rerank,
    )

    _start = time.monotonic()
    try:
        if session_file is not None:
            mode = "Strategic Engagement (session)"
            ctx = orchestrator.run_from_engagement(engagement_spec)
        elif engagement_spec is not None:
            mode = "Strategic Engagement"
            ctx = orchestrator.run_from_engagement(engagement_spec)
        elif goal:
            mode = "Research (goal)"
            ctx = orchestrator.run_from_goal(goal)
        else:
            mode = "Research (question)"
            ctx = orchestrator.run(question)
    except Exception as exc:
        logging.error("Functional agent pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc
    elapsed = time.monotonic() - _start

    # J13.1 — update and persist the explicit session after the run.
    if _cli_session is not None and session_file is not None:
        try:
            from .session import ResearchState, IterationRecord, save_session_file
            trigger = "continuation" if _cli_session.iteration_history else "initial"
            _cli_session.research_state = ResearchState.from_context(ctx)
            _cli_session.add_iteration(IterationRecord(
                iteration_number=len(_cli_session.iteration_history),
                timestamp=datetime.now(timezone.utc).isoformat(),
                trigger=trigger,
                summary=f"Pipeline completed — run_id={getattr(ctx, 'run_id', '')}",
                completed_tasks=[],
                notes="",
            ))
            _cli_session.take_snapshot()
            _cli_session.complete()
            save_session_file(_cli_session, session_file)
        except Exception as exc:
            logging.warning("[Session] session update failed: %s", exc)

    if is_run_dir:
        _write_run_artifacts(ctx, run_dir)

    # DX1 — persist completed AgentContext for isolated agent replay.
    if save_context is not None:
        from .context_snapshot import context_to_jsonable
        try:
            save_context.parent.mkdir(parents=True, exist_ok=True)
            save_context.write_text(
                json.dumps(context_to_jsonable(ctx), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            typer.echo(f"Context saved → {save_context}")
        except Exception as exc:
            raise RuntimeError(
                f"--save-context: failed to write AgentContext to {save_context}: {exc}"
            ) from exc

    _print_run_summary(
        ctx, mode, run_dir, elapsed,
        is_run_dir=is_run_dir,
        session_file=session_file,
    )


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


@app.command("analyze-extraction")
def analyze_extraction_cmd(
    document: Annotated[
        str | None,
        typer.Option("--document", "-d", help="Filter by document name (partial match). Default: all documents."),
    ] = None,
    chunks: Annotated[
        int,
        typer.Option("--chunks", "-n", help="Maximum number of zero-yield chunks to analyze."),
    ] = 10,
    sources: Annotated[
        Path,
        typer.Option("--sources", "-s", exists=True, file_okay=False, dir_okay=True,
                     help="Directory containing source documents."),
    ] = Path("sources"),
    profiles: Annotated[
        str,
        typer.Option("--profiles", help="Comma-separated profile names."),
    ] = "ai_data_centers",
    question: Annotated[
        str,
        typer.Option("--question", "-q", help="Research question used for production-prompt extraction."),
    ] = "What are the power and cooling requirements for AI data centers?",
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output path for Markdown report."),
    ] = Path("outputs/extraction_analysis.md"),
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name."),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Diagnose evidence extraction failures.

    Automatically selects evidence-dense chunks with zero evidence yield,
    runs the production prompt and a simplified permissive prompt against
    each chunk, and writes a side-by-side Markdown report.

    Requires ANTHROPIC_API_KEY.  Never modifies production extraction.
    """
    import json as _json

    _configure_logging(verbose=False, log_level=log_level or "INFO")

    if not os.getenv("ANTHROPIC_API_KEY"):
        typer.echo("Error: ANTHROPIC_API_KEY is required for extraction analysis.", err=True)
        raise typer.Exit(code=1)

    from research_agent.claude_client import ClaudeClient
    from research_agent.loaders import load_sources
    from research_agent.chunker import chunk_documents, compute_chunk_diagnostics
    from research_agent.retrieval import select_top_chunks_multi
    from research_agent.retrieval_planner import RetrievalPlanner
    from research_agent.agent import extract_evidence, _retrieval_scores_to_chunk_scores
    from research_agent.source_quality import build_source_quality_map
    from research_agent.profile import load_profile, get_default_profile
    from research_agent.evidence_recovery import attribute_evidence_to_chunks
    from .extraction_experiment import (
        run_analysis, build_report, build_json_artifact, build_comparison_table,
    )

    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]
    profile = load_profile(profile_names[0]) if profile_names else get_default_profile()

    try:
        client = ClaudeClient(model=model)
    except Exception as exc:
        typer.echo(f"Error: failed to build Claude client: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Loading documents from {sources}…")
    collection = load_sources(sources)
    documents = collection.documents
    if not documents:
        typer.echo(f"Error: no documents found in {sources}", err=True)
        raise typer.Exit(code=1)
    if collection.errors:
        for err in collection.errors:
            typer.echo(f"Warning: {err.path.name}: {err.message}", err=True)
    typer.echo(f"Loaded {len(documents)} documents.")

    source_quality_map = build_source_quality_map(
        [doc.path.name for doc in documents], profile=profile
    )
    all_chunks = chunk_documents(documents)
    planner = RetrievalPlanner(profile=profile)
    retrieval_plan = planner.plan(question)
    selected_chunks, retrieval_scores, _ = select_top_chunks_multi(
        all_chunks, retrieval_plan.queries, top_n=20, source_quality_map=source_quality_map
    )
    typer.echo(f"Chunked: {len(all_chunks)} total, {len(selected_chunks)} selected by retrieval.")

    evidence = extract_evidence(question, documents, source_quality_map=source_quality_map, profile=profile)
    evidence = attribute_evidence_to_chunks(evidence, all_chunks)
    chunk_scores = _retrieval_scores_to_chunk_scores(all_chunks, retrieval_scores)
    chunk_diagnostics = [
        cd.model_dump()
        for cd in compute_chunk_diagnostics(all_chunks, selected_chunks, evidence, chunk_scores)
    ]
    chunks_by_id = {c.chunk_id: c for c in all_chunks}

    zero_yield = sum(
        1 for cd in chunk_diagnostics
        if cd.get("chunk_type") == "evidence_dense" and cd.get("evidence_items_created", 0) == 0
    )
    typer.echo(
        f"Diagnostics: {len(chunk_diagnostics)} chunks, {zero_yield} evidence-dense zero-yield. "
        f"Analyzing top {chunks}…"
    )

    analyses, summary = run_analysis(
        client=client,
        question=question,
        chunk_diagnostics=chunk_diagnostics,
        chunks_by_id=chunks_by_id,
        document_filter=document,
        limit=chunks,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(analyses, summary, question=question, document_filter=document)
    out.write_text(report, encoding="utf-8")
    typer.echo(f"Report written to {out}")

    json_out = out.with_suffix(".json")
    json_out.write_text(_json.dumps(build_json_artifact(analyses, summary), indent=2), encoding="utf-8")
    typer.echo(f"JSON artifact written to {json_out}")

    typer.echo("")
    typer.echo(build_comparison_table(analyses))
    typer.echo("")
    typer.echo(f"Chunks analyzed       : {summary.chunks_analyzed}")
    typer.echo(f"Successful comparisons: {summary.successful_comparisons}")
    typer.echo(f"Tool failures         : {summary.tool_failures}")
    typer.echo(f"Simple prompt wins    : {summary.simple_prompt_wins}")
    typer.echo(f"Production prompt wins: {summary.production_prompt_wins}")
    typer.echo(f"Equivalent            : {summary.equivalent}")
    typer.echo(f"Production evidence   : {summary.production_evidence}")
    typer.echo(f"Simple evidence       : {summary.simple_evidence}")
    typer.echo(f"Average gain          : {summary.average_gain:+.1f} per successful chunk")
    typer.echo(f"Most likely failure   : {summary.most_likely_failure_mode}")
    typer.echo("")
    typer.echo("Diagnosis breakdown:")
    for diag, count in sorted(summary.diagnosis_breakdown.items(), key=lambda x: -x[1]):
        pct = 100 * count // max(summary.chunks_analyzed, 1)
        typer.echo(f"  {diag:<45}: {count} ({pct}%)")


# ---------------------------------------------------------------------------
# session create (J13.1)
# ---------------------------------------------------------------------------

@session_app.command("create")
def session_create_cmd(
    engagement: Annotated[
        Path,
        typer.Option(
            "--engagement",
            help="Path to a Strategic Engagement file (.yaml/.yml/.json).",
        ),
    ],
    session_out: Annotated[
        Path,
        typer.Option("--session", help="Output path for the new session file."),
    ],
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Create a new research session from a Strategic Engagement file.

    The session file stores the engagement spec and is ready to be passed to
    'run --session' for pipeline execution.

    Example:
        python3 -m functional_agents.cli session create \\
            --engagement engagements/my_engagement.yaml \\
            --session outputs/my_session.json
    """
    _configure_logging(verbose=False, log_level=log_level or "WARNING")

    from .engagement_spec import load_engagement_spec, EngagementError
    try:
        spec = load_engagement_spec(engagement)
    except EngagementError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    from .session import ResearchSession, ResearchState, save_session_file

    session = ResearchSession.create(
        metadata={
            "engagement_file": str(engagement),
            "engagement_spec": spec.model_dump(),
            "run_mode": "strategic_engagement",
        },
        research_state=ResearchState(),
    )

    try:
        save_session_file(session, session_out)
    except Exception as exc:
        typer.echo(f"Error: could not write session file: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(f"Session created:  {session_out}")
    typer.echo(f"Session ID:       {session.session_id}")
    typer.echo(f"Engagement:       {engagement}")
    typer.echo(f"Status:           {session.status}")
    typer.echo("")
    typer.echo("Ready to run:")
    typer.echo(f"  python3 -m functional_agents.cli run \\")
    typer.echo(f"      --session {session_out} \\")
    typer.echo(f"      --profiles <profile1,profile2> \\")
    typer.echo(f"      --out <output.md>")


# ---------------------------------------------------------------------------
# session show (J13.1)
# ---------------------------------------------------------------------------

@session_app.command("show")
def session_show_cmd(
    session_path: Annotated[
        Path,
        typer.Option("--session", help="Path to the session file."),
    ],
) -> None:
    """Show research session status, metadata, and iteration history.

    Example:
        python3 -m functional_agents.cli session show \\
            --session outputs/my_session.json
    """
    from .session import load_session_file, SessionNotFoundError

    try:
        session = load_session_file(session_path)
    except SessionNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    state = session.research_state
    meta = session.metadata

    def _present(d: dict) -> str:
        return "present" if d else "empty"

    typer.echo("")
    typer.echo(f"Session:      {session.session_id}")
    typer.echo(f"Status:       {session.status}")
    typer.echo(f"Created:      {session.created_at}")
    typer.echo(f"Updated:      {session.updated_at}")
    typer.echo("")
    typer.echo("Metadata:")
    eng_file = meta.get("engagement_file") or "(not set)"
    typer.echo(f"  Engagement:  {eng_file}")
    profiles = meta.get("profiles") or []
    typer.echo(f"  Profiles:    {', '.join(profiles) if profiles else '(not set)'}")
    typer.echo(f"  Run mode:    {meta.get('run_mode') or '(not set)'}")
    run_id = meta.get("run_id") or ""
    if run_id:
        typer.echo(f"  Run ID:      {run_id}")
    typer.echo("")
    typer.echo("Research State:")
    typer.echo(f"  engagement             [{_present(state.engagement)}]")
    typer.echo(f"  research_object        [{_present(state.research_object)}]")
    typer.echo(f"  decision_model         [{_present(state.decision_model)}]")
    typer.echo(f"  research_gap_analysis  [{_present(state.research_gap_analysis)}]")
    typer.echo(f"  executive_confidence   [{_present(state.executive_confidence)}]")
    typer.echo(f"  iteration_plan         [{_present(state.iteration_plan)}]")
    typer.echo("")
    typer.echo(f"Iterations:   {len(session.iteration_history)}")
    typer.echo(f"Snapshots:    {len(session.snapshots)}")

    if session.iteration_history:
        typer.echo("")
        typer.echo("Iteration History:")
        for rec in session.iteration_history:
            tasks_str = f"  tasks={rec.completed_tasks}" if rec.completed_tasks else ""
            typer.echo(
                f"  [{rec.iteration_number}]  {rec.timestamp[:19]}  "
                f"{rec.trigger:<14}  {rec.summary}{tasks_str}"
            )

    if state.iteration_plan:
        tasks = state.iteration_plan.get("priority_research_tasks") or []
        iteration_needed = state.iteration_plan.get("iteration_needed")
        typer.echo("")
        typer.echo("Iteration Plan:")
        typer.echo(f"  iteration_needed:  {iteration_needed}")
        typer.echo(f"  tasks:             {len(tasks)}")
        if tasks:
            for t in tasks[:5]:
                typer.echo(f"    {t.get('task_id', '?')}  {t.get('task_title', '')[:70]}")
            if len(tasks) > 5:
                typer.echo(f"    … and {len(tasks) - 5} more")

    typer.echo("")


# ---------------------------------------------------------------------------
# dependencies sub-app (J13.1)
# ---------------------------------------------------------------------------

dependencies_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect agent dependency declarations.",
)
app.add_typer(dependencies_app, name="dependencies")


@dependencies_app.command("list")
def dep_list() -> None:
    """List all registered agent dependency declarations."""
    from functional_agents.dependencies import DependencyRegistry
    deps = DependencyRegistry.list_dependencies()
    typer.echo(f"Registered agents ({len(deps)}):")
    for dep in deps:
        typer.echo(
            f"  {dep.agent_name:<40}  "
            f"consumes={len(dep.consumes):<3} "
            f"produces={len(dep.produces):<3} "
            f"invalidates={len(dep.invalidates)}"
        )


@dependencies_app.command("show")
def dep_show(
    agent: str = typer.Option(..., "--agent", help="Agent class name (e.g. EvidenceAgent)"),
) -> None:
    """Show dependency details for a specific agent."""
    from functional_agents.dependencies import DependencyRegistry
    try:
        dep = DependencyRegistry.get_dependency(agent)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{dep.agent_name}")
    typer.echo("")
    typer.echo("  Consumes:")
    for p in dep.consumes:
        typer.echo(f"    - {p}")
    if not dep.consumes:
        typer.echo("    (none)")
    typer.echo("")
    typer.echo("  Produces:")
    for p in dep.produces:
        typer.echo(f"    - {p}")
    if not dep.produces:
        typer.echo("    (none)")
    typer.echo("")
    typer.echo("  Invalidates:")
    for p in dep.invalidates:
        typer.echo(f"    - {p}")
    if not dep.invalidates:
        typer.echo("    (none — terminal agent)")
    typer.echo("")


@dependencies_app.command("affected")
def dep_affected(
    path: str = typer.Option(
        ..., "--path", help="Logical path to query (e.g. research_object.evidence)"
    ),
) -> None:
    """Show agents consuming, producing, or invalidated by a given path."""
    from functional_agents.dependencies import DependencyRegistry

    consumers = DependencyRegistry.agents_consuming(path)
    producers = DependencyRegistry.agents_producing(path)
    invalidated = DependencyRegistry.agents_invalidated_by(path)

    typer.echo(f"\nAffected by: {path}")
    typer.echo("")
    typer.echo(f"  Producers ({len(producers)}) — agents that write this path:")
    for a in producers:
        typer.echo(f"    {a}")
    if not producers:
        typer.echo("    (none)")
    typer.echo("")
    typer.echo(f"  Consumers ({len(consumers)}) — agents that read this path:")
    for a in consumers:
        typer.echo(f"    {a}")
    if not consumers:
        typer.echo("    (none)")
    typer.echo("")
    typer.echo(
        f"  Invalidated by producers ({len(invalidated)}) — "
        "agents that declare this path in their invalidates list:"
    )
    for a in invalidated:
        typer.echo(f"    {a}")
    if not invalidated:
        typer.echo("    (none)")
    typer.echo("")


# ---------------------------------------------------------------------------
# staleness sub-app (J13.2)
# ---------------------------------------------------------------------------

staleness_app = typer.Typer(
    no_args_is_help=True,
    help="Dependency staleness analysis (J13.2).",
)
app.add_typer(staleness_app, name="staleness")


def _print_staleness_plan(plan: object) -> None:
    """Render a StalenessPlan to stdout."""
    typer.echo(f"\nStaleness Plan:  {plan.plan_id}")
    typer.echo(f"Confidence:      {plan.confidence}")
    typer.echo(f"Created:         {plan.created_at}")
    if plan.source_changes:
        typer.echo(f"Source changes:  {', '.join(plan.source_changes)}")
    else:
        typer.echo("Source changes:  (none)")

    typer.echo("")
    typer.echo(f"Changed paths ({len(plan.changed_paths)}):")
    for p in plan.changed_paths:
        from functional_agents.staleness import classify_path
        kind = classify_path(p)
        typer.echo(f"  {p:<45}  [{kind}]")
    if not plan.changed_paths:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"Stale paths ({len(plan.stale_paths)}):")
    for p in plan.stale_paths:
        from functional_agents.staleness import classify_path
        kind = classify_path(p)
        typer.echo(f"  {p:<45}  [{kind}]")
    if not plan.stale_paths:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"Required producers ({len(plan.required_producers)}):")
    for a in plan.required_producers:
        typer.echo(f"  {a}")
    if not plan.required_producers:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"Stale agents ({len(plan.stale_agents)}):")
    for a in plan.stale_agents:
        typer.echo(f"  {a}")
    if not plan.stale_agents:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"Execution-only paths ({len(plan.execution_only_paths)}):")
    for p in plan.execution_only_paths:
        typer.echo(f"  {p}")
    if not plan.execution_only_paths:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"External dependencies ({len(plan.external_dependencies)}):")
    for p in plan.external_dependencies:
        typer.echo(f"  {p}")
    if not plan.external_dependencies:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo("Reasoning:")
    if plan.reasoning:
        for path, reason in sorted(plan.reasoning.items()):
            # Wrap long reasons for readability
            typer.echo(f"  {path}:")
            typer.echo(f"    {reason}")
    else:
        typer.echo("  (none)")
    typer.echo("")


@staleness_app.command("explain")
def staleness_explain(
    session_file: Path = typer.Option(
        ..., "--session", help="Path to a session JSON file."
    ),
) -> None:
    """Analyze all StateChanges in a session and print the StalenessPlan."""
    from functional_agents.session import load_session_file
    from functional_agents.staleness import DependencyReasoner

    try:
        session = load_session_file(session_file)
    except Exception as exc:
        typer.echo(f"Error loading session: {exc}", err=True)
        raise typer.Exit(1)

    if not session.state_changes:
        typer.echo("Session has no StateChanges. Nothing to analyze.")
        raise typer.Exit(0)

    plan = DependencyReasoner().analyze(session.research_state, session.state_changes)
    _print_staleness_plan(plan)


@staleness_app.command("path")
def staleness_path(
    path: str = typer.Option(
        ..., "--path", help="Logical path to analyze (e.g. research_object.evidence)."
    ),
) -> None:
    """Show what would be stale if the given path changed."""
    from functional_agents.session.state_change import ChangeType, StateChange
    from functional_agents.staleness import DependencyReasoner

    synthetic = StateChange.create(
        source="cli",
        change_type=ChangeType.UPDATE,
        affected_paths=[path],
        description=f"Synthetic change for staleness analysis of '{path}'",
    )
    plan = DependencyReasoner().analyze(None, [synthetic])
    _print_staleness_plan(plan)


@staleness_app.command("agent")
def staleness_agent(
    agent: str = typer.Option(
        ..., "--agent", help="Agent class name (e.g. EvidenceAgent)."
    ),
) -> None:
    """Show what would be stale if the given agent re-ran."""
    from functional_agents.dependencies import DependencyRegistry
    from functional_agents.session.state_change import ChangeType, StateChange
    from functional_agents.staleness import DependencyReasoner

    try:
        dep = DependencyRegistry.get_dependency(agent)
    except KeyError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1)

    synthetics = [
        StateChange.create(
            source="cli",
            change_type=ChangeType.REPLACE,
            affected_paths=[p],
            description=f"Synthetic: {agent} re-ran and replaced '{p}'",
        )
        for p in dep.produces
    ]
    plan = DependencyReasoner().analyze(None, synthetics)
    _print_staleness_plan(plan)


# ---------------------------------------------------------------------------
# execution sub-app (J13.3)
# ---------------------------------------------------------------------------

execution_app = typer.Typer(
    no_args_is_help=True,
    help="Execution planning — convert a StalenessPlan into an ExecutionPlan (J13.3).",
)
app.add_typer(execution_app, name="execution")


def _print_execution_plan(plan: object) -> None:
    """Render an ExecutionPlan to stdout."""
    typer.echo(f"\nExecution Plan:  {plan.plan_id}")
    typer.echo(f"Confidence:      {plan.confidence}")
    typer.echo(f"Created:         {plan.created_at}")
    typer.echo(f"Staleness Plan:  {plan.staleness_plan_id}")
    if plan.triggering_state_changes:
        typer.echo(f"State changes:   {', '.join(plan.triggering_state_changes)}")
    else:
        typer.echo("State changes:   (none)")

    typer.echo("")
    typer.echo(f"Required agents ({len(plan.required_agents)}):")
    for a in plan.required_agents:
        typer.echo(f"  {a}")
    if not plan.required_agents:
        typer.echo("  (none)")

    typer.echo("")
    typer.echo(f"Optional agents ({len(plan.optional_agents)}):")
    for a in plan.optional_agents:
        typer.echo(f"  {a}")
    if not plan.optional_agents:
        typer.echo("  (none — all stale agents are required)")

    if plan.blocked_agents:
        typer.echo("")
        typer.echo(f"Blocked agents ({len(plan.blocked_agents)}):")
        for a in plan.blocked_agents:
            reason = plan.blocked_reasons.get(a, "unknown")
            typer.echo(f"  {a}: {reason}")

    typer.echo("")
    typer.echo(f"Execution groups ({plan.estimated_steps} steps):")
    for i, group in enumerate(plan.execution_groups, 1):
        parallel = " [parallel]" if len(group) > 1 else ""
        typer.echo(f"  Step {i}{parallel}:")
        for a in group:
            tag = "(optional)" if a in set(plan.optional_agents) else "(required)"
            typer.echo(f"    {a}  {tag}")
    if not plan.execution_groups:
        typer.echo("  (no agents to execute)")

    typer.echo("")
    typer.echo("Reasoning:")
    if plan.reasoning:
        for agent_name, reason in sorted(plan.reasoning.items()):
            typer.echo(f"  {agent_name}:")
            typer.echo(f"    {reason}")
    else:
        typer.echo("  (none)")
    typer.echo("")


@execution_app.command("plan")
def execution_plan_cmd(
    session_file: Path = typer.Option(
        ..., "--session", help="Path to a session JSON file."
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show full reasoning for each agent."
    ),
) -> None:
    """Compute the execution plan for a session's StateChanges.

    Analyzes all StateChanges in the session via DependencyReasoner, then
    converts the resulting StalenessPlan into a topologically-ordered
    ExecutionPlan showing which agents must run, in what sequence, and
    which can execute in parallel.

    Example:
        python3 -m functional_agents.cli execution plan \\
            --session outputs/sessions/my_session.json
    """
    from functional_agents.session import load_session_file
    from functional_agents.staleness import DependencyReasoner
    from functional_agents.planning import ExecutionPlanner

    try:
        session = load_session_file(session_file)
    except Exception as exc:
        typer.echo(f"Error loading session: {exc}", err=True)
        raise typer.Exit(1)

    if not session.state_changes:
        typer.echo("Session has no StateChanges — nothing to plan.")
        raise typer.Exit(0)

    staleness_plan = DependencyReasoner().analyze(
        session.research_state, session.state_changes
    )
    execution_plan = ExecutionPlanner().plan(staleness_plan)

    if verbose:
        _print_execution_plan(execution_plan)
    else:
        # Compact summary
        typer.echo(f"\nExecution Plan:  {execution_plan.plan_id}")
        typer.echo(f"Confidence:      {execution_plan.confidence}")
        typer.echo(f"Steps:           {execution_plan.estimated_steps}")
        typer.echo(f"Required agents: {len(execution_plan.required_agents)}")
        typer.echo(f"Optional agents: {len(execution_plan.optional_agents)}")
        if execution_plan.blocked_agents:
            typer.echo(f"Blocked agents:  {len(execution_plan.blocked_agents)}")
        typer.echo("")
        typer.echo(f"Execution order ({len(execution_plan.execution_order)} agents):")
        for i, group in enumerate(execution_plan.execution_groups, 1):
            parallel = " [parallel]" if len(group) > 1 else ""
            agents_str = ", ".join(group)
            typer.echo(f"  Step {i}{parallel}: {agents_str}")
        typer.echo("")


# ---------------------------------------------------------------------------
# debug sub-app — developer-only isolated agent debugging (PF1)
# ---------------------------------------------------------------------------

debug_app = typer.Typer(
    no_args_is_help=True,
    help="Developer debugging commands — isolated agent execution.",
)
app.add_typer(debug_app, name="debug")


@debug_app.command("problem-framing")
def debug_problem_framing_cmd(
    engagement: Annotated[
        Path,
        typer.Option(
            "--engagement",
            help="Path to a Strategic Engagement file (.yaml/.yml/.json).",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for debug artifacts."),
    ] = Path("outputs/problem_framing"),
    profiles: Annotated[
        str,
        typer.Option("--profiles", help="Comma-separated profile names."),
    ] = "ai_data_centers",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock client instead of Claude."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run ONLY ProblemFramingAgent against an Engagement YAML and write debug artifacts.

    Writes engagement.json, prompt.txt, raw_response.json, decision_model.json,
    and trace.json to the output directory.  No downstream agents execute.

    Example:
        python3 -m functional_agents.cli debug problem-framing \\
            --engagement engagements/my_engagement.yaml \\
            --out outputs/problem_framing/
    """
    import hashlib
    import os
    import time
    from datetime import datetime, timezone

    _configure_logging(verbose=False, log_level=log_level or "PROGRESS")

    # ---- Load engagement spec -----------------------------------------------
    from .engagement_spec import load_engagement_spec, EngagementError
    try:
        spec = load_engagement_spec(engagement)
    except EngagementError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    # ---- Build profiles / client --------------------------------------------
    from research_agent.profile import load_profile
    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]
    loaded_profiles = []
    for name in profile_names:
        try:
            loaded_profiles.append(load_profile(name))
        except Exception as exc:
            typer.echo(f"Warning: could not load profile '{name}': {exc}", err=True)

    client = _build_client(mock=mock, model=model)

    # ---- Artifact capture state ---------------------------------------------
    _raw_captures: dict[str, object] = {}
    _prompt_captures: dict[str, str] = {}
    _cache_hits: list[str] = []  # operations served from PlanningCache (no LLM call)

    # Wrap real client to capture raw Anthropic API responses (pre-validation).
    _has_anthropic_client = hasattr(client, "_client") and hasattr(
        getattr(client, "_client", None), "messages"
    )
    if _has_anthropic_client:
        _orig_messages = client._client.messages

        class _ResponseCaptor:
            def create(self, **kwargs):
                response = _orig_messages.create(**kwargs)
                tools = kwargs.get("tools") or []
                if tools:
                    op = tools[0].get("name", "unknown") if isinstance(tools[0], dict) else "unknown"
                    for block in getattr(response, "content", []):
                        if getattr(block, "type", None) == "tool_use":
                            _raw_captures[op] = block.input
                            break
                return response

        client._client.messages = _ResponseCaptor()

    # Wrap frame_problem to capture the prompt text and populate raw_response.
    # For real clients: _ResponseCaptor fires on LLM calls; on cache hits it doesn't,
    # so fall back to the validated result to keep raw_response.json non-empty.
    _orig_frame_problem = client.frame_problem

    def _wrapped_frame_problem(goal: str, profiles_context: list) -> object:
        from research_agent.claude_client import _problem_framing_prompt, SYSTEM_PROMPT
        _prompt_captures["problem_framing"] = (
            f"### System\n{SYSTEM_PROMPT}\n\n### User\n{_problem_framing_prompt(goal, profiles_context)}"
        )
        result = _orig_frame_problem(goal, profiles_context)
        if "problem_framing" not in _raw_captures:
            _raw_captures["problem_framing"] = (
                result.model_dump() if hasattr(result, "model_dump") else dict(result)
            )
            if _has_anthropic_client:
                _cache_hits.append("problem_framing")
        return result

    client.frame_problem = _wrapped_frame_problem  # type: ignore[method-assign]

    # Wrap frame_executive_decision similarly.
    _orig_frame_exec = getattr(client, "frame_executive_decision", None)
    if _orig_frame_exec is not None:
        def _wrapped_frame_exec(engagement_dict, decision_model_dict, profiles_context) -> object:
            from research_agent.claude_client import _executive_framing_prompt, SYSTEM_PROMPT
            _prompt_captures["executive_framing"] = (
                f"### System\n{SYSTEM_PROMPT}\n\n"
                f"### User\n{_executive_framing_prompt(engagement_dict, decision_model_dict, profiles_context)}"
            )
            result = _orig_frame_exec(engagement_dict, decision_model_dict, profiles_context)
            if "executive_framing" not in _raw_captures:
                _raw_captures["executive_framing"] = (
                    result.model_dump() if hasattr(result, "model_dump") else dict(result)
                )
                if _has_anthropic_client:
                    _cache_hits.append("executive_framing")
            return result

        client.frame_executive_decision = _wrapped_frame_exec  # type: ignore[method-assign]

    # ---- Build AgentContext (mirrors Orchestrator.run_from_engagement) ------
    from .context import AgentContext

    brief = spec.to_framing_brief()
    ctx = AgentContext(
        question="",
        goal=brief,
        engagement=spec.model_dump(),
        profiles=profile_names,
        execution_profile=profile_names[0] if profile_names else "",
        research_object={},
        run_id="debug-pf",
    )

    # ---- Run ONLY ProblemFramingAgent ---------------------------------------
    from .problem_framing_agent import ProblemFramingAgent

    agent = ProblemFramingAgent(client=client, domain_profiles=loaded_profiles)
    _t0 = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        result = agent.run(ctx)
        ctx = result.context
    except Exception as exc:
        typer.echo(f"Error: ProblemFramingAgent failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    elapsed_s = time.monotonic() - _t0

    # ---- Compute Decision Model fingerprint (SHA-256 over canonical JSON) ---
    dm = ctx.decision_model or {}
    arch = ctx.decision_architecture or {}

    _fingerprint_data = {
        "decision_statement": arch.get("decision_statement", ""),
        "decision_scope": arch.get("decision_scope", {}),
        "decision_areas": dm.get("decision_areas", []),
        "research_questions": dm.get("research_questions", []),
        "evidence_requirements": dm.get("evidence_requirements", []),
        "strategic_themes": arch.get("strategic_themes", []),
        "decision_streams": arch.get("decision_streams", []),
        "executive_unknowns": arch.get("executive_unknowns", []),
        "board_decisions_required": arch.get("board_decisions_required", []),
        "success_definition": arch.get("success_definition", []),
    }
    _canonical = json.dumps(_fingerprint_data, sort_keys=True, ensure_ascii=False)
    _fingerprint = hashlib.sha256(_canonical.encode("utf-8")).hexdigest()

    # ---- Write output artifacts ----------------------------------------------
    out.mkdir(parents=True, exist_ok=True)

    # engagement.json — the parsed engagement spec as supplied
    (out / "engagement.json").write_text(
        json.dumps(spec.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # prompt.txt — all captured prompts concatenated
    _prompt_sections = []
    for op, text in _prompt_captures.items():
        _prompt_sections.append(f"## Operation: {op}\n\n{text}")
    (out / "prompt.txt").write_text(
        "\n\n---\n\n".join(_prompt_sections) or "(no prompts captured — mock client with no frame_problem wrapper)",
        encoding="utf-8",
    )

    # raw_response.json — raw payloads before normalization/condensing
    (out / "raw_response.json").write_text(
        json.dumps(_raw_captures, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # decision_model.json — normalized DM exactly as passed to downstream agents
    (out / "decision_model.json").write_text(
        json.dumps(dm, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # trace.json — run metadata + fingerprint
    _call_traces = getattr(client, "call_traces", [])
    _trace_calls = [
        {"operation": t.operation, "model": t.model_name, "success": t.success,
         "duration_ms": t.duration_ms, "timestamp": t.request_timestamp}
        for t in _call_traces
    ]
    _trace = {
        "model": getattr(client, "model", "mock"),
        "temperature": 0.0 if _has_anthropic_client else None,
        "max_tokens": getattr(client, "max_tokens", None),
        "timestamp": timestamp,
        "elapsed_s": round(elapsed_s, 3),
        "mock": mock or not _has_anthropic_client,
        "profiles": profile_names,
        "engagement_file": str(engagement),
        "llm_calls": _trace_calls,
        "planning_cache_hits": _cache_hits,
        "decision_model_fingerprint": _fingerprint,
    }
    (out / "trace.json").write_text(
        json.dumps(_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    typer.echo(f"ProblemFramingAgent debug run complete ({elapsed_s:.1f}s)")
    typer.echo(f"  decision_areas       : {len(dm.get('decision_areas', []))}")
    typer.echo(f"  research_questions   : {len(dm.get('research_questions', []))}")
    typer.echo(f"  decision_streams     : {len(arch.get('decision_streams', []))}")
    typer.echo(f"  fingerprint          : {_fingerprint[:16]}…")
    typer.echo(f"")
    typer.echo(f"Artifacts written to  : {out}/")
    typer.echo(f"  engagement.json")
    typer.echo(f"  prompt.txt")
    typer.echo(f"  raw_response.json")
    typer.echo(f"  decision_model.json")
    typer.echo(f"  trace.json")


@debug_app.command("research-strategy")
def debug_research_strategy_cmd(
    decision_model: Annotated[
        Path,
        typer.Option(
            "--decision-model",
            help="Path to a frozen decision_model.json produced by 'debug problem-framing'.",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output directory for debug artifacts."),
    ] = Path("outputs/research_strategy"),
    profiles: Annotated[
        str,
        typer.Option("--profiles", help="Comma-separated profile names."),
    ] = "ai_data_centers",
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name."),
    ] = None,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic mock client instead of Claude."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option("--log-level", help="Logging level."),
    ] = None,
) -> None:
    """Run ONLY ResearchStrategyAgent against a frozen decision_model.json and write debug artifacts.

    Writes decision_model_input.json, prompt.txt, raw_response.json,
    research_strategy.json, and trace.json to the output directory.
    No downstream agents execute.

    Example:
        python3 -m functional_agents.cli debug research-strategy \\
            --decision-model outputs/ENG-001_run1/decision_model.json \\
            --out outputs/research_strategy/
    """
    import hashlib
    import time
    from datetime import datetime, timezone

    _configure_logging(verbose=False, log_level=log_level or "PROGRESS")

    # ---- Load frozen Decision Model -----------------------------------------
    if not decision_model.exists():
        typer.echo(f"Error: decision model file not found: {decision_model}", err=True)
        raise typer.Exit(code=1)
    try:
        dm_input = json.loads(decision_model.read_text(encoding="utf-8"))
    except Exception as exc:
        typer.echo(f"Error: could not parse decision_model.json: {exc}", err=True)
        raise typer.Exit(code=1)

    # ---- Build profiles / client --------------------------------------------
    from research_agent.profile import load_profile
    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]
    loaded_profiles = []
    for name in profile_names:
        try:
            loaded_profiles.append(load_profile(name))
        except Exception as exc:
            typer.echo(f"Warning: could not load profile '{name}': {exc}", err=True)

    client = _build_client(mock=mock, model=model)

    # ---- Artifact capture state ---------------------------------------------
    _raw_captures: dict[str, object] = {}
    _prompt_captures: dict[str, str] = {}
    _cache_hits: list[str] = []

    # Wrap real client to capture raw Anthropic API responses (pre-validation).
    _has_anthropic_client = hasattr(client, "_client") and hasattr(
        getattr(client, "_client", None), "messages"
    )
    if _has_anthropic_client:
        _orig_messages = client._client.messages

        class _ResponseCaptor:
            def create(self, **kwargs):
                response = _orig_messages.create(**kwargs)
                tools = kwargs.get("tools") or []
                if tools:
                    op = tools[0].get("name", "unknown") if isinstance(tools[0], dict) else "unknown"
                    for block in getattr(response, "content", []):
                        if getattr(block, "type", None) == "tool_use":
                            _raw_captures[op] = block.input
                            break
                return response

        client._client.messages = _ResponseCaptor()

    # Wrap generate_research_strategy to capture the prompt and handle cache hits.
    _orig_generate = getattr(client, "generate_research_strategy", None)
    if _orig_generate is not None:
        def _wrapped_generate(dm: dict, profiles_ctx: list) -> object:
            from research_agent.claude_client import _strategy_prompt, SYSTEM_PROMPT
            _prompt_captures["generate_research_strategy"] = (
                f"### System\n{SYSTEM_PROMPT}\n\n### User\n{_strategy_prompt(dm, profiles_ctx)}"
            )
            result = _orig_generate(dm, profiles_ctx)
            if "generate_research_strategy" not in _raw_captures:
                _raw_captures["generate_research_strategy"] = (
                    result.model_dump() if hasattr(result, "model_dump") else dict(result)
                )
                if _has_anthropic_client:
                    _cache_hits.append("generate_research_strategy")
            return result

        client.generate_research_strategy = _wrapped_generate  # type: ignore[method-assign]

    # ---- Build AgentContext with frozen Decision Model ----------------------
    from .context import AgentContext

    ctx = AgentContext(
        question="",
        goal="",
        profiles=profile_names,
        execution_profile=profile_names[0] if profile_names else "",
        research_object={},
        run_id="debug-rs",
        decision_model=dm_input,
    )

    # ---- Run ONLY ResearchStrategyAgent -------------------------------------
    from .research_strategy_agent import ResearchStrategyAgent

    agent = ResearchStrategyAgent(client=client, domain_profiles=loaded_profiles)
    _t0 = time.monotonic()
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        result = agent.run(ctx)
        ctx = result.context
    except Exception as exc:
        typer.echo(f"Error: ResearchStrategyAgent failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    elapsed_s = time.monotonic() - _t0

    # ---- Compute Research Strategy fingerprint (SHA-256 over canonical JSON) -
    rs = ctx.research_strategy or {}
    _fingerprint_data = {
        "profile_priorities":             rs.get("profile_priorities", {}),
        "research_question_priorities":   rs.get("research_question_priorities", []),
        "required_evidence":              rs.get("required_evidence", []),
        "source_priorities":              rs.get("source_priorities", []),
        "coverage_targets":               rs.get("coverage_targets", {}),
        "strategy_rationale":             rs.get("strategy_rationale", ""),
    }
    _canonical = json.dumps(_fingerprint_data, sort_keys=True, ensure_ascii=False)
    _fingerprint = hashlib.sha256(_canonical.encode("utf-8")).hexdigest()

    # ---- Write output artifacts ---------------------------------------------
    out.mkdir(parents=True, exist_ok=True)

    # decision_model_input.json — the frozen input as supplied
    (out / "decision_model_input.json").write_text(
        json.dumps(dm_input, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # prompt.txt — rendered prompts
    _prompt_sections = []
    for op, text in _prompt_captures.items():
        _prompt_sections.append(f"## Operation: {op}\n\n{text}")
    (out / "prompt.txt").write_text(
        "\n\n---\n\n".join(_prompt_sections) or "(no prompts captured)",
        encoding="utf-8",
    )

    # raw_response.json — raw payloads before normalization
    (out / "raw_response.json").write_text(
        json.dumps(_raw_captures, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # research_strategy.json — normalized strategy as passed to downstream agents
    (out / "research_strategy.json").write_text(
        json.dumps(rs, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # trace.json — run metadata + fingerprint
    _call_traces = getattr(client, "call_traces", [])
    _trace_calls = [
        {"operation": t.operation, "model": t.model_name, "success": t.success,
         "duration_ms": t.duration_ms, "timestamp": t.request_timestamp}
        for t in _call_traces
    ]
    _trace = {
        "model": getattr(client, "model", "mock"),
        "temperature": 0.0 if _has_anthropic_client else None,
        "max_tokens": 2000 if not mock else None,
        "timestamp": timestamp,
        "elapsed_s": round(elapsed_s, 3),
        "mock": mock or not _has_anthropic_client,
        "profiles": profile_names,
        "decision_model_input": str(decision_model),
        "llm_calls": _trace_calls,
        "planning_cache_hits": _cache_hits,
        "research_strategy_fingerprint": _fingerprint,
    }
    (out / "trace.json").write_text(
        json.dumps(_trace, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    typer.echo(f"ResearchStrategyAgent debug run complete ({elapsed_s:.1f}s)")
    typer.echo(f"  profile_priorities           : {len(rs.get('profile_priorities', {}))}")
    typer.echo(f"  research_question_priorities : {len(rs.get('research_question_priorities', []))}")
    typer.echo(f"  required_evidence            : {len(rs.get('required_evidence', []))}")
    typer.echo(f"  coverage_targets             : {len(rs.get('coverage_targets', {}))}")
    typer.echo(f"  fingerprint                  : {_fingerprint[:16]}…")
    typer.echo(f"")
    typer.echo(f"Artifacts written to  : {out}/")
    typer.echo(f"  decision_model_input.json")
    typer.echo(f"  prompt.txt")
    typer.echo(f"  raw_response.json")
    typer.echo(f"  research_strategy.json")
    typer.echo(f"  trace.json")


if __name__ == "__main__":
    app()
