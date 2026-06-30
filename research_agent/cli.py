"""Command-line interface for research_agent."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from .agent import DEFAULT_TOP_EVIDENCE, DcPowerAgent, rank_evidence_items
from .retrieval import DEFAULT_TOP_CHUNKS
from .claude_client import ClaudeClient, MockClaudeClient
from .evaluator import classify_question_topics
from .loaders import load_sources
from .markdown import memo_to_markdown, write_markdown
from .profile import DomainProfile, list_available_profiles, load_profile
from .schemas import ResearchMemo, SourceDocument
from .trace import MEMO_SECTIONS, build_trace, write_trace
from .decision_model import from_question as _dm_from_question, write_decision_model
from .engagement import from_question as _engagement_from_question, link_decision_model as _link_dm, write_engagement
from .research_object import (
    create_research_object,
    update_research_object,
    write_research_object,
    research_object_trace_stub,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="Research question to answer.")],
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
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Markdown output path."),
    ] = Path("outputs/memo.md"),
    live_llm: Annotated[
        bool,
        typer.Option(
            "--live-llm",
            help="Deprecated; Claude is automatic when ANTHROPIC_API_KEY is set.",
        ),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Anthropic model name for Claude runs."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose diagnostics (shorthand for --log-level DEBUG)."),
    ] = False,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL. Overrides --verbose.",
        ),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option(
            "--show-sources",
            help="Print loaded source file names and extracted character counts.",
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Print a concise run summary after execution."),
    ] = False,
    mock: Annotated[
        bool,
        typer.Option("--mock", help="Use deterministic local mock mode instead of Claude."),
    ] = False,
    profile: Annotated[
        str | None,
        typer.Option(
            "--profile",
            help=(
                "Domain profile name (e.g. 'ai_data_centers', 'smr') or path to a "
                ".yaml profile file.  Defaults to 'ai_data_centers'."
            ),
        ),
    ] = None,
    top_evidence: Annotated[
        int,
        typer.Option(
            "--top-evidence",
            help="Maximum ranked evidence items to pass into memo synthesis.",
        ),
    ] = DEFAULT_TOP_EVIDENCE,
    top_chunks: Annotated[
        int,
        typer.Option("--top-chunks", help="Maximum chunks sent to evidence extraction."),
    ] = DEFAULT_TOP_CHUNKS,
    web_search: Annotated[
        bool,
        typer.Option("--web-search", help="Enable optional web search retrieval (K1.0)."),
    ] = False,
    web_max_results: Annotated[
        int,
        typer.Option("--web-max-results", help="Maximum DuckDuckGo results per query."),
    ] = 5,
    web_max_pages: Annotated[
        int,
        typer.Option("--web-max-pages", help="Maximum pages to download from web results."),
    ] = 5,
) -> None:
    """Analyze local documents and write a Markdown research memo."""

    _configure_logging(verbose or debug, log_level)

    try:
        collection = load_sources(sources)
        if show_sources:
            _echo_loaded_files(collection.documents, sources)
        for error in collection.errors:
            logging.warning("%s: %s", error.path, error.message)

        domain_profile, profile_warnings = _load_profile(profile)
        # K1.0 – CLI flags override profile's web_search section
        if web_search or web_max_results != 5 or web_max_pages != 5:
            from .profile import WebSearchConfig
            domain_profile = domain_profile.model_copy(update={
                "web_search": WebSearchConfig(
                    enabled=web_search,
                    max_results=web_max_results,
                    max_pages=web_max_pages,
                )
            })
        client, startup_warnings = _build_client(mock=mock, live_llm=live_llm, model=model)
        startup_warnings = profile_warnings + startup_warnings
        mock_mode = getattr(client, "is_mock", False)

        # J4.5 – create research object before the run.
        # Use the raw `profile` arg (user-supplied name) as the canonical source.
        # domain_profile always falls back to ai_data_centers, so using its .name
        # would silently record the wrong profile for questions run without --profile.
        # J7.0a – auto-create a minimal engagement for every CLI run.
        engagement = _engagement_from_question(question)
        try:
            write_engagement(engagement)
        except Exception:
            pass  # persistence failure must never block a research run

        # J7.0b – auto-create a minimal Decision Model for every question-driven run.
        # J7.0b1 – also back-link the engagement so decision_model_id is non-null.
        # J7.1a – write_latest=False: simple CLI DMs must not overwrite latest_decision_model.json
        #         which may already contain assumptions from a prior functional-pipeline run.
        dm_id: str | None = None
        try:
            dm = _dm_from_question(question, engagement_id=engagement.engagement_id)
            write_decision_model(dm, write_latest=False)
            dm_id = dm.decision_model_id
            _link_dm(engagement, dm_id)  # re-persists engagement with decision_model_id set
        except Exception:
            pass

        ro = create_research_object(
            question=question,
            profile_name=profile,  # raw --profile string, None if not passed
            profile_source="cli_argument" if profile else "unset",
            sources_dir=sources,
            web_search=web_search,
            mock_mode=mock_mode,
            model_name=model,
            engagement_id=engagement.engagement_id,
            decision_model_id=dm_id,
        )

        memo = DcPowerAgent(
            client=client,
            top_evidence=top_evidence,
            top_chunks=top_chunks,
            profile=domain_profile,
        ).analyze(question, collection.documents)

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

        output_path = write_markdown(memo_to_markdown(memo), out)

        # J4.5 – update and write research object after run
        ro = update_research_object(ro, memo=memo, output_path=output_path)
        # J7.6a – write_latest=False: simple CLI runs must not overwrite
        # latest_research_object.json which belongs to the interactive functional pipeline.
        ro_path = write_research_object(ro, out_dir=out.parent, write_latest=False)

        # Inject research object stub before writing trace
        trace_payload = build_trace(
            question=question,
            source_directory=sources,
            output_path=output_path,
            documents=collection.documents,
            memo=memo,
            mock_mode=mock_mode,
            profile=domain_profile,
        )
        trace_payload["research_object"] = research_object_trace_stub(ro, ro_path)
        # Now update the research object with the trace path (second write)
        ro = update_research_object(ro, memo=memo, output_path=output_path, trace_path=str(Path(output_path).with_suffix(".trace.json")))
        write_research_object(ro, out_dir=out.parent, write_latest=False)

        trace_path = write_trace(trace_payload, output_path)

        if debug:
            _echo_debug_summary(
                question=question,
                source_directory=sources,
                output_path=output_path,
                documents=collection.documents,
                memo=memo,
                trace_path=trace_path,
                domain_profile=domain_profile,
            )

        typer.echo(f"Wrote {output_path}")
        if memo.evaluation_warnings:
            typer.echo(f"{len(memo.evaluation_warnings)} evaluation warning(s).")
    except Exception as exc:
        logging.error("%s", exc)
        raise typer.Exit(code=1) from exc


def _configure_logging(verbose: bool, log_level: str | None = None) -> None:
    from .log import PROGRESS
    if log_level:
        level = logging.getLevelName(log_level.upper())
        if not isinstance(level, int):
            level = logging.INFO
    else:
        level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def _load_profile(
    name_or_path: str | None,
) -> tuple[DomainProfile | None, list[str]]:
    """Load the requested domain profile.

    Returns ``(profile, warnings)`` where *warnings* is a list of
    non-fatal messages (e.g. when falling back to the default profile).
    When *name_or_path* is ``None`` the default profile is loaded silently.
    """
    from .profile import get_default_profile

    if name_or_path is None:
        return get_default_profile(), []

    try:
        p = load_profile(name_or_path)
        return p, []
    except FileNotFoundError as exc:
        available = list_available_profiles()
        warning = (
            f"Profile warning: {exc}  "
            f"Available: {available}.  Falling back to default profile."
        )
        return get_default_profile(), [warning]


def _build_client(*, mock: bool, live_llm: bool, model: str | None, extraction_model: str | None = None):
    if mock:
        return MockClaudeClient(), []

    if not os.getenv("ANTHROPIC_API_KEY"):
        return MockClaudeClient(), [
            "Claude warning: ANTHROPIC_API_KEY is missing; using deterministic mock client."
        ]

    try:
        return ClaudeClient(model=model, extraction_model=extraction_model), []
    except Exception as exc:
        logging.error("Claude client setup failed: %s", exc)
        return MockClaudeClient(), [f"Claude warning: client setup failed: {exc}"]


def _echo_loaded_files(documents: list[SourceDocument], source_root: Path) -> None:
    if not documents:
        typer.echo("Loaded 0 source file(s).")
        return

    typer.echo(f"Loaded {len(documents)} source file(s):")
    for document in documents:
        try:
            display_path = document.path.relative_to(source_root)
        except ValueError:
            display_path = document.path
        typer.echo(f"- {display_path} ({document.char_count} characters)")


def _echo_debug_summary(
    *,
    question: str,
    source_directory: Path,
    output_path: Path,
    documents: list[SourceDocument],
    memo: ResearchMemo,
    trace_path: Path,
    domain_profile: DomainProfile | None = None,
) -> None:
    evidence_items = memo.source_notes or memo.evidence
    evidence_counts = {
        document.path.name: sum(
            1 for item in evidence_items if item.source_document == document.path.name
        )
        for document in documents
    }

    typer.echo("Debug summary:")
    typer.echo(f"Question: {question}")
    if domain_profile is not None:
        typer.echo(f"Domain profile: {domain_profile.name} ({domain_profile.profile_path})")
        typer.echo(f"Profile description: {domain_profile.description}")
        topics = sorted(domain_profile.classify_question_topics(question))
    else:
        topics = sorted(classify_question_topics(question))
    typer.echo(f"Question topics detected: {', '.join(topics) if topics else 'none'}")
    typer.echo(f"Source directory: {source_directory}")
    typer.echo(f"Output path: {output_path}")
    typer.echo(f"Documents loaded: {len(documents)}")
    typer.echo("Documents:")
    if documents:
        for document in documents:
            typer.echo(f"- {document.path.name} ({document.char_count} characters)")
    else:
        typer.echo("- None")
    typer.echo("Evidence items per document:")
    if documents:
        for document in documents:
            typer.echo(f"- {document.path.name}: {evidence_counts.get(document.path.name, 0)}")
    else:
        typer.echo("- None")
    typer.echo(f"Total evidence items: {len(evidence_items)}")
    typer.echo(
        "Evidence items used for synthesis: "
        f"{memo.metadata.get('evidence_items_used_for_synthesis', len(evidence_items))}"
    )
    typer.echo("Top evidence items:")
    top_items = rank_evidence_items(evidence_items)[:5]
    if top_items:
        for item in top_items:
            typer.echo(
                "- "
                f"{item.evidence_id or 'unassigned'} | overall {item.overall_score:.2f} "
                f"(rel {item.relevance_score}, source {item.source_quality_score}, "
                f"spec {item.specificity_score}) | {item.source_document} | "
                f"{_truncate(item.claim, 90)}"
            )
    else:
        typer.echo("- None")
    typer.echo("Memo sections generated:")
    for section in MEMO_SECTIONS:
        typer.echo(f"- {section}")
    typer.echo(f"Evaluation warning count: {len(memo.evaluation_warnings)}")
    chunk_count = memo.metadata.get("chunk_count", 0)
    if chunk_count:
        chunks_selected = memo.metadata.get("chunks_selected", 0)
        avg_size = memo.metadata.get("avg_chunk_size", 0)
        typer.echo(
            f"Chunks: {chunk_count} total, {chunks_selected} sent to Claude "
            f"(avg {avg_size} chars/chunk)"
        )
        typer.echo("Chunks per document:")
        for doc_name, cnt in memo.metadata.get("chunks_per_document", {}).items():
            typer.echo(f"  - {doc_name}: {cnt}")

        diagnostics = memo.metadata.get("chunk_diagnostics", [])
        if diagnostics:
            from collections import Counter
            decisions = Counter(d.get("extraction_decision", "") for d in diagnostics)
            reasons = Counter(
                d.get("rejection_reason") for d in diagnostics if d.get("rejection_reason")
            )
            typer.echo(
                f"Extraction: {decisions.get('accepted', 0)} accepted, "
                f"{decisions.get('rejected', 0)} rejected, "
                f"{decisions.get('not_sent', 0)} not sent"
            )
            if reasons:
                typer.echo("Top rejection reasons:")
                for reason, count in reasons.most_common(5):
                    typer.echo(f"  - {reason}: {count}")
        else:
            epc = memo.metadata.get("evidence_per_chunk", {})
            non_zero = sum(1 for v in epc.values() if v > 0)
            typer.echo(f"Evidence per chunk: {non_zero} of {len(epc)} chunks have evidence")
    retrieval_ranking = memo.metadata.get("retrieval_ranking", [])
    if retrieval_ranking:
        typer.echo("Top chunks selected (by retrieval score):")
        for i, rs in enumerate(retrieval_ranking[:10], start=1):
            typer.echo(
                f"  {i}. {rs['chunk_id']} score={rs['overall_retrieval_score']:.2f} | {rs['document_name']}"
            )
    contradictions = memo.metadata.get("contradictions", [])
    typer.echo(f"Contradictions detected: {len(contradictions)}")
    if contradictions:
        for c in contradictions[:5]:
            typer.echo(
                f"  - {c.get('contradiction_id', '?')} [{c.get('severity', '?')}] "
                f"Topic: {c.get('topic', '?')} | "
                f"{c.get('evidence_a_id', '?')} vs {c.get('evidence_b_id', '?')}"
            )
    coverage_matrix = memo.metadata.get("coverage_matrix", [])
    if coverage_matrix:
        from collections import defaultdict as _dd
        by_level: dict[str, list[str]] = _dd(list)
        for area in coverage_matrix:
            by_level[area.get("coverage_level", "unknown")].append(area.get("topic", "?"))
        typer.echo("Coverage Summary:")
        for level in ("strong", "moderate", "weak", "none"):
            topics = by_level.get(level, [])
            if topics:
                typer.echo(f"  {level.capitalize()}: {', '.join(topics)}")

    sq_map = memo.metadata.get("source_quality_map", {})
    if sq_map:
        from collections import defaultdict
        by_score: dict[int, list[str]] = defaultdict(list)
        for doc_name, sq in sq_map.items():
            score = sq.get("source_quality_score", 0)
            by_score[score].append(doc_name)
        typer.echo("Source Quality Summary:")
        for score in sorted(by_score.keys(), reverse=True):
            docs = by_score[score]
            typer.echo(f"  Score {score}: {len(docs)} document(s)")
            for doc in docs:
                typer.echo(f"    - {doc}")

    research_gaps = memo.metadata.get("research_gaps", [])
    high_gaps = [g for g in research_gaps if g.get("priority") == "high"]
    typer.echo(
        f"Research gaps: {len(research_gaps)} total, {len(high_gaps)} high-priority"
    )
    if research_gaps:
        for g in research_gaps[:6]:
            typer.echo(
                f"  - {g.get('gap_id', '?')} [{g.get('priority', '?')}] "
                f"{g.get('topic', '?')}: {g.get('description', '')[:80]}"
            )
    typer.echo(f"Trace file path: {trace_path}")


def _truncate(text: str, limit: int) -> str:
    compacted = " ".join(text.split())
    if len(compacted) <= limit:
        return compacted
    return compacted[: limit - 3].rstrip() + "..."


if __name__ == "__main__":
    app()
