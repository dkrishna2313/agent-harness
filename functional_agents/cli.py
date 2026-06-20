"""CLI for the functional agent pipeline (J5.0a.7)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

import typer

from research_agent.cli import _configure_logging

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    question: Annotated[str, typer.Argument(help="Research question to answer.")],
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
    """Run the functional agent pipeline and write a Markdown research memo."""

    _configure_logging(verbose=False, log_level=log_level or "INFO")

    profile_names = [p.strip() for p in profiles.split(",") if p.strip()]

    # Build client
    client = _build_client(mock=mock, model=model)

    # Apply web search to profile if requested
    if web_search and profile_names:
        try:
            from research_agent.profile import load_profile
            from research_agent.profile import WebSearchConfig
            base = load_profile(profile_names[0])
            base = base.model_copy(update={
                "web_search": WebSearchConfig(enabled=True, max_results=5, max_pages=5)
            })
        except Exception:
            pass

    from .orchestrator import Orchestrator

    orchestrator = Orchestrator(
        profile_names=profile_names,
        sources_dir=sources,
        out_path=out,
        client=client,
        top_evidence=top_evidence,
        top_chunks=top_chunks,
    )

    try:
        ctx = orchestrator.run(question)
    except Exception as exc:
        logging.error("Functional agent pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc

    agents_run = [h["agent"] for h in ctx.agent_history]
    typer.echo(f"Agents run: {', '.join(agents_run)}")
    typer.echo(f"Profiles:   {', '.join(ctx.profiles)}")
    typer.echo(f"Report:     {ctx.artifacts.get('report_path', ctx.artifacts)}")


def _build_client(*, mock: bool, model: str | None):
    from research_agent.claude_client import ClaudeClient, MockClaudeClient

    if mock:
        return MockClaudeClient()
    if not os.getenv("ANTHROPIC_API_KEY"):
        logging.warning("ANTHROPIC_API_KEY missing — using mock client.")
        return MockClaudeClient()
    try:
        return ClaudeClient(model=model)
    except Exception as exc:
        logging.error("Claude client setup failed: %s — using mock.", exc)
        return MockClaudeClient()


if __name__ == "__main__":
    app()
