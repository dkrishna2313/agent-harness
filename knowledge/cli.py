"""Knowledge Builder CLI.

Usage:
    python3 -m knowledge.builder build \\
        --sources smr_sources/ sources/nvidia/ \\
        --domain smr \\
        --incremental \\
        --workers 2 \\
        --log-level INFO

This CLI is additive — it does not replace or modify any existing CLI commands.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    """Load .env from the CWD if present and ANTHROPIC_API_KEY is not already set."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()

import typer

from .builder import KnowledgeBuilder, infer_domain
from .embeddings import embed_evidence_batch, get_provider
from .reranker import LLMReranker, PassthroughReranker
from .retriever import (
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
    EvidenceRetriever,
)
from .store import KnowledgeStore

app = typer.Typer(
    name="knowledge",
    help="Knowledge Builder — construct and maintain the persistent Knowledge Base.",
    no_args_is_help=True,
)


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("build")
def build(
    sources: Optional[list[Path]] = typer.Option(
        None,
        "--sources",
        help="Source directories to ingest. Defaults to smr_sources/ sources/ if not specified.",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        help="Override domain for all source directories (e.g. smr, ai_data_centers). "
             "Defaults to auto-detection from directory name.",
    ),
    profiles: Optional[list[str]] = typer.Option(
        None,
        "--profiles",
        help="Profile IDs to tag all produced evidence with.",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--no-incremental",
        help="Skip sources whose fingerprint has not changed. Default: on.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Rebuild all sources regardless of fingerprint. Overrides --incremental.",
    ),
    workers: int = typer.Option(
        1,
        "--workers",
        min=1,
        max=16,
        help="Number of concurrent source-processing threads.",
    ),
    store_dir: Path = typer.Option(
        Path("knowledge_store"),
        "--store",
        help="Path to the knowledge store directory.",
    ),
    model: str = typer.Option(
        "claude-sonnet-4-6",
        "--model",
        help="Model identifier recorded in ExtractionRun provenance.",
    ),
    skip_extraction: bool = typer.Option(
        False,
        "--skip-extraction",
        help="Ingest sources without extracting evidence (source indexing only).",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    ),
) -> None:
    """Build or update the Knowledge Base from source directories."""
    _setup_logging(log_level)

    # Default source directories if none given
    if not sources:
        defaults = [
            Path("smr_sources"),
            Path("sources"),
        ]
        sources = [p for p in defaults if p.exists()]
        if not sources:
            typer.echo("No source directories found. Specify --sources.", err=True)
            raise typer.Exit(1)

    # Client — skip if --skip-extraction
    client = None
    if not skip_extraction:
        try:
            from research_agent.claude_client import ClaudeClient
            client = ClaudeClient(model=model)
        except Exception as exc:
            typer.echo(f"Warning: could not initialise ClaudeClient — {exc}. Evidence extraction disabled.", err=True)

    store = KnowledgeStore(store_dir)
    builder = KnowledgeBuilder(
        store=store,
        client=client,
        model_version=model,
        workers=workers,
    )

    domain_overrides: dict[str, str] = {}
    if domain:
        for s in sources:
            domain_overrides[s.name] = domain

    typer.echo(f"Knowledge Builder starting — store={store_dir}  incremental={incremental}  force={force}")
    typer.echo(f"Source directories: {[str(s) for s in sources]}")

    report = builder.build(
        [Path(s) for s in sources],
        domain_overrides=domain_overrides or None,
        incremental=incremental,
        force=force,
        profile_ids=list(profiles) if profiles else None,
    )
    report.print()

    if report.sources_failed > 0:
        raise typer.Exit(1)


@app.command("status")
def status(
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Show current Knowledge Base statistics."""
    _setup_logging(log_level)
    store = KnowledgeStore(store_dir)
    stats = store.read_stats()
    manifest = store.load_manifest()

    print(f"\nKnowledge Base at: {store_dir.resolve()}")
    print(f"Sources indexed:   {len(manifest)}")

    domains: dict[str, int] = {}
    for entry in manifest.values():
        domains[entry.domain] = domains.get(entry.domain, 0) + 1
    for dom, count in sorted(domains.items()):
        print(f"  {dom}: {count} sources")

    if stats:
        print(f"\nLast build:        {stats.get('last_build', 'unknown')}")
        print(f"Evidence objects:  {stats.get('evidence_objects', 0)}")
        print(f"Cache hit ratio:   {stats.get('cache_hit_ratio', 0):.1%}")
        print(f"ExtractionRun ID:  {stats.get('extraction_run_id', 'unknown')}")
    else:
        print("\nNo build has run yet.")


@app.command("list-sources")
def list_sources(
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    domain: Optional[str] = typer.Option(None, "--domain"),
) -> None:
    """List all indexed sources."""
    store = KnowledgeStore(store_dir)
    manifest = store.load_manifest()

    entries = list(manifest.values())
    if domain:
        entries = [e for e in entries if e.domain == domain]

    if not entries:
        print("No sources indexed.")
        return

    print(f"{'source_id':34} {'domain':20} {'evidence':>8}  uri")
    print("-" * 90)
    for entry in sorted(entries, key=lambda e: (e.domain, e.uri)):
        ev_count = len(entry.evidence_ids)
        uri = entry.uri[-50:] if len(entry.uri) > 50 else entry.uri
        print(f"{entry.source_id:34} {entry.domain:20} {ev_count:>8}  ...{uri}")


@app.command("retrieve")
def retrieve(
    query: str = typer.Argument(..., help="Natural-language retrieval query."),
    mode: str = typer.Option(
        RETRIEVAL_MODE_LEXICAL,
        "--mode",
        help="Retrieval mode: lexical | semantic | hybrid.",
    ),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        help="Restrict to a specific domain (e.g. smr, ai_data_centers). Default: all.",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        min=1,
        max=200,
        help="Final number of results to return (after reranking if enabled).",
    ),
    evidence_types: Optional[str] = typer.Option(
        None,
        "--types",
        help="Comma-separated EvidenceType filter, e.g. STRATEGIC,TECHNICAL.",
    ),
    show_source: bool = typer.Option(
        False,
        "--show-source",
        help="Load and display source title/org/version under each result.",
    ),
    all_evidence: bool = typer.Option(
        False,
        "--all",
        help="Include evidence with retrieval_enabled=False (default: excluded).",
    ),
    embed_model: Optional[str] = typer.Option(
        None,
        "--embed-model",
        help="Override embedding model for semantic/hybrid modes.",
    ),
    rerank: bool = typer.Option(
        False,
        "--rerank/--no-rerank",
        help="Apply LLM reranking after retrieval (requires ANTHROPIC_API_KEY).",
    ),
    rerank_candidates: int = typer.Option(
        40,
        "--rerank-candidates",
        min=1,
        max=200,
        help="Number of candidates to retrieve before reranking (default 40).",
    ),
    rerank_model: str = typer.Option(
        "claude-haiku-4-5-20251001",
        "--rerank-model",
        help="Claude model to use for reranking.",
    ),
    show_rationale: bool = typer.Option(
        False,
        "--show-rationale",
        help="Show LLM rationale under each reranked result.",
    ),
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Retrieve Evidence from the Knowledge Base matching a natural-language query.

    With --rerank, retrieves a larger candidate set (--rerank-candidates) and
    uses an LLM to select and reorder the final --top-k results.

    Examples:

        # lexical retrieval, top 10
        python3 -m knowledge retrieve "deployment risks for SMRs" --domain smr

        # hybrid retrieval, top 10
        python3 -m knowledge retrieve "deployment risks for SMRs" --domain smr --mode hybrid

        # hybrid retrieval + LLM reranking, 40 candidates → top 10
        python3 -m knowledge retrieve "deployment risks for SMRs" --domain smr --mode hybrid --rerank
    """
    _setup_logging(log_level)

    if mode not in (RETRIEVAL_MODE_LEXICAL, RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID):
        typer.echo(f"Unknown mode {mode!r}. Use: lexical | semantic | hybrid", err=True)
        raise typer.Exit(1)

    types_filter: Optional[list[str]] = None
    if evidence_types:
        types_filter = [t.strip().upper() for t in evidence_types.split(",")]

    store = KnowledgeStore(store_dir)

    provider = None
    if mode in (RETRIEVAL_MODE_SEMANTIC, RETRIEVAL_MODE_HYBRID):
        provider = get_provider(embed_model)

    retriever = EvidenceRetriever(store, provider=provider)

    # With reranking, retrieve a larger candidate pool first
    retrieval_k = rerank_candidates if rerank else top_k

    result = retriever.retrieve(
        query,
        mode=mode,
        domain=domain,
        top_k=retrieval_k,
        evidence_types=types_filter,
        retrieval_enabled_only=not all_evidence,
        load_sources=show_source,
    )

    if not rerank:
        result.print_summary(show_source=show_source)
        if result.matched_candidates == 0:
            raise typer.Exit(1)
        return

    # --- Reranking path ---
    typer.echo(
        f"\n[RETRIEVAL] {result.matched_candidates} matched, "
        f"passing top {len(result.items)} candidates to reranker…"
    )
    result.print_summary(show_source=False)

    reranker = LLMReranker(model=rerank_model)
    rerank_result = reranker.rerank(query, result.items, top_k=top_k)

    typer.echo("[RERANKED]")
    rerank_result.print_summary(show_rationale=show_rationale)

    if not rerank_result.items:
        raise typer.Exit(1)


@app.command("health")
def health_cmd(
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        help="Validate only a specific domain (default: all).",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Validate a Knowledge Store for runtime readiness.

    Checks manifest, evidence files, evidence counts, index consistency,
    and embedding counts. Exits with code 1 if the store is not ready.

    Example:
        python3 -m knowledge health --store knowledge_store
    """
    _setup_logging(log_level)

    from .health import check_store_health

    store = KnowledgeStore(store_dir)
    report = check_store_health(store, domain=domain)
    report.print()

    if not report.runtime_ready:
        raise typer.Exit(1)


@app.command("embed")
def embed_cmd(
    domain: Optional[str] = typer.Option(
        None,
        "--domain",
        help="Domain to embed. Default: all available domains.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Regenerate embeddings even if they already exist.",
    ),
    embed_model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Override embedding model (default: all-MiniLM-L6-v2).",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        help="Items per model call.",
    ),
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Generate and persist embeddings for all Evidence in a domain.

    Must be run before using --mode semantic or --mode hybrid.
    Already-embedded items are skipped unless --force is given.

    Example:
        python3 -m knowledge embed --domain smr
    """
    _setup_logging(log_level)

    store = KnowledgeStore(store_dir)
    domains = [domain] if domain else store.available_domains()

    if not domains:
        typer.echo("No evidence domains found in the knowledge store.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Loading embedding model… (first run downloads ~80 MB)")
    provider = get_provider(embed_model)
    typer.echo(f"Model: {provider.model_name}  dim={provider.dimension}")

    total_embedded = 0
    total_skipped = 0

    for dom in domains:
        items = list(store.iter_evidence(dom))
        if not items:
            typer.echo(f"  {dom}: no evidence items.")
            continue
        typer.echo(f"  {dom}: {len(items)} items — embedding…")
        embedded, skipped = embed_evidence_batch(
            items, store, provider, force=force, batch_size=batch_size
        )
        typer.echo(f"  {dom}: embedded={embedded}  skipped={skipped}")
        total_embedded += embedded
        total_skipped += skipped

    typer.echo(f"\nDone. Total: embedded={total_embedded}  skipped={total_skipped}")


if __name__ == "__main__":
    app()
