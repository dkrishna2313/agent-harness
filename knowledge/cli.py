"""Knowledge Builder CLI.

Usage:
    python3 -m knowledge.builder build \\
        --sources smr_sources/ sources/nvidia/ \\
        --domain smr \\
        --incremental \\
        --workers 2 \\
        --log-level INFO

    python3 -m knowledge.builder build \\
        --config knowledge/configs/sports.yaml

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


def _load_build_config(config_path: Path) -> tuple[list[Path], list[str]]:
    """Load and validate a YAML build configuration file.

    Returns (sources, profiles).  Raises FileNotFoundError or ValueError on
    any problem so the caller can emit a consistent error and exit.

    Path resolution order for each source entry:
      1. Expand environment variables.
      2. Expand ``~``.
      3. If absolute, use as-is.
      4. Otherwise resolve relative to CWD (NOT relative to the YAML file).
    """
    import yaml  # already a project dependency (pyyaml>=6.0)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__!r}")

    for field in ("name", "profiles", "sources"):
        if field not in raw:
            raise ValueError(f"Config missing required field: '{field}'")

    profiles = raw["profiles"]
    if not isinstance(profiles, list) or not profiles:
        raise ValueError("Config 'profiles' must be a non-empty list")

    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError("Config 'sources' must be a non-empty list")

    cwd = Path.cwd()
    sources: list[Path] = []
    for s in raw_sources:
        expanded = os.path.expandvars(os.path.expanduser(str(s)))
        p = Path(expanded)
        if not p.is_absolute():
            p = cwd / p
        if not p.exists():
            raise FileNotFoundError(f"Source directory does not exist: {p}")
        sources.append(p.resolve())

    return sources, [str(pr) for pr in profiles]


def _setup_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("build")
def build(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help=(
            "Path to a YAML build configuration file. "
            "Mutually exclusive with --sources and --profiles. "
            "Example: knowledge/configs/sports.yaml"
        ),
    ),
    sources: Optional[list[Path]] = typer.Option(
        None,
        "--sources",
        help=(
            "Source directories to ingest. Defaults to smr_sources/ sources/ if not specified. "
            "Mutually exclusive with --config."
        ),
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
        help=(
            "Profile IDs to tag all produced evidence with. "
            "Mutually exclusive with --config (specify profiles in the YAML file instead)."
        ),
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
    """Build or update the Knowledge Base from source directories.

    Sources can be supplied in two ways:

    \b
    1. Declarative YAML config (recommended):
         python3 -m knowledge build --config knowledge/configs/sports.yaml

    \b
    2. Inline flags (backwards-compatible):
         python3 -m knowledge build --sources sports_sources/ --profiles sports
    """
    _setup_logging(log_level)

    # --- Mutual-exclusion guards ---
    if config and sources:
        typer.echo(
            "Error: --config and --sources are mutually exclusive. "
            "Specify sources in the YAML file or use --sources alone.",
            err=True,
        )
        raise typer.Exit(1)
    if config and profiles:
        typer.echo(
            "Error: --config and --profiles are mutually exclusive. "
            "Specify profiles in the YAML file or use --profiles alone.",
            err=True,
        )
        raise typer.Exit(1)

    # --- Config-file path ---
    if config:
        try:
            sources, profiles = _load_build_config(config)
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Loaded config: {config.name}  profiles={profiles}")

    # Auto-load profile config when --profiles given but no sources/config specified
    if profiles and not sources and not config:
        for profile_name in profiles:
            for config_candidate in (
                Path(f"knowledge/configs/{profile_name}.yaml"),
                Path(f"knowledge_sources/configs/{profile_name}.yaml"),
            ):
                if config_candidate.exists():
                    try:
                        sources, profiles = _load_build_config(config_candidate)
                        typer.echo(f"Auto-loaded profile config: {config_candidate}  profiles={profiles}")
                    except (FileNotFoundError, ValueError) as exc:
                        typer.echo(f"Warning: could not load auto-detected config {config_candidate}: {exc}", err=True)
                    break
            if sources:
                break

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


def _evidence_counts_for_profile(store: "KnowledgeStore", profile: str) -> tuple[int, set[str]]:
    """Return (evidence_count, source_id_set) for evidence tagged with *profile*."""
    count = 0
    source_ids: set[str] = set()
    for domain in store.available_domains():
        for ev in store.iter_evidence(domain):
            if profile in ev.profile_ids:
                count += 1
                source_ids.update(ev.supporting_source_ids)
    return count, source_ids


@app.command("status")
def status(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Filter statistics to a specific profile (e.g. sports, smr).",
    ),
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Show current Knowledge Base statistics.

    With --profile, shows counts scoped to evidence tagged with that profile.

    Examples:

        python3 -m knowledge status
        python3 -m knowledge status --profile sports
        python3 -m knowledge status --profile smr
    """
    _setup_logging(log_level)
    store = KnowledgeStore(store_dir)
    stats = store.read_stats()
    manifest = store.load_manifest()

    print(f"\nKnowledge Base at: {store_dir.resolve()}")

    if profile:
        ev_count, source_ids = _evidence_counts_for_profile(store, profile)
        print(f"Profile:           {profile}")
        print(f"Evidence objects:  {ev_count}")
        print(f"Sources with hits: {len(source_ids)}")

        # Break down by domain
        domains: dict[str, int] = {}
        for sid in source_ids:
            entry = manifest.get(sid)
            if entry:
                domains[entry.domain] = domains.get(entry.domain, 0) + 1
        if domains:
            print("Domains:")
            for dom, count in sorted(domains.items()):
                print(f"  {dom}: {count} sources")
    else:
        print(f"Sources indexed:   {len(manifest)}")
        domains_all: dict[str, int] = {}
        for entry in manifest.values():
            domains_all[entry.domain] = domains_all.get(entry.domain, 0) + 1
        for dom, count in sorted(domains_all.items()):
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
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        help="Filter to sources that have at least one evidence item tagged with this profile.",
    ),
) -> None:
    """List all indexed sources.

    With --profile, only sources that contain evidence tagged with that profile
    are shown.  --domain and --profile can be combined.

    Examples:

        python3 -m knowledge list-sources
        python3 -m knowledge list-sources --profile sports
        python3 -m knowledge list-sources --domain smr --profile smr
    """
    store = KnowledgeStore(store_dir)
    manifest = store.load_manifest()

    entries = list(manifest.values())
    if domain:
        entries = [e for e in entries if e.domain == domain]

    if profile:
        _, profile_source_ids = _evidence_counts_for_profile(store, profile)
        entries = [e for e in entries if e.source_id in profile_source_ids]

    if not entries:
        print("No sources indexed.")
        return

    print(f"{'source_id':34} {'domain':20} {'evidence':>8}  uri")
    print("-" * 90)
    for entry in sorted(entries, key=lambda e: (e.domain, e.uri)):
        ev_count = len(entry.evidence_ids)
        uri = entry.uri[-50:] if len(entry.uri) > 50 else entry.uri
        print(f"{entry.source_id:34} {entry.domain:20} {ev_count:>8}  ...{uri}")


@app.command("list-profiles")
def list_profiles(
    store_dir: Path = typer.Option(Path("knowledge_store"), "--store"),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """List all profiles present in the Knowledge Base with evidence and source counts.

    Examples:

        python3 -m knowledge list-profiles
        python3 -m knowledge list-profiles --store knowledge_store
    """
    _setup_logging(log_level)
    store = KnowledgeStore(store_dir)
    manifest = store.load_manifest()

    # Accumulate per-profile stats in a single pass over all evidence
    ev_counts: dict[str, int] = {}
    source_ids_by_profile: dict[str, set[str]] = {}

    for domain in store.available_domains():
        for ev in store.iter_evidence(domain):
            for pid in ev.profile_ids:
                ev_counts[pid] = ev_counts.get(pid, 0) + 1
                source_ids_by_profile.setdefault(pid, set()).update(ev.supporting_source_ids)

    if not ev_counts:
        print("No profiles found. Run 'build --profiles <name>' to tag evidence with a profile.")
        return

    print(f"\n{'Profile':<20} {'Evidence':>9} {'Sources':>8}")
    print("-" * 40)
    for pid in sorted(ev_counts):
        src_count = len(source_ids_by_profile.get(pid, set()))
        print(f"{pid:<20} {ev_counts[pid]:>9} {src_count:>8}")


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
