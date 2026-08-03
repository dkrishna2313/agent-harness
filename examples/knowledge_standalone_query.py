#!/usr/bin/env python3
"""Standalone Knowledge Layer query example.

Demonstrates that the knowledge store can be opened, queried, and results
inspected without importing functional_agents, strategy, or editorial.

Usage (from the repository root):
    python3 examples/knowledge_standalone_query.py \\
        --store knowledge_store \\
        --query "competitive strategy for sports analytics" \\
        --profiles sports \\
        --top-k 5

    # Lexical only (no embeddings download):
    python3 examples/knowledge_standalone_query.py \\
        --store knowledge_store \\
        --query "market sizing" \\
        --mode lexical

    # From outside the repository, add repo root to PYTHONPATH:
    PYTHONPATH=/path/to/agent-harness python3 examples/knowledge_standalone_query.py ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query a knowledge store without the full agent harness."
    )
    parser.add_argument("--store", default="knowledge_store", help="Path to the knowledge store directory.")
    parser.add_argument("--query", required=True, help="Natural-language retrieval query.")
    parser.add_argument("--profiles", help="Comma-separated profile IDs to filter by (e.g. sports).")
    parser.add_argument("--mode", default="lexical", choices=["lexical", "semantic", "hybrid"], help="Retrieval mode.")
    parser.add_argument("--top-k", type=int, default=5, dest="top_k", help="Number of results to return.")
    parser.add_argument("--show-source", action="store_true", help="Load and print source metadata for each result.")
    args = parser.parse_args()

    store_path = Path(args.store)
    if not store_path.exists():
        print(f"ERROR: knowledge store not found at {store_path.resolve()}", file=sys.stderr)
        sys.exit(1)

    # --- The only imports needed for standalone retrieval ---
    from knowledge.store import KnowledgeStore
    from knowledge.retriever import EvidenceRetriever, RETRIEVAL_MODE_LEXICAL

    store = KnowledgeStore(store_path)

    provider = None
    if args.mode in ("semantic", "hybrid"):
        try:
            from knowledge.embeddings import get_provider
            provider = get_provider()
        except ImportError:
            print("WARNING: sentence-transformers not installed; falling back to lexical.", file=sys.stderr)
            args.mode = "lexical"

    retriever = EvidenceRetriever(store, provider=provider)

    profiles = [p.strip() for p in args.profiles.split(",")] if args.profiles else []
    profile = profiles[0] if profiles else None  # EvidenceRetriever.retrieve() takes one profile

    result = retriever.retrieve(
        args.query,
        mode=args.mode,
        profile=profile,
        top_k=args.top_k,
        load_sources=args.show_source,
    )

    print(f"\nQuery:    {result.query!r}")
    print(f"Mode:     {result.mode}  |  Domains: {', '.join(result.domains_searched)}")
    print(f"Matched:  {result.matched_candidates} / {result.total_candidates} candidates  |  Latency: {result.latency_ms:.0f}ms")
    print(f"Results:  {len(result.items)} of {args.top_k} requested\n")

    if not result.items:
        print("  (no results)\n")
        return

    for item in result.items:
        ev = item.evidence
        print(f"  #{item.rank:>2}  score={item.score:.4f}  type={ev.evidence_type}")
        print(f"       {ev.statement[:120]}{'…' if len(ev.statement) > 120 else ''}")
        print(f"       id={ev.evidence_id}  profiles={ev.profile_ids}  domain={item.source_domain}")
        if args.show_source and item.source:
            src = item.source
            print(f"       └─ {src.title}" + (f" · {src.organization}" if src.organization else ""))
        print()


if __name__ == "__main__":
    main()
