"""
Knowledge Layer MCP Server.

Exposes the KnowledgeStore as an MCP server so Claude Desktop (and other
MCP-capable clients) can search and explore the knowledge base interactively.

Three tools are provided:
  search_knowledge  — full-text search with optional profile and domain filters
  list_profiles     — enumerate all profile IDs in the store
  list_domains      — enumerate all domain directories in the store

Configuration
-------------
KNOWLEDGE_STORE_PATH  (required)
    Absolute path to the knowledge_store directory.
    e.g.  export KNOWLEDGE_STORE_PATH=/path/to/knowledge_store

Run
---
    python -m knowledge.mcp_server
or via the installed entry point:
    knowledge-mcp
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from knowledge.retriever import EvidenceRetriever, RETRIEVAL_MODE_LEXICAL
from knowledge.store import KnowledgeStore

# ---------------------------------------------------------------------------
# Store initialisation
# ---------------------------------------------------------------------------

def _get_store() -> KnowledgeStore:
    env_path = os.environ.get("KNOWLEDGE_STORE_PATH", "").strip()
    if not env_path:
        raise RuntimeError(
            "KNOWLEDGE_STORE_PATH environment variable is not set. "
            "Set it to the absolute path of your knowledge_store directory."
        )
    path = Path(env_path)
    if not path.exists():
        raise RuntimeError(f"KNOWLEDGE_STORE_PATH does not exist: {path}")
    return KnowledgeStore(str(path))


# Initialise once at module load so all tools share the same instance.
try:
    _store = _get_store()
    _retriever = EvidenceRetriever(store=_store)
    _init_error: str | None = None
except RuntimeError as _e:
    _store = None  # type: ignore[assignment]
    _retriever = None  # type: ignore[assignment]
    _init_error = str(_e)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server = MCPServer(
    name="knowledge-layer",
    description=(
        "Search and explore the Knowledge Layer evidence store. "
        "Use search_knowledge to retrieve evidence by keyword. "
        "Use list_profiles and list_domains to discover what's available."
    ),
)


def _check_ready() -> str | None:
    """Return an error string if the store is not initialised, else None."""
    return _init_error


def _resolve_source_id(source_id: str) -> tuple[str | None, str | None]:
    """Resolve a full or prefix source ID to a canonical source ID.

    Returns (resolved_id, None) on success, or (None, error_message) on failure.
    Deduplicates matches across domains so the same source stored in multiple
    domains doesn't appear as multiple hits.
    """
    if len(source_id) == 64:
        return source_id, None
    matched: set[str] = set()
    try:
        for domain in _store.available_domains():
            for src in _store.iter_sources(domain):
                if src.source_id.startswith(source_id):
                    matched.add(src.source_id)
    except Exception as exc:
        return None, f"Error resolving source ID: {exc}"
    if not matched:
        return None, (
            f"No source found with ID prefix {source_id!r}. "
            "Use list_sources to see available source IDs."
        )
    if len(matched) > 1:
        return None, (
            f"Prefix {source_id!r} matches {len(matched)} sources — "
            "provide more characters to make it unique."
        )
    return next(iter(matched)), None


# ---------------------------------------------------------------------------
# Tool: search_knowledge
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "Search the knowledge base for evidence relevant to a query. "
        "Returns ranked evidence items with their statements and source information. "
        "Use list_profiles and list_domains first to discover filtering options."
    ),
)
def search_knowledge(
    query: str,
    profile: str | None = None,
    domain: str | None = None,
    top_k: int = 10,
) -> str:
    """Search for evidence matching *query*.

    Parameters
    ----------
    query:
        Natural-language search query (e.g. "GPU price trends 2024").
    profile:
        Filter to evidence tagged with this profile ID (e.g. "ai-research").
        Use list_profiles to see available profiles.
    domain:
        Restrict search to a specific domain directory (e.g. "articles").
        Use list_domains to see available domains.
    top_k:
        Maximum number of results to return (default 10, max 50).
    """
    err = _check_ready()
    if err:
        return f"Error: {err}"

    top_k = min(max(1, top_k), 50)

    try:
        result = _retriever.retrieve(
            query,
            mode=RETRIEVAL_MODE_LEXICAL,
            domain=domain,
            profile=profile,
            top_k=top_k,
            load_sources=True,
        )
    except Exception as exc:
        return f"Error during retrieval: {exc}"

    if not result.items:
        filters = []
        if profile:
            filters.append(f"profile={profile!r}")
        if domain:
            filters.append(f"domain={domain!r}")
        filter_str = f" [{', '.join(filters)}]" if filters else ""
        return (
            f"No results found for {query!r}{filter_str}.\n"
            f"Searched {len(result.domains_searched)} domain(s): "
            f"{', '.join(result.domains_searched)}.\n"
            "Try broadening your query, or check available profiles and domains."
        )

    lines: list[str] = []
    filters = []
    if profile:
        filters.append(f"profile={profile!r}")
    if domain:
        filters.append(f"domain={domain!r}")
    filter_str = f"  [{', '.join(filters)}]" if filters else ""
    lines.append(
        f"Found {len(result.items)} result(s) for {query!r}{filter_str} "
        f"[{result.latency_ms:.0f}ms, {len(result.domains_searched)} domain(s)]"
    )
    lines.append("")

    for item in result.items:
        lines.append(f"#{item.rank}  score={item.score:.3f}  type={item.evidence_type}  domain={item.source_domain}")
        lines.append(f"   {item.statement}")
        if item.source:
            src_label = item.source.title
            if item.source.organization:
                src_label += f"  ·  {item.source.organization}"
            lines.append(f"   Source: {src_label}")
        if item.evidence.profile_ids:
            lines.append(f"   Profiles: {', '.join(item.evidence.profile_ids)}")
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Tool: list_profiles
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "List all profile IDs available in the knowledge base. "
        "Profiles group evidence by topic area (e.g. 'ai-research'). "
        "Pass a profile ID to search_knowledge to restrict results to that topic."
    ),
)
def list_profiles() -> str:
    """Return all unique profile IDs found across the knowledge store."""
    err = _check_ready()
    if err:
        return f"Error: {err}"

    try:
        profiles: dict[str, int] = {}
        for domain in _store.available_domains():
            for ev in _store.iter_evidence(domain):
                for pid in ev.profile_ids:
                    profiles[pid] = profiles.get(pid, 0) + 1
    except Exception as exc:
        return f"Error listing profiles: {exc}"

    if not profiles:
        return "No profiles found in the knowledge store."

    lines = ["Available profiles:"]
    for pid, count in sorted(profiles.items()):
        lines.append(f"  {pid}  ({count} evidence item(s))")
    lines.append("")
    lines.append("Pass a profile name to search_knowledge using the 'profile' parameter.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: list_domains
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "List all domain directories available in the knowledge base. "
        "Domains are top-level groupings of evidence (e.g. 'articles', 'youtube'). "
        "Pass a domain name to search_knowledge to restrict results to that domain."
    ),
)
def list_domains() -> str:
    """Return all domain directories in the knowledge store with evidence counts."""
    err = _check_ready()
    if err:
        return f"Error: {err}"

    try:
        domains = _store.available_domains()
        counts: dict[str, int] = {}
        for d in domains:
            counts[d] = sum(1 for _ in _store.iter_evidence(d))
    except Exception as exc:
        return f"Error listing domains: {exc}"

    if not domains:
        return "No domains found in the knowledge store."

    lines = ["Available domains:"]
    for d in sorted(domains):
        lines.append(f"  {d}  ({counts.get(d, 0)} evidence item(s))")
    lines.append("")
    lines.append("Pass a domain name to search_knowledge using the 'domain' parameter.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool: list_sources
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "List sources (documents, articles, videos) in the knowledge base. "
        "Optionally filter by domain or by a keyword appearing in the title or organization. "
        "Use the source_id from results with get_source_evidence to see what was extracted from it."
    ),
)
def list_sources(
    domain: str | None = None,
    query: str | None = None,
) -> str:
    """Return sources stored in the knowledge base.

    Parameters
    ----------
    domain:
        Restrict to a specific domain (e.g. "articles", "youtube").
        Use list_domains to see available domains.
    query:
        Case-insensitive keyword to filter by title or organization name.
        e.g. "nvidia" or "techcrunch".
    """
    err = _check_ready()
    if err:
        return f"Error: {err}"

    domains = [domain] if domain else _store.available_domains()
    query_lower = query.lower() if query else None

    try:
        sources = []
        for d in domains:
            for src in _store.iter_sources(d):
                if query_lower:
                    haystack = f"{src.title} {src.organization or ''}".lower()
                    if query_lower not in haystack:
                        continue
                sources.append(src)
    except Exception as exc:
        return f"Error listing sources: {exc}"

    if not sources:
        filter_parts = []
        if domain:
            filter_parts.append(f"domain={domain!r}")
        if query:
            filter_parts.append(f"query={query!r}")
        suffix = f" [{', '.join(filter_parts)}]" if filter_parts else ""
        return f"No sources found{suffix}."

    lines = [f"Found {len(sources)} source(s):"]
    lines.append("")
    for src in sources:
        lines.append(f"  [{src.domain}]  {src.title}")
        meta_parts = []
        if src.organization:
            meta_parts.append(src.organization)
        if src.author:
            meta_parts.append(f"by {src.author}")
        if src.publication_date:
            meta_parts.append(str(src.publication_date))
        if src.document_type:
            meta_parts.append(src.document_type)
        if meta_parts:
            lines.append(f"           {' · '.join(meta_parts)}")
        lines.append(f"           ID: {src.source_id[:16]}…")
        lines.append("")

    lines.append("Pass an ID to get_source_evidence to see what was extracted from a source.")
    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Tool: get_source_evidence
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "Return all evidence items extracted from a specific source. "
        "Use list_sources to find source IDs. "
        "Accepts a full source ID or the first 8+ characters as a short prefix."
    ),
)
def get_source_evidence(source_id: str) -> str:
    """Return every evidence statement extracted from the given source.

    Parameters
    ----------
    source_id:
        Full source ID or a unique prefix of at least 8 characters
        (shown as the truncated ID in list_sources output).
    """
    err = _check_ready()
    if err:
        return f"Error: {err}"

    source_id, err = _resolve_source_id(source_id)
    if err:
        return f"Error: {err}"

    # Load source metadata
    source = _store.find_source(source_id)
    if source is None:
        return f"Source {source_id!r} not found in the knowledge store."

    # Collect all evidence referencing this source
    try:
        items = []
        for domain in _store.available_domains():
            for ev in _store.iter_evidence(domain):
                if source_id in ev.supporting_source_ids:
                    items.append((domain, ev))
    except Exception as exc:
        return f"Error reading evidence: {exc}"

    lines = [f"Source: {source.title}"]
    if source.organization:
        lines.append(f"  Organization: {source.organization}")
    if source.author:
        lines.append(f"  Author: {source.author}")
    if source.publication_date:
        lines.append(f"  Published: {source.publication_date}")
    lines.append(f"  Domain: {source.domain}  |  Type: {source.document_type}")
    lines.append(f"  ID: {source.source_id}")
    lines.append(f"  {len(items)} evidence item(s) extracted")
    lines.append("")

    if not items:
        lines.append("No evidence items found for this source.")
        return "\n".join(lines)

    for i, (domain, ev) in enumerate(items, 1):
        lines.append(f"{i}. [{ev.evidence_type}]  {ev.statement}")
        if ev.profile_ids:
            lines.append(f"   Profiles: {', '.join(ev.profile_ids)}")
        lines.append("")

    return "\n".join(lines).rstrip()


# ---------------------------------------------------------------------------
# Tool: get_source_text
# ---------------------------------------------------------------------------

@server.tool(
    description=(
        "Return the full original text of a source document or transcript. "
        "Use this when you need the complete content — speaker names, exact quotes, "
        "context that evidence extraction may have missed. "
        "Accepts a full source ID or the first 8+ characters as a short prefix."
    ),
)
def get_source_text(source_id: str) -> str:
    """Return the complete canonical text of the given source.

    Parameters
    ----------
    source_id:
        Full source ID or a unique prefix of at least 8 characters
        (shown as the truncated ID in list_sources output).
    """
    err = _check_ready()
    if err:
        return f"Error: {err}"

    source_id, err = _resolve_source_id(source_id)
    if err:
        return f"Error: {err}"

    source = _store.find_source(source_id)
    if source is None:
        return f"Source {source_id!r} not found in the knowledge store."

    header_parts = [f"Source: {source.title}"]
    if source.organization:
        header_parts.append(f"Organization: {source.organization}")
    if source.author:
        header_parts.append(f"Author: {source.author}")
    if source.publication_date:
        header_parts.append(f"Published: {source.publication_date}")
    header_parts.append(f"Domain: {source.domain}  |  Type: {source.document_type}")
    header_parts.append(f"ID: {source.source_id}")
    header_parts.append(f"URI: {source.uri}")
    header_parts.append("")
    header_parts.append("--- full text ---")
    header_parts.append("")
    header_parts.append(source.canonical_text)

    return "\n".join(header_parts)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    asyncio.run(server.run_stdio_async())


if __name__ == "__main__":
    main()
