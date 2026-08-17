# Knowledge Layer — Integration Guide

This guide covers two ways to integrate knowledge-layer into your own software:

1. **Python API** — import and call `KnowledgeStore` and `EvidenceRetriever` directly from your Python application
2. **MCP server** — expose the knowledge base as a tool-calling server for Claude Desktop or any MCP-capable LLM client

---

## Table of contents

1. [Python API](#1-python-api)
   - [Installation](#installation)
   - [Core classes](#core-classes)
   - [Basic retrieval](#basic-retrieval)
   - [Filtered retrieval](#filtered-retrieval)
   - [Multi-profile retrieval](#multi-profile-retrieval)
   - [Working with evidence objects](#working-with-evidence-objects)
   - [The adapter pattern (recommended)](#the-adapter-pattern-recommended)
2. [MCP server](#2-mcp-server)
   - [Installation](#installation-1)
   - [Running the server](#running-the-server)
   - [Claude Desktop configuration](#claude-desktop-configuration)
   - [Available tools](#available-tools)
   - [Example tool interactions](#example-tool-interactions)

---

## 1. Python API

### Installation

Install knowledge-layer as a package from its directory:

```bash
pip install -e /path/to/knowledge-layer
# or with semantic retrieval support:
pip install -e "/path/to/knowledge-layer[knowledge]"
```

Then import directly:

```python
from knowledge.store import KnowledgeStore
from knowledge.retriever import EvidenceRetriever
```

### Core classes

| Class | Module | Purpose |
|---|---|---|
| `KnowledgeStore` | `knowledge.store` | Open and read the persistent store |
| `EvidenceRetriever` | `knowledge.retriever` | Search evidence by query |
| `Evidence` | `knowledge.models` | A single extracted claim |
| `Source` | `knowledge.models` | A source document |
| `RetrievedEvidence` | `knowledge.retriever` | Evidence item with retrieval score and rank |
| `RetrievalResult` | `knowledge.retriever` | Full result set from a retrieve() call |

### Basic retrieval

```python
from knowledge.store import KnowledgeStore
from knowledge.retriever import EvidenceRetriever

store = KnowledgeStore("knowledge_store/")        # path to the built store
retriever = EvidenceRetriever(store=store)

result = retriever.retrieve(
    "GPU price trends and AI infrastructure costs",
    top_k=10,
)

for item in result.items:
    print(f"#{item.rank}  score={item.score:.3f}  [{item.evidence_type}]")
    print(f"  {item.statement}")
    print()
```

### Filtered retrieval

```python
from knowledge.retriever import (
    RETRIEVAL_MODE_HYBRID,
    RETRIEVAL_MODE_LEXICAL,
    RETRIEVAL_MODE_SEMANTIC,
)

# Filter by profile (logical tag applied at build time)
result = retriever.retrieve(
    "energy costs for AI data centres",
    profile="my-project",
    top_k=20,
    mode=RETRIEVAL_MODE_HYBRID,   # lexical + semantic
    load_sources=True,            # populate item.source with Source metadata
)

for item in result.items:
    source_title = item.source.title if item.source else "unknown"
    print(f"  [{item.evidence_type}] {item.statement}")
    print(f"  Source: {source_title}")
```

**Retrieval modes:**

| Constant | Value | Description |
|---|---|---|
| `RETRIEVAL_MODE_LEXICAL` | `"lexical"` | BM25 keyword search (default, no embeddings needed) |
| `RETRIEVAL_MODE_SEMANTIC` | `"semantic"` | Embedding similarity search |
| `RETRIEVAL_MODE_HYBRID` | `"hybrid"` | BM25 + semantic, combined score |

Semantic and hybrid modes require embeddings to be generated first (`python3 -m knowledge embed`).

### Multi-profile retrieval

When evidence across multiple profiles needs to be merged:

```python
profiles = ["project-a", "project-b"]
seen: dict[str, RetrievedEvidence] = {}

for profile in profiles:
    result = retriever.retrieve(query, profile=profile, top_k=top_k, load_sources=True)
    for item in result.items:
        eid = item.evidence.evidence_id
        # Keep highest score when the same item appears in multiple profiles
        if eid not in seen or item.score > seen[eid].score:
            seen[eid] = item

merged = sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]
```

### Working with evidence objects

`item.evidence` is an `Evidence` Pydantic model. Key fields:

```python
ev = item.evidence

ev.evidence_id          # str — unique ID (SHA-256 hash)
ev.statement            # str — the extracted claim
ev.evidence_type        # str — "STRATEGIC" | "TECHNICAL" | "PROVENANCE" | "ADMINISTRATIVE"
ev.entity               # str — primary entity (company, technology, policy)
ev.entity_type          # str — e.g. "Company", "Technology", "Policy"
ev.category             # str — topic category (e.g. "Financial", "Technical")
ev.scope                # str — geographic or market scope
ev.profile_ids          # list[str] — profiles this evidence belongs to
ev.supporting_source_ids  # list[str] — source IDs this claim was extracted from
ev.topics               # list[str] — topic tags
ev.evidence_confidence  # str | None — "HIGH" | "MEDIUM" | "LOW"
ev.is_quantitative      # bool — whether the claim contains quantitative data
ev.temporal_reference   # str | None — year or quarter extracted from the text (e.g. "2024", "Q3 2025")
ev.excerpt              # str | None — verbatim source excerpt (max 600 chars)
ev.page_number          # int | None — page in source PDF
```

`item.source` (when `load_sources=True`) is a `Source` Pydantic model:

```python
src = item.source

src.source_id           # str — unique ID
src.title               # str
src.organization        # str | None
src.author              # str | None
src.publication_date    # str | None
src.document_type       # str — e.g. "PDF", "WEB_ARTICLE", "YOUTUBE_TRANSCRIPT"
src.uri                 # str — file path or URL
src.domain              # str — source directory name (e.g. "articles")
src.canonical_text      # str — full extracted text
```

### The adapter pattern (recommended)

For applications that need to swap retrieval backends or handle the knowledge store being unavailable gracefully, wrap `EvidenceRetriever` in an adapter class. This is the pattern used by research-core:

```python
from pathlib import Path
from knowledge.store import KnowledgeStore
from knowledge.retriever import EvidenceRetriever, RETRIEVAL_MODE_LEXICAL


class KnowledgeAdapter:
    """Lazy-initialising wrapper around KnowledgeStore + EvidenceRetriever."""

    def __init__(self, store_root: str | Path, *, load_sources: bool = True) -> None:
        self._store_root = Path(store_root)
        self._load_sources = load_sources
        self._retriever: EvidenceRetriever | None = None

    @staticmethod
    def is_available() -> bool:
        try:
            import knowledge  # noqa: F401
            return True
        except ImportError:
            return False

    def retrieve(self, query: str, *, profile: str | None = None, top_k: int = 10):
        retriever = self._get_retriever()
        return retriever.retrieve(
            query,
            profile=profile,
            top_k=top_k,
            load_sources=self._load_sources,
            mode=RETRIEVAL_MODE_LEXICAL,
        )

    def _get_retriever(self) -> EvidenceRetriever:
        if self._retriever is None:
            store = KnowledgeStore(root=self._store_root)
            self._retriever = EvidenceRetriever(store=store)
        return self._retriever
```

The lazy initialisation means the adapter can be constructed at startup and will only connect to the store on the first `retrieve()` call.

---

## 2. MCP server

The MCP server exposes the knowledge base as a set of tools for any MCP-capable LLM client. It is the recommended way to give Claude Desktop or a custom LLM agent direct access to evidence.

### Installation

The server requires the `mcp` package which is included in the default dependencies:

```bash
pip install -e /path/to/knowledge-layer
```

### Running the server

The server reads from a pre-built knowledge store. Set the store path as an environment variable, then start the server:

```bash
export KNOWLEDGE_STORE_PATH=/absolute/path/to/knowledge_store

# Using the installed entry point:
knowledge-mcp

# Or directly:
python3 -m knowledge.mcp_server
```

The server runs over stdio (standard input/output), which is the protocol Claude Desktop and most MCP clients use.

**The server will fail to start** if `KNOWLEDGE_STORE_PATH` is not set or the path does not exist. Build the knowledge base first with `python3 -m knowledge build`.

### Claude Desktop configuration

Add the server to Claude Desktop's MCP config file.

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "knowledge-layer": {
      "command": "knowledge-mcp",
      "env": {
        "KNOWLEDGE_STORE_PATH": "/absolute/path/to/knowledge_store",
        "GEMINI_API_KEY": "your-key-here"
      }
    }
  }
}
```

If `knowledge-mcp` is not on the system PATH (e.g. installed in a virtualenv), use the full path to the executable:

```json
{
  "mcpServers": {
    "knowledge-layer": {
      "command": "/path/to/venv/bin/knowledge-mcp",
      "env": {
        "KNOWLEDGE_STORE_PATH": "/absolute/path/to/knowledge_store"
      }
    }
  }
}
```

Restart Claude Desktop after editing the config. The tools appear automatically in the tool picker.

### Available tools

The server exposes five tools:

---

#### `search_knowledge`

Search for evidence relevant to a query.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | Yes | Natural-language search query |
| `profile` | string | No | Filter by profile ID (e.g. `"ai-research"`) |
| `domain` | string | No | Filter by domain directory (e.g. `"articles"`) |
| `top_k` | integer | No | Results to return (default 10, max 50) |

Returns ranked evidence statements with source attribution.

---

#### `list_profiles`

List all profile IDs in the knowledge base with evidence counts. Use this before `search_knowledge` to discover valid profile names.

No parameters.

---

#### `list_domains`

List all domain directories in the knowledge base with evidence counts. Use this to discover valid domain names for filtering.

No parameters.

---

#### `list_sources`

List source documents in the knowledge base with titles and metadata.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `domain` | string | No | Filter by domain |
| `query` | string | No | Keyword filter on title or organisation name |

Returns source titles, authors, organisations, and truncated source IDs.

---

#### `get_source_evidence`

Return all evidence extracted from a specific source document.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | Yes | Full source ID or unique prefix (8+ characters) |

Use `list_sources` to find source IDs. The truncated `ID: abc123…` shown in `list_sources` output can be passed directly.

---

#### `get_source_text`

Return the complete original text of a source document.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `source_id` | string | Yes | Full source ID or unique prefix |

Use this when you need the full document context — exact quotes, speaker names, surrounding paragraphs — rather than individual extracted claims.

---

### Example tool interactions

**Discover what's available:**
> "What profiles and domains are in the knowledge base?"

Claude will call `list_profiles` and `list_domains` and report back.

**Search with filters:**
> "Search for evidence about GPU price trends in the ai-research profile"

Claude calls `search_knowledge(query="GPU price trends", profile="ai-research")`.

**Trace a claim back to its source:**
> "Show me the original text of the Goldman Sachs report"

Claude calls `list_sources(query="Goldman Sachs")`, picks the source ID, then calls `get_source_text(source_id="abc123...")`.

**Deep dive into a source:**
> "What did we extract from the Nvidia earnings report?"

Claude calls `list_sources(query="nvidia")`, then `get_source_evidence(source_id="...")`.
