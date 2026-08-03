# Knowledge Layer — Standalone API Reference

The Knowledge Layer is independently usable by any Python application that needs
to query an existing knowledge store. **Read-only retrieval requires zero imports
from `functional_agents`, `strategy`, or `editorial`.**

---

## Supported Use Cases

| Use case | Standalone? |
|---|---|
| Open an existing knowledge store | ✓ fully standalone |
| Lexical query (no embeddings) | ✓ fully standalone |
| Hybrid query (lexical + semantic) | ✓ standalone (requires `sentence-transformers`) |
| Filter by profile | ✓ fully standalone |
| Inspect store health and status | ✓ fully standalone |
| Read provenance-bearing results | ✓ fully standalone |
| Build a store (skip-extraction) | ✓ fully standalone |
| Build a store (with LLM extraction) | ⚠ requires `research_agent` — see caveat below |

---

## Programmatic API

### 1. Open a store and retrieve (lexical, no embeddings required)

```python
from knowledge.store import KnowledgeStore
from knowledge.retriever import EvidenceRetriever, RETRIEVAL_MODE_LEXICAL

store = KnowledgeStore("knowledge_store")          # path is CWD-relative by default
retriever = EvidenceRetriever(store, provider=None) # provider=None → lexical only

result = retriever.retrieve(
    query="competitive strategy for sports analytics",
    mode=RETRIEVAL_MODE_LEXICAL,  # "lexical" | "semantic" | "hybrid"
    profile="sports",             # optional: restrict to this profile_id
    top_k=10,
    evidence_types=["STRATEGIC"], # optional: "STRATEGIC" | "TECHNICAL" | "PROVENANCE" | "ADMINISTRATIVE"
    retrieval_enabled_only=True,  # exclude evidence flagged for non-retrieval
    min_score=0.01,
    load_sources=False,           # set True to eagerly load Source metadata per item
)
```

### 2. Hybrid retrieval (semantic + lexical)

```python
from knowledge.embeddings import get_provider   # downloads sentence-transformers model on first call

provider = get_provider()                        # or get_provider(model_name="all-MiniLM-L6-v2")
retriever = EvidenceRetriever(store, provider=provider)

result = retriever.retrieve(
    query="...",
    mode="hybrid",
    top_k=20,
)
```

### 3. Store utilities

```python
store = KnowledgeStore("knowledge_store")

# Retrieve one evidence item by domain and ID
ev = store.read_evidence("sports", "ev-abc123")      # Evidence | None

# Retrieve one source document
src = store.find_source("src-xyz789")                # Source | None

# Store-level statistics
stats = store.read_stats()                           # dict with cached build stats

# All available domains
domains = store.available_domains()                  # ["ai_data_centers", "sports", ...]

# Evidence count in a domain
n = store.evidence_count("sports")                   # int

# Iterate all evidence in a domain
for ev in store.iter_evidence("sports"):
    print(ev.evidence_id, ev.statement[:80])
```

---

## Return Contract

`retriever.retrieve(...)` returns a `RetrievalResult`:

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The original query string |
| `items` | `list[RetrievedEvidence]` | Ranked result items |
| `domains_searched` | `list[str]` | Domains actually searched |
| `total_candidates` | `int` | Evidence items examined |
| `matched_candidates` | `int` | Items that passed score threshold |
| `retrieval_method` | `str` | e.g. `"lexical-v1"` |
| `mode` | `str` | `"lexical"` / `"semantic"` / `"hybrid"` |
| `latency_ms` | `float` | Wall-clock retrieval time |
| `semantic_model` | `str \| None` | Embedding model name (hybrid/semantic only) |

Each item in `result.items` is a `RetrievedEvidence`:

| Field | Type | Notes |
|---|---|---|
| `evidence` | `Evidence` | Full evidence record |
| `evidence.evidence_id` | `str` | Canonical chunk ID |
| `evidence.statement` | `str` | Chunk text |
| `evidence.evidence_type` | `str` | `"STRATEGIC"` / `"TECHNICAL"` / … |
| `evidence.supporting_source_ids` | `list[str]` | Source document IDs |
| `evidence.profile_ids` | `list[str]` | Profile tags (e.g. `["sports"]`) |
| `evidence.excerpt` | `str \| None` | Original passage (may be absent) |
| `evidence.page_number` | `int \| None` | Page number (PDF sources) |
| `evidence.chunk_id` | `str \| None` | Source chunk identifier |
| `metadata` | `KnowledgeMetadata` | Quality and priority scores |
| `score` | `float` | Combined retrieval score |
| `rank` | `int` | 1-based rank in result set |
| `lexical_score` | `float` | Lexical component (0.0 in semantic-only mode) |
| `semantic_score` | `float` | Semantic component (0.0 in lexical-only mode) |
| `source_domain` | `str` | Knowledge store domain this item came from |
| `metadata_factor` | `float \| None` | Metadata quality multiplier (provenance) |

**Lazy source loading:**

```python
for item in result.items:
    src = item.load_source(store)    # Source | None
    if src:
        print(src.title, src.organization, src.uri)
```

`Source` fields: `source_id`, `title`, `author`, `publisher`, `publication_date`,
`organization`, `uri`, `domain`, `document_type`, `language`.

**Note:** `evidence.provenance` is a `RetrievalProvenance` field that is populated
when `build_retrieval_provenance()` is explicitly called after retrieval (e.g. in
the full research pipeline). It is **not** populated automatically by `retrieve()`.
Use `item.score`, `item.rank`, `item.lexical_score`, `item.semantic_score`, and
`item.source_domain` for traceability in standalone use.

---

## CLI Usage

All read operations are fully standalone (no API key required):

```bash
# Lexical retrieval
python3 -m knowledge retrieve "competitive strategy" --store knowledge_store

# Hybrid retrieval, top 5, sports profile
python3 -m knowledge retrieve "sports analytics competitive strategy" \
  --mode hybrid \
  --store knowledge_store \
  --top-k 5

# Filter to a specific domain
python3 -m knowledge retrieve "deployment risks" --domain smr

# Filter to specific evidence types
python3 -m knowledge retrieve "investment barriers" --types STRATEGIC,TECHNICAL

# Show source titles alongside results
python3 -m knowledge retrieve "market sizing" --show-source

# Store status summary
python3 -m knowledge status --store knowledge_store
python3 -m knowledge status --store knowledge_store --profile sports

# Health check (exits 1 if not ready)
python3 -m knowledge health --store knowledge_store

# Embed evidence (precomputes semantic vectors)
python3 -m knowledge embed --store knowledge_store

# Build a store from source directories (no LLM)
python3 -m knowledge build --config knowledge/configs/sports.yaml --skip-extraction

# Build with LLM extraction (requires ANTHROPIC_API_KEY, see caveat)
python3 -m knowledge build --config knowledge/configs/sports.yaml
```

Common flags available on most commands:

| Flag | Description |
|---|---|
| `--store PATH` | Path to knowledge store (default: `knowledge_store` in CWD) |
| `--log-level TEXT` | `DEBUG` / `INFO` / `WARNING` (default: `WARNING`) |

---

## Build Caveat — `research_agent` Coupling

The build pipeline has **three lazy coupling points** to `research_agent`.
All three are guarded: they only execute when LLM extraction is active.

### Coupling point 1 — `knowledge/cli.py:268`

```python
# Active only when: --skip-extraction flag is NOT set
if not skip_extraction:
    from research_agent.claude_client import ClaudeClient
    client = ClaudeClient(model=model)
```

**Purpose:** ClaudeClient is the LLM client used to extract evidence from source text.

**Standalone impact:** Pass `--skip-extraction` to avoid this import entirely.
With `--skip-extraction`, sources are indexed and manifested without LLM extraction.

### Coupling point 2 — `knowledge/extractor.py:263`

```python
# Active only when: client.extract_evidence() is called (i.e. not skip-extraction)
from research_agent.schemas import SourceDocument
```

**Purpose:** `SourceDocument` is the input type expected by `ClaudeClient.extract_evidence()`.
It adapts the `Source` model to the legacy client interface.

**Standalone impact:** None when using `--skip-extraction` or read-only retrieval.

### Coupling point 3 — `knowledge/reranker.py:276`

```python
# Active only when: LLMReranker._rerank_with_llm() is called
# i.e. only when --rerank flag is used in the CLI or LLMReranker is instantiated
from research_agent.llm_normalize import normalize_llm_items
```

**Purpose:** Normalizes LLM reranker output (handles malformed JSON).

**Standalone impact:** None unless `--rerank` is used or `LLMReranker` is instantiated.
The default retriever does not use reranking.

### Summary

| Mode | `research_agent` imported? |
|---|---|
| Read-only retrieval (lexical or hybrid) | **No** |
| `python3 -m knowledge status/health/embed` | **No** |
| `python3 -m knowledge build --skip-extraction` | **No** |
| `python3 -m knowledge build` (with extraction) | **Yes** (ClaudeClient, SourceDocument) |
| `python3 -m knowledge retrieve --rerank` | **Yes** (llm_normalize) |

---

## Path Behavior

- `KnowledgeStore("knowledge_store")` resolves relative to the **current working directory**.
- `KnowledgeStore("/absolute/path/to/store")` accepts absolute paths.
- `KnowledgeStore()` (no argument) defaults to `Path("knowledge_store")` — CWD-relative.
- The CLI `--store PATH` flag accepts both relative and absolute paths; relative paths
  resolve from CWD at invocation time.
- Config files (`--config knowledge/configs/sports.yaml`) resolve from CWD.
- There are **no hardcoded absolute paths** in the knowledge layer.

---

## Stability Boundary

| Interface | Status | Notes |
|---|---|---|
| `KnowledgeStore(root)` | **Supported public** | Constructor signature stable since J8.0 |
| `KnowledgeStore.read_evidence(domain, id)` | **Supported public** | Returns `Evidence \| None` |
| `KnowledgeStore.find_source(source_id)` | **Supported public** | Cross-domain lookup via manifest |
| `KnowledgeStore.read_source(domain, id)` | **Supported public** | Direct domain-scoped lookup |
| `KnowledgeStore.iter_evidence(domain)` | **Supported public** | Generator; stable contract |
| `KnowledgeStore.available_domains()` | **Supported public** | Returns sorted list |
| `KnowledgeStore.read_stats()` | **Supported public** | Returns dict (cached build stats) |
| `EvidenceRetriever(store, provider)` | **Supported public** | `provider=None` for lexical |
| `EvidenceRetriever.retrieve(...)` | **Supported public** | Primary retrieval API |
| `RetrievedEvidence` | **Supported public** | Stable dataclass; all fields documented above |
| `RetrievalResult` | **Supported public** | Stable dataclass; all fields documented above |
| `RETRIEVAL_MODE_LEXICAL/SEMANTIC/HYBRID` | **Supported public** | String constants |
| `build_retrieval_provenance(...)` | **Supported public** | Use when building pipeline provenance |
| `KnowledgeBuilder` | **Supported public** | For programmatic builds |
| `EvidenceRetriever._score()` | **Internal** | Static helper; do not depend on signature |
| `EvidenceRetriever._metadata_factor()` | **Internal** | Scoring formula; subject to change |
| `KnowledgeStore._atomic_write()` | **Internal** | Write helper; do not call directly |
| `LLMReranker` | **Experimental** | Requires `research_agent`; not stable |
| `tokenize_query()` / `detect_intent()` | **Internal** | Used by retriever; may change |
| `knowledge/assembly.py` | **Internal** | Post-retrieval assembly; not a public API |
| `knowledge/grounding.py` | **Internal** | Grounding helper; not a public API |
| `knowledge/fingerprint.py` | **Internal** | Incremental build; not a public API |

---

## No Adapter Required

`EvidenceRetriever.retrieve()` is directly callable with four lines:

```python
from knowledge.store import KnowledgeStore
from knowledge.retriever import EvidenceRetriever
store = KnowledgeStore("knowledge_store")
result = EvidenceRetriever(store).retrieve("your query", top_k=10)
```

No wrapper or adapter is needed for standalone read-only use. The `profile` filter
and `evidence_types` filter cover the common configuration needs without additional
plumbing.
