# K1.0 – Minimal Internet Retrieval Vertical Slice

You are working on a research harness that currently operates exclusively on local source documents.

Current flow:

```text
User Question
    ↓
Chunk Retrieval
    ↓
Context Assembly
    ↓
LLM Synthesis
    ↓
Answer + Citations
```

The system already supports:

- Local source ingestion
- Chunking
- Retrieval
- Citation tracking
- Trace generation
- Synthetic contradiction testing

DO NOT redesign the architecture.

DO NOT introduce agents.

DO NOT implement query planning, query rewriting, reranking, autonomous search loops, or any "agentic" behavior.

The goal is a minimal vertical slice called K1.0.

---

# Objective

Add internet retrieval as an optional additional retrieval source.

Target flow:

```text
User Question
        ↓
 ┌───────────────┐
 │ Local Search  │
 └───────────────┘
        +
 ┌───────────────┐
 │ Web Search    │
 └───────────────┘
        ↓
Combined Context
        ↓
LLM Synthesis
```

The implementation should be simple, observable, and compatible with the existing architecture.

---

# Requirements

## K1.1 Search Provider

Create a simple search abstraction:

```python
class SearchProvider:
    def search(self, query: str) -> list[SearchResult]:
        ...
```

Implement:

```python
DuckDuckGoSearchProvider
```

Use:

```python
duckduckgo_search
```

or the simplest stable equivalent.

Do not implement multiple providers.

Do not add provider selection logic yet.

---

## K1.2 Web Document Downloader

Given a URL:

```python
download_web_document(url)
```

Retrieve page content.

Use:

- requests
- trafilatura

Preferred extraction:

```python
trafilatura.extract()
```

Return:

```python
WebDocument(
    url=...,
    title=...,
    text=...
)
```

Strip navigation, menus, footers, and boilerplate content.

---

## K1.3 Chunking

Reuse the existing chunking pipeline.

Web documents must become the same internal chunk format as local documents.

Example:

```python
Chunk(
    source_type="web",
    source_url=...,
    text=...
)
```

No special handling.

No separate web chunk format.

---

## K1.4 Retrieval Integration

Add configuration:

```yaml
web_search:
  enabled: true
  max_results: 5
```

When enabled:

1. Run local retrieval.
2. Run web search.
3. Download top N pages.
4. Extract content.
5. Chunk content.
6. Add chunks to retrieval pool.

Keep implementation simple.

No reranking.

No score fusion.

No query expansion.

No semantic filtering.

---

## K1.5 Trace Visibility

Extend trace output.

Current trace shows local retrieval activity.

Add:

```text
WEB SEARCH
-----------
Query:
Results:
Downloaded URLs:
Extracted Characters:
Chunks Created:
```

I must be able to inspect every stage.

Observability is more important than optimization.

---

## K1.6 Citation Metadata

For web-derived chunks include:

```python
source_type="web"
source_url="..."
title="..."
```

Final answers should clearly indicate whether evidence came from:

```text
LOCAL
WEB
```

sources.

Do not replace existing citation behavior.

Only extend it.

---

## K1.7 Configuration

Add settings:

```yaml
web_search:
  enabled: true
  max_results: 5
  max_pages: 5
  timeout_seconds: 20
```

Integrate with the existing config structure.

---

## K1.8 Local Cache

Add a simple disk cache.

Goal:

- Avoid repeatedly downloading the same pages.
- Speed up debugging and development.

Simple implementation is acceptable:

```text
.cache/web/
```

Cache key:

```text
URL hash
```

Store:

- URL
- title
- extracted text
- timestamp

Behavior:

```text
cache hit  -> use cached content
cache miss -> download and cache
```

No cache invalidation logic required yet.

---

## K1.9 Failure Handling

Web retrieval failures must never fail the run.

Examples:

- search timeout
- page timeout
- extraction failure
- invalid URL

Log warning.

Continue execution.

The harness should always degrade gracefully.

---

# Explicit Non-Goals

Do NOT implement:

- agents
- planning
- query rewriting
- recursive search
- multi-hop retrieval
- reranking
- score fusion
- trust scoring
- source quality scoring
- semantic filtering
- citation verification
- autonomous research loops

Those belong to future K-series milestones.

---

# Acceptance Tests

## Test 1

Question:

```text
What is the rack power requirement of NVIDIA GB200 NVL72?
```

Expected:

- local retrieval still works
- web retrieval executes
- trace shows web search activity
- citations include local and/or web sources

---

## Test 2

Question:

```text
What is the status of the BWRX-300 deployment in Canada?
```

Expected:

- answer can be generated even if no local source exists
- web retrieval executes
- trace shows downloaded pages
- web citations appear

---

## Test 3

Configuration:

```yaml
web_search:
  enabled: false
```

Expected:

- behavior matches pre-K1.0 harness
- no web retrieval occurs
- no regression in local retrieval

---

# Deliverables

Provide:

1. Architecture summary
2. Files changed
3. New dependencies
4. Configuration changes
5. Example trace output
6. Example citation output
7. Any technical debt introduced for future K1.x work

Favor:

- minimal code
- maximum observability
- backward compatibility
- incremental architecture

Do not over-engineer the solution.