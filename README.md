# Knowledge Layer

A persistent evidence store for strategic research. Ingest documents and articles, extract structured evidence using an LLM, then query that evidence by keyword or meaning — from the CLI, from Python, or via an MCP server connected to Claude Desktop.

---

## What it does

Knowledge Layer takes unstructured documents (PDFs, web articles, YouTube transcripts) and turns them into a searchable store of atomic evidence claims — facts, figures, risks, and strategic insights that are valuable for future decision-making.

Once built, the store can be queried by any of three methods:

| Access method | Use case |
|---|---|
| **CLI** | Interactive exploration, one-off queries, batch builds |
| **Python API** | Embed retrieval into your own application |
| **MCP server** | Connect to Claude Desktop or any MCP-capable LLM client |

---

## Quick start

### 1. Install

```bash
pip install -e ".[knowledge]"
```

The `[knowledge]` extra installs sentence-transformers for semantic/hybrid retrieval. Omit it if you only need lexical (keyword) search.

### 2. Set API key

Evidence extraction and reranking use an LLM. Set the key for whichever provider you use:

```bash
export GEMINI_API_KEY=...       # for gemini-2.5-flash (default)
export ANTHROPIC_API_KEY=...    # for claude-* models
export OPENAI_API_KEY=...       # for gpt-* models
```

### 3. Organise your sources

Place documents in a source directory. Supported formats: PDF, plain text, Markdown.

```
sources/
  articles/
    nvidia-earnings.pdf
    ai-energy-report.pdf
  papers/
    attention-is-all-you-need.pdf
```

### 4. Build the knowledge base

```bash
python3 -m knowledge build --sources sources/ --profiles my-project
```

This extracts evidence from every document and writes it to `knowledge_store/` (created automatically).

### 5. Query it

```bash
python3 -m knowledge retrieve "GPU price trends" --profile my-project
```

That's it. See [USER_GUIDE.md](docs/USER_GUIDE.md) for the full CLI reference and [INTEGRATION.md](docs/INTEGRATION.md) for Python API and MCP server setup.

---

## Architecture

```
sources/          ← raw documents (PDF, text, markdown)
    │
    ▼
KnowledgeBuilder  ← extracts evidence via LLM tool-use (gemini / claude / openai)
    │
    ▼
knowledge_store/  ← persistent store (JSON files, BM25 index, embeddings)
    │
    ├── CLI         python3 -m knowledge retrieve "..."
    ├── Python API  KnowledgeStore + EvidenceRetriever
    └── MCP server  knowledge-mcp  →  Claude Desktop
```

### Evidence types

Each extracted claim is classified as one of four types:

| Type | Description | Searchable |
|---|---|---|
| `STRATEGIC` | Market, policy, competitive claims | Yes |
| `TECHNICAL` | Specifications, performance parameters, costs | Yes |
| `PROVENANCE` | Authorship, publication metadata | No |
| `ADMINISTRATIVE` | Copyright, revision numbers, boilerplate | No |

Only `STRATEGIC` and `TECHNICAL` evidence surfaces in retrieval by default.

---

## Repository layout

```
knowledge/              ← core library
  builder.py            ← KnowledgeBuilder (orchestrates ingestion)
  store.py              ← KnowledgeStore (read/write persistence)
  retriever.py          ← EvidenceRetriever (lexical / semantic / hybrid)
  reranker.py           ← LLMReranker (optional LLM-assisted reranking)
  extractor.py          ← evidence extraction via LLM tool-use
  llm_client.py         ← provider-agnostic LLM routing (Gemini / Anthropic / OpenAI)
  embeddings.py         ← sentence-transformer embedding backend
  models.py             ← Pydantic data models (Evidence, Source, etc.)
  source_normalizer.py  ← PDF → canonical text, web fetch, YouTube transcripts
  mcp_server.py         ← MCP server entry point
  cli.py                ← Typer CLI entry point
configs/
  knowledge/            ← YAML build configs (one per project)
docs/
  USER_GUIDE.md         ← full CLI and build reference
  INTEGRATION.md        ← Python API and MCP server integration
```

---

## Supported LLM providers

The model string determines the provider automatically — no prefix needed:

| Model string | Provider | Key |
|---|---|---|
| `gemini-2.5-flash` (default) | Google Gemini | `GEMINI_API_KEY` |
| `gemini-2.0-flash` | Google Gemini | `GEMINI_API_KEY` |
| `claude-sonnet-5` | Anthropic | `ANTHROPIC_API_KEY` |
| `claude-haiku-4-5-20251001` | Anthropic | `ANTHROPIC_API_KEY` |
| `gpt-4o-mini` | OpenAI | `OPENAI_API_KEY` |
| `gpt-4o` | OpenAI | `OPENAI_API_KEY` |

Pass any of these to `--model` (extraction) or `--rerank-model` (reranking).
