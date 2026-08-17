# Knowledge Layer — User Guide

This guide covers everything needed to build and query a knowledge base from the command line.

---

## Table of contents

1. [Installation](#1-installation)
2. [Environment variables](#2-environment-variables)
3. [Organising sources](#3-organising-sources)
4. [Build configs (YAML)](#4-build-configs-yaml)
5. [Building the knowledge base](#5-building-the-knowledge-base)
6. [Querying evidence](#6-querying-evidence)
7. [Semantic and hybrid retrieval](#7-semantic-and-hybrid-retrieval)
8. [LLM reranking](#8-llm-reranking)
9. [Knowledge base maintenance](#9-knowledge-base-maintenance)
10. [Choosing a model](#10-choosing-a-model)

---

## 1. Installation

```bash
# Clone or copy the knowledge-layer directory, then install:
pip install -e ".[knowledge]"

# The [knowledge] extra adds sentence-transformers for semantic/hybrid retrieval.
# For lexical-only search, plain install is sufficient:
pip install -e .
```

---

## 2. Environment variables

Set the API key for the LLM provider you want to use for extraction and reranking:

```bash
export GEMINI_API_KEY=...       # Google Gemini (default provider)
export ANTHROPIC_API_KEY=...    # Anthropic Claude
export OPENAI_API_KEY=...       # OpenAI
```

You only need the key for the provider you're using. A `.env` file in the working directory is loaded automatically if the key is not already in the environment.

---

## 3. Organising sources

Create a source directory with subdirectories. Each subdirectory becomes a **domain** — a logical grouping of documents. Supported formats: **PDF**, **plain text**, **Markdown**.

```
sources/
  articles/         ← domain: "articles"
    report-2024.pdf
    market-analysis.pdf
  papers/           ← domain: "papers"
    technical-spec.pdf
  youtube/          ← domain: "youtube"
    transcript.txt
```

The domain name is the directory name. It is used for filtering at query time (`--domain articles`).

### Source metadata

Knowledge Layer infers source metadata (title, author, organisation, publication date) automatically from PDF properties and filename. You can override with a sidecar `.json` file of the same name:

```json
// sources/articles/report-2024.json
{
  "title": "AI Infrastructure Report 2024",
  "organization": "Goldman Sachs",
  "author": "Jane Smith",
  "publication_date": "2024-03"
}
```

---

## 4. Build configs (YAML)

For repeatable builds, use a YAML config file instead of inline flags. Store configs in `configs/knowledge/`.

```yaml
# configs/knowledge/my-project.yaml
name: my-project

profiles:
  - my-project

sources:
  - /absolute/path/to/sources/articles
  - /absolute/path/to/sources/papers
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Human-readable project name |
| `profiles` | Yes | One or more profile IDs to tag all evidence with |
| `sources` | Yes | List of source directory paths (absolute or relative to CWD) |

Environment variables and `~` are expanded in source paths.

---

## 5. Building the knowledge base

### Basic build

```bash
python3 -m knowledge build --sources sources/ --profiles my-project
```

### Build from a YAML config

```bash
python3 -m knowledge build --config configs/knowledge/my-project.yaml
```

### Incremental builds (default)

By default, sources are fingerprinted. If a source file hasn't changed since the last build, it is skipped. To force a full rebuild:

```bash
python3 -m knowledge build --config configs/knowledge/my-project.yaml --force
```

### Build options

| Flag | Default | Description |
|---|---|---|
| `--sources PATH` | — | One or more source directories |
| `--config PATH` | — | YAML build config (mutually exclusive with `--sources`) |
| `--profiles NAME` | — | Profile tag(s) to apply to all extracted evidence |
| `--domain NAME` | auto | Override domain for all sources |
| `--model MODEL` | `gemini-2.5-flash` | LLM for evidence extraction |
| `--store PATH` | `knowledge_store/` | Knowledge store directory |
| `--workers N` | `1` | Concurrent source processing threads (max 16) |
| `--incremental / --no-incremental` | on | Skip unchanged sources |
| `--force` | off | Rebuild all sources regardless of fingerprint |
| `--skip-extraction` | off | Index sources without extracting evidence |
| `--log-level` | `INFO` | Logging verbosity |

### What the build does

For each source file:
1. Extracts text (PDF pages with `[Page N]` markers, web fetch, or raw text)
2. Calls the LLM with a structured extraction tool to produce atomic evidence claims
3. Classifies each claim as `STRATEGIC`, `TECHNICAL`, `PROVENANCE`, or `ADMINISTRATIVE`
4. Deduplicates against existing evidence (by statement fingerprint)
5. Writes evidence and source metadata to `knowledge_store/`

---

## 6. Querying evidence

### Basic lexical search

```bash
python3 -m knowledge retrieve "GPU price growth 2024"
```

### Filter by profile

```bash
python3 -m knowledge retrieve "GPU price growth" --profile my-project
```

### Filter by domain

```bash
python3 -m knowledge retrieve "GPU price growth" --domain articles
```

### Combine filters

```bash
python3 -m knowledge retrieve "GPU price growth" --profile my-project --domain papers --top-k 20
```

### Retrieve options

| Flag | Default | Description |
|---|---|---|
| `--mode` | `lexical` | `lexical` \| `semantic` \| `hybrid` |
| `--profile NAME` | all | Restrict to evidence tagged with this profile |
| `--domain NAME` | all | Restrict to a specific domain |
| `--top-k N` | `10` | Number of results to return |
| `--types LIST` | all | Comma-separated filter: `STRATEGIC,TECHNICAL` |
| `--show-source` | off | Show source title under each result |
| `--all` | off | Include non-retrievable evidence (PROVENANCE, ADMINISTRATIVE) |
| `--store PATH` | `knowledge_store/` | Knowledge store path |
| `--log-level` | `WARNING` | Logging verbosity |

---

## 7. Semantic and hybrid retrieval

Semantic and hybrid modes use sentence-transformer embeddings for meaning-based matching. Embeddings must be generated before use.

### Generate embeddings

```bash
python3 -m knowledge embed --domain articles
# or all domains at once:
python3 -m knowledge embed
```

This runs once. Re-run only when new evidence is added (`--force` regenerates all).

### Use semantic or hybrid mode

```bash
# Semantic only (embedding similarity)
python3 -m knowledge retrieve "energy costs for data centres" --mode semantic

# Hybrid (BM25 + semantic, usually best results)
python3 -m knowledge retrieve "energy costs for data centres" --mode hybrid
```

### Override the embedding model

```bash
python3 -m knowledge embed --model all-mpnet-base-v2
python3 -m knowledge retrieve "..." --mode hybrid --embed-model all-mpnet-base-v2
```

Default embedding model: `all-MiniLM-L6-v2` (fast, 384-dim). Must be consistent between `embed` and `retrieve`.

---

## 8. LLM reranking

After retrieval, an LLM can reorder results by deeper semantic relevance. This is optional but improves quality for complex queries.

```bash
python3 -m knowledge retrieve "deployment risks for small modular reactors" \
  --mode hybrid \
  --rerank \
  --rerank-candidates 40 \
  --top-k 10 \
  --show-rationale
```

This retrieves 40 candidates, then uses the LLM to select and rank the best 10, returning a rationale for each.

### Rerank options

| Flag | Default | Description |
|---|---|---|
| `--rerank / --no-rerank` | off | Enable LLM reranking |
| `--rerank-candidates N` | `40` | Candidate pool size before reranking |
| `--rerank-model MODEL` | `gemini-2.5-flash` | LLM for reranking (can differ from extraction model) |
| `--show-rationale` | off | Print LLM rationale under each result |

---

## 9. Knowledge base maintenance

### Check what's in the store

```bash
# Summary statistics
python3 -m knowledge status

# Filter by profile
python3 -m knowledge status --profile my-project

# List all indexed sources
python3 -m knowledge list-sources

# Filter sources by domain or profile
python3 -m knowledge list-sources --domain articles --profile my-project

# List all profiles with evidence counts
python3 -m knowledge list-profiles
```

### Validate the store

```bash
python3 -m knowledge health
# Exit code 0 = ready; 1 = errors found
```

### Add a profile tag retroactively

If you ingested sources without `--profiles` and want to tag them afterwards:

```bash
python3 -m knowledge retag my-project
python3 -m knowledge retag my-project --domain articles  # specific domain only
```

---

## 10. Choosing a model

The `--model` flag accepts any of these strings. The provider is inferred automatically from the name.

| Model | Provider | Speed | Quality | Notes |
|---|---|---|---|---|
| `gemini-2.5-flash` | Gemini | Fast | High | Default; best cost/quality balance |
| `gemini-2.0-flash` | Gemini | Fast | Good | Older, slightly cheaper |
| `claude-haiku-4-5-20251001` | Anthropic | Fast | Good | Good for high-volume extraction |
| `claude-sonnet-5` | Anthropic | Medium | Very high | Best for complex reranking |
| `gpt-4o-mini` | OpenAI | Fast | Good | Cost-effective alternative |
| `gpt-4o` | OpenAI | Medium | High | Higher quality, higher cost |

Use one model for extraction (`--model`) and optionally a different one for reranking (`--rerank-model`). A smaller model for extraction and a larger one for reranking is a common and cost-effective pattern.
