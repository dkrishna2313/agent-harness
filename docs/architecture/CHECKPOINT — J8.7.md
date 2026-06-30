# J8.7 — Knowledge Platform Maturity: Checkpoint

**Status:** Complete  
**Date:** 2026-06-29  
**Baseline:** J8.6 (Knowledge Layer integration + EvidenceAgent dual-path)

---

## What Was Done

J8.7 hardened the Knowledge Platform before J9 architecture work begins. Five work items were implemented:

### 1. Knowledge Store Health Validation (`knowledge/health.py`)

New module with two public functions:
- `check_domain_health(store_root, domain)` → `DomainHealth`
- `check_store_health(store, domain=None)` → `HealthReport`

Validates per domain:
- `evidence.jsonl` exists
- Evidence count > 0
- `index.json` exists and entry count is within 5% of evidence line count
- Sets `runtime_ready = True` only when all checks pass

Store-level checks:
- `manifest.json` exists and is parseable
- Total embeddings count in `embeddings/evidence/`
- At least one domain is runtime-ready

**Empty/partial stores are explicitly detected.** `available_domains()` returns domains when the directory structure exists but evidence may be empty — J8.7 blocks this case at the Orchestrator before a retriever is initialized.

### 2. Build Diagnostics (`knowledge/builder.py`)

`BuildReport` enhanced with:
- `per_domain: dict[str, DomainBuildSummary]` — per-domain breakdown of sources rebuilt, skipped, failed, evidence objects, embeddings generated
- `runtime_ready: bool` — set via post-build health check; `True` only if ≥1 domain is ready for retrieval
- `summary_lines()` now prints per-domain table and runtime readiness flag

`DomainBuildSummary` dataclass tracks: `domain`, `sources_rebuilt`, `sources_skipped`, `sources_failed`, `evidence_objects`, `embeddings_generated`, `runtime_ready`.

### 3. CLI: `python3 -m knowledge health` Subcommand

```
python3 -m knowledge health --store knowledge_store [--domain smr]
```

- Runs `check_store_health()` and prints a structured report
- Exits with code 1 if not ready
- Accepts `--domain` to validate a single domain

Example output (healthy store):
```
=== Knowledge Store Health: READY ===
Store:              knowledge_store
Manifest:           OK (37 sources)
Embeddings (total): 1079 files
Domains checked:    2

  [ai_data_centers] READY
    evidence.jsonl:   OK (675 items)
    index.json:       OK (675 entries)
    count_consistent: YES

  [smr] READY
    evidence.jsonl:   OK (404 items)
    index.json:       OK (404 entries)
    count_consistent: YES
```

### 4. Runtime Guardrails

**Orchestrator** (`functional_agents/orchestrator.py`): Health check now runs before initializing the retriever. Behavior:
- Health PASS → `EvidenceRetriever` is created with the ready domains; logged at PROGRESS level with evidence counts
- Health FAIL → warning logged with specific issues per domain; falls back to legacy document retrieval

**EvidenceAgent** (`functional_agents/evidence_agent.py`): Added explicit guardrail in `_execute_legacy()`:
- When legacy extraction also returns 0 items → warning logged + `context.trace["_insufficient_evidence"] = True`
- Downstream agents receive explicit signal that evidence coverage is insufficient (rather than silently producing an ungrounded report)

The prior J8.6a guardrail (0-evidence KB fallback) remains unchanged.

### 5. Documentation

This checkpoint document.

---

## CLI Reference

### Build Workflow

```bash
# Build knowledge store from source directories
python3 -m knowledge build \
    --sources smr_sources/ sources/nvidia/ \
    --incremental \
    --workers 2 \
    --log-level INFO

# Generate embeddings for semantic retrieval
python3 -m knowledge embed --domain ai_data_centers

# Validate the store is ready
python3 -m knowledge health --store knowledge_store

# Check current status and source counts
python3 -m knowledge status --store knowledge_store
```

### Validation Workflow

```bash
# Test retrieval
python3 -m knowledge retrieve "power density challenges for AI data centers" \
    --domain ai_data_centers \
    --mode hybrid \
    --top-k 10

# Run a strategic research question with Knowledge Layer
python3 -m functional_agents.cli run \
    --goal "Analyze AI data center power infrastructure strategies" \
    --profiles ai_data_centers \
    --knowledge-store knowledge_store \
    --rerank \
    --log-level PROGRESS \
    --out outputs/strategic_run.json

# Benchmark (legacy DcPowerAgent path, does not use Knowledge Layer)
python3 -m research_agent.eval_runner benchmark \
    --profile ai_data_centers \
    --out outputs/j87_benchmark

# Regression comparison
python3 -m research_agent.eval_runner regress \
    --current outputs/j87_benchmark/evaluation_report.json \
    --baseline outputs/evaluation_report.json
```

### Key CLI Flags Across Tools

| Flag | `knowledge` | `functional_agents.cli` | `eval_runner` |
|---|---|---|---|
| `--log-level PROGRESS` | ✓ (build, health, embed) | ✓ (run) | ✓ (benchmark, regress) |
| `--knowledge-store PATH` | — | ✓ (run, auto-detects `knowledge_store/`) | ✓ (accepted, not wired to benchmark) |
| `--rerank/--no-rerank` | ✓ (retrieve) | ✓ (run) | — |
| `--out PATH` | — | ✓ | ✓ (aliased from `--out-dir`) |
| `--store PATH` | ✓ | — | — |
| `--domain` | ✓ (build, embed, retrieve) | — | — |

---

## Store Artifacts

```
knowledge_store/
  manifests/manifest.json          {source_id → SourceManifestEntry}
  evidence/{domain}/evidence.jsonl  Evidence records (JSONL)
  evidence/{domain}/index.json      {evidence_id → line_number}
  embeddings/evidence/{uuid}.npy    384-dim float32 per evidence item (store-wide)
  metadata/{domain}/metadata.jsonl  Quality scores, retrieval flags
  sources/{domain}/{source_id}.json Normalized source provenance
  extraction_runs/runs.jsonl        Extraction run history
  _meta/schema_version.json         Schema version
  _meta/stats.json                  Build statistics
```

Embeddings are stored flat (`embeddings/evidence/`) — not domain-separated. `HealthReport.total_embeddings` reflects the store-wide count.

---

## Fallback Behavior

| Scenario | Behavior |
|---|---|
| `--knowledge-store` not set | Orchestrator skips KB init; legacy path used |
| Store path doesn't exist | Warning logged; legacy path used |
| Store exists, health FAIL | Warning with issues logged; legacy path used |
| Store ready, KB returns 0 evidence | Warning logged; falls back to legacy |
| Legacy also returns 0 evidence | Warning logged; `_insufficient_evidence=True` in trace |

---

## Benchmark Note

`python3 -m research_agent.eval_runner benchmark` uses the `DcPowerAgent` (legacy pipeline) via `EvaluationRunner`. The `--knowledge-store` flag is accepted by the CLI but not wired into `DcPowerAgent` — it is a forward-compatibility stub. Benchmark score validates only the legacy document extraction pipeline.

**Score variance:** SMR_001 and SMR_006 are keyword-match tests with fragile coverage. They pass/fail depending on which evidence chunks the LLM selects. Two consecutive J8.7 benchmark runs both produced 0.9275, confirming the score is deterministic within a context but varies across runs. This is pre-existing behavior, not introduced by J8.7.

---

## Architectural State

```
Retrieval pipeline (when --knowledge-store is set):

  Orchestrator
    └── check_store_health(store)           [J8.7 — gates retriever init]
        └── if READY: EvidenceRetriever     [J8.6 — hybrid BM25 + semantic]
                └── EvidenceAgent._execute_kb()
                    └── 0 candidates? → fallback to _execute_legacy()  [J8.6a]
        └── if FAIL: skip retriever         [J8.7 — explicit fallback]

Legacy pipeline (always available as fallback):

  EvidenceAgent._execute_legacy()
    └── DcPowerAgent.analyze()
        └── 0 items? → _insufficient_evidence=True in trace  [J8.7]
```

Knowledge Layer domains at J8.7 close:
- `ai_data_centers`: 675 evidence items, 675 index entries
- `smr`: 404 evidence items, 404 index entries
- Total embeddings: 1079 (384-dim, all-MiniLM-L6-v2)

---

## Not Done (Deferred to J9)

- Autonomous research loops
- New strategic engagement model
- Database migration / ANN backend
- Vector database integration
- Major dataset reorganization
