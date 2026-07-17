# Knowledge Layer Provenance Audit

**Document Purpose:** Architectural specification for provenance throughout the Knowledge Layer  
**Audit Date:** 2026-07-16  
**Milestone:** PH5.3  
**Status:** Authoritative — this document supersedes any prior informal provenance notes

---

## Table of Contents

1. [Current Provenance Model](#1-current-provenance-model)
2. [Provenance Levels Table](#2-provenance-levels-table)
3. [Provenance Flow Diagram](#3-provenance-flow-diagram)
4. [Architectural Gaps](#4-architectural-gaps)
5. [Canonical Provenance Model](#5-canonical-provenance-model)
6. [Overall Assessment](#6-overall-assessment)
7. [Roadmap](#7-roadmap)

---

## 1. Current Provenance Model

This section describes what provenance information is actually present at each phase of the Knowledge Layer, based on direct inspection of the implementation.

### 1.1 Ingestion Phase (`knowledge/builder.py`, `knowledge/models.py`)

**What is captured:**

| Field | Location | Description |
|-------|----------|-------------|
| `source_id` | `Source.source_id` | First 32 hex characters of the SHA-256 of `canonical_text`; content-addressed |
| `uri` | `Source.uri` | Original file path or URL |
| `title` | `Source.title` | Document title, derived from filename or metadata |
| `canonical_text` | `Source.canonical_text` | Full document text; for PDFs contains `[Page N]\n` text markers embedded inline |
| `page_count` | `Source.page_count` | Total page count (integer); structural metadata only |
| `domain` | `Source.domain` | Knowledge domain classifier (e.g., `smr`, `ai_data_centers`) |
| `extraction_run_id` | `Evidence.extraction_run_id` | UUID linking evidence to the run that produced it |
| `supporting_source_ids` | `Evidence.supporting_source_ids` | List of `source_id` values the evidence came from |
| `statement_fingerprint` | `Evidence.statement_fingerprint` | SHA-256[:16] of the normalized statement; used for extraction-time deduplication |
| `evidence_id` | `Evidence.evidence_id` | UUID; primary key for evidence items |
| `run_id → source_ids → evidence_ids_produced` | `ExtractionRun` | Narrow provenance record per extraction run |

**What is not captured at ingestion:**

| Missing Field | Impact |
|---------------|--------|
| Chunk identifier | Cannot trace evidence to a specific passage within the source document |
| Page number (structured) | Page info appears only as `[Page N]\n` text markers in `canonical_text`; not a queryable field |
| Excerpt / verbatim passage | No verbatim text preserved; only the extracted claim statement |
| Section heading | Document structure (chapter, section, subsection) not captured |
| Character offset | No start/end byte or character offset into the source document |
| Token range | No tokenization boundary preserved |

**Key observation on page numbers:** The builder writes `f"[Page {i}]\n{text}"` into `canonical_text` during PDF extraction. This makes page information recoverable only by text parsing — it is not a structured field on `Source` or `Evidence`. Page numbers exist in the corpus but are not addressable.

### 1.2 Storage Phase (`knowledge/store.py`)

**What the store persists:**

| Artifact | Path | Content |
|----------|------|---------|
| Source documents | `sources/{domain}/{source_id}.json` | Full `Source` model including `canonical_text` |
| Evidence items | `evidence/{domain}/evidence.jsonl` | `Evidence` model; append-order in JSONL determines corpus-build order |
| Evidence index | `evidence/{domain}/index.json` | `{evidence_id → line_number}`; lookup index into JSONL |
| Extraction runs | `extraction_runs/runs.jsonl` | `ExtractionRun` records; run_id, source_ids, evidence_ids, model_version, prompt_version |
| Source manifest | `manifests/manifest.json` | `{source_id → SourceManifestEntry}`; tracks domain, fingerprint, uri, evidence_ids per source |
| Embeddings | `embeddings/evidence/{evidence_id}.npy` | Sentence-level embedding per evidence item; domain not tracked in filename |

**Provenance gap at storage:** JSONL append order is not stored as a queryable field. The line number in `index.json` is an implementation artifact, not a provenance field. If evidence is re-indexed or migrated, the effective corpus-build order changes, and tiebreak behavior in the retriever changes silently.

**What the store does not persist:**

- Retrieval scores (score, rank, lexical_score, semantic_score) — these are runtime-only
- Reranker rationale from `LLMReranker` — computed during reranking, discarded after the run
- Chunk-level intermediate representations — full text is stored, but chunking is done at query time by the retriever

### 1.3 Retrieval Phase (`knowledge/retriever.py`)

**What the retriever computes at runtime:**

| Field | On | Description |
|-------|----|-------------|
| `score` | `RetrievedEvidence` | Hybrid final score: `0.4 × lex_relevance + 0.6 × sem_similarity × metadata_factor` |
| `rank` | `RetrievedEvidence` | Integer rank in the retrieved set (1-indexed) |
| `lexical_score` | `RetrievedEvidence` | BM25 or TF-IDF lexical relevance score |
| `semantic_score` | `RetrievedEvidence` | Cosine similarity between query embedding and evidence embedding |
| `source_domain` | `RetrievedEvidence` | Domain of the source document; carried from `Source.domain` |
| `statement` | `RetrievedEvidence` | The evidence statement used for retrieval matching |

**What happens to these fields:** None of `score`, `rank`, `lexical_score`, `semantic_score` are written back to the `Evidence` schema. They exist only in the `RetrievedEvidence` dataclass during the active retrieval call. Once the retriever returns, these values are available to the caller but are not persisted anywhere.

**Tiebreak provenance gap:** When two evidence items have identical hybrid scores, the retriever resolves the tie by JSONL iteration order (corpus-build order). This means the effective ranking depends on which source was added to the corpus first — a provenance dependency that has no representation in any schema field.

**Metadata factor provenance gap:** The `metadata_factor ∈ [0.65, 1.45]` is computed from source quality, priority, and strategic signals but is not recorded on `RetrievedEvidence`. A caller cannot determine why one item outranked another if the difference is entirely in the metadata factor.

### 1.4 Reranking Phase (`knowledge/reranker.py`)

**Two reranker implementations exist:**

**PassthroughReranker (default):**
- Identity operation; preserves retrieval order and all scores
- `RankedEvidence` wraps `RetrievedEvidence` with `relevance_score = None` and `rationale = None`
- No new provenance information added; no provenance lost

**LLMReranker:**
- Calls Claude Haiku via `tool_use` interface without temperature control
- Assigns a `relevance_score` (LLM-assigned, float) and `rationale` (free-text) per candidate
- These fields are on `RankedEvidence` but are NOT written back to `Evidence` schema
- The rationale represents the LLM's judgment about why an item is relevant — this is provenance information that exists during reranking but is not persisted

### 1.5 Agent Assembly Phase (`functional_agents/evidence_agent.py`)

**What the EvidenceAgent carries forward (KB path):**

When converting `RetrievedEvidence` to the downstream dict format, the agent explicitly maps:

| Output Field | Source | Notes |
|-------------|--------|-------|
| `evidence_id` | `RetrievedEvidence.evidence.evidence_id` | Preserved |
| `claim` | `RetrievedEvidence.statement` | The retrieval statement, not the original stored statement |
| `category` | `_safe_category(evidence.category or evidence_type)` | Coerced via category normalization |
| `topics` | Hardcoded `[]` | Always empty in KB path; `Evidence.topics` is not propagated |
| `relevance_score` | `_score_to_5(c.score)` | Hybrid score compressed to integer 1–5 scale |
| `source_document` | `evidence.supporting_source_ids[0]` | Only the first source_id; subsequent IDs in multi-source evidence are dropped |
| `source_domain` | `RetrievedEvidence.source_domain` | Preserved for profile attribution |

**What is dropped at assembly:**

| Dropped | Was Available On | Impact |
|---------|-----------------|--------|
| `rank` | `RetrievedEvidence` | Cannot reconstruct retrieval ranking downstream |
| `lexical_score` | `RetrievedEvidence` | Cannot distinguish lexical vs semantic contribution to ranking |
| `semantic_score` | `RetrievedEvidence` | Same |
| `metadata_factor` | Computed in retriever | Cannot explain ranking differences |
| `supporting_source_ids[1:]` | `Evidence.supporting_source_ids` | Multi-source evidence loses all sources beyond the first |
| `extraction_run_id` | `Evidence.extraction_run_id` | Cannot trace assembled evidence back to the extraction run |
| `statement_fingerprint` | `Evidence.statement_fingerprint` | Not exposed downstream |

**ResearchMemo:** The agent builds a `ResearchMemo` for ReportAgent compatibility. The memo's `source_notes` contains `EvidenceItem` objects with `evidence_id`, `claim`, `source_document`, and `evidence_snippet` (= claim[:300]). The memo carries no additional provenance beyond what is in the item dict.

### 1.6 Boundary Phase (`functional_agents/evidence_boundary.py`)

**What the Evidence boundary validates:**

| Check | Validated? | Notes |
|-------|-----------|-------|
| `evidence_id` present on every item | YES | Hard requirement; items missing evidence_id are dropped |
| `source_document` present on every item | DIAGNOSTIC only | Recorded in `citations_present` / `citations_missing`; not a hard failure |
| Mapping references point to valid evidence_ids | YES | Dangling references are removed |
| Planner subquestion alignment | DIAGNOSTIC only | Non-fatal; records whether mapped keys are a subset of plan subquestions |
| Chunk-level provenance (page, excerpt, chunk_id) | NOT CHECKED | These fields do not exist |
| Retrieval score presence | NOT CHECKED | Scores are not in the evidence note schema |

**Boundary provenance gap:** The boundary validates citation integrity at the source-document level (`source_document` field) but has no mechanism to validate or require passage-level provenance. This is because the schema does not define these fields — the boundary can only validate what the schema requires.

---

## 2. Provenance Levels Table

This table rates each provenance capability in the current implementation.

| Provenance Capability | Level | Evidence |
|-----------------------|-------|---------|
| Source document identity (`source_id`) | **GREEN** | Content-addressed (SHA-256 of `canonical_text`); stable across reingest of same content |
| Source attribution on evidence item (`source_document`) | **GREEN** | Validated by evidence boundary; present on all KB-path items |
| Extraction run traceability (`extraction_run_id` on Evidence) | **GREEN** | Persisted to schema; `ExtractionRun` records run_id → source_ids → evidence_ids_produced |
| Statement deduplication (`statement_fingerprint`) | **GREEN** | SHA-256[:16] of normalized statement; applied at extraction time |
| Evidence-to-source manifest linkage | **GREEN** | `SourceManifestEntry.evidence_ids` tracks which evidence came from each source |
| Domain attribution (`source_domain` on RetrievedEvidence) | **GREEN** | Carried from Source.domain through retrieval; used for profile attribution |
| Passage-level origin (page number, paragraph, chunk) | **RED** | No structured field; page numbers are text markers in `canonical_text` only |
| Verbatim excerpt (original text fragment) | **RED** | Not stored; `claim` is the LLM-extracted statement, not a verbatim quote |
| Retrieval rank persistence | **RED** | Runtime-only; not written back to Evidence or any persisted schema |
| Retrieval score persistence (lexical, semantic, hybrid) | **RED** | Runtime-only on `RetrievedEvidence`; compressed to 1–5 integer at assembly; full scores lost |
| Multi-source evidence completeness | **RED** | Only `supporting_source_ids[0]` survives to downstream agents |
| Extraction run traceability on assembled evidence | **YELLOW** | `extraction_run_id` exists on `Evidence` schema but is dropped at agent assembly |
| Page number (structured, queryable) | **RED** | Not a field; recoverable only by regex parsing of `canonical_text` |
| Reranker rationale persistence | **YELLOW** | `LLMReranker` produces rationale per candidate; not persisted; PassthroughReranker default avoids the issue |
| Corpus-build order traceability | **YELLOW** | JSONL append order is an implicit provenance dependency; not represented as a schema field |
| Section / document structure | **RED** | Not captured at ingestion; document structure is not in the schema |
| Topics field on evidence | **YELLOW** | `Evidence.topics` field exists in schema; always `[]` in KB path; EvidenceAgent hardcodes `topics: []` |

**Summary:**

| Level | Count | Capabilities |
|-------|-------|-------------|
| GREEN | 6 | Source identity, source attribution, extraction run traceability, deduplication, manifest linkage, domain attribution |
| YELLOW | 4 | Extraction run at assembly, reranker rationale, corpus-build order, topics field population |
| RED | 7 | Passage-level origin, verbatim excerpt, retrieval rank, retrieval scores, multi-source completeness, page number (structured), section structure |

The Knowledge Layer has solid document-level provenance but no passage-level provenance. The RED tier represents structural gaps — capabilities that require new schema fields, not just configuration changes.

---

## 3. Provenance Flow Diagram

The following diagram traces what provenance information enters, survives, and is lost at each Knowledge Layer phase.

```
SOURCE DOCUMENT
  │
  │  Ingested as:
  │    Source.source_id       ← SHA-256 of canonical_text (content-addressed)
  │    Source.uri             ← original file path / URL
  │    Source.canonical_text  ← full text; PDF: "[Page N]\n{text}" markers
  │    Source.page_count      ← integer (structural only)
  │    Source.domain
  │
  ▼
EXTRACTION (builder.py + LLM)
  │
  │  Produced:
  │    Evidence.evidence_id           ← UUID per evidence item
  │    Evidence.statement             ← LLM-extracted claim (not verbatim)
  │    Evidence.supporting_source_ids ← list of source_ids
  │    Evidence.extraction_run_id     ← run UUID
  │    Evidence.statement_fingerprint ← SHA-256[:16] for dedup
  │    Evidence.category / entity / scope
  │
  │  LOST:
  │    ✗ chunk_id           (never captured)
  │    ✗ page_number        (text marker in canonical_text; not a field)
  │    ✗ excerpt            (verbatim text fragment; not stored)
  │    ✗ section_heading    (never captured)
  │    ✗ character_offset   (never captured)
  │
  ▼
STORAGE (store.py)
  │
  │  Persisted:
  │    All Evidence fields above
  │    ExtractionRun (run_id → source_ids → evidence_ids_produced)
  │    SourceManifestEntry (source_id → domain, fingerprint, evidence_ids)
  │    Evidence.npy per evidence_id (embedding)
  │
  │  Implicit:
  │    JSONL append order = corpus-build order (affects tiebreaks; not a schema field)
  │
  ▼
RETRIEVAL (retriever.py)
  │
  │  Computed (runtime only — NOT persisted):
  │    RetrievedEvidence.score         ← hybrid: 0.4×lex + 0.6×sem × metadata_factor
  │    RetrievedEvidence.rank          ← integer rank in result set
  │    RetrievedEvidence.lexical_score ← BM25 / TF-IDF component
  │    RetrievedEvidence.semantic_score← cosine similarity component
  │    RetrievedEvidence.source_domain ← carried from Source.domain
  │    RetrievedEvidence.statement     ← evidence statement (used for matching)
  │
  │  LOST at call return:
  │    ✗ metadata_factor breakdown     (quality × priority × strategic; not on dataclass)
  │    ✗ tiebreak identity            (JSONL order; not recorded)
  │
  ▼
RERANKING (reranker.py)
  │
  │  PassthroughReranker (default):
  │    Identity — all RetrievedEvidence fields preserved
  │    relevance_score = None, rationale = None on RankedEvidence
  │
  │  LLMReranker (non-default):
  │    Adds relevance_score (float) and rationale (str) per candidate
  │    These are on RankedEvidence — NOT written back to Evidence schema
  │
  │  LOST at reranker return (LLMReranker):
  │    ✗ rationale                    (LLM's reasoning; ephemeral)
  │    ✗ relevance_score              (not persisted anywhere)
  │
  ▼
AGENT ASSEMBLY (evidence_agent.py KB path)
  │
  │  Carried forward to downstream dict:
  │    evidence_id        ← from Evidence
  │    claim              ← from RetrievedEvidence.statement
  │    category           ← coerced via _safe_category()
  │    topics             ← hardcoded [] (Evidence.topics NOT propagated)
  │    relevance_score    ← _score_to_5(score) → integer 1–5
  │    source_document    ← supporting_source_ids[0] only
  │    source_domain      ← from RetrievedEvidence
  │
  │  DROPPED at assembly:
  │    ✗ rank              (runtime; not in output dict)
  │    ✗ lexical_score     (runtime; not in output dict)
  │    ✗ semantic_score    (runtime; not in output dict)
  │    ✗ source_ids[1:]    (multi-source evidence truncated to first source)
  │    ✗ extraction_run_id (exists on Evidence schema; not propagated)
  │    ✗ statement_fingerprint (not propagated)
  │
  ▼
BOUNDARY VALIDATION (evidence_boundary.py)
  │
  │  Validated:
  │    evidence_id present per item (hard requirement)
  │    source_document present per item (diagnostic; non-fatal)
  │    mapping references point to valid evidence_ids
  │    planner subquestion alignment (diagnostic; non-fatal)
  │
  │  NOT VALIDATED (schema does not define these):
  │    — page_number, chunk_id, excerpt
  │    — retrieval scores
  │    — multi-source completeness
  │
  ▼
DOWNSTREAM AGENTS
(HypothesisAgent, QAAgent, ReportAgent, AssumptionAgent, ...)

  Available per evidence item:
    evidence_id         ← stable UUID anchor
    claim               ← LLM-extracted statement
    category            ← normalized category
    topics              ← always []
    relevance_score     ← integer 1–5 (compressed from hybrid score)
    source_document     ← first source_id
    source_domain       ← domain classifier
    profile             ← attributed profile

  NOT available downstream:
    ✗ page_number       ← never captured as field
    ✗ excerpt           ← never stored
    ✗ rank              ← dropped at assembly
    ✗ full score        ← compressed to 1–5
    ✗ run_id            ← dropped at assembly
    ✗ all source_ids    ← truncated to first
```

---

## 4. Architectural Gaps

This section describes the four principal provenance gaps, their current impact, and their structural nature.

### Gap 1: No Passage-Level Provenance

**Nature:** Schema-level. The `Evidence` model has no fields for chunk_id, page_number, excerpt, section_heading, or character offset. This is not a configuration gap — these fields do not exist.

**Current state:** Page numbers appear as `[Page N]\n` text markers embedded in `Source.canonical_text` during PDF ingestion. They are parseable by regex but not addressable as structured fields.

**Impact on the Knowledge Layer contracts:**

- **Provenance Preservation contract (YELLOW):** An evidence item can be traced to its source document but not to the specific passage within that document. An analyst receiving evidence item `EVD-abc123` knows it came from source `SRC-xyz` but cannot determine whether it came from page 3 or page 47, or from the executive summary versus the appendix.
- **Citation Integrity contract (YELLOW):** The `source_document` field provides document-level citation. Passage-level citation (the standard for academic and regulatory citation) is structurally impossible given the current schema.
- **Grounding contract (YELLOW):** A downstream agent cannot ground a claim to a specific text span in the source. The only grounding available is the extracted `claim` statement itself, which is LLM-synthesized, not verbatim.

**Downstream consequence:** ReportAgent citations point to source documents, not pages or passages. Human reviewers cannot verify a specific claim without reading the entire source document.

### Gap 2: Retrieval Provenance Not Persisted

**Nature:** Lifecycle gap. Retrieval scores (`score`, `rank`, `lexical_score`, `semantic_score`) are computed during retrieval and are available on `RetrievedEvidence` but are not written back to `Evidence` or any persisted schema.

**What is lost:**
- The hybrid score that determined why item A ranked above item B
- The lexical vs. semantic contribution split (the weights `0.4 × lex + 0.6 × sem` are fixed in code, but the realized scores are ephemeral)
- The `metadata_factor` component (not even captured on `RetrievedEvidence`)
- The retrieval rank

**What survives:** The hybrid score is compressed to a 1–5 integer by `_score_to_5()` in the agent assembly phase. This is a lossy compression: `_score_to_5(0.41) = 2` and `_score_to_5(0.49) = 2` — distinct rankings that become identical relevance labels.

**Impact on contracts:**

- **Ranking Stability contract (YELLOW):** Without persisted scores, it is impossible to audit whether the same query against the same corpus produces the same ranking. Only the compressed 1–5 label survives for comparison.
- **Reproducibility contract (YELLOW):** A reproducibility audit that compares two runs of the same query can only compare evidence_ids and compressed scores. Any ranking difference within the same 1–5 tier is invisible.

**LLMReranker amplification:** When `LLMReranker` is active (non-default), the LLM rationale per candidate — the only explanation of why the reranker changed the order — is discarded after the reranking call. The final order is observable; the reasoning is not.

### Gap 3: Extraction Run Traceability Lost at Assembly

**Nature:** Propagation gap. `Evidence.extraction_run_id` exists as a schema field and is persisted to the store. However, this field is not included in the downstream dict format assembled by `EvidenceAgent._execute_kb()`.

**Impact:** When a downstream agent receives an evidence item, it has:
- `evidence_id` — a UUID that is stable and can be used to look up the full `Evidence` record in the store
- `source_document` — the first source_id

But the agent does NOT have `extraction_run_id` directly. To determine which extraction run produced a specific evidence item, a downstream agent must:
1. Receive the `evidence_id`
2. Query the KnowledgeStore for the `Evidence` record
3. Read `Evidence.extraction_run_id`
4. Query `ExtractionRun` records to find context

This is possible but not automatic. The assembled evidence dict, which is what all downstream agents see, does not carry run traceability.

**Impact on contracts:**

- **Reproducibility contract (YELLOW):** To reproduce a specific research output, one needs to know which extraction runs contributed to the evidence set. This requires a lookup back to the store rather than being readable from the assembled evidence.

### Gap 4: Multi-Source Evidence Truncation

**Nature:** Data loss gap. `Evidence.supporting_source_ids` is a list — a single evidence item may be supported by multiple source documents. However, in `evidence_agent.py`'s KB path assembly:

```python
"source_document": (
    c.evidence.supporting_source_ids[0]
    if c.evidence.supporting_source_ids else "knowledge_store"
),
```

Only `supporting_source_ids[0]` survives. Any evidence item citing two or more sources is reduced to citing only its first source.

**Impact on contracts:**

- **Citation Integrity contract (YELLOW):** A claim supported by sources A and B is cited as supported by source A only. This is a citation integrity failure for multi-source evidence.
- **Provenance Preservation contract (YELLOW):** The provenance of the second, third, etc. sources is permanently lost at assembly time. The store retains the full list, but downstream agents never see it.

**Scope:** This gap affects only evidence items with `len(supporting_source_ids) > 1`. The frequency of multi-source evidence depends on the LLM extraction prompt; it is a structural risk regardless of current incidence.

### Gap 5: Topics Field Silent Impairment

**Nature:** Propagation gap (secondary). `Evidence.topics` exists in the schema as a list of topic keywords. In the KB path, the agent assembly hardcodes `topics: []`. This affects the `_map_evidence_to_areas()` function in `evidence_agent.py`, which uses `topics` as one of its mapping signals:

```python
signal_texts = [category] + list(topics) + [claim]
```

With `topics: []`, the area mapping relies only on `category` and `claim` text. The `_CATEGORY_TO_AREA` lookup compensates for some of this, but composite and cross-cutting investigation areas that rely on topic keywords receive weaker mapping signals.

This gap was noted in the KNOWLEDGE_LAYER_GAP_ANALYSIS.md (PH5.2) as a silent impairment.

---

## 5. Canonical Provenance Model

This section defines the provenance model the Knowledge Layer should implement. It is expressed as a schema and contract specification — the authoritative target state.

### 5.1 Principles

**P1 — Traceability to original text.** Every evidence item must carry sufficient provenance to allow a human reviewer to locate the specific passage in the source document from which the claim was extracted. Source-document-level citation is a minimum; passage-level citation is the target.

**P2 — Retrieval provenance persistence.** The scores and rank that determined which evidence items were selected for a given query are provenance. They must be preserved — either on the evidence item or in a per-run retrieval manifest — so that ranking differences across runs can be audited.

**P3 — Run traceability must be propagated.** Extraction run identity must be available to downstream agents without requiring a store lookup. It is part of the assembled evidence record.

**P4 — Citation completeness.** All source documents supporting an evidence item must be cited. Truncation to a single source is a citation integrity failure.

**P5 — Topics must not be suppressed.** If `Evidence.topics` carries meaningful signal at extraction time, it must be propagated through retrieval and assembly. Hardcoding `topics: []` in the assembly path makes topics a dead field.

### 5.2 Canonical Evidence Schema (Target)

The canonical `Evidence` model should carry the following fields:

**Required fields (currently present):**
- `evidence_id: str` — UUID; primary key
- `statement: str` — extracted claim
- `supporting_source_ids: list[str]` — all supporting sources (must not be truncated)
- `extraction_run_id: str` — extraction run UUID; must be propagated to assembled evidence
- `statement_fingerprint: str` — SHA-256[:16] for dedup
- `category: str` — evidence category
- `entity: str`, `entity_type: str`, `scope: str` — entity classification

**Required fields (currently absent):**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | `str \| None` | Identifier for the text chunk or passage the claim was extracted from; links to passage-level provenance |
| `page_number` | `int \| None` | Page number (for paged documents); structured integer, not text marker |
| `excerpt` | `str \| None` | Verbatim or near-verbatim text fragment from the source that supports the claim; maximum ~500 characters |
| `section_heading` | `str \| None` | Nearest section heading in the source document (for structured documents) |
| `char_offset_start` | `int \| None` | Character offset within `canonical_text` where the supporting passage begins |
| `char_offset_end` | `int \| None` | Character offset where the supporting passage ends |

All new fields are nullable to maintain backward compatibility with existing evidence items that predate passage-level extraction.

### 5.3 Canonical RetrievalProvenance Schema (Target)

Retrieval provenance should be a first-class record, either embedded on the assembled evidence item or written to a per-query retrieval manifest. The canonical fields:

| Field | Type | Description |
|-------|------|-------------|
| `retrieval_query` | `str` | The query string that retrieved this item |
| `retrieval_rank` | `int` | Integer rank in the retrieved set |
| `hybrid_score` | `float` | Final hybrid score (full precision; not compressed) |
| `lexical_score` | `float` | BM25 / TF-IDF component score |
| `semantic_score` | `float` | Cosine similarity score |
| `metadata_factor` | `float` | The multiplier applied from quality, priority, and strategic signals |
| `retrieval_mode` | `str` | `"hybrid"` \| `"lexical"` \| `"semantic"` |
| `reranker` | `str` | `"passthrough"` \| `"llm"` \| `"none"` |
| `rerank_relevance_score` | `float \| None` | LLM reranker relevance score if LLMReranker was active |

### 5.4 Canonical Assembled Evidence Item (Target)

The evidence item dict that downstream agents receive should include:

| Field | Source | Status |
|-------|--------|--------|
| `evidence_id` | `Evidence.evidence_id` | Present |
| `claim` | `RetrievedEvidence.statement` | Present |
| `category` | `Evidence.category` (coerced) | Present |
| `topics` | `Evidence.topics` | MISSING — hardcoded `[]` |
| `source_document` | `Evidence.supporting_source_ids[0]` | Present but truncated |
| `supporting_source_ids` | `Evidence.supporting_source_ids` | MISSING — full list dropped |
| `extraction_run_id` | `Evidence.extraction_run_id` | MISSING — dropped at assembly |
| `page_number` | `Evidence.page_number` | MISSING — field does not exist |
| `excerpt` | `Evidence.excerpt` | MISSING — field does not exist |
| `chunk_id` | `Evidence.chunk_id` | MISSING — field does not exist |
| `retrieval_rank` | `RetrievedEvidence.rank` | MISSING — dropped at assembly |
| `retrieval_score` | `RetrievedEvidence.score` | MISSING — compressed to 1–5 |
| `source_domain` | `RetrievedEvidence.source_domain` | Present |

### 5.5 Canonical Provenance Chain (Target)

The complete provenance chain from source document to downstream agent should be:

```
Source document
  → Source.source_id (content-addressed SHA-256)
    → Evidence.evidence_id (UUID)
      → Evidence.extraction_run_id (links to ExtractionRun record)
      → Evidence.supporting_source_ids (all sources, complete)
      → Evidence.chunk_id (passage identifier)
      → Evidence.page_number (structured integer)
      → Evidence.excerpt (verbatim text fragment)
        → RetrievedEvidence (runtime; carries rank, scores)
          → Assembled evidence item (carries all above; no dropping)
            → Downstream agents (full provenance available)
```

In the canonical model, no provenance information is lost at any phase transition.

---

## 6. Overall Assessment

### 6.1 Provenance Maturity Rating

| Dimension | Rating | Rationale |
|-----------|--------|-----------|
| Document-level provenance | **ADEQUATE** | Source attribution is content-addressed, validated by the evidence boundary, and persisted. |
| Extraction-level provenance | **ADEQUATE** | ExtractionRun records link runs to sources and evidence_ids. The statement fingerprint provides extraction-time dedup. |
| Passage-level provenance | **ABSENT** | No chunk_id, page_number, or excerpt fields exist. Page markers are unstructured text in canonical_text. |
| Retrieval-level provenance | **INADEQUATE** | Scores and rank are runtime-only; compressed to 1–5 at assembly. |
| Assembly-level propagation | **PARTIAL** | evidence_id and source_domain propagate correctly; extraction_run_id, topics, multi-source IDs, and scores are dropped. |
| Downstream citation completeness | **PARTIAL** | Source document cited; page/passage not cited; multi-source evidence truncated. |

### 6.2 Fitness for Purpose

**For current analytical use:** The Knowledge Layer's document-level provenance is fit for purpose. An analyst can identify which source documents contributed to a given analysis, trace evidence_ids to specific source documents, and audit the extraction run that produced an evidence item (via store lookup). This is adequate for most research contexts.

**For regulatory and auditability requirements:** The Knowledge Layer is not yet fit for purpose. Regulatory-grade citation requires passage-level provenance — the ability to point a reviewer to a specific page and paragraph, not just a document. For climate investment decisions (the platform's domain), regulators and institutional clients may require this level of auditability.

**For reproducibility auditing:** The Knowledge Layer is inadequate. Without persisted retrieval scores and ranks, it is impossible to determine whether two independent runs of the same query against the same corpus produced the same evidence ranking. The current gap analysis (KNOWLEDGE_LAYER_GAP_ANALYSIS.md, PH5.2) rated the Reproducibility contract as YELLOW on this basis.

### 6.3 Risk Register

| Risk | Severity | Likelihood | Description |
|------|----------|-----------|-------------|
| Multi-source citation truncation | HIGH | CERTAIN | Any evidence item citing multiple sources is silently truncated to its first source; all downstream citations are incomplete for these items |
| Retrieval score loss | MEDIUM | CERTAIN | The full hybrid score is irreversibly compressed to 1–5 at assembly; ranking audit is impossible without a separate retrieval log |
| Passage-level citation gap | HIGH | CERTAIN | No page or passage citation is possible with the current schema; this is a structural gap |
| LLMReranker rationale loss | MEDIUM | CONDITIONAL | When LLMReranker is active (non-default), the reranking rationale is ephemeral and unauditable |
| Corpus-build order dependency | LOW | LATENT | Tiebreaks depend on JSONL append order; changing corpus-build order or re-indexing can change rankings silently |
| Topics field propagation | LOW | CERTAIN | `topics: []` in KB path weakens area mapping; composite investigation areas affected |

---

## 7. Roadmap

The following provenance gaps are ordered by severity and structural independence. Each item identifies the gap it closes and the contracts it improves.

### PH5.3-R1 — Fix Multi-Source Citation Truncation

**Closes:** Gap 4  
**Contract improvement:** Citation Integrity (YELLOW → GREEN)  
**Nature:** Propagation fix. `evidence_agent.py` assembly must carry `supporting_source_ids` as the complete list. The `source_document` field can be kept as an alias for `supporting_source_ids[0]` for backward compatibility, but the full list must be present.  
**Prerequisite:** None.

### PH5.3-R2 — Propagate Extraction Run ID to Assembled Evidence

**Closes:** Gap 3  
**Contract improvement:** Reproducibility (YELLOW → approaching GREEN)  
**Nature:** Propagation fix. `extraction_run_id` must be included in the assembled evidence item dict. The field already exists on the `Evidence` schema.  
**Prerequisite:** None.

### PH5.3-R3 — Propagate Topics from Evidence to Assembly

**Closes:** Gap 5  
**Contract improvement:** Retrieval Completeness (YELLOW → approaching GREEN)  
**Nature:** Propagation fix. The hardcoded `topics: []` in `_execute_kb()` must read from `Evidence.topics`. This immediately improves investigation area mapping quality.  
**Prerequisite:** None.

### PH5.3-R4 — Persist Retrieval Provenance

**Closes:** Gap 2  
**Contract improvement:** Ranking Stability, Reproducibility (both YELLOW → GREEN)  
**Nature:** Schema addition + propagation. Two approaches:

1. **Per-item approach:** Add retrieval provenance fields (`retrieval_rank`, `hybrid_score`, `lexical_score`, `semantic_score`, `metadata_factor`, `retrieval_mode`) to the assembled evidence item dict. This makes retrieval provenance available to all downstream agents without store lookup.

2. **Per-query manifest approach:** Write a `RetrievalManifest` record (query → evidence_ids → scores → ranks) to the store at retrieval time. The assembled evidence item carries a `retrieval_manifest_id` reference.

The per-item approach is simpler and recommended for the first iteration. The per-query manifest approach is more space-efficient for large corpora.  
**Prerequisite:** None (schema changes; no existing downstream consumers of these fields).

### PH5.3-R5 — Add Passage-Level Provenance Fields to Evidence Schema

**Closes:** Gap 1  
**Contract improvement:** Provenance Preservation (YELLOW → GREEN), Citation Integrity (→ GREEN), Grounding (YELLOW → approaching GREEN)  
**Nature:** Schema addition (extraction-side change required). New nullable fields must be added to `Evidence`: `chunk_id`, `page_number`, `excerpt`, `section_heading`, `char_offset_start`, `char_offset_end`.

The extraction prompt must be updated to populate these fields when possible. For PDFs, `page_number` is parseable from the existing `[Page N]` markers in `canonical_text` at extraction time. `excerpt` should be a verbatim or near-verbatim text fragment (≤500 characters) from the passage that supports the claim.

**Prerequisite:** R1 (canonical assembly already carries `supporting_source_ids`), to ensure passage provenance is bound to the correct source.

### PH5.3-R6 — Structure Page Numbers in Source Schema

**Closes:** Gap 1 (partial, independent path)  
**Contract improvement:** Provenance Preservation  
**Nature:** Schema addition to `Source`. The builder embeds page markers as text; a `page_index: dict[int, int]` (page_number → char_offset in canonical_text) should be extracted at ingestion time and stored as a structured field. This enables structured page lookup without modifying the retrieval or extraction path.  
**Prerequisite:** None (additive; does not break existing consumers).

### PH5.3-R7 — Persist LLMReranker Rationale (Conditional)

**Closes:** Partial gap in reranker provenance  
**Contract improvement:** Ranking Stability (conditional)  
**Nature:** Storage addition. When `LLMReranker` is active, the reranking rationale per item should be written to a `RerankerLog` record (query → evidence_id → relevance_score → rationale). This rationale is currently the only explanation for how the reranker changed the retrieval order.  
**Prerequisite:** This gap is only active when `LLMReranker` is in use. `PassthroughReranker` (the default) does not have this gap. This work is conditional on `LLMReranker` being promoted to production use.

---

## Appendix A — File Reference

| File | Role in Provenance Chain |
|------|--------------------------|
| `knowledge/models.py` | Canonical schema definitions: Evidence, Source, ExtractionRun, SourceManifestEntry |
| `knowledge/builder.py` | Ingestion pipeline; PDF page marker embedding; extraction LLM call; statement fingerprint |
| `knowledge/store.py` | Persistence layer; JSONL evidence storage; index.json; manifest; embeddings |
| `knowledge/retriever.py` | RetrievedEvidence dataclass; hybrid scoring formula; tiebreak by JSONL order |
| `knowledge/reranker.py` | RankedEvidence; PassthroughReranker; LLMReranker with rationale |
| `functional_agents/evidence_agent.py` | KB path assembly; dict conversion; topics hardcoded `[]`; multi-source truncation |
| `functional_agents/evidence_boundary.py` | Normalization/validation; citation integrity check; mapping integrity check |

## Appendix B — Contract Coverage

| Knowledge Layer Contract | Current Rating | Primary Gap(s) | Target Rating (post-roadmap) |
|--------------------------|---------------|----------------|------------------------------|
| Retrieval Completeness | YELLOW | Gap 5 (topics), Gap 2 (scores) | GREEN (after R3, R4) |
| Provenance Preservation | YELLOW | Gap 1 (passage-level) | GREEN (after R5, R6) |
| Citation Integrity | YELLOW | Gap 4 (multi-source), Gap 1 (passage) | GREEN (after R1, R5) |
| Ranking Stability | YELLOW | Gap 2 (score persistence) | GREEN (after R4) |
| Grounding | YELLOW | Gap 1 (excerpt/verbatim) | YELLOW → GREEN (after R5) |
| Reproducibility | YELLOW | Gap 2 (scores), Gap 3 (run_id) | GREEN (after R2, R4) |
