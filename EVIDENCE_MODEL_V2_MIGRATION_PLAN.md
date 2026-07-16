# Evidence Model v2 Migration Plan

**Document Purpose:** Implementation blueprint for migrating the Knowledge Layer from Evidence v1 to the Canonical Evidence Model (v2)  
**Milestone:** PH5.5  
**Status:** Authoritative planning document — architecture already approved (PH5.4)  
**Reference:** CANONICAL_EVIDENCE_MODEL.md (approved target architecture)  
**Prerequisite reads:** KNOWLEDGE_LAYER_PROVENANCE_AUDIT.md, KNOWLEDGE_LAYER_GAP_ANALYSIS.md

> This document is concerned exclusively with migrating the *implementation* toward the approved canonical architecture. No architectural decisions are made here. The canonical model is a given.

---

## Table of Contents

1. [Field-by-Field Comparison: v1 vs v2](#1-field-by-field-comparison)
2. [Implementation Packages](#2-implementation-packages)
3. [Dependency Analysis](#3-dependency-analysis)
4. [Migration Strategy](#4-migration-strategy)
5. [Backward Compatibility](#5-backward-compatibility)
6. [Architectural Risks](#6-architectural-risks)
7. [Implementation Roadmap](#7-implementation-roadmap)

---

## 1. Field-by-Field Comparison

**v1 key files:**
- `knowledge/models.py` — `Evidence`, `KnowledgeMetadata`, `Source`, `ExtractionRun`
- `knowledge/retriever.py` — `RetrievedEvidence`, `RetrievalResult`
- `functional_agents/evidence_agent.py` — assembly dict (7-field lossy conversion)
- `research_agent/schemas.py` — `EvidenceItem` (downstream consumer schema)

### Section A — Identity

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `evidence_id` | Exists | None | Present on `Evidence` model as UUID string. |
| `content_fingerprint` | Partial | Rename + expand to full SHA-256 | v1 has `statement_fingerprint` as a computed field returning `hashlib.sha256(…).hexdigest()[:16]` (16 chars, not 256-bit). v2 requires the full 64-char SHA-256. Used for deduplication via `get_statement_fingerprints()`. |
| `source_id` | Exists | None | Present on `Source` model; propagated as `supporting_source_ids[0]` in retrieval. |
| `corpus_version` | Missing | New field + infrastructure | No concept of corpus versioning exists. `KnowledgeStore` has `SCHEMA_VERSION = "1.0.0"` on the store, not on individual Evidence records. |
| `schema_version` | Partial | Propagate to Evidence records | Store-level `schema_version.json` exists in `_meta/`. Not a field on `Evidence` or surfaced in assembled output. |

### Section B — Provenance

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `source_uri` | Missing | New field (propagate from Source) | `Source.uri` exists in `knowledge/models.py:73` but is never propagated to `Evidence`. Assembly converts to `source_document = supporting_source_ids[0]` (source_id, not URI). |
| `source_title` | Missing | New field (propagate from Source) | `Source.title` exists but never propagated to `Evidence`. Report citations lack title. |
| `source_domain` | Partial | Promote from assembly to Evidence model | Present on `RetrievedEvidence.source_domain` via domain_map. Carried in assembly dict as `"source_domain"` key. Not a field on the `Evidence` Pydantic model. |
| `supporting_source_ids` | Partial | Stop truncation at assembly | Exists on `Evidence` model as `list[str]`. DROPPED at assembly: `evidence_agent.py:615` uses only `c.evidence.supporting_source_ids[0]` as `source_document`. All but the first source are silently discarded. |
| `extraction_run_id` | Partial | Stop dropping at assembly | `Evidence.extraction_run_id` exists. DROPPED at assembly: `evidence_agent.py:607–622` dict does not include it. |
| `page_number` | Missing | New field — requires extraction update | No such field anywhere in `knowledge/models.py`, `evidence_agent.py`, or retrieval path. |
| `section_heading` | Missing | New field — requires extraction update | No such field. |
| `chunk_id` | Missing | New field — requires extraction update | Chunk is an implementation detail of `KnowledgeExtractor`; docstring states "Chunk is an implementation detail of this extractor; it never enters the KB." (extractor.py:10) |
| `char_offset_start` | Missing | New field — requires extraction update | No such field. |
| `char_offset_end` | Missing | New field — requires extraction update | No such field. |
| `excerpt` | Missing | New field — highest priority gap | No verbatim source passage anywhere in the KB path. `EvidenceItem.evidence_snippet` in legacy path is `c.statement[:300]` — a truncation of the synthesized claim, not verbatim source text (evidence_agent.py:754). |

### Section C — Content

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `statement` | Exists | None | `Evidence.statement` exists and flows correctly. |
| `category` | Partial | Enum migration | `Evidence.category` is a free string (not a Literal). Assembly applies `_safe_category()` which maps to domain-specific strings: `"reactor design"`, `"bwrx"`, `"licensing"`, etc. v2 uses semantic categories: `MARKET`, `TECHNICAL`, `REGULATORY`, `FINANCIAL`, `RISK`. Different enum; existing values need migration. |
| `entity` | Exists | Propagation gap | `Evidence.entity` exists (defaults `""`). DROPPED at assembly for KB path: `evidence_agent.py:607–622` dict does not include it. |
| `entity_type` | Exists | Propagation gap | Same as `entity`. Exists on model; dropped at assembly. |
| `scope` | Exists | Propagation gap | Same pattern. `Evidence.scope` exists; not in assembly dict. |
| `topics` | Missing | New field — hardcoded empty | `evidence_agent.py:611`: `"topics": []`. Hardcoded as empty list for all KB path evidence. The comment at evidence_agent.py:143–147 explicitly acknowledges this as a structural gap. |
| `temporal_reference` | Missing | New field — requires extraction update | No such field. |

### Section D — Retrieval (target: RetrievalProvenance record)

These fields are classified as Runtime only in the v2 Evidence schema. They belong in a new `RetrievalProvenance` record. **None of these exist as a persisted record today.** The underlying data exists transiently in `RetrievedEvidence` but is never persisted.

| Field (v2 RetrievalProvenance) | Exists Today | Implementation Gap | Notes |
|-------------------------------|-------------|-------------------|-------|
| `evidence_id` | Exists | Carry-through only | Would reference existing `evidence_id`. |
| `retrieval_query` | Missing | New — requires persistence | The query string is passed to `_execute_kb()` as `primary_query` but never written anywhere. |
| `retrieval_mode` | Partial | Promote from log to record | `mode` variable in `evidence_agent.py:477`. Logged at PROGRESS level but not persisted to any output. |
| `retrieval_model_version` | Missing | New — requires surfacing | `RetrievalResult.semantic_model` captures the model name; never written to a persistent record. |
| `retrieval_rank` | Partial | Promote from transient to record | `RetrievedEvidence.rank` exists (retriever.py:157). DROPPED at assembly; not in items_dicts. |
| `hybrid_score` | Partial | Stop converting to 1–5 integer | `RetrievedEvidence.score` (float) exists. Assembly converts it to a 1–5 int via `_score_to_5()` (evidence_agent.py:299,610), losing precision. |
| `lexical_score` | Partial | Promote from transient to record | `RetrievedEvidence.lexical_score` exists. Dropped at assembly. |
| `semantic_score` | Partial | Promote from transient to record | `RetrievedEvidence.semantic_score` exists. Dropped at assembly. |
| `metadata_factor` | Missing | New — requires surfacing | Computed in `EvidenceRetriever._metadata_factor()` (retriever.py:540) but not stored as a field on `RetrievedEvidence`. |
| `reranker` | Missing | New | `self._use_reranker` bool in EvidenceAgent but not recorded per item. |
| `rerank_score` | Missing | New | Reranker returns scores internally in `LLMReranker` but they are not surfaced per evidence item. |
| `rerank_rationale` | Missing | New | Appears in trace normalization (`context.trace["_llm_normalization"]`) as a list, not as a per-item record. |

### Section E — Scoring

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `source_quality_score` | Partial | Scale change + move to Evidence | `KnowledgeMetadata.source_quality_score` (float 1.0–5.0). v2 uses `float [0.0, 1.0]`. Different scale; different location (v2: on Evidence; v1: on KnowledgeMetadata, separate from Evidence). |
| `source_retrieval_priority` | Partial | Scale/type change + move | `KnowledgeMetadata.retrieval_priority` (int 1–5). v2 uses `HIGH/MEDIUM/LOW` enum. |
| `source_strategic_value` | Partial | Type change + move | `KnowledgeMetadata.strategic_value` (float 0.0–1.0). v2 uses `HIGH/MEDIUM/LOW` enum. |
| `evidence_confidence` | Partial | Move to Evidence model | `KnowledgeMetadata.confidence` (float 0.0–1.0) and `credibility` (HIGH/MEDIUM/LOW) together approximate this. v2 requires `HIGH/MEDIUM/LOW` enum explicitly on Evidence indicating claim explicitness. |
| `is_quantitative` | Missing | New field | `EvidenceItem.quantitative_score` (int 1–5) exists in the legacy `schemas.py:145`. Not extracted or stored in KB path. |

### Section F — Grounding (Derived fields)

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `subquestion_assignments` | Partial | Move to per-item field | `evidence_by_subquestion` mapping exists at run output level (e.g., `research_object["evidence_by_subquestion"]`). Not a field on individual evidence items. |
| `investigation_area_assignments` | Partial | Move to per-item field | `evidence_by_area` mapping exists at run output level. Not per-item. |
| `grounding_strength` | Missing | New derived field | No such concept. |
| `coverage_contribution` | Partial | Move to per-item field | `coverage_by_subquestion` exists at run level. Not per-item. |

### Section G — Relationships

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `contradicts` | Partial | Name change + scope | `Evidence.contradiction_ids` exists. v2 names it `contradicts` (list of evidence_ids that are incompatible). Same concept; rename + confirm semantics. |
| `corroborates` | Missing | New field | No corroboration detection exists. |
| `superseded_by` | Exists | None | `Evidence.superseded_by: str | None` matches v2 exactly. |
| `supersedes` | Partial | Type change: list → string | `Evidence.supersedes: list[str]` in v1. v2 specifies `supersedes: string | null` (single predecessor). Type mismatch. |
| `informed_hypotheses` | Missing | New forward-traceability field | No such field. Not populated by reasoning agents. |
| `informed_recommendations` | Missing | New forward-traceability field | No such field. |

### Section H — Lifecycle

| Field (v2) | Exists Today | Implementation Gap | Notes |
|------------|-------------|-------------------|-------|
| `status` | Partial | Move from KnowledgeMetadata to Evidence + enum update | `KnowledgeMetadata.state: EvidenceState` exists with enum: `ACTIVE/SUPERSEDED/LOW_CONFIDENCE/RETRACTED/ARCHIVED`. v2 `status` on Evidence has `ACTIVE/SUPERSEDED/RETRACTED/PENDING_REVIEW`. Different location and different enum values (LOW_CONFIDENCE/ARCHIVED not in v2; PENDING_REVIEW not in v1). |
| `acceptance_policy` | Missing | New governance field | `KnowledgeMetadata.review_status` (UNREVIEWED/AUTO_REVIEWED/HUMAN_REVIEWED) is adjacent but not equivalent. v2 requires `AUTO/HUMAN_APPROVED/PENDING`. |
| `extracted_at` | Missing | New timestamp | `KnowledgeMetadata.created_at` exists but is on the mutable metadata, not on Evidence itself. |
| `accepted_at` | Missing | New timestamp | No such field. |
| `retracted_at` | Missing | New timestamp | No such field. |
| `retraction_reason` | Missing | New field | No such field. |

---

### Gap Summary

| Status | Evidence Fields (A–H) | RetrievalProvenance Fields | Total |
|--------|----------------------|---------------------------|-------|
| **Exists** | 7 | 1 (evidence_id) | 8 |
| **Partial** | 20 | 5 (mode, rank, hybrid_score, lex_score, sem_score) | 25 |
| **Missing** | 17 | 6 | 23 |
| **Total** | 44 | 12 | 56 |

---

## 2. Implementation Packages

Changes are grouped by architectural concern. Each package is independently implementable once its prerequisite packages are complete.

---

### Package 1 — Schema

**What it is:** All changes to Pydantic model definitions in `knowledge/models.py` and `research_agent/schemas.py`. This package defines the target data structures that all other packages produce and consume.

**v1 fields affected:**
- `statement_fingerprint` (computed, SHA-256[:16]) → replace with `content_fingerprint` (full SHA-256)
- `supersedes: list[str]` → `supersedes: str | None`
- `evidence_type` enum → retained; verify alignment with v2 `category` enum migration
- `KnowledgeMetadata.state` → coordinate with new `Evidence.status`
- `EvidenceState` literal — add `PENDING_REVIEW`, remove `LOW_CONFIDENCE` and `ARCHIVED`

**v2 fields introduced:**
- On `Evidence` model: `corpus_version`, `schema_version`, `source_uri`, `source_title`, `source_domain`, `excerpt`, `page_number`, `section_heading`, `chunk_id`, `char_offset_start`, `char_offset_end`, `temporal_reference`, `is_quantitative`, `evidence_confidence` (enum), `source_quality_score` (float 0–1), `source_retrieval_priority` (enum), `source_strategic_value` (enum), `subquestion_assignments`, `investigation_area_assignments`, `grounding_strength`, `coverage_contribution`, `corroborates`, `informed_hypotheses`, `informed_recommendations`, `status` (moved from KnowledgeMetadata), `acceptance_policy`, `extracted_at`, `accepted_at`, `retracted_at`, `retraction_reason`
- New model: `RetrievalProvenance` dataclass — `evidence_id`, `retrieval_query`, `retrieval_mode`, `retrieval_model_version`, `retrieval_rank`, `hybrid_score`, `lexical_score`, `semantic_score`, `metadata_factor`, `reranker`, `rerank_score`, `rerank_rationale`

**Approach:** Add new fields with `Optional`/`None` defaults first. Do not remove or rename existing fields until downstream consumers are updated. This allows the model to be v2-forward while remaining v1-compatible during the transition.

---

### Package 2 — Extraction

**What it is:** Changes to the LLM extraction prompt and `KnowledgeExtractor` logic to capture passage-level provenance at extraction time.

**v1 fields affected:**
- `_KB_EXTRACTION_QUESTION` (extractor.py:32) — current prompt does not request excerpt, page_number, section_heading, topics, temporal_reference, is_quantitative, evidence_confidence
- `_PROMPT_VERSION` — bump from `"kb-v2.0"` to `"kb-v3.0"` to track prompt lineage

**v2 fields introduced:**
- `excerpt` — verbatim source passage (≤600 chars), extracted by LLM from source chunk
- `page_number` — derived from chunk's position in `page_index`
- `section_heading` — nearest heading above the passage
- `chunk_id` — persisted from the passage window identifier
- `char_offset_start`, `char_offset_end` — from chunk `start_offset`/`end_offset` (already in `Chunk` schema)
- `topics` — keyword list extracted from claim content
- `temporal_reference` — time period stated or inferred in claim
- `is_quantitative` — boolean, from LLM claim classification
- `evidence_confidence` — HIGH/MEDIUM/LOW, from LLM claim classification
- `corpus_version` — stamped at extraction time from store
- `schema_version` — stamped at extraction time

**Approach:** This package introduces a KB rebuild dependency. Evidence records produced before this change lack passage-level fields. The rebuild is a one-time full re-extraction pass.

---

### Package 3 — Knowledge Store

**What it is:** Changes to `KnowledgeStore` (`knowledge/store.py`) to persist new v2 fields, add corpus versioning, and store `RetrievalProvenance` records.

**v1 fields affected:**
- `SCHEMA_VERSION` — bump to `"2.0.0"` to signal v2 schema in `_meta/schema_version.json`
- `get_statement_fingerprints()` — must return `content_fingerprint` (full SHA-256) after Package 1 rename

**New v2 infrastructure:**
- `corpus_version` — written to `_meta/corpus_version.json` at build time; stamped on each Evidence record at extraction
- `page_index` — per-source lookup of `page_number → char_offset`; stored in `sources/{domain}/{source_id}_page_index.json`
- `RetrievalProvenance` store — new collection at `retrieval_provenance/{run_id}.jsonl`; write-once per research run

**Store layout changes:**
```
knowledge_store/
  _meta/
    corpus_version.json          ← NEW
    schema_version.json          ← bump to "2.0.0"
  retrieval_provenance/          ← NEW collection
    {run_id}.jsonl
  sources/{domain}/
    {source_id}_page_index.json  ← NEW
```

**No existing collection layouts change.** Evidence JSONL rows gain new optional fields; JSONL is forward-compatible.

---

### Package 4 — Retrieval

**What it is:** Changes to `EvidenceRetriever` and `RetrievedEvidence` to surface scoring provenance and retrieval configuration as first-class fields — making them available for persistence at assembly time.

**v1 fields affected:**
- `RetrievedEvidence` — add `metadata_factor: float` field
- `EvidenceRetriever.retrieve()` — add `retrieval_model_version` to result

**v2 fields introduced (on `RetrievedEvidence`):**
- `metadata_factor` — the combined quality/priority/strategic multiplier, currently computed in `_metadata_factor()` but discarded after use

**v2 fields introduced (on `RetrievalResult`):**
- `retrieval_model_version` — promote from `semantic_model: str | None` to a required field with `"lexical"` sentinel when no embedding model is used

**No scoring algorithm changes.** Weights, formula, and thresholds are unchanged. This package only surfaces data that already exists.

---

### Package 5 — Evidence Assembly

**What it is:** Changes to `EvidenceAgent._execute_kb()` to stop the lossy 7-field dict conversion and pass the complete v2 Evidence object to downstream consumers.

**v1 code affected:**
- `evidence_agent.py:607–622` — the `items_dicts = [...]` list comprehension that converts `RetrievedEvidence` to a 7-key dict. This is where the majority of provenance is lost.
- `_build_synthetic_memo()` — constructs `EvidenceItem` with only `evidence_id`, `claim`, `source_document`, `evidence_snippet`, `category`, `relevance`, `confidence`, `relevance_score`

**What changes:**
- Replace 7-field dict conversion with a v2-compliant dict carrying all fields from the Evidence model
- Preserve `supporting_source_ids` as a list (not truncated to `[0]`)
- Carry `extraction_run_id` in assembly output
- Carry `entity`, `entity_type`, `scope` from Evidence model
- Load Source object for selected items (`load_sources=True`) to populate `source_uri` and `source_title`
- Write a `RetrievalProvenance` record (via Package 3 store) for every assembled item

**Downstream consumer impact:**
- Items gain new fields. Existing consumers that access `claim`, `category`, `evidence_id`, `source_document`, `relevance_score`, `source_domain` continue working unchanged.
- `source_document` (source_id string) must be retained alongside `source_uri` for backward compatibility until all consumers migrate.

---

### Package 6 — Grounding

**What it is:** Computing `subquestion_assignments`, `investigation_area_assignments`, `grounding_strength`, and `coverage_contribution` as per-item fields, not just as aggregate run-level mappings.

**v1 code affected:**
- `_map_evidence_to_subquestions()` and `_map_evidence_to_areas()` produce `dict[str, list[evidence_ids]]` (aggregate). This is correct and should be retained for the run-level summary. **In addition**, each evidence item must carry its own assignment list.
- `_coverage_level()` — already correct for aggregate computation; `coverage_contribution` per-item uses rank-within-subquestion-pool

**v2 fields computed:**
- `subquestion_assignments: list[str]` — all subquestions this item maps to (primary + secondary via `_SQ_SECONDARY_THRESHOLD`)
- `investigation_area_assignments: list[str]` — all investigation areas this item maps to
- `grounding_strength` — derived from `evidence_confidence` + `is_quantitative`:
  - `STRONG`: `evidence_confidence=HIGH` and `is_quantitative=True`
  - `MODERATE`: `evidence_confidence=HIGH` or (`MEDIUM` + `is_quantitative=True`)
  - `WEAK`: otherwise
- `coverage_contribution` — derived from `retrieval_rank` within the item's primary subquestion pool:
  - Rank 1–2 in pool → `STRONG`
  - Rank 3–5 → `MODERATE`
  - Rank 6+ → `WEAK`

**The aggregate `evidence_by_subquestion`, `evidence_by_area`, and `coverage_by_subquestion` are preserved** in the assembled output for backward compatibility.

---

### Package 7 — Serialization

**What it is:** Changes to `EvidenceOutput`, `EvidenceBoundaryValidator`, `ResearchMemo` builder, and Research Object update logic to carry and emit v2 fields.

**v1 code affected:**
- `functional_agents/evidence_boundary.py:38–53` — `EvidenceOutput` model: `evidence_items: list[dict]` is untyped. After this package, the dict schema is enforced.
- `evidence_agent.py:699–773` — `_build_synthetic_memo()` constructs a `ResearchMemo` with a minimal `EvidenceItem`. Must be updated to carry `excerpt`, `page_number`, `source_uri` so downstream ReportAgent can build passage-level citations.
- Research Object update at `evidence_agent.py:713–722` — must add per-item provenance fields to the persisted research object.

**v2 changes:**
- `EvidenceItem` in `research_agent/schemas.py` gains: `source_uri`, `source_title`, `page_number`, `excerpt`, `subquestion_assignments`, `investigation_area_assignments`, `grounding_strength`
- `ResearchMemo.source_notes` carries the updated `EvidenceItem`
- Research Object gains top-level `retrieval_provenance_run_id` pointing to the persisted `RetrievalProvenance` record

---

### Package 8 — Validation

**What it is:** Changes to `evidence_boundary.py` validation logic to enforce v2 Required field presence and produce informative diagnostics for provenance-impaired items.

**v1 code affected:**
- `validate_evidence_output()` (evidence_boundary.py:162) — currently only checks `evidence_id` presence and mapping integrity. Does not check `excerpt`, `corpus_version`, `schema_version`, `topics`, or `extraction_run_id`.

**v2 field checks added:**
- `excerpt` must be non-empty string
- `topics` must be non-empty list
- `corpus_version` must be present
- `schema_version` must be present
- `extraction_run_id` must be present
- `page_number` absence becomes a WARNING (not ERROR) — valid for HTML/API sources

**Gate policy:** Items produced by the new extraction path (v2 prompt) must pass all Required field checks. Items sourced from legacy KB records (pre-v2 extraction) are permitted to pass with a DEGRADED status flag during the transition period.

---

## 3. Dependency Analysis

### Package 1 — Schema

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | None — independent foundation |
| **Downstream impact** | All other packages depend on Schema. No other package can implement v2 fields until the models are defined. |
| **Affected contracts** | All 7 contracts (new fields enable or partially resolve every gap identified in PH5.2) |

### Package 2 — Extraction

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Package 1 (Schema) — `Evidence` model must carry new fields before extractor can populate them |
| **Downstream impact** | Package 3 (Knowledge Store) — must store new fields; Package 5 (Assembly) — new fields flow through only after new evidence is extracted; triggers KB rebuild |
| **Affected contracts** | Provenance Preservation (excerpt, page_number, chunk_id), Citation Integrity (excerpt), Reproducibility (corpus_version, schema_version), Retrieval Completeness (topics) |

### Package 3 — Knowledge Store

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Package 1 (Schema) — must know about new Evidence fields and RetrievalProvenance model |
| **Downstream impact** | Package 4 (Retrieval) reads from store; Package 5 (Assembly) writes RetrievalProvenance; Package 2 (Extraction) stamps corpus_version from store |
| **Affected contracts** | Reproducibility (corpus_version), Provenance Preservation (RetrievalProvenance persistence), Auditability (forward-traceability storage) |

### Package 4 — Retrieval

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Package 3 (Knowledge Store) — reads new store structure; Package 1 (Schema) — RetrievedEvidence gains metadata_factor |
| **Downstream impact** | Package 5 (Assembly) — can now pass metadata_factor and retrieval_model_version into RetrievalProvenance record |
| **Affected contracts** | Ranking Stability (metadata_factor now auditable), Reproducibility (retrieval_model_version), Recommendation Traceability (retrieval context preserved) |

### Package 5 — Evidence Assembly

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Packages 1, 3, 4 — depends on Schema (v2 Evidence dict), Store (RetrievalProvenance write), Retrieval (metadata_factor on RetrievedEvidence) |
| **Downstream impact** | Packages 6, 7, 8 — all downstream packages consume the assembled output. This is the highest-impact package: it is the point at which v1 provenance loss occurs. |
| **Affected contracts** | All 7 contracts. This package alone closes 5 of the 7 structural gaps identified in PH5.3 (drops: supporting_source_ids truncation, extraction_run_id drop, score drop, rank drop). |

### Package 6 — Grounding

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Package 5 (Assembly) — per-item fields computed from assembled evidence; Package 1 (Schema) — grounding fields must be defined |
| **Downstream impact** | Package 7 (Serialization) — grounding fields must be serialized; Package 8 (Validation) — grounding fields are validated |
| **Affected contracts** | Grounding (primary), Citation Integrity (grounding_strength), Recommendation Traceability (subquestion_assignments per item) |

### Package 7 — Serialization

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Packages 5 (Assembly) and 6 (Grounding) — must assemble before serializing |
| **Downstream impact** | Package 8 (Validation) — validates the serialized output; ReportAgent — consumes EvidenceItem with new fields |
| **Affected contracts** | Citation Integrity (excerpt/page_number in report citations), Provenance Preservation (evidence items carry full provenance to report), Recommendation Traceability (evidence_ids in report recommendations) |

### Package 8 — Validation

| Dimension | Detail |
|-----------|--------|
| **Prerequisites** | Packages 1 (Schema) and 7 (Serialization) — validates the complete assembled, serialized output |
| **Downstream impact** | Final gate; nothing downstream. Incorrect configuration here silently degrades all downstream contract guarantees. |
| **Affected contracts** | All contracts — validation is the enforcement mechanism for all of them |

---

## 4. Migration Strategy

**Governing principle:** Additive changes before breaking changes. A pipeline that runs correctly at every step is preferable to a theoretically optimal pipeline that has a long broken-window period.

**Recommended sequence:**

```
Step 1:  Package 1 — Schema
         (all new fields added as Optional; no existing field removed or renamed)

Step 2:  Package 3 — Knowledge Store infrastructure
         (corpus_version, page_index structure, RetrievalProvenance store)
         (no existing evidence.jsonl layout changes)

Step 3:  Package 4 — Retrieval surfacing
         (metadata_factor on RetrievedEvidence; retrieval_model_version on result)
         (zero change to scoring algorithm or existing retrieval consumers)

Step 4:  Package 2 — Extraction
         (updated LLM prompt; new fields populated at extraction time)
         (triggers KB rebuild)

Step 5:  Package 5 — Evidence Assembly
         (stop lossy dict conversion; pass v2 fields through)
         (source loading for URI/title; RetrievalProvenance write)

Step 6:  Package 6 — Grounding
         (per-item grounding fields computed at assembly)

Step 7:  Package 7 — Serialization
         (EvidenceItem, EvidenceOutput, ResearchMemo updated)

Step 8:  Package 8 — Validation
         (enforcement of Required v2 fields)
         (DEGRADED mode for legacy KB records during transition)

KB Rebuild:
         Full re-extraction pass triggered after Step 4.
         Required before Steps 5–8 can be validated end-to-end.
```

**Justification for this order:**

1. Schema first because all other packages depend on it being defined. Adding Optional fields does not break existing models.
2. Store and Retrieval before Extraction because new evidence records need the store infrastructure to exist before they can be written.
3. Extraction (Step 4) is the KB rebuild trigger. It is placed as late as possible before assembly, so the KB rebuild only happens once — not after each package.
4. Assembly (Step 5) is the critical path change. It is placed after the KB rebuild so that the new evidence in the store actually carries the fields the assembly layer is now trying to pass through.
5. Grounding (Step 6) before Serialization (Step 7) because grounding fields are computed at assembly and serialized into output.
6. Validation (Step 8) last because it enforces the complete v2 contract. Running validation before assembly is ready produces spurious failures.

**Breaking change boundary:** Steps 1–3 are additive and backward-compatible. The breaking change occurs at Step 4 (Extraction), which changes the KB data model and triggers the rebuild. After Step 4, the system has a mixed-vintage KB (old records lack passage-level fields; new records carry them). The DEGRADED flag in Package 8 accommodates this transition period.

---

## 5. Backward Compatibility

### Safely additive — add without breaking anything

These fields can be added to the `Evidence` Pydantic model with `Optional`/`None` defaults and no downstream effect. Existing KB records will deserialize successfully with `None` for these fields.

```
corpus_version
schema_version (on Evidence record)
source_uri
source_title
source_domain (promote from assembly dict to model)
excerpt
page_number
section_heading
chunk_id
char_offset_start
char_offset_end
temporal_reference
is_quantitative
corroborates
informed_hypotheses
informed_recommendations
acceptance_policy
extracted_at
accepted_at
retracted_at
retraction_reason
```

Additionally, the entire `RetrievalProvenance` record is new infrastructure; no existing code reads from it.

### Requires schema migration — changes existing field structure

These changes affect existing fields and must be coordinated with all consumers.

| Change | Affected v1 Field | Migration Action |
|--------|------------------|-----------------|
| Rename + expand content_fingerprint | `statement_fingerprint` (computed, SHA-256[:16]) | Rename field; expand to full SHA-256. Update `get_statement_fingerprints()` in `KnowledgeStore`. Existing deduplication entries reference the truncated fingerprint — all KB records must be re-indexed against the new fingerprint format to avoid duplicate insertions. |
| Enum migration for category | `Evidence.category` (free string with _safe_category) | Map existing domain-specific values to v2 semantic categories. Requires a migration pass over all existing evidence records. |
| status moved to Evidence | `KnowledgeMetadata.state` | Add `Evidence.status` as a new field; populate from `KnowledgeMetadata.state` on read. Retain `KnowledgeMetadata.state` for backward compat until migration is validated. |
| supersedes: list → string | `Evidence.supersedes: list[str]` | Type change. Existing records with `supersedes: [...]` must migrate to `supersedes: str | None`. Multi-predecessor records (rare) must select a canonical predecessor. |
| source_quality_score scale | `KnowledgeMetadata.source_quality_score` (1.0–5.0) | Scale conversion: `new = (old - 1.0) / 4.0` to produce [0.0, 1.0]. Also moves from KnowledgeMetadata to Evidence. |
| source_retrieval_priority type | `KnowledgeMetadata.retrieval_priority` (int 1–5) | Map 4–5 → HIGH, 2–3 → MEDIUM, 1 → LOW. |
| source_strategic_value type | `KnowledgeMetadata.strategic_value` (float 0.0–1.0) | Map ≥0.7 → HIGH, ≥0.4 → MEDIUM, <0.4 → LOW. |

### Requires Knowledge Store rebuild

These changes cannot be satisfied by model-layer migration alone — the missing data was never stored and must be re-extracted from source documents.

```
excerpt            — verbatim passage; not in any existing KB record
page_number        — not in any existing KB record
section_heading    — not in any existing KB record
chunk_id           — not persisted to KB (extractor.py docstring: "Chunk never enters KB")
char_offset_start  — not in any existing KB record
char_offset_end    — not in any existing KB record
topics             — hardcoded [] for all existing KB evidence
```

Additionally: `content_fingerprint` change (from SHA-256[:16] to full SHA-256) requires re-indexing all existing evidence records in the deduplication index, which is effectively a rebuild of the fingerprint-based deduplication state.

**Rebuild scope:** All domains. The rebuild must use the updated extraction prompt (Package 2). This is a one-time full re-extraction of all source documents.

### Requires report changes

| Field | Report impact |
|-------|---------------|
| `excerpt` | Report citations should quote the supporting passage. `ReportAgent` builds citations using `source_document` (source_id string) today. After v2, citations must include `excerpt` and `page_number`. `_build_synthetic_memo()` must pass these fields through `EvidenceItem.evidence_snippet`. |
| `page_number` | Report citation format changes from `(source_id)` to `(source_title, p. N)`. |
| `source_uri` | Report citations gain a retrievable link. |
| `grounding_strength` | Report may weight evidence citations by grounding_strength in narrative construction. |

### Requires benchmark updates

| Test area | Impact |
|-----------|--------|
| `statement_fingerprint` assertion | Any test asserting on `statement_fingerprint` value breaks when field is renamed to `content_fingerprint` and expanded to full SHA-256. |
| Evidence item field count assertions | Any test asserting `len(item.keys()) == 7` or similar breaks when assembly output expands. |
| ResearchMemo `source_notes` field checks | Tests asserting on `EvidenceItem` field structure break when new fields are added. |
| PH4/PH5.1 determinism tests | **Not affected.** `test_ph51_determinism.py` tests `ResearchGapAgent` and `IterationPlanAgent` fingerprints. These agents consume `coverage_by_subquestion` (run-level, not per-item). Schema changes to Evidence items do not flow into their outputs. |
| `evaluation_report.json` regression baseline | Evidence count and structure assertions in evaluation baselines may need updating after KB rebuild produces different evidence items from updated extraction prompt. |

---

## 6. Architectural Risks

### Risk 1 — KB Compatibility Risk

**What it is:** After the `content_fingerprint` rename and expansion (SHA-256[:16] → full SHA-256), the deduplication index in `KnowledgeStore.get_statement_fingerprints()` returns the old truncated fingerprints for existing evidence records. A KB rebuild using the new extraction prompt would compute full-SHA-256 fingerprints for extracted claims. Since the old and new fingerprints are structurally different (16-char hex vs 64-char hex), the deduplication check will fail to detect duplicates — the same claim will be inserted twice, once under the old fingerprint and once under the new.

**Trigger:** Any rebuild of KB domains that have existing evidence records, if `content_fingerprint` is changed before the old records are migrated.

**Blast radius:** Duplicate evidence records across all rebuilt domains. Retrieval precision degrades. Contradiction detection may flag a claim against itself.

---

### Risk 2 — Extraction Prompt Hallucination Risk

**What it is:** The updated extraction prompt requests `page_number`, `section_heading`, and `excerpt` from the LLM. For source documents with ambiguous or absent structure (HTML pages without headings, API responses, flat TXT files), the LLM may hallucinate page numbers or section headings that do not correspond to any actual document structure.

**Trigger:** Running the v2 extraction prompt against structurally poor source documents (HTML, TXT, synthetic sources).

**Blast radius:** Evidence records with fabricated provenance fields are more dangerous than records with null provenance fields. A `page_number: 47` that doesn't exist in the source document misleads an auditor who reads the claim and attempts to verify it. The severity of this risk is higher than the "missing field" risk it replaces.

---

### Risk 3 — Assembly Contract Risk

**What it is:** `EvidenceAgent._execute_kb()` currently produces a 7-field dict that all downstream consumers (`HypothesisAgent`, `QAAgent`, `ReportAgent`, `DecisionAnalysisAgent`, etc.) consume via `item.get("claim")`, `item.get("category")`, `item.get("source_document")`, etc. After Package 5 removes the lossy conversion, the assembly output expands. If any downstream consumer accesses fields by position (unlikely in Python dicts) or relies on the absence of fields, unexpected behavior may result.

**Trigger:** Package 5 (Assembly non-truncation) changes the dict schema.

**Blast radius:** Low probability; Python dict access by key is not position-sensitive. But `claim` vs `statement` is a real naming mismatch: v2 uses `statement`; all downstream consumers use `claim`. Until `claim` is retained as an alias, downstream agents fail silently (get `None` for claims).

---

### Risk 4 — Fingerprint Invalidation Risk

**What it is:** The PH4 and PH5.1 determinism tests register canonical SHA-256 fingerprints of agent outputs. These fingerprints cover `ResearchStrategyAgent`, `PlannerAgent`, `ResearchGapAgent`, and `IterationPlanAgent`. These agents operate before `EvidenceAgent` in the pipeline. However, if any Evidence model field names are added to the output of `ResearchGapAgent` (e.g., if `coverage_by_subquestion` format changes), those fingerprints would invalidate.

**Trigger:** Any change to the schema of `coverage_by_subquestion`, `evidence_summary`, or `evidence_by_subquestion` at the aggregate level (not per-item). The run-level aggregates are what ResearchGapAgent reads.

**Blast radius:** Invalidated fingerprints require re-running the 23 `test_ph51_determinism.py` tests and registering new canonical values. Not a functional failure but a validation regression.

---

### Risk 5 — Schema Migration Risk (Mixed-Vintage KB)

**What it is:** During the transition period between Package 2 (Extraction, which produces v2 records) and the completion of the full KB rebuild, the knowledge store will contain a mix of v1 and v2 evidence records. V1 records lack `excerpt`, `page_number`, `topics`, etc. If the v2 validation enforcement (Package 8) is activated before the rebuild completes, all v1 records fail validation and the pipeline rejects them.

**Trigger:** Package 8 (Validation hardening) activated before KB rebuild completes.

**Blast radius:** 0 evidence items returned from Knowledge Layer for any domain with legacy records. Pipeline falls back to legacy document retrieval (if available) or produces empty evidence sets.

---

### Risk 6 — Retrieval Ranking Stability Risk

**What it is:** The v2 Extraction package changes the LLM prompt (`_KB_EXTRACTION_QUESTION`). A changed prompt changes what claims are extracted, their phrasing, and their `statement` text. Embedding vectors were computed against the current statements. After re-extraction, statement text may differ, making existing embeddings stale. Semantic retrieval would compute cosine similarity between new query embeddings and old statement embeddings — producing incorrect results.

**Trigger:** KB rebuild using updated extraction prompt without regenerating embeddings.

**Blast radius:** Semantic and hybrid retrieval scores degrade for new evidence items until embeddings are regenerated. Ranking order becomes inconsistent: some items score well (if statement text is similar) and others score poorly (if statement text changed significantly).

---

### Risk 7 — Report Citation Backward Compatibility Risk

**What it is:** Reports generated from existing KB evidence (v1 records) will lack `page_number` and `excerpt`. If the `ReportAgent` is updated to reference these fields (Package 7), reports generated against a v1 KB will have blank or null citation fields — visible to users as incomplete citations.

**Trigger:** Package 7 (Serialization) activated before KB rebuild produces v2 evidence.

**Blast radius:** All reports generated against a v1 KB will have degraded citations. Depending on ReportAgent's handling of null fields, this could produce visible output errors or silently omit citation details.

---

## 7. Implementation Roadmap

Each phase has one architectural objective, leaves the system in a valid working state, and is independently testable.

---

### PH5.5a — Schema Foundation

**Architectural objective:** Add all v2 Optional and new fields to the `Evidence` Pydantic model and introduce the `RetrievalProvenance` dataclass, without changing any existing field or breaking any existing validation.

**Scope:**
- `knowledge/models.py`: add all new fields with `Optional`/`None` defaults
- `knowledge/models.py`: add `RetrievalProvenance` dataclass
- `knowledge/store.py`: bump `SCHEMA_VERSION` to `"2.0.0"`; existing evidence JSONL continues to deserialize (new Optional fields default to None)
- No changes to extractors, retrieval, assembly, or agents

**Testable at completion:**
- Existing KB deserializes without error
- New Evidence objects with v2 fields can be constructed and serialized
- Round-trip test: write Evidence with excerpt, page_number, chunk_id; read back and confirm values
- `RetrievalProvenance` can be constructed and serialized

---

### PH5.5b — Knowledge Store Provenance Infrastructure

**Architectural objective:** Add corpus versioning, page index, and RetrievalProvenance persistence to KnowledgeStore without altering any existing storage layout.

**Scope:**
- `knowledge/store.py`: write/read `_meta/corpus_version.json`; expose `current_corpus_version()` method
- `knowledge/store.py`: write/read `sources/{domain}/{source_id}_page_index.json`
- `knowledge/store.py`: write/read `retrieval_provenance/{run_id}.jsonl`

**Testable at completion:**
- `KnowledgeStore.current_corpus_version()` returns current version string
- Write a `RetrievalProvenance` record; read it back by run_id
- Existing source and evidence reads are unaffected

---

### PH5.5c — Retrieval Provenance Surfacing

**Architectural objective:** Surface `metadata_factor` and `retrieval_model_version` as first-class fields on retrieval result objects so they are available for assembly-time persistence.

**Scope:**
- `knowledge/retriever.py`: add `metadata_factor: float` to `RetrievedEvidence` dataclass; populate in `retrieve()` loop
- `knowledge/retriever.py`: promote `semantic_model` to `retrieval_model_version: str`; use sentinel `"lexical"` when no embedding model
- No changes to scoring formula, weights, or thresholds

**Testable at completion:**
- `RetrievedEvidence.metadata_factor` is populated on every result item
- `RetrievalResult.retrieval_model_version` is a non-null string in all modes
- Existing retrieval tests pass without modification

---

### PH5.5d — Extraction Prompt and Passage-Level Fields

**Architectural objective:** Update the KB extraction LLM prompt and KnowledgeExtractor to capture verbatim passage, page number, section heading, and content characterization fields at extraction time.

**Scope:**
- `knowledge/extractor.py`: update `_KB_EXTRACTION_QUESTION` to request excerpt, page_number, section_heading, topics (list), temporal_reference, is_quantitative, evidence_confidence
- `knowledge/extractor.py`: populate new v2 fields on produced `Evidence` objects
- `knowledge/extractor.py`: populate `char_offset_start`/`char_offset_end` from `Chunk.start_offset`/`end_offset`; `chunk_id` from `Chunk.chunk_id`
- `knowledge/extractor.py`: stamp `corpus_version` (from store) and `schema_version` on each Evidence
- `knowledge/extractor.py`: bump `_PROMPT_VERSION` to `"kb-v3.0"`
- **Trigger full KB rebuild** after this phase

**Testable at completion:**
- New extraction produces Evidence records with non-null `excerpt` and `page_number`
- New extraction produces `topics` as a non-empty list
- New extraction produces `is_quantitative` as a boolean
- New extraction produces `corpus_version` and `schema_version`
- `chunk_id` matches the Chunk used for extraction

---

### PH5.5e — Assembly Non-Truncation

**Architectural objective:** Remove the lossy 7-field dict conversion in `EvidenceAgent._execute_kb()` so that all Evidence model fields survive through to the assembled output, and write a `RetrievalProvenance` record per research run.

**Scope:**
- `functional_agents/evidence_agent.py`: replace `items_dicts` list comprehension with v2-compliant field pass-through
- Preserve `claim` as an alias for `statement` for backward compatibility with all downstream consumers
- Preserve `source_document` as an alias for `supporting_source_ids[0]` for backward compatibility
- Add `supporting_source_ids` (full list), `extraction_run_id`, `entity`, `entity_type`, `scope` to assembly output
- Load Source objects for top-N evidence items to populate `source_uri`, `source_title`
- Write `RetrievalProvenance` to store for each assembled item

**Testable at completion:**
- Assembly output includes `extraction_run_id` for each item
- Assembly output includes full `supporting_source_ids` list
- Assembly output includes `source_uri` and `source_title` for items with loadable sources
- `RetrievalProvenance` records persist to store after an evidence retrieval run
- All downstream agents continue working (they access `claim`, `category`, etc., which remain present)

---

### PH5.5f — Grounding Field Computation

**Architectural objective:** Compute `subquestion_assignments`, `investigation_area_assignments`, `grounding_strength`, and `coverage_contribution` as first-class per-item fields at assembly time.

**Scope:**
- `functional_agents/evidence_agent.py`: after `_map_evidence_to_subquestions()` and `_map_evidence_to_areas()`, back-populate each item dict with its own assignment lists
- Compute `grounding_strength` from `evidence_confidence` and `is_quantitative` per item
- Compute `coverage_contribution` from retrieval rank within primary subquestion pool
- Retain aggregate `evidence_by_subquestion`, `evidence_by_area`, `coverage_by_subquestion` in output for backward compatibility

**Testable at completion:**
- Each evidence item in assembly output carries `subquestion_assignments` (non-empty where mapped)
- Each evidence item carries `grounding_strength` as HIGH/MEDIUM/LOW
- Each evidence item carries `coverage_contribution`
- Aggregate mappings still present in `EvidenceOutput`

---

### PH5.5g — Validation Hardening

**Architectural objective:** Update `EvidenceBoundaryValidator` to enforce v2 Required field presence for new evidence and emit structured DEGRADED diagnostics for legacy KB records during the transition period.

**Scope:**
- `functional_agents/evidence_boundary.py`: add Required field checks to `validate_evidence_output()`
- Add excerpt non-empty check
- Add topics non-empty check
- Add corpus_version and schema_version presence checks
- Add extraction_run_id presence check
- Implement DEGRADED mode: items from legacy KB records (schema_version absent or `"1.0.0"`) pass with a `"provenance_grade": "DEGRADED"` diagnostic flag rather than hard failure
- Legacy items continue to reach downstream agents; auditors are informed of missing provenance

**Testable at completion:**
- Validation rejects items missing `excerpt` (v2 records)
- Validation rejects items missing `topics` (v2 records)
- Legacy KB records (no excerpt) pass with DEGRADED flag
- Boundary diagnostics include `provenance_grade` field

---

*End of EVIDENCE_MODEL_V2_MIGRATION_PLAN.md*
