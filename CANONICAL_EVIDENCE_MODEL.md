# Canonical Evidence Model

**Document Purpose:** Architectural specification for the canonical Evidence artifact in the Strategic Research Harness  
**Milestone:** PH5.4  
**Status:** Authoritative — this is the target architecture, independent of current implementation  
**Design Principle:** Evidence is designed from the requirements of downstream reasoning, not from retrieval convenience

---

## Table of Contents

1. [Responsibilities of a Canonical Evidence Object](#1-responsibilities)
2. [Canonical Schema](#2-canonical-schema)
3. [Field Rationale and Contract Dependencies](#3-field-rationale)
4. [Evidence Lifecycle](#4-lifecycle)
5. [Field Classification](#5-field-classification)
6. [Canonical Evidence Object](#6-canonical-evidence-object)
7. [Sufficiency Assessment](#7-sufficiency-assessment)

---

## 1. Responsibilities

A canonical Evidence object is the atomic unit of grounded knowledge in the Strategic Research Harness. Every reasoning agent that produces a finding, hypothesis, recommendation, or conclusion must be able to trace that output to one or more canonical Evidence objects. If a claim cannot be traced, it is not grounded.

The canonical Evidence object must carry enough information to satisfy four downstream properties simultaneously.

---

### 1.1 Grounded

A downstream reasoning agent can refer to an evidence item and have every consumer — human or machine — understand exactly what claim is being made, from what source, from what passage within that source, and with what confidence.

This requires:
- The claim itself (the extracted statement)
- The verbatim passage it was extracted from (for verification)
- The exact location within the source (page, section, character offset or chunk)
- The quality of the original source
- The score that determined this item's relevance to the query

Without verbatim passage and passage location, grounding is nominal rather than real. An analyst can cite the document but cannot confirm the claim without reading the entire document.

---

### 1.2 Explainable

A downstream reasoning agent can explain why a specific evidence item was selected for a specific question. This requires the evidence item to carry:
- The query or subquestion it was retrieved for
- The score that placed it in the retrieved set
- The decomposition of that score (lexical contribution, semantic contribution, quality adjustment)
- The rank it occupied in the result set

Without retrieval provenance, an explanation like "this recommendation is supported by evidence E042" is unverifiable — an auditor cannot determine whether E042 was the top-ranked item for the relevant question or a marginal item that barely made the threshold.

---

### 1.3 Reproducible

Given the same source corpus and the same query, the same evidence item should be produced with the same score and rank. Reproducibility requires:
- A content-addressed identity for the claim (so the same claim from the same source always has the same identity regardless of re-extraction order)
- A content-addressed identity for the source
- The retrieval configuration that produced this item (mode, model version, scoring weights)
- The corpus version against which retrieval was performed

Without corpus version and retrieval configuration, two runs that produce the same evidence_id cannot be confirmed to have used the same retrieval path.

---

### 1.4 Auditable

An auditor — a human reviewer, a regulator, or an automated compliance check — can independently verify every claim by tracing backward from a report finding to a recommendation, to a hypothesis, to the evidence that grounded it, to the specific passage in the source document that supports it.

This requires all provenance fields to survive every phase transition. No provenance may be dropped as the evidence item moves from the retrieval layer through the reasoning agents to the final report. Auditability is a property of the complete chain, not of any single layer.

---

## 2. Canonical Schema

Fields are grouped into eight logical sections. Within each section, fields are listed in dependency order.

---

### Section A — Identity

Fields that uniquely identify the evidence item and its source, independent of any retrieval or reasoning context.

| Field | Type | Description |
|-------|------|-------------|
| `evidence_id` | `string (UUID)` | Primary key. Assigned once at extraction time; never reassigned. Stable across re-extractions if content is identical. |
| `content_fingerprint` | `string (SHA-256)` | SHA-256 of the normalized claim statement. Content-addressed identity. Two items with identical fingerprints are considered equivalent claims; only one enters the corpus. |
| `source_id` | `string (SHA-256[:32])` | Content-addressed identity of the source document. Stable across re-ingestion of the same document. |
| `corpus_version` | `string` | Semantic version of the knowledge corpus at extraction time. Enables reproducibility auditing across corpus generations. |
| `schema_version` | `string` | Version of this evidence schema. Enables forward-compatible evolution without silent field drops. |

---

### Section B — Provenance

Fields that establish where the claim came from, at every level of granularity from document to passage.

| Field | Type | Description |
|-------|------|-------------|
| `source_uri` | `string` | Original location of the source document (file path, URL, or DOI). Stable; not derived from content. |
| `source_title` | `string` | Human-readable title of the source document. |
| `source_domain` | `string` | Knowledge domain of the source (e.g., `smr`, `ai_data_centers`, `regulatory`). Used for profile attribution. |
| `supporting_source_ids` | `list[string]` | All source documents that support this claim. Never truncated. First entry is the primary source. |
| `extraction_run_id` | `string (UUID)` | UUID of the extraction run that produced this item. Links to an ExtractionRun record containing model version, prompt version, timestamp, and source_ids processed. |
| `page_number` | `integer \| null` | Page number within the source document where the supporting passage appears. Null for unpaged sources. |
| `section_heading` | `string \| null` | The nearest heading in the document structure above the supporting passage. |
| `chunk_id` | `string \| null` | Identifier of the text chunk or passage window from which the claim was extracted. Stable across re-chunking if the chunking strategy is versioned. |
| `char_offset_start` | `integer \| null` | Character offset of the start of the supporting passage within the full source text. |
| `char_offset_end` | `integer \| null` | Character offset of the end of the supporting passage. |
| `excerpt` | `string` | Verbatim or near-verbatim text fragment (≤600 characters) from the source passage that supports the claim. This is the auditable anchor — the text a human reviewer can read to confirm the claim. |

---

### Section C — Content

Fields that describe what the evidence item says and how to interpret it.

| Field | Type | Description |
|-------|------|-------------|
| `statement` | `string` | The extracted claim. An LLM-synthesized distillation of the supporting passage — not necessarily verbatim, but faithfully representing the source claim. |
| `category` | `string (enum)` | The semantic category of the claim (e.g., `MARKET`, `TECHNICAL`, `REGULATORY`, `FINANCIAL`, `RISK`). Used for investigation area mapping. |
| `entity` | `string \| null` | Primary entity referenced in the claim (company, technology, regulation, geography). |
| `entity_type` | `string \| null` | Type of the primary entity (e.g., `COMPANY`, `TECHNOLOGY`, `REGULATOR`, `COUNTRY`). |
| `scope` | `string \| null` | Geographic or organizational scope of the claim (e.g., `UK`, `GLOBAL`, `EU`). |
| `topics` | `list[string]` | Topic keywords derived from the claim content. Used for investigation area mapping and cross-domain synthesis. Never an empty list for a valid evidence item. |
| `temporal_reference` | `string \| null` | Time period the claim pertains to, if stated or inferable (e.g., `"2024"`, `"Q3 2023"`, `"next 5 years"`). |

---

### Section D — Retrieval

Fields that record how this evidence item was selected for a specific query context. These fields are query-specific — the same evidence item may appear with different retrieval fields when retrieved for different questions.

| Field | Type | Description |
|-------|------|-------------|
| `retrieval_query` | `string` | The query string (subquestion or investigation area) that retrieved this item. |
| `retrieval_mode` | `string (enum)` | The retrieval strategy used: `hybrid` \| `lexical` \| `semantic`. |
| `retrieval_rank` | `integer` | Integer rank in the retrieved set for this query (1-indexed). |
| `hybrid_score` | `float` | Final hybrid relevance score (full precision). |
| `lexical_score` | `float` | Lexical retrieval score component (BM25 or TF-IDF). |
| `semantic_score` | `float` | Semantic similarity score (cosine similarity against query embedding). |
| `metadata_factor` | `float` | Quality, priority, and strategic multiplier applied to the score. |
| `retrieval_model_version` | `string` | Version of the embedding model used for semantic scoring. Required for reproducibility. |
| `reranker` | `string (enum)` | Reranker applied: `passthrough` \| `llm` \| `none`. |
| `rerank_score` | `float \| null` | Reranker-assigned relevance score. Null when reranker is passthrough or none. |
| `rerank_rationale` | `string \| null` | LLM reranker's free-text rationale. Null when reranker is passthrough or none. Retained for audit when LLMReranker is active. |

---

### Section E — Scoring

Fields that characterize the quality and reliability of the evidence item independently of the retrieval context. These are source-quality signals, not retrieval relevance signals.

| Field | Type | Description |
|-------|------|-------------|
| `source_quality_score` | `float [0.0, 1.0]` | Overall quality assessment of the source document (e.g., peer-reviewed = 0.9, press release = 0.4). |
| `source_retrieval_priority` | `string (enum)` | Priority tier for retrieval: `HIGH` \| `MEDIUM` \| `LOW`. Set at ingestion based on source type and relevance classification. |
| `source_strategic_value` | `string (enum)` | Strategic relevance classification for this engagement domain: `HIGH` \| `MEDIUM` \| `LOW`. |
| `evidence_confidence` | `string (enum)` | Confidence in the extracted claim relative to the source: `HIGH` (claim is explicit) \| `MEDIUM` (claim is implied) \| `LOW` (claim is inferred). Set at extraction time. |
| `is_quantitative` | `boolean` | Whether the claim contains a quantitative assertion (number, percentage, timeline). Quantitative claims are higher-confidence grounding anchors. |

---

### Section F — Grounding

Fields that make the evidence item usable as a grounding anchor in downstream reasoning. These fields are the primary interface between the Knowledge Layer and the Reasoning Layer.

| Field | Type | Description |
|-------|------|-------------|
| `subquestion_assignments` | `list[string]` | The research subquestions this evidence item is assigned to. Assignment is based on semantic relevance to the subquestion text. A single item may serve multiple subquestions. |
| `investigation_area_assignments` | `list[string]` | The investigation areas this evidence item maps to. Derived from `category`, `topics`, and subquestion assignments. |
| `grounding_strength` | `string (enum)` | Assessment of how strongly this item grounds any claim that cites it: `STRONG` (verbatim or near-verbatim support) \| `MODERATE` (implied support) \| `WEAK` (tangential). |
| `coverage_contribution` | `string (enum)` | The coverage tier this item contributes to for its assigned subquestion: `STRONG` \| `MODERATE` \| `WEAK`. Distinct from `grounding_strength` — a high-quality item may contribute WEAK coverage if the subquestion has many competing items. |

---

### Section G — Relationships

Fields that express connections to other evidence items and to the analytical artifacts they have informed.

| Field | Type | Description |
|-------|------|-------------|
| `contradicts` | `list[string]` | evidence_ids of items that assert a claim incompatible with this item's statement. Populated by a contradiction-detection pass; empty until that pass runs. |
| `corroborates` | `list[string]` | evidence_ids of items that assert a compatible or reinforcing claim. Provides triangulation signals for confidence scoring. |
| `superseded_by` | `string \| null` | evidence_id of the item that supersedes this one (e.g., after an updated extraction from the same source). Null if this item is current. |
| `supersedes` | `string \| null` | evidence_id of the item this one supersedes. Null if this is the original extraction. |
| `informed_hypotheses` | `list[string]` | hypothesis_ids of hypotheses that referenced this evidence item. Populated as downstream agents run. Enables forward traceability. |
| `informed_recommendations` | `list[string]` | recommendation_ids that cited this evidence item. Populated as downstream agents run. |

---

### Section H — Lifecycle

Fields that govern the operational state of the evidence item and its eligibility for reasoning use.

| Field | Type | Description |
|-------|------|-------------|
| `status` | `string (enum)` | Current lifecycle state: `ACTIVE` \| `SUPERSEDED` \| `RETRACTED` \| `PENDING_REVIEW`. Only `ACTIVE` items are eligible for retrieval. |
| `acceptance_policy` | `string (enum)` | Governance policy under which this item was accepted: `AUTO` (accepted by pipeline) \| `HUMAN_APPROVED` \| `PENDING`. Enables evidence governance auditing. |
| `extracted_at` | `string (ISO 8601)` | Timestamp of extraction. |
| `accepted_at` | `string \| null (ISO 8601)` | Timestamp of acceptance into the active corpus. Null if `PENDING`. |
| `retracted_at` | `string \| null (ISO 8601)` | Timestamp of retraction, if applicable. |
| `retraction_reason` | `string \| null` | Free-text reason for retraction. Null if not retracted. |

---

## 3. Field Rationale and Contract Dependencies

This section explains why each section exists and which downstream contracts depend on it.

---

### Section A — Identity

**Why it exists:** Evidence items must be uniquely, stably, and content-addressably identified. Without content-addressed identity, re-extracting the same claim from the same source produces a new evidence item with a different UUID — inflating the corpus and making reproducibility impossible. `content_fingerprint` prevents this. `corpus_version` and `schema_version` allow downstream systems to know exactly what generation of knowledge they are reasoning against.

**Contracts served:** Reproducibility, Auditability.

---

### Section B — Provenance

**Why it exists:** Every downstream agent that cites an evidence item is making an implicit claim that the cited statement appears in the cited source. Without `excerpt`, this claim is unverifiable without reading the entire document. Without `page_number` and `chunk_id`, a human reviewer has no starting point for verification. Without `extraction_run_id`, the extraction conditions (model, prompt, timestamp) cannot be reconstructed.

**Contracts served:** Provenance Preservation (primary), Citation Integrity, Grounding, Auditability, Reproducibility.

**Key field:** `excerpt` is the single most important field for auditability. It is the verbatim anchor that allows a human reviewer to confirm that the extracted `statement` faithfully represents the source.

---

### Section C — Content

**Why it exists:** The claim content must be characterized well enough for downstream agents to assign it to the correct analytical context (subquestion, investigation area, hypothesis). `category`, `topics`, `entity`, and `scope` are the semantic signals that drive this assignment. Without these fields, area mapping degrades to keyword overlap on the raw statement text — a brittle fallback.

**Contracts served:** Retrieval Completeness, Grounding.

**Key field:** `topics` is a load-bearing mapping signal. If it is empty, area mapping relies only on `category` and statement text.

---

### Section D — Retrieval

**Why it exists:** Downstream reasoning agents need to know not just what evidence was retrieved, but why it was selected over alternatives. `retrieval_rank`, `hybrid_score`, and its decomposition (`lexical_score`, `semantic_score`, `metadata_factor`) are the scoring provenance that makes this explicable. Without these fields, there is no architectural basis for asking whether the right evidence was selected.

`retrieval_query` binds the evidence item to the specific analytical question it answered, preventing context confusion when the same item is retrieved for multiple subquestions.

`retrieval_model_version` is required for reproducibility: a semantic score computed with embedding model v1 is not comparable to a score computed with v2.

**Contracts served:** Ranking Stability, Reproducibility, Recommendation Explainability.

---

### Section E — Scoring

**Why it exists:** Retrieval scores measure relevance-to-query. Source quality scores measure credibility-of-source. These are orthogonal dimensions. A low-quality source can be highly relevant to a query; a high-quality source can be retrieved with a low relevance score. `evidence_confidence` distinguishes explicit claims from inferred ones — an important signal for hypothesis generation.

**Contracts served:** Grounding, Auditability.

---

### Section F — Grounding

**Why it exists:** Grounding fields are the interface between the Knowledge Layer and the Reasoning Layer. Without `subquestion_assignments` and `investigation_area_assignments` as first-class fields on the evidence item itself, downstream agents must re-derive this mapping from scratch — and may derive it differently on each run. By committing the assignment at the Knowledge Layer, the system ensures that every downstream agent operates from the same grounding baseline.

`grounding_strength` is the qualitative signal that tells a downstream agent how confidently it can cite this item as support for a claim. `WEAK` grounded evidence should generate lower-confidence hypotheses than `STRONG` grounded evidence.

**Contracts served:** Grounding, Recommendation Traceability, Citation Integrity.

---

### Section G — Relationships

**Why it exists:** Evidence does not exist in isolation. A recommendation is stronger when multiple independent evidence items corroborate each other and weaker when contradicted. The contradiction and corroboration fields make these relationships first-class rather than requiring downstream agents to compute them ad-hoc.

Forward traceability fields (`informed_hypotheses`, `informed_recommendations`) enable an auditor to ask: "which downstream conclusions depended on this specific evidence item?" This is the architectural basis for impact analysis when evidence is updated or retracted.

**Contracts served:** Auditability, Recommendation Traceability, Evidence Governance.

---

### Section H — Lifecycle

**Why it exists:** Evidence enters and leaves active use. A source document may be superseded by a newer version; an extracted claim may be found to be erroneous. The `status` field gates retrieval — only `ACTIVE` items are eligible. The `acceptance_policy` field distinguishes automatically accepted evidence from human-reviewed evidence, which is required for evidence governance and regulatory compliance.

**Contracts served:** Auditability, Evidence Governance, Reproducibility.

---

## 4. Evidence Lifecycle

The following diagram identifies at every phase which evidence metadata must survive the transition and what is introduced.

```
SOURCE DOCUMENT
  │
  │  Required input:
  │    URI, title, domain, page_count
  │    (qualitative source metadata set externally)
  │
  ↓
INGESTION
  │
  │  Produced:
  │    source_id          ← SHA-256 of canonical_text
  │    source_uri
  │    source_title
  │    source_domain
  │    corpus_version     ← corpus state at ingestion time
  │
  │  Structural metadata indexed:
  │    page_index (page_number → char_offset)
  │
  ↓
EXTRACTION (LLM)
  │
  │  Produced:
  │    evidence_id        ← UUID
  │    content_fingerprint← SHA-256 of normalized statement
  │    statement          ← extracted claim
  │    excerpt            ← verbatim supporting passage
  │    page_number        ← from page_index lookup
  │    section_heading    ← from document structure
  │    chunk_id           ← passage window identifier
  │    char_offset_start/end
  │    category, entity, entity_type, scope, topics
  │    temporal_reference
  │    evidence_confidence, is_quantitative
  │    supporting_source_ids
  │    extraction_run_id
  │    extracted_at, status = PENDING, acceptance_policy = AUTO
  │    schema_version, corpus_version
  │
  ↓
DEDUPLICATION & ACCEPTANCE
  │
  │  content_fingerprint checked → duplicate suppressed
  │  status → ACTIVE (auto) or PENDING (governance gate)
  │  accepted_at set
  │
  │  All fields above must survive. Nothing dropped.
  │
  ↓
RETRIEVAL
  │
  │  Produced (query-specific):
  │    retrieval_query
  │    retrieval_mode, retrieval_model_version
  │    retrieval_rank
  │    hybrid_score, lexical_score, semantic_score, metadata_factor
  │
  │  All identity and provenance fields must survive into RetrievedEvidence.
  │
  ↓
RERANKING
  │
  │  Added (if LLMReranker):
  │    reranker = "llm"
  │    rerank_score, rerank_rationale
  │
  │  If PassthroughReranker:
  │    reranker = "passthrough"
  │    rerank_score = null, rerank_rationale = null
  │
  │  All prior fields must survive. rerank_rationale must be persisted, not ephemeral.
  │
  ↓
ASSEMBLY (Evidence Layer Output)
  │
  │  The assembled Evidence artifact is the canonical unit handed to the Reasoning Layer.
  │  It must carry the COMPLETE canonical evidence object — all sections A through H.
  │  Nothing may be dropped. Retrieval fields are added; no prior fields are removed.
  │
  │  At this transition:
  │    subquestion_assignments    ← computed from semantic assignment
  │    investigation_area_assignments ← computed from category + topics + subquestion assignments
  │    grounding_strength         ← computed from evidence_confidence + coverage
  │    coverage_contribution      ← computed from retrieval rank within subquestion pool
  │
  ↓
REASONING AGENTS
  │
  │  HypothesisAgent, AssumptionAgent, ChallengeAgent, RecommendationAgent, etc.
  │
  │  These agents read evidence items.
  │  On write, they MUST record which evidence_ids informed their output.
  │
  │  Populated (reverse traceability):
  │    informed_hypotheses  ← hypothesis_id appended when an agent cites this item
  │    informed_recommendations ← recommendation_id appended when cited
  │
  │  At every reasoning step: evidence_id, excerpt, page_number, source_id
  │  must remain available for citation.
  │
  ↓
REPORT FINDING
  │
  │  A citation in the final report must carry:
  │    evidence_id         ← stable identifier
  │    source_title        ← human-readable
  │    page_number         ← human-readable location
  │    excerpt             ← verifiable anchor
  │    source_uri          ← retrievable reference
  │
  │  A citation without page_number and excerpt is unverifiable.
  │  A citation without source_uri is irreproducible.
```

**The invariant across every transition:** once a field is populated, it is never dropped. The only fields that grow are relationships (`informed_hypotheses`, `informed_recommendations`) as the evidence item is consumed by downstream agents.

---

## 5. Field Classification

### Required

These fields must be present on every canonical Evidence item. An item missing any Required field is structurally invalid and must not enter the active corpus or be made available to retrieval.

```
evidence_id
content_fingerprint
source_id
corpus_version
schema_version
source_uri
source_title
source_domain
supporting_source_ids
extraction_run_id
excerpt
statement
category
topics
evidence_confidence
is_quantitative
status
acceptance_policy
extracted_at
```

### Optional

These fields should be present when the source structure supports them. Their absence must be explicit (null), not silent (missing key). An item may be valid without them, but a corpus where these fields are systematically null is a provenance-impaired corpus.

```
page_number
section_heading
chunk_id
char_offset_start
char_offset_end
entity
entity_type
scope
temporal_reference
source_quality_score
source_retrieval_priority
source_strategic_value
accepted_at
retracted_at
retraction_reason
superseded_by
supersedes
```

### Derived (computed from other fields at a defined pipeline stage)

These fields are not set at extraction but are computed deterministically from other fields at a defined pipeline stage. They must be computed before the item is handed to the Reasoning Layer.

```
subquestion_assignments
investigation_area_assignments
grounding_strength
coverage_contribution
contradicts
corroborates
```

### Runtime only (query-specific; not part of the persisted Evidence record)

These fields exist in the retrieval result but are not persisted to the `Evidence` schema. Instead, they are persisted in a `RetrievalProvenance` record that is associated with the assembled output artifact, not with the evidence item itself. This separation keeps the evidence corpus clean of query-specific state.

```
retrieval_query
retrieval_mode
retrieval_model_version
retrieval_rank
hybrid_score
lexical_score
semantic_score
metadata_factor
reranker
rerank_score
rerank_rationale
```

*Note on retrieval fields:* Although classified as Runtime only in the persisted Evidence schema, they are Required in the assembled Evidence artifact that the Reasoning Layer receives. The `RetrievalProvenance` record must accompany every assembled evidence set.

### Populated by downstream agents (grow over lifecycle)

```
informed_hypotheses
informed_recommendations
```

---

## 6. Canonical Evidence Object

The following is the target schema. This is the canonical Evidence object the Strategic Research Harness should produce as the output of the Knowledge Layer.

```json
{
  "schema_version": "1.0",
  "corpus_version": "2024-Q3.1",

  "identity": {
    "evidence_id": "e7c3a1b2-...",
    "content_fingerprint": "sha256:3fa8c2...",
    "source_id": "src:a1b2c3d4..."
  },

  "provenance": {
    "source_uri": "https://example.com/report.pdf",
    "source_title": "UK SMR Regulatory Pathway Assessment 2024",
    "source_domain": "smr",
    "supporting_source_ids": ["src:a1b2c3d4...", "src:e5f6g7h8..."],
    "extraction_run_id": "run:9d3e7f...",
    "page_number": 47,
    "section_heading": "3.2 Generic Design Assessment Timeline",
    "chunk_id": "chunk:p47-para3",
    "char_offset_start": 142300,
    "char_offset_end": 142680,
    "excerpt": "The ONR has indicated that the Generic Design Assessment process for SMR designs submitted by 2024 is expected to complete by Q4 2028, subject to design maturity and applicant responsiveness. Historical GDA timelines have ranged from 3.5 to 6 years."
  },

  "content": {
    "statement": "UK Generic Design Assessment for SMR designs submitted in 2024 is expected to complete by Q4 2028, with historical timelines of 3.5–6 years.",
    "category": "REGULATORY",
    "entity": "ONR",
    "entity_type": "REGULATOR",
    "scope": "UK",
    "topics": ["GDA", "SMR", "regulatory timeline", "ONR", "nuclear licensing"],
    "temporal_reference": "Q4 2028"
  },

  "scoring": {
    "source_quality_score": 0.85,
    "source_retrieval_priority": "HIGH",
    "source_strategic_value": "HIGH",
    "evidence_confidence": "HIGH",
    "is_quantitative": true
  },

  "grounding": {
    "subquestion_assignments": [
      "What are the regulatory timelines for SMR deployment in the UK?"
    ],
    "investigation_area_assignments": ["Regulatory Pathway", "Risk Assessment"],
    "grounding_strength": "STRONG",
    "coverage_contribution": "STRONG"
  },

  "relationships": {
    "contradicts": [],
    "corroborates": ["e7c3a100-...", "e7c3a200-..."],
    "superseded_by": null,
    "supersedes": null,
    "informed_hypotheses": [],
    "informed_recommendations": []
  },

  "lifecycle": {
    "status": "ACTIVE",
    "acceptance_policy": "AUTO",
    "extracted_at": "2026-07-16T14:23:00Z",
    "accepted_at": "2026-07-16T14:23:01Z",
    "retracted_at": null,
    "retraction_reason": null
  }
}
```

The associated `RetrievalProvenance` record (not stored on the Evidence object; stored per query run):

```json
{
  "evidence_id": "e7c3a1b2-...",
  "retrieval_query": "What are the regulatory timelines for SMR deployment in the UK?",
  "retrieval_mode": "hybrid",
  "retrieval_model_version": "all-MiniLM-L6-v2:1.2",
  "retrieval_rank": 1,
  "hybrid_score": 0.842,
  "lexical_score": 0.731,
  "semantic_score": 0.903,
  "metadata_factor": 1.35,
  "reranker": "passthrough",
  "rerank_score": null,
  "rerank_rationale": null
}
```

---

## 7. Sufficiency Assessment

> If the Strategic Research Harness were built from scratch today, would this Evidence object be sufficient to support grounding, provenance, citation integrity, auditability, reproducibility, and recommendation traceability?

**YES.**

---

**Grounding:** The canonical Evidence object is sufficient for grounding. Every downstream finding that cites an evidence item carries `evidence_id`, `statement`, `excerpt`, `page_number`, `section_heading`, and `grounding_strength`. A reasoning agent knows not just what the claim is but how strongly it is anchored to the source.

**Provenance:** The canonical Evidence object is sufficient for provenance. Every level of the provenance chain — source document, extraction run, passage, page, section, character offset — is a first-class field. No level is implicit or recoverable only by parsing.

**Citation Integrity:** The canonical Evidence object is sufficient for citation integrity. `supporting_source_ids` carries all sources (no truncation). `excerpt` and `page_number` enable passage-level citation. `source_uri` enables the citation to be independently retrieved.

**Auditability:** The canonical Evidence object is sufficient for audit. An auditor can move from a report finding to a `recommendation_id`, to an `evidence_id`, to `excerpt + page_number + source_uri`, and verify the claim in the original document without any additional lookup. The `informed_hypotheses` and `informed_recommendations` fields enable forward traceability — identifying all downstream conclusions that depended on a given evidence item before it is updated or retracted.

**Reproducibility:** The canonical Evidence object combined with the `RetrievalProvenance` record is sufficient for reproducibility. `corpus_version`, `retrieval_model_version`, `retrieval_mode`, and the full score decomposition allow an auditor to reconstruct the exact retrieval context that produced a given evidence set.

**Recommendation Traceability:** The canonical Evidence object is sufficient for recommendation traceability. `informed_recommendations` provides forward traceability from evidence to recommendations. `subquestion_assignments` and `investigation_area_assignments` establish the analytical context that connects evidence to the reasoning chain.

---

*End of CANONICAL_EVIDENCE_MODEL.md*
