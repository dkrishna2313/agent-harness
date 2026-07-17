# KNOWLEDGE_LAYER_GAP_ANALYSIS.md

Strategic Research Harness — Knowledge Layer Architecture Gap Analysis  
Classification Date: 2026-07-16  
Analysis Version: PH5.2  
Contract Reference: `docs/architecture/KNOWLEDGE_LAYER_ARCHITECTURAL_CONTRACT.md`

---

## Scope and Method

This document audits the current Knowledge Layer implementation against the six architectural contracts defined in `KNOWLEDGE_LAYER_ARCHITECTURAL_CONTRACT.md`. It is a read-only architectural audit — no code was modified as part of this analysis.

**Files examined:**

| File | Lines | Purpose |
|------|-------|---------|
| `functional_agents/evidence_agent.py` | 939 | Primary retrieval and coverage logic |
| `functional_agents/evidence_boundary.py` | 243 | PH2.2 boundary: normalize → validate → typed output |
| `knowledge/retriever.py` | 571 | Lexical, semantic, and hybrid retrieval modes |
| `knowledge/reranker.py` | 416 | LLMReranker (Claude Haiku) and PassthroughReranker |
| `knowledge/models.py` | 265 | Evidence, Source, KnowledgeMetadata, ExtractionRun schema |
| `knowledge/store.py` | partial | JSONL persistence, iter_evidence(), atomic writes |
| `functional_agents/report_agent.py` | partial | Citation building, grounding counts |

---

## Part 1 — Contract Status Table

| # | Contract Property | Status | Summary |
|---|------------------|--------|---------|
| 1 | Retrieval Completeness | **YELLOW** | Coverage tracking (STRONG/MODERATE/WEAK/NONE) exists; area mapping is impaired by `topics: []` always empty in KB path; thresholds are heuristics with no formal validation |
| 2 | Provenance Preservation | **YELLOW** | Source attribution exists at document level; no sub-document location (chunk, page, excerpt) — a gap explicitly called out in the contract |
| 3 | Citation Integrity | **YELLOW** | Citation markers built deterministically; `citations_missing` is a non-fatal diagnostic not surfaced as pipeline warning; area-mapping gaps produce silent citation voids |
| 4 | Ranking Stability | **YELLOW** | Lexical/hybrid scoring is deterministic on fixed corpus; JSONL iteration order introduces corpus-build dependency; LLMReranker breaks stability whenever enabled |
| 5 | Grounding | **YELLOW** | Grounding counts computed at report time only; no per-finding grounding score at the evidence layer; subquestion assignment via token overlap can silently fail |
| 6 | Reproducibility | **YELLOW** | Fixed corpus + disabled reranker produces repeatable ranked sets; JSONL append order creates corpus-build dependency; no canonical evidence fingerprint registered |

**Legend:** GREEN = contract fully met and validated. YELLOW = partial implementation, meaningful gaps against contract. RED = contract unimplemented or systematically violated.

No contract is GREEN. All six contracts have meaningful implementation but none have been formally validated against the contract. No contract is RED — each has a real, working implementation.

---

## Part 2 — Per-Contract Deep Dive

---

### 2.1 Retrieval Completeness

**What exists:**

- Four-level coverage taxonomy: STRONG (≥4 items), MODERATE (≥2), WEAK (≥1), NONE (0) computed per subquestion in `evidence_agent.py:_coverage_level()`
- Per-subquestion expansion in `_execute_kb()`: primary query + one expansion query per subquestion from the plan
- `coverage_by_subquestion` and `evidence_by_subquestion` maps produced on every KB path run
- 0-evidence guardrail: KB→legacy fallback; legacy→`_insufficient_evidence=True` prevents downstream agents from reasoning on empty evidence
- Investigation area mapping in `_map_evidence_to_areas()` using category, topics, and claim fields

**What is missing:**

- **Area mapping is structurally impaired.** In the KB path, all evidence items have `topics: []` (empty list). The `_map_evidence_to_areas()` method relies on topics as a first-class signal. With topics always empty, area mapping falls back to category and claim content only — a weaker signal that silently undercounts area coverage.
- **STRONG≥4 is an unvalidated heuristic.** The thresholds (STRONG≥4, MODERATE≥2, WEAK≥1) are arbitrary constants in the implementation. There is no calibration study showing that STRONG coverage on a subquestion correlates with sufficient evidence quality for a strategic recommendation.
- **No cross-subquestion deduplication.** Evidence items mapped to multiple subquestions via the secondary assignment (threshold=3 shared tokens) inflate multi-subquestion counts without adding unique evidence. This can produce MODERATE/STRONG coverage from a smaller actual evidence pool.
- **Coverage does not imply relevance.** A subquestion with STRONG coverage means four or more items were retrieved; it does not mean those items are topically relevant. There is no relevance filter between retrieval score and the coverage tier assigned.

**Why it matters architecturally:**

Retrieval completeness is the contract that determines whether the pipeline has an adequate evidence foundation. If coverage tiers do not reflect actual topical relevance — especially with `topics: []` impairing area mapping — every downstream agent that conditions on coverage (ResearchGapAgent, QAAgent, IterationPlanAgent) reasons on unreliable signals. The 0-evidence guardrail is a floor; the completeness contract is about the floor being high enough to support strategic analysis.

---

### 2.2 Provenance Preservation

**What exists:**

- Every `Evidence` record has `evidence_id` (immutable, assigned at extraction time) and `supporting_source_ids` linking the evidence claim to one or more `Source` records
- Every `Source` record has `source_id` (first 32 chars of SHA-256 content fingerprint), `uri`, `title`, `fingerprint`, and `canonical_text`
- `ExtractionRun` records `source_ids`, `model_version`, `prompt_version`, and `evidence_ids_produced` — a narrow provenance record that connects extraction batch to inputs and outputs
- The `statement_fingerprint` field on `Evidence` supports deduplication and implicit lineage

**What is missing:**

- **No sub-document location.** Neither `Evidence` nor `Source` nor any model in `knowledge/models.py` has a field for chunk index, page number, paragraph offset, or excerpt reference. The contract says "chunk or excerpt reference if applicable" — for documents where this is applicable (multi-page PDFs, long reports), it is never recorded. Provenance stops at the source document boundary.
- **`supporting_source_ids` is not page/chunk granular.** The field links evidence to a source record, not to a location within that source. A citation like `[Source: IPCC AR6, Evidence: E042]` cannot be independently verified to a specific section without reading the entire source document.
- **No extraction chain from evidence to raw text excerpt.** `canonical_text` on the Source record stores the full cleaned source text, not the specific passage that generated a given evidence statement. There is no field recording which passage or sentences were the basis for a particular extraction.

**Why it matters architecturally:**

Provenance preservation is the contract that makes evidence auditable. For strategic investment decisions (SMR Go/No-Go, energy infrastructure), the ability to trace a recommendation back to a specific claim, in a specific document, in a specific passage, is a legal and governance requirement — not just an engineering quality signal. The current implementation provides source-level attribution only. For any engagement requiring evidence accountability beyond document identity, this is a structural gap.

---

### 2.3 Citation Integrity

**What exists:**

- `_format_citations()` in `report_agent.py` builds citation markers in the format `[Source: <name>, Evidence: <eid>]` deterministically from a list of evidence IDs and a `source_document` lookup map
- The evidence boundary (`evidence_boundary.py`) validates that `source_document` references are non-dangling — every cited `source_document` value must resolve to a known source
- `citations_present` and `citations_missing` are non-fatal diagnostics produced by the boundary. `citations_missing` counts items whose `source_document` field is empty after normalization
- `_build_key_findings()` in ReportAgent counts `supported_findings`, `unsupported_findings`, and `citation_count` at report construction time

**What is missing:**

- **`citations_missing` is non-fatal and invisible downstream.** When evidence items arrive at the boundary with an empty `source_document`, they are counted in `citations_missing` and allowed to pass. No downstream agent sees this count; it is a diagnostic only. Pipeline warnings are not emitted. A report can cite evidence that has no source attribution without any alarm.
- **Citation uses source_id, not human-readable title.** `source_document` stores a source_id (SHA-256 prefix), not the title. `_format_citations()` receives the `id_to_source` map which maps `evidence_id → source_document`, but the source_document is the raw source_id. In the benchmark format `[Source: <name>, Evidence: E001]`, `<name>` is actually the source_id string unless a title lookup is done. This creates citations that are technically traceable but not human-readable in isolation.
- **Silent citation gaps from area-mapping failure.** Evidence items that fail area mapping (because `topics: []`) are not cited in area-specific report sections. These are not flagged as citation gaps — they simply do not appear in citations for those areas.
- **No citation coverage requirement.** There is no contract check that a minimum fraction of findings in the report have citations. A finding backed by MODERATE/STRONG coverage but with empty `source_document` on all items will appear uncited, and this is not detected as a violation.

**Why it matters architecturally:**

Citation integrity is what makes the pipeline's output accountable. The current implementation guarantees that citations are not structurally invalid (no dangling references), but it does not guarantee that citations are complete, human-readable, or that citation gaps are surfaced. For benchmark scoring, citation format correctness is measured — missing citations directly reduce scores in the current evaluation framework.

---

### 2.4 Ranking Stability

**What exists:**

- Hybrid scoring in `retriever.py`: `0.4 × lexical_relevance + 0.6 × semantic_similarity × metadata_factor`. All three components are deterministic for fixed corpus and fixed query
- Metadata factor: `metadata_factor ∈ [0.65, 1.45]` computed from `quality_factor × priority_factor × strategic_factor` via `KnowledgeMetadata` fields — deterministic for fixed metadata state
- Tiebreak: `(score, overall_score)` descending — a deterministic tiebreak that resolves equal-score items consistently for fixed corpus
- `PassthroughReranker`: identity reranker that preserves retrieval order — deterministic by construction

**What is missing:**

- **JSONL iteration order introduces corpus-build dependency.** `store.iter_evidence()` reads the JSONL knowledge base file in insertion/append order. There is no sort. If two knowledge base builds ingest the same documents in a different order (e.g., after an incremental update), tiebreaks at equal scores resolve differently. The ranked output is build-order dependent, not content-order dependent.
- **LLMReranker is non-deterministic by design.** `LLMReranker` calls Claude Haiku via tool_use. Temperature is not explicitly set to 0.0. The reranker produces a reordering that is not reproducible across calls. When `use_reranker=True`, the ranking stability contract is broken. The CLI defaults and test harness default to `use_reranker=False` (PassthroughReranker), which is why empirical runs appear stable — but the architecture contains an unstable path.
- **Semantic embeddings are pre-computed but not versioned.** Semantic similarity uses pre-computed embeddings. If embedding model or chunking strategy changes, semantic scores change, and rankings change. There is no embedding model version recorded in the retrieval output, so ranking provenance cannot be reconstructed.
- **No stability metric.** There is no test that measures how much ranking changes when a small perturbation is applied to the corpus (adding one document, changing one metadata score). The system has no declared tolerance for ranking drift.

**Why it matters architecturally:**

Ranking stability determines whether two pipeline runs on the same question, against the same knowledge base, produce the same evidence set — and therefore whether the pipeline's analytical conclusions are reproducible. An LLM reranker in the critical path means reproducibility is contingent on not enabling a feature. This is a latent fragility: the stable path works until someone enables the reranker for quality improvement, at which point reproducibility silently breaks.

---

### 2.5 Grounding

**What exists:**

- `_build_key_findings()` in `report_agent.py` counts `supported_findings` (findings with MODERATE/STRONG coverage and usable claim text) vs `unsupported_findings` (MODERATE/STRONG coverage but no usable claim)
- `citation_count` tracks the number of `[Source: ...]` markers appended to findings
- Evidence-to-subquestion mapping in `_map_evidence_to_subquestions()` uses token overlap with winner-take-all assignment: the subquestion with the most shared tokens wins the evidence item; a secondary threshold (≥3 shared tokens) allows additional assignments
- `_insufficient_evidence` flag in EvidenceAgent guards against evidence-free reports

**What is missing:**

- **Grounding is computed at report time, not at evidence time.** The evidence layer (EvidenceAgent, EvidenceBoundary) produces no per-finding grounding score. Grounding is a post-hoc count computed in ReportAgent. This means there is no grounding signal available to any reasoning agent (HypothesisAgent, AssumptionAgent, RecommendationAgent) at the time they generate their outputs.
- **No per-finding grounding score in evidence notes.** The `evidence_notes` output contains coverage tiers and item lists, but no explicit "this finding is grounded at X%" field. Agents downstream of EvidenceAgent have no grounding signal to condition on unless they are ReportAgent.
- **Token-overlap subquestion assignment can silently fail.** Evidence items that have fewer than 3 shared tokens with any subquestion go to `_unmapped`. Unmapped items contribute to corpus size but not to any subquestion's coverage count or finding list. There is no log entry or diagnostic that counts how many items are unmapped in a given run.
- **No requirement that recommendations trace to evidence.** The contract says that recommendations should be grounded in evidence. But RecommendationAgent is a generative agent that receives evidence notes as context — it is not required to output `supporting_evidence_ids` for each recommendation. Whether it does depends on the LLM's behavior, not on a schema constraint. The evidence layer does not produce a pre-computed mapping from recommendation topics to evidence IDs that would make grounding enforcement tractable.

**Why it matters architecturally:**

Grounding is the contract that ties the pipeline's analytical output to its evidence base. Without grounding scores at the evidence layer, downstream agents are free to generate unsupported outputs without any structural check. The current implementation surfaces grounding information only in the final report — after all reasoning is complete — making it a diagnostic rather than a constraint.

---

### 2.6 Reproducibility

**What exists:**

- Lexical retrieval on a fixed corpus with fixed parameters is deterministic: same query, same corpus, same weights → same ranked list (subject to JSONL iteration order — see Ranking Stability)
- `_execute_kb()` in EvidenceAgent is deterministic for fixed corpus and fixed embeddings: per-subquestion expansion uses a deterministic seen-ID set that prevents duplicate retrieval
- Evidence deduplication via `statement_fingerprint` prevents content-identical items from re-entering the corpus on subsequent builds
- The KB path does not call an LLM during retrieval (LLM is called downstream by HypothesisAgent and others, not by EvidenceAgent itself in the KB path)

**What is missing:**

- **No canonical evidence output fingerprint registered.** Unlike the Planning Layer (PH4) and YES-deterministic agents (PH5.1), no canonical SHA-256 fingerprint has been registered for the evidence output of a fixed question on a fixed corpus. Without a registered fingerprint, there is no formal baseline against which to detect drift.
- **Reproducibility depends on build order.** JSONL iteration order is append order. If the knowledge base is rebuilt (e.g., after adding new sources), existing evidence items may appear in a different JSONL line order, causing tiebreaks to resolve differently and producing a different ranked set — even if the evidence content is identical.
- **Seen-ID set during subquestion expansion is insertion-order dependent.** In `_execute_kb()`, each subquestion expansion retrieves additional items and adds them to a `seen` set. If two subquestions retrieve the same item at the same score, the one processed first (primary query or first subquestion) claims the item, and the subsequent subquestion does not receive it. Subquestion ordering in the plan is deterministic given a fixed plan; but plan ordering itself follows PlannerAgent output order, which is only deterministic with PlanningCache enabled.
- **No reproducibility test for KB path.** There is no test that runs EvidenceAgent twice on the same frozen corpus and compares the evidence output fingerprint. The PH5.1 methodology (freeze inputs → run N times → compare fingerprints) has not been applied to the KB path.

**Why it matters architecturally:**

Reproducibility is what allows a pipeline run to be audited — to confirm that a prior analytical output could be generated again from the same inputs. For regulatory, legal, or governance contexts, reproducibility is a requirement, not a quality-of-life feature. The current implementation is reproducible in practice when the corpus is fixed and the reranker is disabled, but this guarantee is unregistered, untested, and fragile.

---

## Part 3 — Undocumented Properties

The following properties are implemented in the Knowledge Layer but are not addressed in `KNOWLEDGE_LAYER_ARCHITECTURAL_CONTRACT.md`. These are not gaps in the implementation — they are implemented capabilities without a contract. No contract modifications are proposed here.

---

### 3.1 Profile Attribution via Keyword Scoring

`evidence_agent.py` assigns a `source_profile` field to each evidence item using keyword overlap between the item's content and each loaded profile's vocabulary. This creates per-item profile attribution that feeds the MultiProfileAgent downstream. The contract does not address profile attribution — it is an undocumented property of the retrieval layer that affects cross-domain traceability.

### 3.2 Multi-Domain Evidence Isolation via Scratch Contexts

For engagements with multiple domain profiles, EvidenceAgent runs retrieval in isolated "scratch contexts" — one per secondary domain — before merging into the primary evidence set. This prevents evidence from a secondary domain from crowding out primary domain evidence in the ranked list. The isolation mechanism is undocumented in the contract and does not have formal isolation criteria.

### 3.3 Intent Detection and Vocabulary Boosting in Retriever

`retriever.py` detects the intent type of a query (e.g., MARKET, REGULATORY, TECHNICAL) using keyword matching and applies a vocabulary boost to lexical scores for intent-aligned terms. The intent boosting parameter and vocabulary lists are hardcoded constants. This is a retrieval quality mechanism that affects ranking but is not surfaced as a tunable contract property.

### 3.4 Metadata Factor (Quality × Priority × Strategic Multiplier)

The hybrid retrieval formula applies a `metadata_factor ∈ [0.65, 1.45]` derived from three `KnowledgeMetadata` fields: `overall_score`, `retrieval_priority`, and `strategic_value`. This multiplier is a first-class architectural mechanism that can suppress or amplify retrieval scores by up to ±80%. It is not documented in the contract and has no formal calibration.

### 3.5 Zero-Evidence Legacy Fallback

When the KB path returns zero evidence items, EvidenceAgent falls back to the legacy `DcPowerAgent` path. When the legacy path also returns zero, `_insufficient_evidence` is set to True and downstream agents receive a guard condition. This two-tier fallback is an undocumented reliability mechanism with no contract coverage.

### 3.6 Statement-Level Deduplication via `statement_fingerprint`

`Evidence.statement_fingerprint` (SHA-256 of normalized statement text) prevents duplicate claims from entering the corpus across extraction runs. This is a knowledge base quality property that affects coverage counts — a duplicate statement that would count twice without deduplication counts once with it. The contract does not address extraction-time deduplication or its effect on coverage tiers.

---

## Part 4 — Overall Assessment

### Strongest Part: Citation Integrity

The citation integrity implementation is the most structurally sound of the six contracts. Citation markers are built deterministically. The evidence boundary validates referential integrity (no dangling source references). The benchmark-compatible format `[Source: <name>, Evidence: <eid>]` is explicitly implemented and tested in the evaluation framework. The implementation is complete enough to be directly testable.

### Weakest Part: Provenance Preservation

Provenance preservation has the largest absolute gap between what is implemented and what the contract requires. The implementation traces evidence to a source document — a strong foundation. But for the contract to be met, traceability must extend to sub-document location (chunk, page, excerpt). The knowledge models have no fields for this. Closing this gap requires schema changes to `Evidence` and `Source`, changes to the extraction pipeline that populates them, and updates to the boundary validator that enforces them. No other contract requires schema-level changes to close its gap.

### Second Weakest: Ranking Stability

Ranking stability has a latent architectural risk that is not visible in normal operation. The `LLMReranker` is an enabled, production-ready component that breaks the stability contract whenever it is activated. The fact that CLI defaults use `PassthroughReranker` means the gap is invisible in typical runs. But the architecture contains an unstable path that will be triggered if reranker quality improvements are pursued. This is a different category of gap than the others — it is not "partially implemented" but "correctly implemented with a structural exception that breaks the contract."

### Which Single Gap to Address First

**Provenance Preservation** is the highest-priority gap to address.

The reasoning is architectural leverage: provenance is the foundation that makes citation integrity, grounding, and reproducibility auditable. A citation without sub-document location cannot be independently verified. A grounding claim without excerpt reference cannot be traced. A reproducibility guarantee without provenance granularity cannot be confirmed by an auditor. Closing the provenance gap at the schema level — adding chunk/page/location fields to `Evidence` and `Source` — enables a cascade of improvements to the other five contracts.

Ranking stability is technically more urgent in the sense that it creates an active risk (enabling LLMReranker silently breaks the contract). But the provenance gap is broader in impact: every engagement, every evidence item, every citation is affected by the absence of sub-document location — not just runs that happen to enable the reranker.

---

## Part 5 — Roadmap

Ordered by architectural importance. No effort estimates. No code. Architectural sequencing only.

---

### PH5.3 — Provenance Audit

**Objective:** Formally characterize the sub-document location gap and define what provenance means for this system.

**Scope:**
- Audit the `Evidence`, `Source`, and `ExtractionRun` schemas against the contract requirement for chunk/excerpt reference
- Define what sub-document location means for each source type used in the system (PDF reports, web articles, structured data files) — not all source types have the same granularity concept
- Document the boundary between what provenance the extraction pipeline can produce and what the contract requires
- Produce a provenance specification: what fields must exist, what values are required vs optional, what null/absent means for an evidence item
- Register the gap as a formal contract deviation with a severity classification

**Architectural significance:** PH5.3 produces the specification that PH5.4–PH5.7 depend on. Without a formal provenance specification, citation integrity, grounding, and reproducibility audits have no stable reference point.

---

### PH5.4 — Ranking Stability Audit

**Objective:** Characterize the stability gap introduced by JSONL iteration order and the LLMReranker, and define a canonical execution path.

**Scope:**
- Measure how many evidence items are affected by JSONL iteration order dependency in a tiebreak scenario on a canonical corpus (ENG-002 SMR knowledge base)
- Characterize LLMReranker variance: run on identical inputs N times and measure rank-order distance between runs
- Define the canonical execution path for reproducible retrieval: which retrieval mode, which reranker, which corpus state
- Audit whether subquestion expansion ordering creates reproducibility sensitivity given PlannerAgent's planning cache
- Produce a stability specification: what retrieval configuration guarantees stable ranking, and what is out-of-scope for the stability contract

**Architectural significance:** PH5.4 resolves the latent instability risk. Without a defined canonical execution path, any pipeline improvement that enables the LLMReranker will silently break reproducibility. PH5.4 makes the stable path explicit and auditable.

---

### PH5.5 — Retrieval Completeness Audit

**Objective:** Validate coverage tier calibration and address the `topics: []` area-mapping impairment.

**Scope:**
- Audit the STRONG≥4 / MODERATE≥2 / WEAK≥1 coverage thresholds against evidence quality on the ENG-002 canonical engagement
- Measure how many evidence items fail area mapping due to `topics: []` on the KB path — characterize the scope of the impairment
- Determine whether area mapping via category + claim (without topics) produces acceptable coverage or systematically undercounts specific areas
- Audit unmapped evidence rate: what fraction of retrieved items go to `_unmapped` and are excluded from subquestion and area coverage counts
- Produce a completeness specification: minimum coverage thresholds, area-mapping accuracy criteria, and unmapped item tolerance

**Architectural significance:** PH5.5 validates whether the coverage signals that ResearchGapAgent, QAAgent, and IterationPlanAgent rely on are trustworthy. If coverage tiers are miscalibrated or systematically impaired by the `topics: []` gap, all three downstream agents reason on bad inputs — and their validated fingerprints (PH4 and PH5.1) guarantee reproducible wrong answers.

---

### PH5.6 — Grounding Audit

**Objective:** Define what "grounded" means at the evidence layer and assess evidence-to-reasoning traceability.

**Scope:**
- Define a formal grounding criterion: what conditions must an evidence item satisfy for a downstream finding to be considered evidence-grounded
- Audit the token-overlap subquestion assignment mechanism against a semantic relevance baseline: how often does the winner-take-all assignment correctly assign evidence to the most relevant subquestion
- Measure the unmapped item rate and its impact on grounding counts in the ReportAgent
- Audit whether MODERATE/STRONG coverage subquestions reliably produce at least one finding with a citation in the final report
- Produce a grounding specification: minimum grounding requirements for findings, recommendations, and hypotheses

**Architectural significance:** PH5.6 connects the evidence layer to the reasoning and decision layers. Until grounding is specified at the evidence boundary — not just measured at the report — the pipeline has no mechanism to prevent unsupported reasoning from reaching the final output. This is the contract that matters most for audit and governance use cases.

---

### PH5.7 — Reproducibility Contract

**Objective:** Register a canonical evidence fingerprint for a fixed corpus and define the reproducibility boundary for the Knowledge Layer.

**Scope:**
- Register a canonical SHA-256 fingerprint for the EvidenceAgent output on the ENG-002 SMR engagement with the canonical frozen corpus
- Define what "reproducible" means for the Knowledge Layer: the canonical execution path (retrieval mode, reranker setting, corpus state, embedding version) that produces a stable fingerprint
- Define the reproducibility boundary: what the pipeline guarantees to reproduce (ranked set, coverage tiers, source attribution) and what is explicitly out of scope (LLMReranker output, narrative phrasing in subsequent agents)
- Extend the PH5 validation methodology to the KB path: freeze corpus, run N times, compare evidence fingerprints
- Produce a reproducibility specification aligned with PH5.3 (provenance), PH5.4 (ranking), and PH5.5 (completeness) specifications

**Architectural significance:** PH5.7 closes the validation gap for the Knowledge Layer that PH4 and PH5.1 closed for the Planning Layer and YES-deterministic agents. Without a registered canonical fingerprint, the Knowledge Layer has no formal reproducibility claim — only empirical confidence that re-running a fixed corpus produces similar results.

---

*End of KNOWLEDGE_LAYER_GAP_ANALYSIS.md*
