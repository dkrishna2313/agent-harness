# Architecture Object Model — Canonical Objects & Information Flow

**Phase A · Review 3 — Inventory & Classification**

Status: inventory only. This document catalogues the platform's architecturally
significant objects and classifies each. It proposes **no** changes. All
file:line references reflect `main` at the J10.8 checkpoint.

---

## Classification legend

**Type**
- **Canonical Business Object (CBO)** — a first-class object the platform is
  *about*; has identity, a defined lifecycle, and is (usually) persisted.
- **Working / Transport Object (WTO)** — carries state between stages within a
  single run; not itself a durable business record.
- **Derived Object (DO)** — computed/rendered from other objects; can be
  regenerated and holds no independent authority.

**Lifetime**
- **Persistent** — written to disk (`knowledge_store/` or `outputs/`), survives
  the run.
- **Engagement-scoped** — one per Strategic Engagement / research run; persisted
  as a run artifact.
- **Execution-scoped** — in-memory only, discarded when the process ends.

---

## Layer map (information flow)

```
KNOWLEDGE (persistent)        Source → Evidence (+KnowledgeMetadata) → Contradiction
                                     ↑ ExtractionRun (provenance)   ↑ Embedding index
                                     │
RESEARCH / RETRIEVAL          Document → Chunk → EvidenceItem → ResearchMemo → Research Object
                                     │ (RetrievalResult when KB-backed)
                                     │
STRATEGIC (engagement run)    Strategic Engagement → Executive Framing → Decision Architecture
                                → Decision Domains → Reasoning Targets
                                → Planner Output → Evidence Output → Hypothesis Output
                                → Strategic Synthesis → Recommendation
                                → Decision Analysis → Executive Confidence   (all on Decision Model v2)
                                     │
EXECUTION                     AgentContext (carries all of the above) · Functional Agent Contract
                                     │
COMMUNICATION                 Executive Report (markdown) · Trace Output (.trace.json)
```

---

# 1. Knowledge Layer

### 1.1 Knowledge Source
- **Name / class:** `Source` — `knowledge/models.py:62`
- **Purpose:** An original document exactly as ingested from the world; content-addressed by SHA-256 `fingerprint`, written once and never modified.
- **Primary owner:** `KnowledgeBuilder` (`knowledge/builder.py`) at ingestion.
- **Primary consumers:** `EvidenceRetriever` (source-loading), Evidence provenance (`supporting_source_ids`), citation rendering.
- **Lifetime:** **Persistent** — `knowledge_store/sources/{domain}/{source_id}.json`.
- **Upstream:** raw input files (ingestion).
- **Downstream:** Evidence (via ExtractionRun), Citation markers.
- **Type:** **Canonical Business Object.**
- **Notes:** Distinct from `SourceDocument` (§2.1). `Source` is the durable KB record; `SourceDocument` is the transient loaded-file form used by the legacy/research path. Overlap is intentional (two layers), but the names are easy to confuse.

### 1.2 Document
- **Name / class:** `SourceDocument` — `research_agent/schemas.py:62`
- **Purpose:** Extracted text + metadata for a local source file during a research run (frozen record).
- **Primary owner:** `research_agent/loaders.py` (`load_sources`).
- **Primary consumers:** `chunk_documents()`, legacy `DcPowerAgent` extraction, trace builder.
- **Lifetime:** **Execution-scoped** (in-memory; loaded from filesystem per run).
- **Upstream:** filesystem source files.
- **Downstream:** Chunk, EvidenceItem (legacy path), Trace `documents`.
- **Type:** **Working / Transport Object.**
- **Notes:** Only used by the legacy document-extraction path; the Knowledge-Layer path retrieves pre-extracted Evidence and does not load Documents.

### 1.3 Chunk
- **Name / class:** `Chunk` — `research_agent/schemas.py:86`
- **Purpose:** Fixed-size text slice of a Document (with web-retrieval provenance) fed to evidence extraction.
- **Primary owner:** `research_agent/chunker.py` (`chunk_documents`).
- **Primary consumers:** `select_top_chunks_multi()`, `extract_evidence_from_chunks()`.
- **Lifetime:** **Execution-scoped** (in-memory builder artifact).
- **Upstream:** Document (§1.2).
- **Downstream:** EvidenceItem.
- **Type:** **Working / Transport Object.**
- **Notes:** Per the J8.0 frozen decision, a Chunk is *a builder artifact only, never a KB first-class object*. Confirmed — not persisted to `knowledge_store/`.

### 1.4 Evidence
- **Name / class:** `Evidence` — `knowledge/models.py:110` (canonical); `EvidenceItem` — `research_agent/schemas.py:115` (transient research form).
- **Purpose:** One atomic, source-backed claim. `Evidence` is the append-only KB record; `EvidenceItem` is the richly-annotated in-run form (scores, entity/scope, topics, recovery flags).
- **Primary owner:** `KnowledgeBuilder` (Evidence); `DcPowerAgent`/`EvidenceAgent` (EvidenceItem).
- **Primary consumers:** `EvidenceRetriever`, HypothesisAgent, RecommendationAgent, Citation markers.
- **Lifetime:** `Evidence` **Persistent** — `knowledge_store/evidence/{domain}/evidence.jsonl` (+ `index.json`); `EvidenceItem` **Execution-scoped**.
- **Upstream:** Source + ExtractionRun (Evidence); Chunk (EvidenceItem).
- **Downstream:** KnowledgeMetadata, Contradiction, RetrievalResult, Hypotheses, Recommendations.
- **Type:** `Evidence` = **Canonical Business Object**; `EvidenceItem` = **Working / Transport Object.**
- **Notes / overlap:** The two-model split is deliberate (immutable KB record vs mutable in-run annotations) but is the platform's most significant naming overlap. `EvidenceItem.overall_score` etc. correspond to fields KnowledgeMetadata owns canonically.

### 1.5 KnowledgeMetadata
- **Name / class:** `KnowledgeMetadata` — `knowledge/models.py:147`
- **Purpose:** Mutable lifecycle + quality wrapper for one Evidence record (state ACTIVE/SUPERSEDED/RETRACTED…, scores, review_status, retrieval flags).
- **Primary owner:** `KnowledgeBuilder`; future review tooling.
- **Primary consumers:** `EvidenceRetriever` (retrieval_enabled/priority, scores), health checks.
- **Lifetime:** **Persistent** — `knowledge_store/metadata/{domain}/metadata.jsonl`.
- **Upstream:** Evidence (1:1 by `evidence_id`).
- **Downstream:** retrieval ranking, RetrievedEvidence.
- **Type:** **Canonical Business Object** (the *mutable* half of the Evidence pair, by J8.0 design).
- **Notes:** Separation exists precisely to keep Evidence immutable while quality/lifecycle evolves.

### 1.6 Citation
- **Name / class:** *No dedicated class.* Inline marker string `[Source: <document>, Evidence: <evidence_id>]`, generated by `_citation_marker`/`_format_citations` (`research_agent/agent.py:1302`, `functional_agents/report_agent.py`).
- **Purpose:** Ties a rendered claim back to its source + evidence for traceability and scoring.
- **Primary owner:** synthesis/report rendering.
- **Primary consumers:** benchmark scorer (`eval_runner.py:23` regex), human readers, `report_grounding_score` citation count.
- **Lifetime:** **Execution-scoped** (embedded in memo/report text; persists only as substrings of the report artifact).
- **Upstream:** Evidence / EvidenceItem (`source_document`, `evidence_id`).
- **Downstream:** Executive Report, Trace `citation_count`.
- **Type:** **Derived Object.**
- **Notes / ambiguity:** Citations are a *string convention*, not a modelled object. Counting is marker-based (occurrences of `[Source:`), not unique-source-based — flagged in J8.10. This is the main "object that architecturally exists but is not typed."

### 1.7 ExtractionRun
- **Name / class:** `ExtractionRun` — `knowledge/models.py:183`
- **Purpose:** Records parameters + outcome of one KnowledgeBuilder execution; provides provenance for Evidence construction/supersession (narrow, not a general audit log — J8.0).
- **Primary owner:** `KnowledgeBuilder`.
- **Primary consumers:** Evidence (`extraction_run_id`), build reporting.
- **Lifetime:** **Persistent** — `knowledge_store/extraction_runs/runs.jsonl`.
- **Type:** **Canonical Business Object** (provenance record).

### 1.8 Contradiction
- **Name / class:** `Contradiction` — `knowledge/models.py:213` (canonical) and `research_agent/schemas.py:168` (research-stage).
- **Purpose:** A known conflict between two Evidence records. KB form is computed offline and stored; research form is the rich in-run detection trace.
- **Primary owner:** `KnowledgeBuilder` (KB); contradiction detection in the research pipeline (research form).
- **Primary consumers:** RecommendationAgent (validated contradictions), QA, trace.
- **Lifetime:** KB form **Persistent** — `knowledge_store/contradictions/contradictions.jsonl`; research form **Execution-scoped**.
- **Type:** KB form = **Canonical Business Object**; research form = **Working / Transport Object.**
- **Notes:** Same conceptual object, two representations (mirrors the Evidence/EvidenceItem split).

### 1.9 Embedding Index *(supporting)*
- **Purpose:** Evidence-level vectors for semantic/hybrid retrieval (J8.0 Option C).
- **Owner/consumer:** `KnowledgeBuilder` / `EvidenceRetriever` (+ `LocalEmbeddingProvider`).
- **Lifetime:** **Persistent** — `knowledge_store/embeddings/`.
- **Type:** **Derived Object** (regenerable from Evidence).

---

# 2. Research / Retrieval

### 2.1 Research Object
- **Name / class:** dict from `create_research_object()` — `research_agent/research_object.py:157`
- **Purpose:** The durable per-run record: question, profiles, run_config, evidence_ids, findings, contradictions, decision_model/decision_architecture, report block, and engagement/decision-model linkage.
- **Primary owner:** Orchestrator (creates it); mutated in-place by ~8 agents; finalised by `update_research_object()` in ReportAgent.
- **Primary consumers:** ReportAgent, benchmark harness, persistence, downstream analytics.
- **Lifetime:** **Persistent / Engagement-scoped** — `outputs/research_objects/{research_id}.json` (+ `latest_research_object.json`).
- **Upstream:** question/goal/engagement, all agent outputs.
- **Downstream:** Executive Report, Trace `research_object` block.
- **Type:** **Canonical Business Object.**
- **Notes / ambiguity:** It is a plain `dict`, not a Pydantic model — the largest CBO with the weakest schema guarantee. Many writers, no formal contract; carries both durable business state and some derived fields (e.g. `findings`).

### 2.2 Research Question
- **Name / representation:** *No standalone object.* Appears as `DecisionModelPayload.research_questions` (`claude_client.py:87`), `context.question` (primary), `plan["subquestions"]`, `research_object["subquestions"]`, and — since J9.2 — as children of `DecisionStream.research_questions`.
- **Purpose:** The analytical question(s) driving retrieval and reasoning.
- **Owner:** ProblemFramingAgent (produces); PlannerAgent (decomposes).
- **Consumers:** Planner, Evidence, QA, Report (via `context.question`).
- **Lifetime:** **Execution-scoped** (values persist inside Research Object / Decision Architecture).
- **Type:** **Derived Object** (a field/value, not a modelled entity).
- **Notes / ambiguity:** Represented in ≥4 places; only `research_questions[0]`→`context.question` drives the primary execution path. J10.3 reframed them as children of Decision Domains, so the same values now live in two structures (decision_model list *and* decision_stream lists).

### 2.3 Research Findings
- **Name / representation:** *No standalone object.* `research_object["findings"]` derived from `ResearchMemo.confirmed_facts` (top-5, citation-stripped) in `update_research_object()` (`research_object.py:276`).
- **Purpose:** Human-facing summary of what the evidence established.
- **Owner:** synthesis (ResearchMemo) → `update_research_object`.
- **Consumers:** Executive Report, Research Object readers.
- **Lifetime:** **Persistent** (as a Research Object field).
- **Type:** **Derived Object.**

### 2.4 Research Memo *(carrier)*
- **Name / class:** `ResearchMemo` — `research_agent/schemas.py:254`
- **Purpose:** Structured synthesis output (executive_summary, confirmed_facts, inferences, domain implications, open_questions) bridging evidence → report.
- **Owner:** `synthesize_memo()` (legacy) / synthetic memo builder (KB path, `EvidenceAgent._build_synthetic_memo`).
- **Consumers:** ReportAgent, `evaluate_memo`, Trace.
- **Lifetime:** **Execution-scoped** (carried in `context.trace["_memo"]`).
- **Type:** **Working / Transport Object.**

### 2.5 RetrievalResult *(carrier)*
- **Name / class:** `RetrievalResult` (+ `RetrievedEvidence`) — `knowledge/retriever.py:182` / `:151`
- **Purpose:** Output of `EvidenceRetriever.retrieve()` — ranked evidence + retrieval diagnostics (candidates, latency, mode, semantic model).
- **Owner:** `EvidenceRetriever`.
- **Consumers:** `EvidenceAgent._execute_kb`, `LLMReranker`.
- **Lifetime:** **Execution-scoped.**
- **Type:** **Working / Transport Object.**

---

# 3. Strategic

### 3.1 Strategic Engagement
- **Name / class:** `EngagementSpec` (input) — `functional_agents/engagement_spec.py`; `StrategicEngagement` (runtime record) — `research_agent/engagement.py:54`
- **Purpose:** The consulting brief the platform runs from. `EngagementSpec` is the validated YAML/JSON input contract; `StrategicEngagement` is the persisted runtime record that links Decision Model + Research Objects.
- **Primary owner:** CLI/loader (EngagementSpec); Orchestrator (StrategicEngagement).
- **Primary consumers:** ProblemFramingAgent (framing), engagement linkage.
- **Lifetime:** `StrategicEngagement` **Persistent** — `outputs/engagements/{engagement_id}.json` (+ `latest_engagement.json`); `EngagementSpec` **Execution-scoped** (also lives on `AgentContext.engagement` as a dict).
- **Upstream:** client-authored engagement file (or auto-created from a question).
- **Downstream:** Decision Model (`engagement_id`), Decision Architecture, Research Object.
- **Type:** **Canonical Business Object** (top of the hierarchy).
- **Notes / ambiguity:** Two classes for one concept — input contract vs runtime record. They do not share a schema; `_engagement_from_spec` maps one to the other. This is a deliberate but noteworthy duplication.

### 3.2 Executive Framing
- **Name / representation:** A *process/stage*, not a stored object. Implemented by `ProblemFramingAgent._build_executive_architecture` via `ClaudeClient.frame_executive_decision` → `DecisionArchitecturePayload` (`claude_client.py`), with a deterministic builder fallback.
- **Purpose:** Reason an engagement into an executive decision framing (J9.3).
- **Owner:** ProblemFramingAgent.
- **Output:** Decision Architecture (§3.3).
- **Lifetime:** **Execution-scoped** (its product persists).
- **Type:** **Derived Object** (a transformation; its output is the durable artifact).

### 3.3 Decision Architecture
- **Name / class:** `DecisionArchitecture` (+ `DecisionStream`, `DecisionScope`) — `functional_agents/decision_architecture.py`; persisted as a dict on `DecisionModel.decision_architecture` (`decision_model.py:356`) and `research_object["decision_architecture"]`; `context.decision_architecture`.
- **Purpose:** Executive framing of the engagement: decision_statement, scope, success_definition, strategic_themes, decision_streams, executive_unknowns, board_decisions_required.
- **Primary owner:** ProblemFramingAgent.
- **Primary consumers:** `get_reasoning_targets()` (produces Decision Domains), StrategicSynthesisAgent, ReportAgent Section 1.
- **Lifetime:** **Persistent** (on Decision Model + Research Object) / **Engagement-scoped**.
- **Upstream:** Executive Framing (engagement + framing payload).
- **Downstream:** Reasoning Targets, Strategic Synthesis, Executive Report headline.
- **Type:** **Canonical Business Object** (the executive spine introduced in J9.2).

### 3.4 Decision Domain
- **Name / representation:** `DecisionStream` inside Decision Architecture (`decision_architecture.py`); surfaced as `ReasoningTarget(kind="decision_domain")`.
- **Purpose:** An executive workstream (title, executive_objective, related themes, child research_questions, expected_outputs) — the unit the J10 reasoning spine iterates over.
- **Owner:** Decision Architecture builder.
- **Consumers:** `get_reasoning_targets()`, and (per-domain) Planner/Evidence/Hypothesis via the `domain_*` collections.
- **Lifetime:** **Persistent** (as part of Decision Architecture).
- **Type:** **Canonical Business Object** (structural sub-object of Decision Architecture; the J10 "primary unit of reasoning").
- **Notes:** Not independently persisted or identified beyond `domain-N`; exists only within Decision Architecture.

### 3.5 Reasoning Target
- **Name / class:** `ReasoningTarget` — `functional_agents/reasoning_target.py`; produced by `AgentContext.get_reasoning_targets()`.
- **Purpose:** The compatibility seam (J10.1) between the reasoning spine and downstream agents: one target from `context.question` (research mode) or one per Decision Domain (engagement mode).
- **Owner:** `AgentContext`.
- **Consumers:** PlannerAgent (J10.2+).
- **Lifetime:** **Execution-scoped** (computed on demand; not persisted).
- **Upstream:** `context.question` / Decision Architecture streams.
- **Downstream:** Planner Output.
- **Type:** **Working / Transport Object.**

### 3.6 Planner Output
- **Name / representation:** `context.plan` (primary) + `context.domain_plans` (per-domain, J10.4); `ResearchPlanningPayload` (`claude_client.py`).
- **Purpose:** research_type, subquestions, investigation_areas, reasoning for the primary target and each Decision Domain.
- **Owner:** PlannerAgent.
- **Consumers:** EvidenceAgent, QA, ReportAgent (primary plan → RO fields).
- **Lifetime:** **Execution-scoped** (primary fields mirrored into Research Object).
- **Type:** **Working / Transport Object** (`domain_plans` are organizational-only).

### 3.7 Evidence Output
- **Name / representation:** `context.evidence_notes` (primary) + `context.domain_evidence` (per-domain, J10.5).
- **Purpose:** Retrieved/mapped evidence collection + coverage per target/domain.
- **Owner:** EvidenceAgent.
- **Consumers:** HypothesisAgent, RecommendationAgent, QA, Report.
- **Lifetime:** **Execution-scoped** (primary summary mirrored into Research Object).
- **Type:** **Working / Transport Object.**

### 3.8 Hypothesis Output
- **Name / representation:** `context.hypotheses` (primary) + `context.domain_hypotheses` (per-domain, J10.6); `HypothesisPayload`/`HypothesisItem` (`claude_client.py`).
- **Purpose:** Competing hypotheses with evidence links, confidence, decision implications.
- **Owner:** HypothesisAgent.
- **Consumers:** ChallengeAgent, RecommendationAgent, StrategicSynthesisAgent.
- **Lifetime:** **Execution-scoped** (persisted into `research_object["hypotheses"]`).
- **Type:** **Working / Transport Object.**

### 3.9 Strategic Synthesis
- **Name / class:** `StrategicSynthesisPayload` (`claude_client.py`); `context.strategic_synthesis`; `research_object["strategic_synthesis"]`.
- **Purpose:** Cross-domain executive integration (J10.7): executive_summary, cross_domain_findings/dependencies/conflicts, strategic_levers, dominant_constraints, emerging_themes. Executive reasoning only — no recommendations.
- **Owner:** StrategicSynthesisAgent.
- **Consumers:** RecommendationAgent (J10.8, bounded prompt section), trace.
- **Lifetime:** **Persistent** (on Research Object) / **Engagement-scoped**.
- **Upstream:** domain_plans, domain_evidence, domain_hypotheses, Decision Architecture.
- **Downstream:** Recommendations (reasoning influence, not evidence).
- **Type:** **Derived Object** (integration of prior reasoning) — arguably a CBO as the first executive-reasoning artifact; classified Derived because it is regenerable from its inputs and holds no linkage identity.

### 3.10 Recommendation
- **Name / class:** `RecommendationItem` / `RecommendationPayload` / `RecommendationPortfolio` (`claude_client.py`); `context.recommendations`, `context.recommendation_portfolio`; `research_object["recommendations"]`.
- **Purpose:** Actionable recommendations grounded in surviving hypotheses + evidence, classified by horizon/priority, now shaped by Strategic Synthesis (J10.8).
- **Owner:** RecommendationAgent (+ Improvement/Synthesis refiners downstream).
- **Consumers:** StrategicOptionAgent, DecisionAnalysisAgent, ExecutiveConfidenceAgent, ReportAgent.
- **Lifetime:** **Persistent** (Research Object; linked into Decision Model options).
- **Type:** **Canonical Business Object** (a primary platform output).

### 3.11 Decision Model (v2) *(container)*
- **Name / class:** `DecisionModel` — `research_agent/decision_model.py:326`
- **Purpose:** Canonical decision object linking engagement → strategic reasoning. Holds objectives, criteria, investigation_areas, constraints, out_of_scope, decision_architecture, and the J7 sub-objects (assumptions, risks, opportunities, options, decision_analysis, executive_confidence).
- **Owner:** ProblemFramingAgent (creates); J7 agents append their sub-objects via `model_copy(update=)`.
- **Consumers:** PlannerAgent, ResearchStrategyAgent, ReportAgent, persistence.
- **Lifetime:** **Persistent** — `outputs/decision_models/{id}.json` (+ `latest_decision_model.json`).
- **Type:** **Canonical Business Object** (the strategic aggregate root).
- **Notes:** Bidirectionally linked to Strategic Engagement (`engagement_id` ↔ `decision_model_id`) and referenced by Research Object.

### 3.12 Decision Analysis
- **Name / class:** `DecisionAnalysis` (+ `DecisionMatrixEntry`) — `decision_model.py:270`; field `DecisionModel.decision_analysis`.
- **Purpose:** Explicit comparison of Strategic Options (matrix, rankings, tradeoffs, sensitivity, recommended option) — "why B over A" (J7.6).
- **Owner:** DecisionAnalysisAgent.
- **Consumers:** ExecutiveConfidenceAgent, ReportAgent (executive report).
- **Lifetime:** **Persistent** (on Decision Model).
- **Type:** **Derived Object** (reasoning over the existing graph) persisted on a CBO.

### 3.13 Executive Confidence
- **Name / class:** `ExecutiveConfidence` — `decision_model.py:303`; field `DecisionModel.executive_confidence`.
- **Purpose:** Synthesis over the completed decision graph (J7.7): overall confidence, decision readiness, board recommendation, validation priorities — "should we act now?".
- **Owner:** ExecutiveConfidenceAgent.
- **Consumers:** ReportAgent (executive report).
- **Lifetime:** **Persistent** (on Decision Model).
- **Type:** **Derived Object** persisted on a CBO.
- **Notes:** Along with §3.9/§3.12, part of a family of executive synthesis artifacts that are all derived-but-persisted; the boundary between "derived" and "canonical" is softest here.

---

# 4. Execution

### 4.1 AgentContext
- **Name / class:** `AgentContext` — `functional_agents/context.py:58`
- **Purpose:** The single mutable state object threaded through every functional agent; holds inputs, all per-agent outputs, the `domain_*` collections, `strategic_synthesis`, and a `trace` scratch dict.
- **Primary owner:** Orchestrator (creates once).
- **Primary consumers:** every functional agent.
- **Lifetime:** **Execution-scoped** (its durable contents are copied into Research Object / Decision Model / Trace at the end).
- **Upstream:** engagement/goal/question + profiles.
- **Downstream:** Research Object, Decision Model, Trace, Executive Report.
- **Type:** **Working / Transport Object** (the master transport object).
- **Notes:** `context.trace` mixes durable data with non-serialisable scratch (`_client`, `_perf_tracker`) — the `_`-prefix convention marks what to exclude from serialisation.

### 4.2 Functional Agent Contract
- **Name / representation:** `FunctionalAgent` base (`functional_agents/base.py`), `AgentResult` (`context.py:201`), validators in `functional_agents/contract.py`.
- **Purpose:** The uniform interface: subclasses implement `_execute(ctx)->ctx`; `run()` wraps timing/history/perf and returns an `AgentResult` (`status, next_action, summary, context, outputs, metrics, trace`).
- **Owner:** framework.
- **Consumers:** Orchestrator (routing), contract validation trace block.
- **Lifetime:** N/A (interface); `AgentResult` instances are **Execution-scoped**.
- **Type:** **Contract / Interface** (not a data object) — `AgentResult` itself is a **Working / Transport Object.**

---

# 5. Communication

### 5.1 Executive Report
- **Name / representation:** Markdown produced by `_build_j7_executive_report()` (14-section J7 report) or the legacy memo path (`functional_agents/report_agent.py`); written to the `--out` path.
- **Purpose:** The human-facing deliverable — executive decision statement, options, decision analysis, confidence, supporting evidence.
- **Owner:** ReportAgent.
- **Primary consumers:** end users / stakeholders.
- **Lifetime:** **Persistent** (markdown file on disk).
- **Upstream:** Decision Model, Research Object, Strategic Synthesis, Recommendations, Decision Analysis, Executive Confidence.
- **Downstream:** none (terminal artifact).
- **Type:** **Derived Object.**

### 5.2 Trace Output
- **Name / representation:** `.trace.json` assembled by `build_trace()` (`research_agent/trace.py:40`) plus ~35 additive keys appended in `ReportAgent` (planner, evidence_agent, qa_agent, problem_framing, decision_architecture, reasoning_targets, planner_reasoning, evidence_reasoning, hypothesis_reasoning, strategic_synthesis, recommendation_strategy_context, llm_normalization, performance, contract_validation, …). Per-LLM-call detail = `ClaudeCallTrace` (`schemas.py`).
- **Purpose:** Full-run observability + benchmark scoring input.
- **Owner:** ReportAgent (assembles); every agent contributes via `context.trace["_*"]`.
- **Primary consumers:** benchmark/eval harness, engineers, this review series.
- **Lifetime:** **Persistent** (`.trace.json` next to the report).
- **Type:** **Derived Object.**
- **Notes / ambiguity:** No formal schema (~77 top-level keys, additive by convention); the legacy `build_trace` core is still `ResearchMemo`/document-shaped, with functional richness bolted on afterward. The prompt's "Knowledge Layer → Trace" and "Communication → Trace Output" refer to the same object; `RetrievalResult` (§2.5) is its retrieval-level sub-component.

---

## Cross-cutting observations (ambiguity & overlap only — no changes proposed)

1. **Dual representations for KB concepts.** Evidence/`EvidenceItem` and Contradiction (KB/research) each exist in two forms — immutable canonical vs richly-annotated transient. Deliberate (J8.0), but the shared field names invite confusion.
2. **Strategic Engagement has two classes** — `EngagementSpec` (input) and `StrategicEngagement` (runtime record) — with no shared schema.
3. **Research Object and Trace are the two largest untyped objects** (plain dicts, many writers, additive-by-convention). Highest schema-drift surface.
4. **Citation and Research Question/Findings are values, not modelled entities** — architecturally significant but represented as strings/fields across multiple locations.
5. **Executive synthesis family** (Strategic Synthesis, Decision Analysis, Executive Confidence) are all *derived-but-persisted*; the derived/canonical line is softest here.
6. **Decision Domain** is a first-class reasoning unit (J10) but exists only as a `DecisionStream` sub-structure of Decision Architecture, identified only as `domain-N`.
7. **`domain_*` collections** (plans/evidence/hypotheses) are organizational-only today: every stage keeps a byte-identical *primary* that drives the pipeline; the per-domain sets are carried but not yet independently consumed downstream of their producing stage.

---

*End of inventory. No architectural changes proposed.*
