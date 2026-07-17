# AGENT_ARCHITECTURAL_CONTRACTS.md

Strategic Research Harness — Agent Architectural Contract Registry  
Classification Date: 2026-07-16  
Pipeline version: J12.2 (22 agents)

---

## Agent Contract Table

Pipeline order follows orchestrator execution sequence.

| # | Agent | Primary Responsibility | Architectural Layer | Input Artifact | Output Artifact | Role | Expected Behavioral Contract | Should Be Deterministic? | Primary Validation Property |
|---|-------|----------------------|---------------------|---------------|----------------|------|------------------------------|--------------------------|----------------------------|
| 1 | ProblemFramingAgent | Converts strategic engagement / business goal into structured Decision Model | Planning | Strategic Engagement (goal, profiles, engagement spec) | Decision Model (objective, decision areas, critical uncertainties, research questions) | Compiler | Deterministic transformation | **YES** — same goal must always produce the same Decision Model; PlanningCache + temperature=0.0 implemented | Determinism |
| 2 | ResearchStrategyAgent | Converts Decision Model into prioritized research strategy | Planning | Decision Model | Research Strategy (profile priorities, research question priorities, required evidence types) | Compiler | Deterministic transformation | **YES** — same Decision Model must always produce the same strategy; PlanningCache + temperature=0.0 implemented | Determinism |
| 3 | PlannerAgent | Classifies research type and decomposes question into subquestions and investigation areas | Planning | Research question, profiles context, Decision Model, Research Strategy | Execution Plan (research_type, subquestions, investigation_areas) | Compiler | Deterministic transformation | **YES** — same research strategy must always produce the same plan; PlanningCache + temperature=0.0 implemented | Determinism |
| 4 | EvidenceAgent | Retrieves, validates, and organizes evidence from the knowledge base around the research plan | Knowledge | Execution Plan (subquestions, investigation_areas), domain profiles | Evidence Notes (evidence items mapped to subquestions and investigation areas, with source attribution) | Knowledge | Retrieval completeness | **PARTIALLY** — KB retrieval is stable for a fixed corpus; evidence ordering and coverage quality may vary across corpus versions or retrieval path changes | Retrieval completeness |
| 5 | HypothesisAgent | Generates competing strategic hypotheses from evidence and decision context | Reasoning | Evidence Notes, Execution Plan, Decision Model | Hypotheses (competing interpretations with supporting/contradicting evidence, evidence gaps, confidence) | Reasoning | Controlled creativity | **NO** — competing hypotheses are valid strategic interpretations; diversity is a quality signal, not a defect | Hypothesis quality |
| 6 | ResearchGapAgent | Deterministic research completeness and decision-support assessment using pure heuristics | Reasoning | Evidence Notes, Execution Plan, Hypotheses, Research Object | Research Gap Analysis (coverage scores, weak subquestions, contradiction flags, decision-support gaps) | Reasoning | Deterministic transformation | **YES** — no LLM calls; pure heuristic scoring over pipeline context fields; identical inputs must produce identical gap report | Determinism |
| 7 | StrategicSynthesisAgent | Integrates per-domain reasoning into a single executive strategic perspective | Reasoning | Evidence Notes, Hypotheses, Research Gap Analysis, Execution Plan | Strategic Synthesis (executive narrative integrating all domain evidence) | Reasoning | Robust synthesis | **NO** — synthesis requires creative integration across domains; narrative voice and framing are intentionally variable | Grounding |
| 8 | ChallengeAgent | Adversarially challenges each hypothesis to surface hidden assumptions and weak evidence | Reasoning | Hypotheses, Evidence Notes | Challenges (per-hypothesis: challenge summary, hidden assumptions, weak evidence, counter-evidence, survivability rating) | Reasoning | Controlled creativity | **NO** — adversarial framing requires genuine creative judgment; the same hypothesis may yield different but equally valid challenge angles | Coverage |
| 9 | AssumptionAgent | Derives strategic assumptions from challenged hypotheses — conditions that must hold for recommendations to remain valid | Reasoning | Challenged Hypotheses, Evidence Notes | Decision Assumptions (evidence-supported, confidence-rated, conflict-flagged) | Reasoning | Evidence provenance | **PARTIALLY** — same challenged hypothesis set should yield structurally equivalent assumptions; exact wording and ordering vary | Evidence provenance |
| 10 | RecommendationAgent | Derives actionable recommendations from surviving hypotheses and challenges | Decision | Hypotheses, Challenges, Assumptions, Evidence Notes | Recommendations (prioritized, time-horizoned, hypothesis-linked, evidence-supported) | Decision | Robust synthesis | **NO** — recommendation framing requires strategic judgment; different valid framings exist for the same evidence base | Grounding |
| 11 | RiskAgent | Produces strategic risks from assumptions — events that could cause assumptions to fail | Decision | Assumptions, Recommendations | Strategic Risks (linked to assumptions and recommendations, with severity and likelihood) | Decision | Evidence provenance | **PARTIALLY** — same assumption set should yield equivalent risk categories; severity scores and framing may vary | Evidence provenance |
| 12 | OpportunityAgent | Produces strategic opportunities from upside assumption scenarios | Decision | Assumptions, Recommendations | Strategic Opportunities (linked to assumptions and enabled recommendations) | Decision | Controlled creativity | **NO** — opportunity identification requires creative extrapolation beyond the evidence base | Grounding |
| 13 | MultiProfileAgent | Validates and propagates profile attribution from evidence through hypotheses to recommendations | Knowledge | Evidence Notes (with source_profile), Hypotheses, Recommendations, Profiles | Multi-Profile Analysis (cross-profile traceability, attribution completeness) | Knowledge | Evidence provenance | **PARTIALLY** — provenance tracing over fixed inputs is stable; LLM-driven attribution narrative may vary | Evidence provenance |
| 14 | ScenarioAgent | Generates plausible scenarios and stress-tests recommendations under each | Reasoning | Recommendations, Research Object | Scenarios, Scenario Analysis, Scenario Validation (per-recommendation survivability under each scenario) | Reasoning | Controlled creativity | **NO** — scenario construction is genuinely creative; different plausible scenarios are valid | Recommendation robustness |
| 15 | QAAgent | Validates research coverage, evidence sufficiency, contradictions, and recommendation quality; produces confidence assessment | Knowledge | Execution Plan, Evidence Notes, Research Object | QA Results, QA Notes, Recommendation Evaluation, Confidence Assessment | Knowledge | Consistent scoring | **PARTIALLY** — coverage checks and contradiction detection are structurally stable; confidence rating and recommendation scores may shift on regeneration | Consistent scoring |
| 16 | RecommendationImprovementAgent | Improves weak recommendations using QA evaluator feedback | Decision | Recommendations, QA Recommendation Evaluation | Improved Recommendations (addressed weaknesses, strengthened evidence links) | Decision | Controlled creativity | **NO** — improvement requires generative judgment within evidence constraints | Grounding |
| 17 | RecommendationSynthesisAgent | Forces cross-profile integration into the final recommendation set | Decision | Improved Recommendations, Multi-Profile Analysis, Profiles | Enriched Recommendations (cross-profile attribution added, integration verified) | Decision | Evidence provenance | **PARTIALLY** — integration logic is LLM-driven; same inputs may yield equivalent but non-identical attribution narratives | Evidence provenance |
| 18 | StrategicOptionAgent | Produces ~3 genuinely distinct strategic options as coherent postures | Decision | Assumptions, Risks, Opportunities, Recommendations (full J7 graph) | Strategic Options (~3, with implementation complexity, time horizon, capital requirements, linked artifacts) | Decision | Controlled creativity | **NO** — strategic option design requires creative synthesis; distinctness is a first-class quality requirement | Recommendation robustness |
| 19 | DecisionAnalysisAgent | Produces explicit decision analysis — rates options across 10 dimensions, ranks options, surfaces tradeoffs | Decision | Strategic Options, Assumptions, Risks, Opportunities, Recommendations, Decision Model | Decision Analysis (dimension ratings, explicit ranking, tradeoffs, sensitivity to assumption failures) | Decision | Consistent scoring | **PARTIALLY** — same option set should produce stable ranking; dimension scores may shift; the ranking is the primary stability guarantee | Ranking stability |
| 20 | ExecutiveConfidenceAgent | Produces executive confidence assessment from the full J7 graph | Decision | Full J7 context (all prior outputs) | Executive Confidence (High/Medium/Low rating, decision readiness, board recommendation, due diligence checklist, critical unknowns) | Decision | Consistent scoring | **PARTIALLY** — confidence rating should be stable for the same evidence base; narrative framing may vary | Consistent scoring |
| 21 | IterationPlanAgent | Deterministic iteration planning — converts validation priorities and gaps into ranked IRT tasks | Planning | Executive Confidence, Assumptions, Recommendations, Risks, Strategic Options, Decision Analysis, Research Gap Analysis | Iteration Plan (ranked IRT tasks with urgency, expected confidence gain, artifact links) | Compiler | Deterministic transformation | **YES** — no LLM calls; pure priority-scoring heuristics over structured inputs; identical inputs must produce identical plan | Determinism |
| 22 | ReportAgent | Synthesizes all pipeline outputs into executive communication artifacts | Communication | Full pipeline context (all prior outputs, orchestrator summary) | Executive Report (executive summary, key findings, key risks, open questions, report confidence, evidence traceability) | Communication | Executive communication | **NO** — executive narrative synthesis is intentionally expressive; quality and tone are not reproducibility targets | Executive explainability |

---

## Determinism Rationale by Category

### YES — Fully Deterministic (4 agents)

**ProblemFramingAgent, ResearchStrategyAgent, PlannerAgent** — Planning compilers. These are structural transformations: the output is uniquely determined by the input. Non-determinism in these agents would silently corrupt all downstream artifacts. PlanningCache + temperature=0.0 provides the guarantee. Validated in PH4.1–PH4.3.

**ResearchGapAgent, IterationPlanAgent** — No LLM calls. Pure heuristic scoring over typed context fields. Determinism is an intrinsic property of the implementation, not a configuration choice. No cache is needed; the same inputs always produce the same output by construction.

### NO — Generative (7 agents)

**HypothesisAgent, ChallengeAgent, StrategicSynthesisAgent, OpportunityAgent, RecommendationAgent, StrategicOptionAgent, RecommendationImprovementAgent, ReportAgent** — These agents perform genuine creative or strategic synthesis. Reproducibility is not a goal. Hypothesis diversity, recommendation framing quality, scenario plausibility, and narrative voice are quality signals that require LLM variability. Constraining these to deterministic outputs would reduce strategic value.

### PARTIALLY — Scoring or Provenance Stable (7 agents)

**AssumptionAgent, EvidenceAgent, MultiProfileAgent, QAAgent, RiskAgent, RecommendationSynthesisAgent, DecisionAnalysisAgent, ExecutiveConfidenceAgent** — These agents produce structured assessments where the categorical result (coverage level, risk severity, ranking, confidence tier) should be stable across runs for the same input, but the exact wording, ordering within a tier, or narrative explanation may vary. The primary validation target is the structural output (ranking, scores, traceability links), not byte-identical reproduction.

---

## Conclusions

### 1. Which agents should be validated using the PH4 fingerprint methodology?

The PH4 fingerprint methodology (SHA-256 over canonical JSON of output fields, validated across N independent runs with PlanningCache) is appropriate for all **YES — Fully Deterministic** agents:

| Agent | Status | Notes |
|-------|--------|-------|
| ProblemFramingAgent | Validated (PH4.1) | Fingerprint confirmed across 5 archetypes |
| ResearchStrategyAgent | Validated (PH4.2) | Fingerprint confirmed for ENG-002 frozen Decision Model |
| PlannerAgent | Validated (PH4.3) | Fingerprint confirmed for ENG-002 frozen Research Strategy |
| ResearchGapAgent | Validated (PH5.1) | Fingerprint 3ccbd120… confirmed across 5 runs for ENG-002 frozen planner |
| IterationPlanAgent | Validated (PH5.1) | Fingerprint a4e28499… confirmed across 5 runs for ENG-002 frozen context |

### 2. Which agents require a different validation methodology?

| Agent Category | Validation Methodology |
|---------------|----------------------|
| **EvidenceAgent** | Retrieval completeness audit — measure coverage rate across subquestions; validate that evidence items have correct source_profile and evidence_ids; check that no subquestion has NONE coverage for a question that has known KB coverage |
| **HypothesisAgent** | Hypothesis quality review — validate that each hypothesis references real evidence_ids; validate competing hypothesis diversity (at least two distinct strategic interpretations); validate no unsupported high-confidence hypotheses |
| **ChallengeAgent** | Coverage audit — validate that every hypothesis has at least one challenge; validate that hidden_assumptions are non-trivial; validate challenge quality against weak-evidence flag |
| **AssumptionAgent, RiskAgent, OpportunityAgent** | Evidence provenance audit — validate that every output item references valid evidence_ids or assumption IDs from the pipeline; validate traceability links are non-empty and point to real artifacts |
| **RecommendationAgent, RecommendationImprovementAgent** | Grounding audit — validate that each recommendation traces to at least one hypothesis; validate that priority ratings are consistent with evidence strength |
| **QAAgent** | Scoring consistency test — run QA on the same frozen context twice; compare coverage scores and confidence tier; accept ±5% variance on numeric scores, require identical categorical tier (High/Medium/Low) |
| **DecisionAnalysisAgent** | Ranking stability test — run decision analysis on the same frozen Strategic Options twice; require that the top-ranked option is identical across runs; accept ±1 position variance for non-preferred options |
| **ExecutiveConfidenceAgent** | Scoring consistency test — run on the same frozen full context twice; require identical confidence tier (High/Medium/Low) and identical decision_readiness flag; accept narrative variance |
| **StrategicOptionAgent, ScenarioAgent, StrategicSynthesisAgent** | Structural completeness audit — validate artifact schema completeness, option count (target ~3), scenario count, evidence coverage; do not validate content stability |
| **MultiProfileAgent, RecommendationSynthesisAgent** | Provenance completeness audit — validate that cross-profile attribution covers all loaded profiles; validate that profile-specific evidence is traceable to at least one recommendation |
| **ReportAgent** | Executive communication audit — validate that executive_summary is non-empty, key_findings is non-empty, report_confidence is present and traces to QA output; validate that supporting_evidence_ids are present on each key finding |

### 3. Which agent should be investigated next?

**EvidenceAgent (Knowledge Layer).**

All YES-deterministic agents are now validated (PH5.1). The next investigation target is the EvidenceAgent, which is the entry point to the Knowledge Layer. Unlike the Planning Layer and the pure heuristic agents, EvidenceAgent interacts with external retrieval systems and requires a different validation methodology: retrieval completeness audit, provenance audit, and ranking stability — as specified in KNOWLEDGE_LAYER_ARCHITECTURAL_CONTRACT.md.

This investigation defines PH5.x (Knowledge Layer validation).
