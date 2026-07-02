# Harness Engineering Checkpoint — J8 Complete
**Checkpoint:** J8.0 – Knowledge Platform, Performance & Traceability Complete
**Date:** 30 June 2026

---

# Executive Summary

J8 completes the transition from a document-centric research pipeline into a **Knowledge-Layer-driven strategic reasoning platform**.

The system now has:

- persistent Knowledge Store
- hybrid semantic + lexical retrieval
- Functional Agent architecture
- strategic reasoning workflow
- Decision Model generation
- Research Object persistence
- performance instrumentation
- benchmark instrumentation
- benchmark optimization
- configurable benchmark extraction model
- end-to-end evidence traceability
- executive strategic reporting

The platform is now considered stable enough to begin **J9 – Strategic Engagement Engine**.

---

# Current Architecture

```
                Source Documents
                      │
                      ▼
            Knowledge Builder
                      │
          Evidence Extraction
                      │
          Embeddings Generated
                      │
                      ▼
             Knowledge Store
        (Evidence + Embeddings)

                      │
                      ▼

             Evidence Retriever
     Hybrid Lexical + Semantic Search
        Optional Claude Reranking

                      │
                      ▼

            Functional Agents

ProblemFraming
ResearchStrategy
Planner
Evidence
Hypothesis
Challenge
Assumption
Recommendation
Risk
Opportunity
StrategicOption
DecisionAnalysis
ExecutiveConfidence
MultiProfile
Scenario
QA
RecommendationImprovement
RecommendationSynthesis
Report

                      │
                      ▼

 Executive Strategic Report
 Decision Model
 Research Object
 Trace
```

---

# J8 Milestones

## J8.6
Knowledge Layer integrated into Functional Agents.

EvidenceAgent now consumes Knowledge Store evidence rather than raw document extraction.

---

## J8.6a

Fixed incorrect zero-evidence fallback.

Discovery:

The retriever worked correctly.

The reranker occasionally eliminated every candidate.

Added safe fallback to retrieval ordering.

---

## J8.7

Knowledge Platform Maturity.

Added:

- Knowledge Store health checks
- build diagnostics
- runtime validation
- CLI health command
- runtime guardrails

---

## J8.7b

Fixed Knowledge Layer retrieval diagnostics.

Added:

- retrieval instrumentation
- candidate counts
- reranking diagnostics

Confirmed:

Functional Agents genuinely consume Knowledge Layer evidence.

---

## J8.8

Performance work for Functional Agents.

### J8.8a

Pipeline instrumentation.

Added:

- per-agent timings
- LLM timings
- prompt/completion tokens
- sub-phase timings
- performance summaries
- trace performance block

Discovery:

Pipeline is overwhelmingly LLM-bound.

---

### J8.8b

Prompt & context optimization.

Reduced:

- prompt size
- completion verbosity
- redundant context

Result:

Functional runtime reduced dramatically while preserving quality.

---

## J8.9

Benchmark optimization.

### J8.9a

Benchmark instrumentation.

Created stable benchmark subset.

Development benchmark:

- NVIDIA_003
- NVIDIA_007
- NVIDIA_010
- NVIDIA_012

Performance metrics now collected for every benchmark question.

---

### J8.9b

Initial benchmark optimization.

Discovered benchmark regressions.

---

### J8.9b1

Root cause analysis.

Fixed:

- unbounded UNION of retrieved chunks
- recovery path producing larger prompts than success path

New invariant:

Recovery must never produce more evidence than successful execution.

---

### J8.9c

Architecture investigation.

Conclusion:

Single extraction call remains correct.

No batching.

No incremental extraction.

Legacy benchmark architecture is transitional.

---

### J8.9d

Haiku extraction experiment.

Result:

Haiku performs benchmark extraction successfully without measurable benchmark degradation on representative benchmark subset.

---

### J8.9e

Added benchmark extraction CLI option.

New CLI:

```
--extraction-model
```

Benchmark extraction model is now configurable without environment variables.

Trace records benchmark extraction model.

---

## J8.10

Citation propagation.

Root cause:

Report renderer read citation count from the wrong object.

Fixed:

Executive report now uses the same grounding metadata as the trace.

Report now includes:

- correct citation counts
- evidence-backed findings
- source IDs
- evidence IDs

Trace and report now agree.

---

# Functional Pipeline

Current agent sequence:

ProblemFraming

↓

ResearchStrategy

↓

Planner

↓

Evidence

↓

Hypothesis

↓

Challenge

↓

Assumption

↓

Recommendation

↓

Risk

↓

Opportunity

↓

StrategicOption

↓

DecisionAnalysis

↓

ExecutiveConfidence

↓

MultiProfile

↓

Scenario

↓

QA

↓

RecommendationImprovement

↓

RecommendationSynthesis

↓

Report

---

# Knowledge Store

Current domains:

```
knowledge_store/

    ai_data_centers/

    smr/
```

Current evidence:

```
ai_data_centers

675 evidence objects

smr

404 evidence objects
```

Health checks implemented.

Runtime validated.

---

# Benchmark Strategy

Two benchmark modes now exist conceptually.

## Fast Benchmark

Development only.

Stable subset:

```
NVIDIA_003

NVIDIA_007

NVIDIA_010

NVIDIA_012
```

Used for:

- instrumentation
- optimization
- debugging

---

## Full Benchmark

Run only:

- milestone completion
- regression validation
- release readiness

---

# Benchmark CLI

Current benchmark supports:

```
--knowledge-store

--only

--workers

--extraction-model

--log-level PROGRESS
```

Example:

```bash
python3 -m research_agent.eval_runner benchmark \
  --profile ai_data_centers \
  --knowledge-store knowledge_store \
  --only NVIDIA_003,NVIDIA_007,NVIDIA_010,NVIDIA_012 \
  --extraction-model claude-haiku-4-5-20251001 \
  --log-level PROGRESS \
  --out outputs/j89e_benchmark
```

---

# Functional CLI

Current standard command:

```bash
python3 -m functional_agents.cli run \
  --goal "..." \
  --profiles ai_data_centers \
  --knowledge-store knowledge_store \
  --rerank \
  --log-level PROGRESS \
  --out outputs/report.md
```

Project conventions:

- use `python3`
- always use `--log-level PROGRESS`
- Functional runs use Knowledge Layer
- benchmark development uses fast benchmark subset

---

# Performance Summary

Functional pipeline:

- Knowledge Layer functioning
- Retrieval no longer bottleneck
- Runtime dominated by strategic reasoning LLM calls

Benchmark:

- Instrumented
- Deterministic
- Configurable extraction model
- Fast benchmark established
- Haiku validated for benchmark extraction

---

# Architectural Decisions

Accepted:

✓ Knowledge Layer

✓ Hybrid retrieval

✓ Functional Agents

✓ Single retrieval pass

✓ Single benchmark extraction call

✓ Knowledge Store health validation

✓ Performance instrumentation

✓ Evidence traceability

✓ Configurable benchmark extraction model

Rejected:

✗ Batched extraction

✗ Incremental benchmark retrieval

✗ Benchmark redesign

---

# Technical Debt

## Minor

`evidence_summary.citation_count`

Still not populated.

Current renderer uses report grounding score instead.

---

Legacy report path

Non-J7 report rendering still differs.

Not blocking.

---

Reranker

Still creates its own Anthropic client.

Should eventually use dependency injection.

---

# Parking Lot

## Knowledge Layer

Future migration from JSONL to database.

Potential candidates:

- SQLite
- DuckDB
- PostgreSQL

Only when scale justifies it.

Current JSONL implementation is sufficient.

---

## Source Management

Current source layout:

```
sources/

smr_sources/

...
```

Eventually move to unified source registry.

Not currently blocking because Knowledge Store abstracts retrieval.

---

## Reasoning Graph Optimization

Future investigation:

Can strategic reasoning quality be maintained with fewer LLM interactions?

Possible future work:

- shared reasoning context
- merged reasoning stages
- reasoning graph optimization

---

## Dual Benchmark Strategy

Long-term:

Legacy benchmark:

Documents

↓

Extraction

↓

Reasoning

Functional benchmark:

Knowledge Layer

↓

Reasoning

↓

Executive Report

Maintain both for historical continuity.

---

# J9 Preview

J9 shifts focus from answering a research question toward supporting an executive strategic engagement.

Rather than a single question:

```
"What is..."
```

the system will accept:

- business situations
- strategic context
- objectives
- constraints
- success criteria

and produce:

- strategic analysis
- decision options
- tradeoff analysis
- recommendations
- executive-ready strategy

The Functional Agent architecture created during J8 becomes the execution engine for J9.

---

# Overall Status

Knowledge Layer:

✅ Complete

Functional Agents:

✅ Complete

Performance:

✅ Complete

Benchmark Framework:

✅ Complete

Evidence Traceability:

✅ Complete

J8:

# COMPLETE