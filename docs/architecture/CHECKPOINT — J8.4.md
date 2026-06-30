# CHECKPOINT — J8.4
**Date:** June 2026

---

# Project Vision

The project has evolved from a document research harness into a layered strategic reasoning platform.

Current architecture:

Knowledge Layer
        ↓
Reasoning Layer
        ↓
Decision Layer
        ↓
Presentation Layer

The long-term objective is a persistent knowledge platform capable of supporting strategic decision making across multiple domains while separating knowledge construction from knowledge consumption.

---

# Repository Layout

Major directories:

```
functional_agents/
    reasoning engine

research_agent/
    shared models
    orchestration
    evaluation
    CLI

knowledge/
    knowledge platform implementation

knowledge_store/
    persistent knowledge base

profiles/
    domain profiles

eval/
    benchmark suite

outputs/
    generated artifacts

docs/
    architecture documentation
```

---

# J7 Summary (Complete)

J7 introduced a complete strategic reasoning layer.

Final decision graph:

```
Evidence
    ↓
Strategic Assumptions
    ├── Risks
    ├── Opportunities
    └── Recommendations
            ↓
Strategic Options
            ↓
Decision Analysis
            ↓
Executive Confidence
```

Major milestones completed:

- Strategic Engagement
- Decision Model v2
- Strategic Assumptions
- Recommendation linkage
- Strategic Risks
- Strategic Opportunities
- Strategic Options
- Decision Analysis
- Executive Confidence
- Executive Reporting

J7 is considered architecturally complete.

---

# J8 Summary

Chronological progression:

```
J8.0
Knowledge Ontology

↓

J8.1
Knowledge Platform

↓

J8.2
Knowledge Quality

↓

J8.3
Knowledge Retrieval

↓

J8.4
Hybrid Retrieval
```

Status:

## J8.0

Knowledge ontology frozen.

## J8.1

Persistent Knowledge Platform implemented.

Includes:

- Sources
- Evidence
- Metadata
- ExtractionRuns
- persistent store
- incremental builds

## J8.2

Strategic evidence extraction.

Document metadata removed from Evidence.

Source normalization added.

Evidence quality significantly improved.

## J8.3

Knowledge Retrieval.

EvidenceRetriever abstraction introduced.

Lexical retrieval.

Intent-aware lexical ranking.

## J8.4

Hybrid retrieval.

Semantic embeddings.

Hybrid lexical + semantic retrieval.

Metadata reranking.

Retrieval abstraction complete.

---

# Current Knowledge Ontology

```
Source (immutable)

↓

Evidence (immutable)

↓

KnowledgeMetadata (mutable)
```

Supporting entities:

- ExtractionRun
- Embeddings
- Contradictions

Important design principles:

- Source owns provenance.
- Evidence owns knowledge.
- Metadata owns mutable state.

---

# Current Retrieval Pipeline

```
Query

↓

Lexical Retrieval

+

Semantic Retrieval

↓

Merge

↓

Metadata Ranking

↓

Evidence
```

Planner integration has NOT yet occurred.

The functional reasoning pipeline still uses the legacy retrieval path.

---

# Knowledge Store

```
knowledge_store/

    sources/

    evidence/

    metadata/

    embeddings/

    extraction_runs/

    manifests/

    indexes/

    contradictions/

    cache/

    _meta/
```

Purpose:

sources/
    normalized source objects

evidence/
    canonical evidence

metadata/
    mutable evidence state

embeddings/
    evidence vectors

extraction_runs/
    provenance

manifests/
    incremental build tracking

_meta/
    schema version and statistics

---

# Major Architectural Decisions

The following decisions are considered frozen:

✓ Evidence is the semantic unit.

✓ Evidence embeddings instead of chunk embeddings.

✓ Immutable Evidence.

✓ Mutable KnowledgeMetadata.

✓ Source normalization belongs on Source.

✓ Retrieval abstraction separates storage from reasoning.

✓ Query operations never mutate canonical artifacts.

✓ Strategic evidence rather than exhaustive transcription.

---

# Current CLI

Knowledge:

```bash
python3 -m knowledge.cli build

python3 -m knowledge.cli retrieve

python3 -m knowledge.cli embed
```

Reasoning:

```bash
python3 -m functional_agents.cli run \
    ... \
    --profiles ai_data_centers \
    --web-search \
    --out outputs/...
```

Benchmark:

```bash
python3 -m research_agent.eval_runner benchmark ...
```

Regression:

```bash
python3 -m research_agent.eval_runner regress ...
```

---

# Validation Strategy

Knowledge milestones:

- inspect ontology
- inspect retrieval
- inspect evidence

Reasoning milestones:

- benchmark
- regression

Decision milestones:

- inspect Decision Model
- inspect Executive Report

---

# Current Open Issues

- Placeholder PDF title normalization.
- Hybrid retrieval currently uses weighted lexical + semantic scoring.
- Planner still retrieves through legacy pipeline.
- LLM evidence reranking not yet implemented.

---

# Next Milestone

## J8.5 — Intelligent Evidence Reranking

Architecture:

```
Hybrid Retrieval

↓

Candidate Evidence

↓

LLM Evidence Reranker

↓

Planner
```

Purpose:

Move from score-based evidence ordering to intent-aware evidence selection.

The bottleneck has moved from retrieval to evidence selection.

---

# Expected Roadmap

```
J8.5
Evidence Reranking

↓

J8.6
Planner Integration

↓

J8.7
Knowledge Maintenance

↓

J8.8
Performance & Scale
```

---

# Architectural Principles

Always preserve:

- Ontology before implementation.
- Freeze architecture before coding.
- One architectural concept per milestone.
- Immutable knowledge.
- Mutable metadata.
- Evidence is the canonical semantic unit.
- Separate knowledge construction from knowledge consumption.
- Query operations never mutate canonical artifacts.
- Reasoning never depends on storage implementation.

---

# Current Assessment

J7:
Architecturally complete.

Knowledge Layer:
Architecturally complete.

Hybrid Retrieval:
Implemented.

Current bottleneck:
Evidence selection.

Next milestone:
LLM-assisted Evidence Reranking.