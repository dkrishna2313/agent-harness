# CHECKPOINT — J8.6

**Harness Engineering Project**

---

# Project Vision

The project has evolved from a document research harness into a layered strategic reasoning platform.

Current architecture:

```
Knowledge Layer
        ↓
Reasoning Layer
        ↓
Decision Layer
        ↓
Presentation Layer
```

The long-term objective is a persistent Knowledge Platform capable of supporting strategic decision making across multiple domains.

Knowledge is built once.

Reasoning consumes persistent knowledge.

---

# Repository Layout

```
functional_agents/
    Strategic reasoning engine

research_agent/
    Shared models
    Evaluation
    Orchestration
    CLI

knowledge/
    Knowledge platform implementation

knowledge_store/
    Persistent knowledge base

profiles/
    Domain profiles

eval/
    Benchmark suite

outputs/
    Generated run artifacts

docs/
    Architecture documentation
```

---

# J7 Summary (Complete)

J7 introduced the complete strategic reasoning layer.

Decision graph:

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

Completed:

* Strategic Engagement
* Decision Model
* Strategic Assumptions
* Recommendation linkage
* Risks
* Opportunities
* Strategic Options
* Decision Analysis
* Executive Confidence
* Executive Reporting

J7 is architecturally complete.

---

# J8 Progress

Completed:

## J8.0

Knowledge Ontology

Ontology frozen.

---

## J8.1

Knowledge Platform

Implemented:

* Sources
* Evidence
* Metadata
* ExtractionRuns
* Incremental builder
* Persistent knowledge_store

---

## J8.2

Knowledge Quality

Implemented:

* Strategic evidence extraction
* Source normalization
* Source provenance
* Strategic Evidence classification

---

## J8.3

Knowledge Retrieval

Implemented:

* EvidenceRetriever abstraction
* Lexical retrieval
* Intent-aware lexical ranking

---

## J8.4

Hybrid Retrieval

Implemented:

* Sentence-transformer embeddings
* Semantic retrieval
* Hybrid lexical + semantic retrieval
* Metadata scoring
* Pluggable embedding providers

---

## J8.5

Intelligent Evidence Reranking

Implemented:

* EvidenceReranker abstraction
* Claude Haiku reranker
* Hallucinated ID rejection
* Rationale generation
* Candidate reranking
* Provenance preservation

Knowledge retrieval stack is now architecturally complete.

---

# Knowledge Ontology

```
Source (immutable)

↓

Evidence (immutable)

↓

KnowledgeMetadata (mutable)
```

Supporting objects:

* ExtractionRun
* Embeddings
* Contradictions

Design principles:

* Source owns provenance.
* Evidence owns knowledge.
* Metadata owns lifecycle state.

---

# Retrieval Pipeline

Current retrieval architecture:

```
Query

↓

Lexical Retrieval

+

Semantic Retrieval

↓

Hybrid Scoring

↓

Top Candidate Set

↓

LLM Evidence Reranker

↓

Evidence
```

The retrieval subsystem is considered complete.

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
Normalized documents

evidence/
Canonical knowledge

metadata/
Mutable evidence state

embeddings/
Persistent evidence vectors

extraction_runs/
Provenance

manifests/
Incremental build tracking

_meta/
Schema version
Statistics

---

# Current CLI

Knowledge:

```bash
python3 -m knowledge.cli build

python3 -m knowledge.cli embed

python3 -m knowledge.cli retrieve
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

Knowledge milestones

* inspect ontology
* inspect retrieval
* inspect reranking

Reasoning milestones

* benchmark
* regression

Decision milestones

* inspect Decision Model
* inspect Executive Report

---

# Current Open Issues

Only genuine remaining issues:

* Planner still retrieves via legacy document pipeline.
* Functional reasoning engine does not yet consume Knowledge Layer.
* Hybrid retrieval currently uses weighted hybrid scoring before reranking (acceptable, but may evolve).

No architectural issues remain in the Knowledge Layer.

---

# Next Milestone

## J8.6 — Planner Integration

Objective:

Replace document-first evidence acquisition with Knowledge-first retrieval.

Target architecture:

```
Planner

↓

EvidenceRetriever

↓

Hybrid Retrieval

↓

LLM Evidence Reranker

↓

EvidenceAgent

↓

Reasoning Pipeline
```

The Planner should request evidence.

Not documents.

---

# Roadmap

```
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

Never violate:

* Ontology before implementation.
* Freeze architecture before coding.
* One architectural concept per milestone.
* Immutable knowledge.
* Mutable metadata.
* Evidence is the semantic unit.
* Knowledge construction separate from knowledge consumption.
* Query operations never mutate canonical artifacts.
* Reasoning never depends on storage implementation.

---

# Current Assessment

J7

✓ Complete

Knowledge Layer

✓ Complete

Retrieval Layer

✓ Complete

Current bottleneck

Planner still consumes legacy retrieval.

Next objective

Integrate the Planner with the completed Knowledge Layer.
