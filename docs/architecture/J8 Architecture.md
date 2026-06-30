# J8 Architecture
Knowledge Platform Architecture

---

# Purpose

J8 introduces a persistent Knowledge Layer beneath the strategic reasoning engine.

The system is evolving from document processing into knowledge-driven reasoning.

Knowledge is built once.

Knowledge is reused many times.

---

# Overall Architecture

```
Knowledge Layer
        ↓
Reasoning Layer
        ↓
Decision Layer
        ↓
Presentation Layer
```

Each layer has clearly defined ownership.

---

# Knowledge Layer

Responsible for:

- Source ingestion
- Source normalization
- Evidence extraction
- Metadata
- Embeddings
- Knowledge retrieval

The Knowledge Layer owns persistent knowledge.

It performs no strategic reasoning.

---

# Knowledge Ontology

```
Source

↓

Evidence

↓

KnowledgeMetadata
```

Supporting entities:

- ExtractionRun
- Embeddings
- Contradictions

---

# Source

Immutable.

Represents an original document.

Owns:

- provenance
- metadata
- canonical text

Never represents strategic knowledge.

---

# Evidence

Immutable.

Represents one reusable strategic or technical claim.

Owns:

- statement
- supporting sources
- evidence type
- extraction provenance

Evidence is the canonical semantic unit.

---

# KnowledgeMetadata

Mutable.

Owns:

- confidence
- credibility
- lifecycle state
- retrieval priority
- strategic value
- quality scores

Evidence never changes.

Metadata evolves.

---

# Retrieval

Current architecture:

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

Future:

```
Hybrid Retrieval

↓

LLM Evidence Reranker

↓

Planner
```

---

# Ownership

Knowledge owns:

- Sources
- Evidence
- Metadata
- Embeddings
- ExtractionRuns

Reasoning owns:

- Engagement
- Decision Model
- Research Object
- Decision Graph

Knowledge never writes canonical reasoning artifacts.

Reasoning never accesses storage directly.

Communication occurs through the EvidenceRetriever abstraction.

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

This structure is considered stable.

---

# Design Principles

1. Ontology before implementation.

2. Stable abstractions.

3. Immutable knowledge.

4. Mutable metadata.

5. Evidence-first reasoning.

6. Retrieval abstraction isolates storage.

7. Incremental knowledge construction.

8. Strategic evidence rather than document transcription.

---

# Current State

Implemented:

✓ Persistent knowledge

✓ Incremental builds

✓ Source normalization

✓ Strategic evidence

✓ Hybrid retrieval

Planned:

• LLM evidence reranking

• Planner integration

• Knowledge maintenance

• Performance optimization

---

# Long-term Vision

```
Knowledge Construction

↓

Persistent Knowledge

↓

Evidence Retrieval

↓

Strategic Reasoning

↓

Decision Graph

↓

Executive Report
```

The Knowledge Layer becomes the long-term memory of the platform.

The Reasoning Layer becomes an intelligent consumer of that memory rather than reconstructing knowledge from documents on every run.