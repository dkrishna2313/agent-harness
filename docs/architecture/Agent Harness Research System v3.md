# Agent Harness Research System v3

## Complete Retrospective and Architecture Guide

### Phase 1: Foundations Through J2.3a

---

# Executive Summary

The Agent Harness Research System began with a simple question:

> How do we move from using an LLM as a chatbot to using an LLM as part of a reliable research system?

The central insight was that research quality depends on far more than model capability. A high-quality research system requires:

* Retrieval
* Evidence management
* Source governance
* Contradiction detection
* Gap identification
* Evaluation
* Regression testing
* Observability

Rather than relying on:

```text
Question
↓
Prompt
↓
LLM
↓
Answer
```

the project evolved into:

```text
Question
↓
Domain Profile
↓
Retrieval
↓
Evidence Extraction
↓
Evidence Ranking
↓
Contradiction Detection
↓
Research Gap Detection
↓
Coverage Analysis
↓
Memo Synthesis
↓
Evaluation
↓
Regression Testing
↓
Trace Generation
```

The project's most important achievement is not any individual feature.

It is the creation of a reusable Harness Engineering architecture that can operate across multiple research domains while maintaining traceability, quality controls, evaluation, and regression protection.

The harness has successfully operated across:

### AI Infrastructure

* NVIDIA Blackwell
* GB200 NVL72
* Vera Rubin
* Data center power
* Cooling
* Networking

and:

### Small Modular Reactors

* Licensing
* Construction
* Economics
* Fuel supply
* Grid integration

without changing the underlying research engine.

Only the domain profile changed.

This is the defining characteristic of a successful Harness Engineering implementation.

---

# Core Concepts and Terminology

## Document Loading

The process of ingesting source material into the harness.

Examples:

* PDFs
* Markdown files
* HTML pages
* Downloaded web sources

Purpose:

```text
Make source material available for analysis.
```

---

## Chunking

Breaking large documents into smaller sections.

Example:

```text
200-page PDF
↓
500 chunks
```

Purpose:

```text
Enable scalable retrieval and evidence extraction.
```

---

## Retrieval

Selecting the most relevant chunks for a question.

Instead of:

```text
Question
↓
All Documents
```

the harness performs:

```text
Question
↓
Relevant Chunks
```

Purpose:

```text
Reduce noise and improve relevance.
```

---

## Evidence Extraction

Identifying factual claims within retrieved chunks.

Examples:

```text
120 kW rack power
```

```text
BWRX-300 = 300 MWe
```

```text
HALEU supply shortage
```

Purpose:

```text
Convert raw text into structured research findings.
```

---

## Evidence Item

A structured research finding.

Example:

```json
{
  "id": "E001",
  "claim": "GB200 NVL72 requires ~120 kW",
  "source": "...",
  "confidence": 0.92
}
```

Purpose:

```text
Create reusable research building blocks.
```

---

## Evidence Ranking

Scoring evidence according to:

* relevance
* specificity
* confidence
* source quality

Purpose:

```text
Allow stronger evidence to influence conclusions more heavily.
```

---

## Source Grounding

Maintaining traceability from conclusions back to source material.

Purpose:

```text
Prevent unsupported claims.
```

---

## Source Weighting

Assigning credibility scores to sources.

Example:

```text
NVIDIA Technical Documentation
>
Vendor Marketing Material
>
Forum Discussion
```

Purpose:

```text
Improve trustworthiness.
```

---

## Memo Synthesis

Combining evidence into a coherent research report.

Purpose:

```text
Transform evidence into usable knowledge.
```

---

## Contradiction Detection

Identifying evidence that makes incompatible claims.

Example:

```text
24–36 months
vs
8–12 years
```

Purpose:

```text
Expose uncertainty and disagreement.
```

---

## Contradiction Normalization

Determining whether an apparent contradiction is actually valid.

Examples:

Not contradictions:

```text
300 GW target
vs
13 GW/year licensing throughput
```

```text
Rack power
vs
Power shelf power
```

Purpose:

```text
Reduce false positives.
```

---

## Research Gap Detection

Identifying important unanswered questions.

Example:

```text
No UPS information found.
```

Purpose:

```text
Guide future research.
```

---

## Coverage Analysis

Measuring how thoroughly a topic has been researched.

Example:

```text
Power      Strong
Cooling    Moderate
Network    Weak
```

Purpose:

```text
Measure completeness.
```

---

## Domain Profile

A configuration describing a research domain.

Examples:

```text
ai_data_centers.yaml
smr.yaml
```

Purpose:

```text
Separate research engine from domain knowledge.
```

---

## Topic Detection

Determining which research topics are relevant.

Example:

AI Infrastructure:

```text
Power
Cooling
Networking
Operations
```

SMR:

```text
Licensing
Construction
Economics
Fuel
```

---

## Entity Extraction

Identifying the subject of a claim.

Examples:

```text
GB200 NVL72
```

```text
Power Shelf
```

```text
BWRX-300
```

Purpose:

```text
Enable accurate comparison.
```

---

## Scope Extraction

Determining the level at which a claim applies.

Examples:

```text
component
rack
cluster
unit
fleet
site
```

Purpose:

```text
Prevent invalid comparisons.
```

---

## Metric Normalization

Converting measurements into comparable forms.

Example:

```text
24 months
↓
24 months

2 years
↓
24 months
```

Purpose:

```text
Enable reliable contradiction analysis.
```

---

## Evaluation

Benchmarking system performance against known expectations.

Purpose:

```text
Measure quality.
```

---

## Regression Testing

Comparing current benchmark results to historical results.

Purpose:

```text
Detect quality improvements or degradations.
```

---

# Design Philosophy

The harness treats the LLM as:

```text
A component
```

not:

```text
The system
```

The harness itself provides:

* workflow
* governance
* observability
* evaluation
* regression protection
* portability

The model performs reasoning.

The harness provides reliability.

---

# Architecture Evolution

## Version 0

```text
Question
↓
LLM
↓
Answer
```

Problems:

* no traceability
* no retrieval
* no evaluation
* no reproducibility

---

## Early Harness

```text
Question
↓
Documents
↓
Evidence
↓
Memo
```

Solved:

* source grounding
* repeatability

---

## Mid Harness

```text
Question
↓
Retrieval
↓
Evidence
↓
Contradictions
↓
Memo
```

Solved:

* scalability
* conflict detection

---

## Current Harness

```text
Question
↓
Profile
↓
Retrieval
↓
Evidence
↓
Evidence Ranking
↓
Contradictions
↓
Research Gaps
↓
Coverage Analysis
↓
Memo
↓
Evaluation
↓
Regression
↓
Trace
```

Solved:

* portability
* observability
* quality measurement

---

# Major Inflection Points

## Inflection Point 1

Harness vs Prompting

The realization that prompting alone cannot provide reliability.

---

## Inflection Point 2

Domain Profiles

The separation of research engine from domain knowledge.

---

## Inflection Point 3

Web Retrieval (K1)

The transition from static corpora to dynamic research.

---

## Inflection Point 4

Evaluation (J2)

The transition from intuition to measurement.

---

## Inflection Point 5

Regression (J2.3)

The transition from measurement to controlled evolution.

---

# Chronological Evolution

## Foundations

### Initial Harness

Created:

```text
Documents
↓
Evidence
↓
Memo
```

---

### Step A – Observability

Added:

* debug mode
* trace generation
* source reporting

Purpose:

```text
Understand system behavior.
```

---

### Step B – Claude Integration

Added:

* planning
* evidence extraction
* synthesis

Purpose:

```text
Move from templates to reasoning.
```

---

### Step C – Source Grounding

Added:

```text
E001
E002
...
```

Purpose:

```text
Trace every conclusion.
```

---

### Step D – Question-Aware Evaluation

Purpose:

```text
Measure answers against the actual question.
```

---

### Step E – Regression Foundations

Created benchmark evaluation capability.

---

## Scaling

### F1–F3

Added:

* evaluation preservation
* evidence ranking
* chunking

Purpose:

```text
Scale research beyond a few documents.
```

---

### G1–G3

Added:

* retrieval
* contradiction detection
* research gaps

Purpose:

```text
Improve research quality.
```

---

### H1–H2

Added:

* source weighting
* coverage matrix

Purpose:

```text
Measure evidence quality and completeness.
```

---

## Domain Portability

### J1.0–J1.5

Added:

* domain profiles
* profile-based topics
* profile-based source weighting
* profile-aware contradictions

Purpose:

```text
Make the harness reusable.
```

---

## Web Retrieval

### K1.0

Added:

```text
Question
↓
Web Search
↓
Download
↓
Evidence
```

Purpose:

```text
Allow dynamic research.
```

---

## Contradiction Maturity

### J1.6

Added:

* entity extraction
* scope extraction
* contradiction suppression

Purpose:

```text
Reduce false contradictions.
```

---

### J1.6.1b

Added:

* rack vs component power distinction
* metric taxonomy

Purpose:

```text
Fix GB200 false positives.
```

---

## Evaluation Era

### J2.1

Created gold benchmark datasets.

Domains:

* NVIDIA
* SMR
* Contradictions

---

### J2.2

Created automated benchmark execution.

Purpose:

```text
Measure quality.
```

---

### J2.2a

Added:

* benchmark validation
* failure diagnostics
* evaluation traces

Purpose:

```text
Explain failures.
```

---

## Regression Era

### J2.3

Added:

* baseline management
* regression reports
* pass/fail logic

Purpose:

```text
Detect changes in quality.
```

---

### J2.3a

Validated:

* synthetic regressions
* real regressions

Result:

```text
Regression framework proven.
```

---

# Current Capability Matrix

| Capability              | Status |
| ----------------------- | ------ |
| Document Loading        | ✅      |
| Chunking                | ✅      |
| Retrieval               | ✅      |
| Web Retrieval           | ✅      |
| Evidence Extraction     | ✅      |
| Evidence Ranking        | ✅      |
| Source Weighting        | ✅      |
| Source Grounding        | ✅      |
| Contradiction Detection | ✅      |
| Entity Extraction       | ✅      |
| Scope Extraction        | ✅      |
| Research Gap Detection  | ✅      |
| Coverage Analysis       | ✅      |
| Memo Synthesis          | ✅      |
| Domain Profiles         | ✅      |
| Evaluation              | ✅      |
| Regression Testing      | ✅      |
| Trace Generation        | ✅      |

---

# Lessons Learned

## What Worked

* Domain profiles
* Evaluation
* Traceability
* Regression testing

---

## What Failed

* Early contradiction logic
* Scope-insensitive comparisons
* Benchmark brittleness
* Blind quality assessment

---

## Surprises

* Web retrieval mattered more than expected.
* Contradiction quality depended heavily on entity/scope awareness.
* Evaluation became more valuable than adding new features.

---

# Current Known Limitations

### CONTRA_006

Still a known contradiction limitation.

---

### Suppression Taxonomy

Some suppressions still classify as:

```text
entity_mismatch
```

instead of:

```text
scope_mismatch
```

---

### Benchmark Brittleness

Some benchmark failures are benchmark-definition issues rather than harness issues.

---

### Retrieval Gaps

Certain SMR benchmark questions still achieve only partial coverage.

---

# Current Benchmark State

Latest validated benchmark:

```text
Overall Score:            ~97%
Fact Coverage:            ~96%
Citation Coverage:        100%
Contradiction Accuracy:   100%
```

Regression framework validated through:

* synthetic degradation
* real no-web-search degradation

---

# Phase 1 Outcome

Phase 1 established:

```text
Reliable Research Infrastructure
```

The harness can:

* ingest sources
* retrieve evidence
* reason over evidence
* identify contradictions
* identify gaps
* evaluate itself
* detect regressions

across multiple domains.

---

# Phase 2 Roadmap

## J3.0

Retrieval Quality Improvement

Focus:

```text
Query Expansion
```

---

## J3.1

Evidence Quality Improvement

---

## J3.2

Retrieval Diversity

---

## J3.3

Source Ranking Improvements

---


# Final Assessment

The project successfully evolved from:

```text
Prompt
↓
Answer
```

to:

```text
Research System
```

with:

* observability
* portability
* evaluation
* regression protection

The completion of J2.3a marks the end of Phase 1 and the beginning of Phase 2: improving research quality rather than infrastructure.


# Agent Harness Research System v3

## Addendum: Phase 2 Progress Update (J3.0 → J3.1c.1)

---

# Phase 2 Overview

Phase 1 focused on:

```text
Research Infrastructure
```

including:

* Retrieval
* Evidence Extraction
* Contradiction Detection
* Evaluation
* Regression Testing
* Web Retrieval

At the completion of J2.3a the system achieved approximately:

```text
Overall Score: ~95%
```

The objective of Phase 2 was to improve:

```text
Research Quality
```

rather than infrastructure.

---

# J3.0 – Query Expansion and Retrieval Planning

## Goal

Improve evidence retrieval coverage.

Previous architecture:

```text
Question
↓
Single Retrieval Query
↓
Evidence
```

Target architecture:

```text
Question
↓
Retrieval Planner
↓
Multiple Retrieval Queries
↓
Evidence
```

---

## Result

Partial Success.

Benefits:

* Improved SMR evidence coverage.
* Improved exploratory research questions.

Problems:

* Retrieval precision degraded.
* NVIDIA benchmark regressions introduced.

Key lesson:

```text
More retrieval
≠
Better retrieval
```

The planner increased breadth but also increased noise.

---

# J3.0a – Planner Precision Control

## Goal

Separate:

```text
Fact Lookup
```

from:

```text
Exploratory Research
```

and adjust retrieval breadth accordingly.

Added:

* Query classification
* Entity locking
* Metric locking
* Planner modes

---

## Result

Partial Success.

The planner became more disciplined.

However:

```text
Retrieval Planning
```

was not ultimately identified as the primary benchmark bottleneck.

The evaluation traces suggested the remaining benchmark gaps were occurring later in the pipeline.

---

# Major Phase 2 Discovery

The project originally assumed:

```text
Retrieval
↓
Evidence
↓
Answer
```

and that retrieval quality was the dominant bottleneck.

The J3.0 and J3.0a traces demonstrated:

```text
Retrieval quality was generally adequate.
```

The larger issues were:

* Evidence extraction completeness
* Benchmark matching quality

This discovery redirected Phase 2.

---

# J3.1 – Evidence Quality Improvement

## Goal

Improve extraction of useful facts from retrieved evidence.

Added:

* Evidence typing
* Multi-claim extraction
* Topic tagging
* Metric extraction
* Entity coverage
* Evidence density metrics

---

## Result

Success.

Benchmark improvements:

```text
NVIDIA coverage improved.
NVIDIA_003 fixed.
Regression framework passed.
```

This was the first Phase 2 milestone that survived the regression gate.

---

# J3.1a – Semantic Benchmark Matching

## Goal

Replace brittle string matching with semantic matching.

Examples:

```text
economy of scale
```

vs

```text
economics-of-scale disadvantage
```

and:

```text
load following
```

vs

```text
grid flexibility
```

---

## Result

Major Success.

Benchmark improvements:

```text
Fact Coverage: 100%
Q&A Passes increased
Hallucination penalties reduced
```

The project discovered that benchmark evaluation itself had become a bottleneck.

---

# Major Phase 2 Discovery

The evaluation traces revealed:

```text
Research Quality
>
Benchmark Score
```

In several cases.

The benchmark was underestimating answer quality.

This led to:

```text
Benchmark Quality Engineering
```

becoming a new workstream.

---

# J3.1b – Semantic Matcher Hardening

## Goal

Improve trustworthiness of semantic matching.

Added:

* Synonym registry
* Anti-synonym registry
* Confidence bands
* Semantic auditing

Examples:

```text
economy of scale
```

should not match:

```text
learning rate
```

---

## Result

Success after stabilization.

The matcher became:

* Explainable
* Auditable
* More trustworthy

---

# J3.1c – Benchmark Hygiene

## Goal

Reduce benchmark false positives.

Previous behavior:

```text
String appears
↓
Fail
```

New behavior:

```text
Incorrect claim
↓
Fail

Valid contextual mention
↓
Pass
```

Examples:

```text
GDDR
```

mentioned as a contrast to HBM.

```text
PCIe
```

mentioned as a technology replaced by NVLink-C2C.

---

## Added Capabilities

* Prohibited-term auditing
* Context detection
* Context exemptions
* Benchmark explainability
* Benchmark hygiene reporting

---

# J3.1c.1 – NVIDIA_011 Fix

## Goal

Resolve the final benchmark hygiene regression.

Problem:

```text
PCIe connection
```

was still treated as prohibited even when used in valid technical context.

---

## Result

Success.

Final benchmark:

```text
Overall Score        0.9928
Fact Coverage        0.9783
Hallucination Rate   0.0000
Citation Coverage    1.0000
Contradiction Acc.   1.0000
Q&A Passed           23 / 23
```

Regression status:

```text
PASS
```

---

# Current System Capability Matrix

| Capability                  | Status |
| --------------------------- | ------ |
| Retrieval                   | ✅      |
| Retrieval Planning          | ✅      |
| Web Retrieval               | ✅      |
| Evidence Extraction         | ✅      |
| Evidence Typing             | ✅      |
| Multi-Claim Extraction      | ✅      |
| Source Weighting            | ✅      |
| Contradiction Detection     | ✅      |
| Entity Extraction           | ✅      |
| Scope Extraction            | ✅      |
| Research Gap Detection      | ✅      |
| Coverage Analysis           | ✅      |
| Evaluation                  | ✅      |
| Regression Testing          | ✅      |
| Semantic Benchmark Matching | ✅      |
| Benchmark Hygiene           | ✅      |
| Trace Generation            | ✅      |

---

# Updated Lessons Learned

## What Worked

* Evaluation-first development
* Regression protection
* Domain profiles
* Semantic benchmark matching
* Benchmark explainability

---

## What Did Not Work

* Aggressive retrieval expansion
* Retrieval breadth without precision controls
* Pure string matching evaluation

---

## Biggest Discovery

The project initially believed:

```text
Retrieval
```

was the dominant bottleneck.

Phase 2 demonstrated:

```text
Benchmark Quality
```

was responsible for a significant portion of apparent failures.

Improving evaluation quality produced some of the largest benchmark gains of the entire project.

---

# Current Known Limitations

## CONTRA_006

Remains a known limitation.

Tracked separately.

---

## Suppression Classification

Examples:

```text
scope_mismatch
```

still occasionally appear as:

```text
entity_mismatch
```

This affects explanation quality rather than correctness.

---

## Semantic Edge Cases

Examples remain where:

```text
load following
```

coverage is not always recognized optimally.

These are isolated evaluation edge cases rather than major benchmark deficiencies.

---

# Current Project State

The harness has successfully evolved from:

```text
Prompt
↓
Answer
```

to:

```text
Research Platform
```

with:

* Retrieval
* Evidence Management
* Contradiction Detection
* Evaluation
* Regression Protection
* Semantic Benchmarking
* Benchmark Hygiene

Phase 2 has validated that the system can be continuously improved while maintaining objective quality measurement.

---

# Next Planned Milestone

## J3.2 – Retrieval Diversity

Focus:

```text
Different evidence perspectives
```

rather than:

```text
More evidence
```

Examples:

* Technical
* Economic
* Operational
* Regulatory
* Supply Chain

The goal is to improve research completeness while preserving the benchmark gains achieved in J3.1.


##Potential directions after this

Now that J2 is done, I'd recommend **not** adding more infrastructure.

The evaluation traces already told us where the biggest weaknesses are.

If I rank potential J3 directions by expected impact:

| Candidate                      | Impact     | Recommendation |
| ------------------------------ | ---------- | -------------- |
| Retrieval quality              | Very High  | ⭐ J3.0         |
| Evidence extraction quality    | High       | J3.1           |
| Contradiction sophistication   | Medium     | Later          |
| Better prompting               | Medium     | Later          |
| Multi-agent workflows          | Low-Medium | Much later     |
| More evaluation infrastructure | Low        | Stop here      |

