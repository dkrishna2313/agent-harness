# Agent Harness Research System

# Architecture Checkpoint — J7 Complete

**Version:** 6

**Milestone:** J7 Complete

**Date:** June 2026

---

# Executive Summary

This document captures the architecture of the Agent Harness Research System at the completion of **J7**.

The project has now reached an important architectural milestone.

The system is no longer best described as a Retrieval-Augmented Generation (RAG) system, nor simply as a collection of AI agents.

It has evolved into an **evidence-based strategic decision support platform**.

The purpose of the system is no longer merely to answer questions.

Its purpose is to support executive decision-making through explicit, traceable reasoning.

The architecture now separates:

* knowledge acquisition
* strategic reasoning
* decision analysis
* executive communication

into distinct layers with clearly defined responsibilities.

Every significant reasoning step is now represented as a first-class object.

---

# Current Vision

The long-term objective of the project is:

> Produce board-quality strategic advice for CEOs, investors and senior government decision-makers through explicit, evidence-based reasoning.

The system is intended to answer four executive questions.

1. What should we do?

2. Why should we do it?

3. How confident are we?

4. What would change our decision?

Every architectural decision since J5 has been made in service of answering these four questions more explicitly.

---

# Evolution of the Architecture

The project has evolved through four distinct phases.

## J1–J4 — Research Harness

Focus:

* evidence extraction
* citation generation
* contradiction detection
* evaluation
* retrieval quality

Output:

Research memorandum.

---

## J5 — Functional Agent Architecture

Focus:

* explicit functional agents
* orchestrator
* shared context
* agent contracts

Output:

Structured research pipeline.

---

## J6 — Strategic Reasoning

Focus:

* problem framing
* research strategy
* hypotheses
* challenge
* recommendations
* scenarios

Output:

Structured reasoning.

---

## J7 — Decision Intelligence

Focus:

* Strategic Engagement
* Decision Model
* Strategic Assumptions
* Risks
* Opportunities
* Strategic Options
* Decision Analysis
* Executive Confidence

Output:

Executive decision support.

---

# Current Architecture

The system now consists of four logical layers.

```
Strategic Layer

Strategic Engagement

↓

Decision Model



Research Layer

Evidence

↓

Hypotheses

↓

Challenge



Decision Layer

Assumptions

↓

Recommendations

↓

Risks

↓

Opportunities

↓

Strategic Options

↓

Decision Analysis

↓

Executive Confidence



Presentation Layer

Executive Report
```

This layering deliberately separates reasoning from presentation.

The Executive Report is no longer responsible for creating intelligence.

It communicates intelligence that already exists inside the Decision Model.

---

# Canonical Objects

The architecture now revolves around four canonical objects.

## Strategic Engagement

Represents the business problem.

Owns:

* client
* brief
* strategic question
* engagement context

---

## Decision Model

The canonical strategic reasoning object.

Owns:

* objectives
* assumptions
* risks
* opportunities
* strategic options
* decision analysis
* executive confidence

This object now represents the "brain" of the system.

---

## Research Object

Represents the research process.

Owns:

* evidence
* hypotheses
* challenges
* recommendations
* scenarios
* QA
* execution history

The Research Object serves as the audit trail for how the Decision Model was produced.

---

## Executive Report

Presentation layer.

Consumes:

* Decision Model
* Research Object

Generates:

board-ready narrative.

No strategic reasoning is generated inside the report.

---

# Decision Graph

The completed decision graph is:

```
Evidence

↓

Hypotheses

↓

Challenge

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

This graph represents the core intellectual property of the project.

Every object is explicitly represented.

Every relationship is traceable.

---

# Functional Agent Pipeline

Current execution order:

```
Problem Framing

↓

Research Strategy

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

Recommendation Linkage

↓

Risk

↓

Opportunity

↓

Strategic Options

↓

Decision Analysis

↓

Executive Confidence

↓

Scenario

↓

Multi-Profile

↓

QA

↓

Report
```

Each agent owns a single responsibility.

Agents communicate exclusively through the shared AgentContext.

---

# Current Repository

Important directories:

```
research_agent/

Core research engine


functional_agents/

Production reasoning pipeline


profiles/

Domain profiles


eval/

Evaluation suite


baseline/

Regression baseline


outputs/

Generated artefacts


tests/

Automated validation
```

---

# Current Entry Points

Primary production command:

```bash
python3 -m functional_agents.cli run \
  "<goal>" \
  --profiles <profile> \
  --web-search \
  --out outputs/report.md \
  --log-level PROGRESS
```

Benchmark:

```bash
python3 -m research_agent.eval_runner benchmark
```

Regression:

```bash
python3 -m research_agent.eval_runner regress
```

The legacy `research_agent.cli` remains for compatibility and isolated workflows, but the functional agent CLI is now the canonical production entry point.

---

# Quality Status

At completion of J7:

* 23 / 23 benchmark questions passing
* 8 / 8 contradiction tests passing
* 100% fact coverage
* 100% citation coverage
* 0% hallucination rate
* Regression PASS

Development remains evaluation-driven.

No architectural milestone is considered complete without benchmark and regression validation.

---

# Design Principles

The project now follows several architectural principles.

## Evidence before opinion

Recommendations emerge from evidence.

---

## Explicit reasoning

Reasoning exists as persistent objects.

Not hidden prompts.

---

## Canonical ownership

Each concept has one owner.

Strategic Engagement.

Decision Model.

Research Object.

Executive Report.

---

## Deterministic relationships

Where possible,

relationships are computed,

not inferred by an LLM.

---

## Evaluation-first development

Architecture evolves under benchmark protection.

---

## Executive-first outputs

The intended audience is:

* CEOs
* investors
* boards
* senior government officials

---

# Current Strengths

The project now provides:

* persistent reasoning graph

* explicit assumptions

* explicit risks

* explicit opportunities

* explicit strategic options

* executive decision analysis

* executive confidence

* traceability from evidence to recommendation

* benchmark-driven validation

This combination differentiates the system from conventional RAG architectures.

---

# Known Technical Debt

Several items have been intentionally deferred.

## Performance

Knowledge acquisition and reasoning remain tightly coupled.

J8 will separate:

offline knowledge construction

from

online reasoning.

---

## Report Compression

The report currently presents all reasoning.

Future work will distinguish:

Executive Report

from

Technical Appendix.

---

## Knowledge Base

Evidence is still reconstructed on each run.

Future architecture will introduce a persistent knowledge layer.

---

## Scalability

Current corpus sizes remain relatively small.

J8 will focus heavily on indexing, caching and incremental evidence construction.

---

# Roadmap

J7 is complete.

The next architectural phase is J8.

Primary goals:

* persistent knowledge base
* incremental indexing
* retrieval scalability
* executive narrative synthesis
* report compression
* performance optimisation
* enterprise-scale research workspaces

Unlike J7,

which focused on improving reasoning,

J8 will focus on improving scalability and usability.

---

# State of the Codebase

At this checkpoint the Agent Harness Research System should be considered a complete strategic decision engine.

The remaining work is no longer about adding reasoning capabilities.

Instead it is about making the existing reasoning engine:

* faster
* more scalable
* easier to use
* easier to maintain

The architecture described here should therefore be treated as the stable foundation upon which future development is built.

Changes after J7 should preserve these architectural principles unless there is a compelling reason to evolve them.

---

*"Architecture is no longer the constraint. Knowledge, scale, and usability are the next frontier."*
