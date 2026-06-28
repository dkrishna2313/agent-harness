# Agent Harness Research System

## Architecture Checkpoint — J7.5

**Version:** 5  
**Checkpoint:** J7.5 (Strategic Options Complete)  
**Date:** June 2026

---

# Executive Summary

This document describes the architecture, evolution, philosophy, and current implementation state of the Agent Harness Research System as of milestone **J7.5**.

Unlike a conventional software design document, this checkpoint serves two purposes.

First, it is intended to be a complete architectural specification for the current system. A new engineer—or even a future version of the project team—should be able to understand how the system works, why it was designed this way, and how future work should extend it without reading months of historical discussions.

Second, it captures the architectural reasoning behind the project. Source code explains *how* a system behaves. It rarely explains *why* the architecture evolved the way it did. This document attempts to preserve those decisions.

This checkpoint intentionally occurs after completion of **J7.5** because the project has now crossed an important architectural threshold.

The system is no longer simply a document research harness.

It is no longer merely a retrieval-augmented generation system.

It is no longer simply a collection of cooperating AI agents.

Instead, it has evolved into an explicit **strategic decision support system** built around a persistent reasoning graph.

Rather than generating recommendations directly from retrieved evidence, the system now explicitly models:

- Strategic engagements
- Decision models
- Evidence
- Competing hypotheses
- Challenges
- Strategic assumptions
- Risks
- Opportunities
- Recommendations
- Strategic options

Each object is represented explicitly, persisted independently, and linked through deterministic relationships wherever possible.

This architectural shift is the defining characteristic of the J7 series.

---

# Vision

The long-term objective of the Agent Harness Research System is straightforward.

> Produce board-quality strategic advice for CEOs, investors and senior government decision-makers through explicit, evidence-based reasoning.

This vision deliberately differs from that of most contemporary AI research systems.

Most research systems attempt to answer questions.

The Harness attempts to support decisions.

This distinction is subtle but fundamental.

A research system attempts to determine:

> What information is relevant?

A decision-support system attempts to determine:

> Given the available evidence, what should be done, why should it be done, what assumptions does that recommendation depend upon, what risks threaten it, and what alternatives should be considered?

These are fundamentally different problems.

Throughout the evolution of the project, the architecture has steadily shifted away from information retrieval toward explicit strategic reasoning.

This shift has influenced nearly every major architectural decision.

---

# Architectural North Star

The project is evolving toward the following conceptual architecture.

```
Strategic Engagement

        ↓

Decision Model

        ↓

Evidence Graph

        ↓

Strategic Reasoning Graph

        ↓

Decision Analysis

        ↓

Executive Report

        ↓

Interactive Decision Support
```

Each layer represents a different level of abstraction.

The lower layers focus on evidence acquisition.

The middle layers transform evidence into reasoning.

The upper layers transform reasoning into decisions.

Eventually, the system should become capable of supporting iterative strategic conversations in which recommendations evolve as assumptions, evidence, or external conditions change.

This architecture deliberately separates knowledge acquisition from strategic reasoning.

Doing so enables each layer to evolve independently.

---

# Guiding Principles

Several principles have remained consistent throughout the project.

## Evidence before opinion

Recommendations should emerge from evidence rather than from the language model's prior knowledge.

Whenever practical, every important statement should be traceable back to explicit evidence.

---

## Explicit reasoning

Earlier generations of LLM systems often produce recommendations without exposing the intermediate reasoning process.

The Harness intentionally models intermediate reasoning objects explicitly.

Examples include:

- hypotheses
- challenges
- assumptions
- risks
- opportunities

Rather than hiding reasoning inside a prompt, the reasoning becomes part of the persistent architecture.

---

## Canonical objects

Every major concept should exist only once.

For example:

- Strategic Engagement
- Decision Model
- Research Object

Each owns a clearly defined responsibility.

Objects should reference one another through identifiers rather than duplicating information.

This dramatically simplifies future evolution.

---

## Deterministic relationships whenever possible

The project intentionally avoids repeated LLM reasoning when deterministic computation can establish relationships.

Examples include:

Recommendation linkage.

Rather than asking the LLM which assumptions support a recommendation, the system derives these relationships by traversing evidence identifiers and hypothesis identifiers already present in the graph.

This improves:

- reproducibility
- observability
- testing
- maintainability

---

## Additive evolution

Major architectural milestones should be additive rather than destructive.

Each milestone should introduce a new capability while preserving backwards compatibility whenever practical.

This philosophy has allowed the project to evolve continuously without repeatedly rewriting earlier subsystems.

---

## Observability first

Reasoning that cannot be observed cannot be trusted.

Nearly every major subsystem therefore produces structured trace information.

This philosophy first emerged during the H-series hardening work and has remained a defining characteristic of the project.

---

## Board-level outputs

The intended audience is not software engineers.

The intended audience is:

- CEOs
- Investors
- Boards of Directors
- Senior Government Officials

Consequently, architectural decisions increasingly optimize for strategic reasoning rather than technical explanation.

---

# Evolution of the Architecture

The project did not begin as a strategic reasoning system.

It evolved there gradually.

Understanding that evolution explains why the architecture looks the way it does today.

---

# J1 — Evidence Foundation

The earliest milestones focused on creating a reliable evidence extraction pipeline.

Core capabilities introduced during J1 included:

- document ingestion
- chunking
- evidence extraction
- citation generation
- contradiction detection
- traceability

At this stage the system behaved similarly to a conventional retrieval-augmented generation pipeline.

Evidence was extracted directly from local documents and synthesized into research memoranda.

Although relatively simple compared to later milestones, J1 established two architectural principles that still define the system today:

1. Evidence should be explicitly represented rather than embedded inside prompts.

2. Every synthesized statement should remain traceable back to its originating source wherever possible.

These two principles underpin every subsequent architectural decision.

---

# J2 — Evaluation Harness

Once evidence extraction became reliable, the next challenge was measuring quality.

J2 introduced the project's evaluation framework.

Rather than manually judging output quality, the project began using:

- gold-standard question sets
- benchmark scoring
- citation validation
- hallucination detection
- contradiction evaluation
- regression testing

This transformed development.

Instead of asking

> "Does this output look good?"

the project could ask

> "Did this architectural change objectively improve the system?"

This evaluation-first philosophy remains one of the project's greatest strengths.

Nearly every subsequent milestone has been validated against the same benchmark suite, allowing architecture to evolve while maintaining measurable quality.

---

# J3 — Persistent Research Objects

As reasoning became more sophisticated, simple Markdown output became insufficient.

J3 introduced the Research Object.

Rather than treating a research run as transient text, the system began representing each investigation as structured data.

Research Objects became persistent containers for:

- research questions
- investigation areas
- evidence
- findings
- contradictions
- recommendations
- metadata

This represented the project's first major transition from document generation toward knowledge representation.

The Research Object later became one of the three canonical objects in the overall architecture.

---

# J4 — Retrieval Intelligence

Early retrieval relied primarily on relevance ranking.

J4 expanded this into a considerably more sophisticated retrieval subsystem.

Major additions included:

- retrieval planning
- evidence ranking
- evidence recovery
- coverage analysis
- source quality evaluation
- extraction diagnostics
- retrieval observability

The emphasis shifted from

> finding documents

to

> understanding whether the system possessed sufficient evidence to support strategic conclusions.

These capabilities later became critical for higher-level reasoning introduced in J6 and J7.

---

# J5 — Functional Agent Architecture

J5 represents one of the most important architectural transitions in the project.

Earlier milestones relied primarily on a monolithic research pipeline.

J5 decomposed reasoning into explicit functional agents operating over shared context.

The initial functional architecture introduced agents responsible for:

- planning
- evidence gathering
- quality assurance
- reporting

An orchestrator coordinated execution.

Shared context replaced implicit prompt state.

Agent contracts formalized responsibilities.

This decomposition had an important consequence.

It became possible to insert entirely new reasoning stages into the workflow without redesigning the entire pipeline.

In hindsight, J5 laid the architectural foundation for everything that followed.

Without explicit functional agents, the higher-order reasoning introduced during J6 and J7 would have been extremely difficult to implement cleanly.

# J6 — Strategic Reasoning

J5 established a functional agent architecture.

J6 transformed those agents into a strategic reasoning pipeline.

The central architectural question changed.

Earlier milestones attempted to answer:

> What does the evidence say?

J6 began asking:

> What does the evidence imply?

This distinction marks the beginning of the system's evolution from research automation toward decision support.

Rather than synthesizing evidence directly into recommendations, the system began constructing intermediate reasoning layers.

These intermediate layers became explicit architectural objects rather than implicit prompt engineering.

---

## Problem Framing

The first major addition was the **Problem Framing Agent**.

Earlier versions assumed that users supplied a well-defined research question.

Experience quickly demonstrated that this assumption was unrealistic.

Real strategic engagements rarely begin with a concise research question.

Instead they begin with:

- board papers
- investment theses
- policy documents
- client briefs
- strategic objectives

The Problem Framing Agent converts these into a structured decision problem.

Rather than beginning with:

> "Find evidence."

the system now begins with:

> "What decision is actually being made?"

This represented the first movement away from information retrieval and toward decision support.

---

## Research Strategy

Once the decision problem is understood, the system must determine how to investigate it.

The Research Strategy Agent separates:

what should be learned

from

how it should be learned.

This distinction significantly improved planning quality because retrieval strategies became dependent upon decision context rather than keyword matching alone.

---

## Hypothesis-driven reasoning

Perhaps the largest conceptual shift introduced during J6 was explicit hypothesis generation.

Instead of moving directly from evidence to recommendations, the system now generates competing explanations.

Evidence

↓

Hypotheses

The purpose is not prediction.

The purpose is structured reasoning.

Hypotheses expose uncertainty explicitly.

They also make subsequent challenge possible.

---

## Challenge

Recommendations generated without challenge frequently exhibit confirmation bias.

The Challenge Agent intentionally attempts to invalidate every hypothesis.

Rather than asking

> Why is this true?

the agent asks

> Why might this be wrong?

Surviving hypotheses therefore represent reasoning that has already been stress-tested.

This adversarial stage dramatically improved recommendation robustness.

---

## Recommendations

Only after hypotheses survive challenge are recommendations generated.

Recommendations therefore become downstream consequences of explicit reasoning rather than direct summaries of evidence.

This distinction remains fundamental throughout the current architecture.

---

## Recommendation Evaluation

Generating recommendations is insufficient.

The system also evaluates recommendation quality.

Dimensions include:

- evidence support

- reasoning quality

- trade-off awareness

- risk awareness

- actionability

These dimensions later became observable metrics inside evaluation reports.

---

## Recommendation Improvement

Evaluation initially served only as measurement.

Later milestones converted it into an active feedback loop.

Weak recommendations are revised and re-evaluated.

This transformed recommendation evaluation from passive scoring into an optimization process.

---

## Scenario Analysis

Real strategic decisions rarely depend upon a single future.

Scenario Analysis therefore evaluates recommendations under:

- Base Case

- Upside Case

- Downside Case

This shifted recommendation quality from

> "good under one future"

to

> "robust across multiple plausible futures."

Scenario analysis later became an important prerequisite for strategic option generation.

---

## Multi-profile reasoning

Earlier retrieval operated primarily within a single domain profile.

J6 expanded reasoning across multiple domain profiles.

Examples include simultaneously reasoning over:

- AI infrastructure

- transmission

- energy

- nuclear

Cross-profile synthesis represented another step away from document summarization and toward strategic reasoning.

---

## J6 Outcome

By the conclusion of J6 the system no longer behaved like a traditional research assistant.

Instead it possessed an explicit reasoning pipeline.

```
Evidence

↓

Hypotheses

↓

Challenges

↓

Recommendations

↓

Scenario Analysis
```

Although sophisticated, recommendations still existed largely as isolated outputs.

They lacked explicit representation of:

- assumptions

- risks

- opportunities

- strategic alternatives

These limitations motivated the J7 series.

---

# H-Series — Architectural Hardening

Between J6 and J7 several hardening milestones substantially improved architectural robustness.

Unlike feature milestones, these focused on correctness, observability and maintainability.

Examples included:

- parser robustness

- candidate pool preservation

- extraction redesign

- quantitative evidence scoring

- atomic evidence extraction

- contradiction precision

Although these milestones introduced relatively little user-visible functionality, they significantly strengthened confidence in the underlying reasoning architecture.

This distinction between

feature work

and

architectural hardening

has become an important project philosophy.

Major architectural changes are now routinely followed by dedicated validation milestones.

---

# J7 — Decision Intelligence

If J6 transformed the system into a reasoning engine,

J7 transforms it into a decision engine.

The defining architectural change introduced during J7 is explicit representation of strategic reasoning.

Rather than generating recommendations directly,

the system now constructs a persistent decision graph.

---

## Strategic Engagement

Every strategic investigation now begins with a Strategic Engagement.

An engagement captures:

- client

- strategic brief

- decision context

- stakeholders

- constraints

- deadlines

The engagement is intentionally broader than a research question.

Research questions become one consequence of an engagement rather than its starting point.

---

## Decision Model

The Decision Model became the canonical strategic object.

Rather than embedding decision context throughout the system,

the Decision Model owns:

- objectives

- investigation areas

- decision criteria

- constraints

- assumptions

- risks

- opportunities

- strategic options

Every subsequent reasoning object ultimately relates back to the Decision Model.

This significantly simplified architectural evolution because new reasoning capabilities now extend the Decision Model rather than inventing parallel representations.

---

## Strategic Assumptions

Earlier versions generated recommendations directly from evidence.

J7 introduced explicit assumptions.

Every recommendation now depends upon assumptions.

Assumptions answer:

> What must be true for this recommendation to remain valid?

This architectural decision fundamentally changed the system.

Reasoning became explicit.

Future capabilities—including risks, opportunities and confidence—naturally attach to assumptions.

---

## Recommendation Linkage

Recommendations are now linked explicitly to supporting assumptions.

Importantly,

these relationships are deterministic.

Rather than repeatedly asking the language model,

the system derives linkage from existing reasoning objects.

This philosophy reduces cost,

improves reproducibility,

and simplifies testing.

---

## Strategic Risks

Risks are derived from assumptions.

Not from recommendations.

This is intentional.

Risks describe

what could invalidate an assumption,

not merely

what could prevent implementation.

This distinction produces considerably more meaningful strategic analysis.

---

## Strategic Opportunities

Opportunities complement risks.

Rather than describing failure,

they describe additional value created when assumptions prove more favourable than expected.

Together,

Risks

and

Opportunities

form the positive and negative futures surrounding every strategic assumption.

---

## Strategic Options

The final major J7 capability completed at this checkpoint is Strategic Options.

Earlier versions generated recommendations directly.

The system now constructs multiple coherent strategic options.

Each option represents an internally consistent strategic posture.

Examples include:

- aggressive investment

- phased deployment

- conservative adoption

Only after evaluating competing options does the system recommend a preferred strategic posture.

This architecture much more closely resembles real consulting practice.

---

## J7.5 Refinements

Several refinement milestones followed.

These included:

- evidence-driven option cardinality

- strategic assumption prioritisation

- opportunity persistence hardening

These milestones did not substantially alter architecture.

Instead they improved:

- graph quality

- graph consistency

- executive relevance

By the end of J7.5 the system had evolved into an explicit strategic reasoning graph.

```
Evidence

↓

Strategic Assumptions

├──────── Risks

├──────── Opportunities

└──────── Recommendations

↓

Strategic Options
```

Unlike earlier architectures,

this graph is persistent,

traceable,

observable,

and designed for future expansion.

It represents the architectural foundation upon which the remaining decision-support capabilities (Decision Analysis, Executive Confidence and Executive Reporting) will be constructed.

# Current System Architecture

At the conclusion of J7.5, the Agent Harness Research System consists of three major architectural layers.

```
Strategic Layer

    Strategic Engagement

            ↓

      Decision Model



Reasoning Layer

Evidence

↓

Hypotheses

↓

Challenges

↓

Assumptions

├──────── Risks

├──────── Opportunities

└──────── Recommendations

↓

Strategic Options



Execution Layer

Functional Agents

↓

Research Object

↓

Evaluation

↓

Executive Output
```

Each layer owns a distinct responsibility.

The Strategic Layer defines the decision being made.

The Reasoning Layer determines what should be recommended.

The Execution Layer performs the work required to construct that reasoning.

This separation has become one of the defining characteristics of the project.

---

# Repository Structure

The repository has gradually evolved into a collection of relatively independent subsystems.

The current top-level layout is:

```text
baseline/
eval/
evals/
functional_agents/
outputs/
profile_editor/
profiles/
research_agent/
research_objects/
scripts/
sources/
smr_sources/
tests/
```

The most important directories are:

### research_agent/

Contains the core research engine.

Responsibilities include:

- document loading
- chunking
- evidence extraction
- retrieval
- contradiction detection
- evaluation
- research objects
- decision models
- engagement objects
- CLI
- benchmark framework

This package should increasingly be viewed as the project's core library rather than its primary execution path.

---

### functional_agents/

Contains the production reasoning pipeline.

Each reasoning stage is implemented as an explicit functional agent.

Current agents include:

- ProblemFramingAgent
- ResearchStrategyAgent
- PlannerAgent
- EvidenceAgent
- HypothesisAgent
- ChallengeAgent
- AssumptionAgent
- RecommendationAgent
- RecommendationImprovementAgent
- RiskAgent
- OpportunityAgent
- StrategicOptionAgent
- ScenarioAgent
- MultiProfileAgent
- QAAgent
- ReportAgent

The orchestrator coordinates these agents through a shared AgentContext.

This directory now represents the primary production implementation.

---

### profiles/

Domain profiles.

Examples include:

- ai_data_centers
- transmission
- smr

Profiles define retrieval priorities and domain-specific planning behaviour.

---

### eval/

Gold benchmark dataset.

Used by the evaluation harness.

---

### baseline/

Reference benchmark results.

Used for regression testing.

---

### outputs/

Contains all generated artifacts.

Examples include:

- executive reports
- traces
- research objects
- decision models
- engagements
- evaluation reports
- regression reports

---

### tests/

Automated validation.

As of this checkpoint the project contains approximately 1,755 passing tests.

The test suite validates:

- contracts
- orchestration
- persistence
- reasoning objects
- recommendation quality
- strategic options
- assumptions
- benchmark behaviour

---

# Canonical Objects

The architecture deliberately limits the number of primary persistent objects.

---

## Strategic Engagement

Represents the client engagement.

Typical contents include:

- client
- strategic brief
- stakeholders
- engagement type
- constraints
- deadlines

The Engagement answers:

> Why are we performing this research?

---

## Decision Model

The Decision Model is now the canonical strategic object.

It represents the decision itself.

Current contents include:

- objectives
- investigation areas
- decision criteria
- assumptions
- risks
- opportunities
- strategic options

Future reasoning capabilities should extend the Decision Model whenever practical rather than creating parallel representations.

---

## Research Object

The Research Object represents execution.

Typical contents include:

- evidence
- hypotheses
- challenges
- recommendations
- evaluation
- traces
- metadata

Where the Decision Model captures strategic reasoning,

the Research Object captures the research process that produced it.

---

# Functional Agent Pipeline

The current production pipeline is:

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

Scenario

↓

Multi-Profile

↓

QA

↓

Report
```

The ordering is intentional.

Every downstream reasoning object depends upon upstream reasoning.

Future agents should preserve this philosophy.

---

# Current Entry Points

Three execution paths currently exist.

---

## Legacy Research CLI

```
python -m research_agent.cli
```

Originally the primary interface.

Now primarily useful for:

- debugging
- isolated research
- backwards compatibility

---

## Functional Agent CLI

This is now the preferred production entry point.

```
python3 -m functional_agents.cli run \
    "<goal>" \
    --profiles ai_data_centers \
    --web-search \
    --out outputs/report.md \
    --log-level PROGRESS
```

This path executes the complete functional reasoning pipeline.

All new architectural development should assume this is the canonical execution path.

---

## Benchmark Runner

```
python3 -m research_agent.eval_runner benchmark
```

Runs the gold evaluation dataset.

---

## Regression Runner

```
python3 -m research_agent.eval_runner regress
```

Compares current benchmark results against the baseline.

Regression testing is mandatory for all architectural milestones.

---

# Evaluation Workflow

The recommended development workflow is now:

```
Implement

↓

Functional Agent Run

↓

Benchmark

↓

Regression

↓

Architecture Review
```

This workflow has proven extremely effective.

Major architectural improvements have been made while preserving objective quality measurements.

---

# Current Quality

At the time of this checkpoint:

- QA benchmark: 23 / 23
- Contradiction benchmark: 8 / 8
- Overall benchmark score: 100%
- Regression status: PASS

These metrics provide confidence that the architectural evolution has not compromised factual quality.

---

# Design Philosophy

Several architectural themes have emerged organically during development.

These are worth preserving.

---

## Explicit reasoning over implicit prompting

The project increasingly represents reasoning as persistent objects.

Reasoning should not disappear inside prompts.

---

## Graphs over pipelines

The system began as a pipeline.

It is becoming a graph.

Graphs are easier to:

- inspect
- validate
- extend
- reason about

Future work should continue this transition.

---

## Deterministic relationships

Whenever relationships can be computed rather than inferred by an LLM,

they should be computed.

This improves:

- repeatability
- explainability
- testing

---

## Canonical ownership

Every important concept should have a single owner.

Examples:

Decision Model

Research Object

Strategic Engagement

Avoid duplicated truth.

---

## Executive-first thinking

The intended audience is:

- CEOs
- investors
- senior government officials

The architecture should increasingly optimise for strategic clarity rather than technical completeness.

---

# Current Limitations

Several capabilities remain intentionally incomplete.

These include:

- explicit comparison between strategic options
- decision confidence
- executive report redesign
- interactive decision support

These limitations define the remaining work for J7 and J8.

---

# Roadmap

The immediate roadmap is:

```
J7.6

Decision Analysis

↓

J7.7

Executive Confidence

↓

J8

Executive Reporting
```

Decision Analysis will compare competing strategic options explicitly.

Executive Confidence will assess confidence in the selected strategy rather than individual reasoning objects.

J8 will redesign report generation around the completed decision graph.

The objective is no longer to produce a research memorandum.

The objective is to produce a board-quality strategic decision paper.

---

# Lessons Learned

Looking back across the evolution of the project, several lessons stand out.

The first is that architecture matters more than prompts.

Many capabilities initially appeared to require increasingly sophisticated prompt engineering.

In practice, introducing explicit reasoning objects produced larger improvements than prompt optimisation alone.

The second is that validation milestones are essential.

Hardening work such as H1, H2, J6.8a, J7.1a, and J7.5c repeatedly uncovered architectural issues that feature development alone would not have exposed.

The third is that persistent reasoning objects dramatically simplify future evolution.

Each new capability introduced during J7 attached naturally to the Decision Model because earlier milestones had established clear ownership boundaries.

Finally, the project has steadily shifted from information retrieval toward strategic reasoning.

This was never planned as a single milestone.

Instead, it emerged organically through successive architectural improvements.

That evolution now defines the identity of the system.

---

# Closing Remarks

The Agent Harness Research System began as an experiment in evidence extraction.

It evolved into a research platform.

It then became a functional multi-agent reasoning system.

At the conclusion of J7.5, it has become something more significant:

a strategic decision-support platform built around explicit, persistent, evidence-based reasoning.

Future work will focus less on retrieving information and increasingly on helping decision-makers understand uncertainty, compare strategic choices, and make better long-term decisions.

The architecture described in this document should therefore be viewed not as the conclusion of the project, but as the foundation upon which the next generation of executive decision-support capabilities will be built.