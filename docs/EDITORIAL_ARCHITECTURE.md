# Editorial Architecture

**Version:** PH6.2  
**Status:** Architecture Specification  
**Scope:** All executive deliverables produced by the Agent Harness

---

## 1. Guiding Principle

Separate Executive Reasoning from Executive Communication.

**Reasoning** determines: *What should leadership decide?*  
**Editorial** determines: *How should leadership consume that decision?*

The reasoning pipeline is an epistemic process. It synthesises evidence, evaluates options, identifies risks, and produces a recommendation. It must not be influenced by how the output will look.

The editorial layer is a communication process. It takes structured executive knowledge and renders it for a specific audience, format, and rhetorical purpose. It must not alter the knowledge it receives.

These are different concerns. Conflating them inside `ReportAgent` produces a class of defects that cannot be fixed without touching reasoning: truncation, grammar failures, awkward board prose, and brittle format-specific logic embedded in analysis code.

---

## 2. Current Architecture

```
Reasoning Pipeline
   ├── ResearchStrategyAgent
   ├── PlannerAgent
   ├── DecisionAnalysisAgent
   ├── EvidenceAgent
   └── (further agents)
         │
         ▼
   ReportAgent
   (reasoning + communication mixed)
         │
         ▼
   Markdown string
         │
         ▼
   docx_export.py
   (format conversion)
```

**Problems with the current model:**

- `ReportAgent` contains both reasoning (e.g. which risks are critical, how to weight assumptions) and presentation (e.g. section prose templates, truncation logic, board-prose lookups).
- Format-specific logic leaks into analysis code. The `[:140]` truncation bug, the `_BOARD_PROSE` dict, and the "If either fails" grammar defect are all presentation decisions embedded inside reasoning functions.
- Adding a new deliverable format (PPTX, HTML, Board Paper) requires modifying `ReportAgent` and risking reasoning regressions.
- Testing is difficult: a test that asserts prose output is implicitly testing both reasoning and presentation, making failures ambiguous.

---

## 3. Proposed Architecture

```
Reasoning Pipeline
   ├── ResearchStrategyAgent
   ├── PlannerAgent
   ├── DecisionAnalysisAgent
   ├── EvidenceAgent
   └── (further agents)
         │
         ▼
   Editorial Brief
   (structured executive knowledge — no formatting)
         │
         ▼
   Editorial Pipeline
   ├── ExecutiveSummaryWriter
   ├── DecisionAnalysisWriter
   ├── RecommendationWriter
   ├── RiskWriter
   ├── OpportunityWriter
   ├── ConfidenceWriter
   ├── AppendixWriter
   └── PresentationWriter
         │
         ▼
   Deliverable Renderers
   ├── MarkdownRenderer    → .md
   ├── DocxRenderer        → .docx
   ├── PptxRenderer        → .pptx
   ├── HtmlRenderer        → .html
   └── BoardPaperRenderer  → .docx (board template)
```

The Editorial Brief is the single handoff boundary. Everything to the left of it is reasoning. Everything to the right is communication.

---

## 4. The Editorial Brief

The Editorial Brief is the canonical handoff object between the reasoning pipeline and all deliverables. It contains structured executive knowledge. It contains no formatting instructions, no prose templates, and no presentation logic.

### 4.1 Schema

```
EditorialBrief
│
├── executive_summary
│   ├── recommended_option_title: str
│   ├── board_recommendation: str          # enum value, e.g. "Proceed with Conditions"
│   ├── decision_readiness: str            # e.g. "Needs Additional Validation"
│   ├── overall_confidence: str            # e.g. "Low"
│   ├── why_this_option: str               # full reasoning sentence — no truncation
│   └── key_conditions: list[str]          # conditions attached to the recommendation
│
├── decision
│   ├── question: str
│   ├── analysis_id: str
│   ├── recommended_option_id: str
│   └── rationale: str
│
├── strategic_options: list[StrategicOption]
│   └── (option_id, title, rationale, posture, time_horizon, rankings position,
│       decision_matrix scores, strengths, weaknesses)
│
├── decision_analysis
│   ├── comparison_dimensions: list[str]
│   ├── option_rankings: list[str]         # ordered list of option_ids
│   ├── key_tradeoffs: list[str]
│   ├── key_uncertainties: list[str]
│   ├── sensitivity_analysis: str
│   └── confidence_summary: str
│
├── recommendations: list[Recommendation]
│   └── (recommendation_id, title, summary, time_horizon, priority)
│
├── risks: list[Risk]
│   └── (risk_id, statement, severity, likelihood, mitigation_notes,
│       related_assumption_ids, affected_recommendation_ids)
│
├── opportunities: list[Opportunity]
│   └── (opportunity_id, statement, category, likelihood, impact)
│
├── assumptions: list[Assumption]
│   └── (assumption_id, statement, importance, confidence, evidence_support)
│
├── executive_confidence
│   ├── overall_confidence: str
│   ├── decision_readiness: str
│   ├── board_recommendation: str
│   └── validation_priorities: list[str]   # full strings — no truncation
│
└── supporting_evidence
    ├── total_evidence_items: int
    ├── citation_count: int
    ├── profiles: list[str]
    ├── evidence_topics: dict[str, int]
    └── citations: list[Citation]
```

### 4.2 Invariants

- All string fields are stored at full length. Truncation is a presentation decision and belongs in writers, not in the brief.
- All enum fields (board_recommendation, decision_readiness) carry their canonical string values. Prose rendering is a writer responsibility.
- The brief is read-only from the perspective of all writers. No writer may mutate it.
- The brief is format-agnostic. It contains no Markdown, no HTML, no DOCX-specific constructs.

---

## 5. The Editorial Pipeline

The Editorial Pipeline is an ordered set of independent section writers. Each writer owns exactly one rhetorical style and one section of the executive narrative.

Writers are stateless. Given the same Editorial Brief, a writer produces the same output. Writers do not call each other.

### 5.1 Writer Registry

| Writer | Rhetorical Style | Input Fields Consumed |
|---|---|---|
| `ExecutiveSummaryWriter` | Board-level executive opener | `executive_summary`, `decision` |
| `DecisionAnalysisWriter` | Analytical comparison narrative | `decision_analysis`, `strategic_options` |
| `RecommendationWriter` | Action-oriented, timeframe-grouped | `recommendations` |
| `RiskWriter` | Risk enumeration with mitigation context | `risks`, `assumptions` |
| `OpportunityWriter` | Opportunity framing with market context | `opportunities` |
| `ConfidenceWriter` | Confidence statement and validation priorities | `executive_confidence` |
| `AppendixWriter` | Evidence provenance and citation list | `supporting_evidence` |
| `PresentationWriter` | Slide-optimised condensed narrative | `executive_summary`, `decision_analysis`, `strategic_options` |

### 5.2 Writer Contract

Each writer satisfies the following contract:

```
Input:  One or more named fields from EditorialBrief (read-only)
Output: ExecutiveSection
   ├── section_id: str
   ├── title: str
   ├── content: str          # format-agnostic prose
   ├── structured_data: dict # tables, lists, decision matrices — raw data, not formatted
   └── metadata: dict        # word count, section type, rhetorical mode
```

The `content` field is natural executive prose. The `structured_data` field carries data that the downstream renderer will format for its target medium (Markdown table, DOCX table, PPTX slide, etc.).

**A writer may:**
- improve readability within its section
- improve structure and flow
- improve executive tone and precision
- choose appropriate rhetorical emphasis for board consumption

**A writer may NOT:**
- alter the reasoning, conclusions, or recommendations it receives
- invent conditions, risks, or opportunities not present in the brief
- remove caveats or weaken specificity
- change evidence or citation data
- make formatting decisions that belong to the renderer

---

## 6. Deliverable Renderers

Renderers are format-specific. They take `list[ExecutiveSection]` (the output of the editorial pipeline) and produce a deliverable. Renderers contain no rhetorical logic — only format translation.

### 6.1 Markdown Renderer

Translates `ExecutiveSection` objects to GitHub-flavoured Markdown. Produces the `.md` output that currently comes from `ReportAgent`. Section prose maps to paragraphs and blockquotes. Structured data maps to pipe tables and ordered lists.

### 6.2 DOCX Renderer

Translates `ExecutiveSection` objects to a Word document via `python-docx`. The existing `docx_export.py` becomes a thin renderer that consumes structured sections rather than parsing Markdown. Table data maps to formatted Word tables. Blockquotes map to styled Word paragraphs with the blue-grey left border. No Markdown parsing is required.

### 6.3 PPTX Renderer

Translates `ExecutiveSection` objects — specifically those produced by `PresentationWriter` — to a PowerPoint deck via `python-pptx`. Each section maps to one or more slides. The renderer applies slide templates, applies master styling, and positions content boxes. No section writer is invoked inside the renderer.

### 6.4 HTML Renderer

Translates `ExecutiveSection` objects to a self-contained HTML file. Supports embedded CSS for executive styling. Suitable for browser-based review and email distribution.

### 6.5 Board Paper Renderer

A specialised DOCX renderer that applies a formal board-paper template: numbered sections, defined headers/footers, regulated typography. Consumes the same `list[ExecutiveSection]` as the standard DOCX renderer. Board Paper format is a renderer concern, not a writer concern — the editorial content is identical.

---

## 7. Writer Isolation

The principle of writer isolation has two dimensions:

**Input isolation.** Each writer consumes only the fields it needs. `RiskWriter` receives `risks` and `assumptions`. It does not receive `strategic_options`. This prevents a writer from accidentally incorporating content that belongs to a different section, and makes the data dependency of each section explicit.

**Output isolation.** Each writer produces its section independently. No writer depends on the output of another writer. The editorial pipeline may run writers in any order (or in parallel) and assemble sections at the rendering stage.

This isolation makes each writer independently testable. A `RiskWriter` test asserts that a given `risks` list produces a specific risk narrative — without running the rest of the pipeline.

---

## 8. Editorial Principles

The editorial layer operates under a strict separation of responsibilities.

The editorial layer exists to make executive knowledge consumable. It does not exist to reinterpret, soften, or amplify that knowledge. A board reading the deliverable should be reading the reasoning pipeline's conclusions — expressed clearly, not altered.

### 8.1 Permitted editorial actions

- Restructure prose for clarity and flow within a section
- Replace passive constructions with active voice
- Replace enum strings with natural board prose (e.g. "Proceed with Conditions" → "Leadership should proceed — with the conditions outlined below")
- Group recommendations by time horizon for scanability
- Lead with the most important finding in each section
- Omit empty sections (e.g. no opportunities recorded → no opportunities section)

### 8.2 Prohibited editorial actions

- Alter recommendations
- Invent conclusions not present in the brief
- Remove important caveats or qualifications
- Weaken specificity (e.g. replacing a precise confidence level with vague language)
- Change evidence item counts or citation data
- Introduce risk or assumption IDs not present in the brief
- Truncate strings that the reasoning pipeline produced at full length

---

## 9. Migration Path

The Editorial Architecture does not require a full rewrite. The migration can proceed incrementally:

**Phase 1 — Define the Editorial Brief.** Introduce `EditorialBrief` as a Pydantic model. Add a `build_editorial_brief(ctx: AgentContext) -> EditorialBrief` function that extracts structured knowledge from the existing `AgentContext`. `ReportAgent` continues to produce Markdown as before; the brief is produced in parallel as a validation artifact.

**Phase 2 — Extract writers.** Migrate section-building functions out of `ReportAgent` into standalone writer classes. Each writer consumes `EditorialBrief` fields directly. `ReportAgent` becomes an orchestrator that calls writers and assembles their output into Markdown.

**Phase 3 — Introduce the Markdown Renderer.** Separate the Markdown formatting logic from the writer prose logic. Writers produce `ExecutiveSection` objects; the Markdown Renderer produces the `.md` file. `ReportAgent` is now a thin coordinator.

**Phase 4 — Add DOCX and PPTX Renderers.** `docx_export.py` is refactored to consume `list[ExecutiveSection]` directly, eliminating Markdown parsing. The PPTX renderer is introduced as a new deliverable.

**Phase 5 — Retire ReportAgent.** The `ReportAgent` function is replaced by the full Editorial Pipeline. The pipeline is the entry point; the renderer is selected by the caller.

---

## 10. Future Extensions

### 10.1 PresentationWriter as a parallel editorial track

`PresentationWriter` consumes the same `EditorialBrief` as every other writer but produces a condensed, slide-optimised narrative. A board presentation requires different rhetorical choices than a written report: shorter sentences, higher-level claims, fewer supporting details per slide. `PresentationWriter` applies these choices without access to any format-specific knowledge. The PPTX Renderer then translates the writer's output into slides.

This means the same reasoning pipeline run produces both a 10-section written report and a board slide deck — from the same `EditorialBrief`, with zero duplication of reasoning code.

### 10.2 The Editorial Brief as the canonical communication object

The `EditorialBrief` should become the canonical persistence object for executive communication. Storing it enables:

- **Re-rendering**: regenerate a DOCX or PPTX from a historical brief without re-running the reasoning pipeline.
- **Diffing**: compare two briefs from different pipeline runs to show what changed in the executive knowledge, independent of formatting.
- **Audit trail**: the brief captures the exact structured knowledge that was communicated to leadership, separate from the pipeline's internal reasoning state.
- **Format flexibility**: add a new deliverable format (e.g. a regulatory board paper, an investor memo) by writing a new renderer — no reasoning code is touched.

### 10.3 Writer versioning

Each writer can carry an independent version. A `ConfidenceWriter v2` can be introduced alongside `ConfidenceWriter v1` and A/B tested against the same brief without touching any other part of the pipeline. This enables controlled iteration on editorial quality without reasoning risk.

---

## 11. Glossary

| Term | Definition |
|---|---|
| **Editorial Brief** | The canonical handoff object. Structured executive knowledge, no formatting. |
| **Writer** | A stateless function that consumes one or more Editorial Brief fields and produces an `ExecutiveSection`. |
| **ExecutiveSection** | The output of a single writer: section title, prose content, structured data, metadata. |
| **Renderer** | A format-specific translator that consumes `list[ExecutiveSection]` and produces a deliverable file. |
| **Editorial Pipeline** | The ordered set of writers that transforms an Editorial Brief into a list of sections. |
| **Rhetorical Style** | The communication approach a writer applies: e.g. analytical, action-oriented, board-level summary. |
| **Reasoning Pipeline** | The upstream agent chain that produces the knowledge captured in the Editorial Brief. |

---

*This document specifies the Editorial Architecture at the design level. It does not prescribe implementation details, class hierarchies, or file structure. Those decisions belong to the implementation phase.*
