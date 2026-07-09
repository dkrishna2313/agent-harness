# Project Purpose

Build a local AI research harness focused on AI data center infrastructure.

Primary research domains:

- NVIDIA Rubin
- NVIDIA Vera
- NVIDIA Blackwell
- AI factories
- Data center power systems
- Power distribution
- Cooling infrastructure
- Networking infrastructure
- Rack architecture

# Current Version

Version: v1

Goals:

- Local files only
- Claude API
- Markdown output
- Warning-mode evaluation

Non-goals:

- Web search
- Vector databases
- Multi-agent systems
- Memory systems
- Dashboards

# Workflow

Question
→ Research Plan
→ Evidence Extraction
→ Synthesis
→ Evaluation
→ Markdown Memo

# Output Requirements

Every memo must contain:

- Executive Summary
- Confirmed Facts
- Inferences
- Power Implications
- Cooling Implications
- Open Questions
- Source Notes
- Evaluation Warnings

# Research Session Architecture (J13.0)

The primary unit of work is a **ResearchSession**, not a pipeline execution.

Every `Orchestrator` run transparently creates a transient `ResearchSession`,
persists it, and saves the final state after the pipeline completes.

## Session Package

`functional_agents/session/` contains:

| Module | Class | Responsibility |
|---|---|---|
| `research_state.py` | `ResearchState` | Canonical mutable state owning reasoning artifacts |
| `iteration_record.py` | `IterationRecord` | Immutable record of one pipeline execution |
| `snapshot.py` | `Snapshot` | Full ResearchState capture at a point in time |
| `research_session.py` | `ResearchSession` | First-class persistent unit of work |
| `session_store.py` | `SessionStore` | JSON-backed persistence layer |

## ResearchState artifacts

ResearchState owns the current reasoning artifacts:

| Field | Source |
|---|---|
| `engagement` | `context.engagement` |
| `research_object` | `context.research_object` |
| `decision_model` | `context.decision_model` |
| `research_gap_analysis` | `context.research_gap_analysis` |
| `executive_confidence` | `context.executive_confidence` |
| `iteration_plan` | `context.iteration_plan` |

## Session lifecycle

```
ResearchSession.create()
  └── initial ResearchState from context (before pipeline)
  └── IterationRecord(trigger="initial")
  └── SessionStore.create() → outputs/sessions/{session_id}.json

Orchestrator.run()
  └── pipeline executes unchanged

Post-run:
  └── session.research_state = ResearchState.from_context(result_ctx)
  └── session.take_snapshot()
  └── session.complete()
  └── SessionStore.save()
```

## Session files

Sessions are persisted as JSON in `outputs/sessions/{session_id}.json`.

Session IDs use format: `SS-{YYYYMMDD}-{HHMMSS}-{hex6}`

## Design constraints

- No execution logic in ResearchState.
- No agent sees a ResearchSession — it is orchestrator-level only.
- All session operations are best-effort: failures are logged, never propagated.
- Backward compatibility: existing CLI, benchmarks, and agent contracts unchanged.

# Engineering Principles

- Keep implementation simple.
- Prefer readable code over clever code.
- Use Pydantic schemas.
- Add tests for core logic.
- Fail gracefully.
- Log useful diagnostics.

# Domain Guidance

For infrastructure questions, prioritize:

1. NVIDIA primary sources
2. OCP specifications
3. Infrastructure vendor documentation
4. Independent technical analysis

Clearly distinguish:
- Confirmed facts
- Reasoned inferences
- Speculation