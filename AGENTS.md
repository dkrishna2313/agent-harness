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

# Dependency Reasoner Architecture (J13.2)

J13.2 introduces a deterministic staleness engine that analyzes StateChanges to
produce a StalenessPlan. It is analysis-only — the pipeline is unchanged.

## Path Classification

Every dependency path has exactly one kind:

| Kind | Meaning |
|---|---|
| `PERSISTED` | Stored in ResearchState; survives pipeline runs |
| `EXECUTION_ONLY` | In AgentContext only; not persisted; must be re-produced each run |
| `EXTERNAL` | Provided by environment (knowledge store, user); never recomputed |

## StalenessPlan

Output of `DependencyReasoner.analyze(research_state, state_changes)`:

| Field | Description |
|---|---|
| `changed_paths` | Paths explicitly changed (expanded from StateChange.affected_paths) |
| `stale_paths` | All PERSISTED + EXECUTION_ONLY paths now stale |
| `stale_agents` | Agents whose produces overlap stale_paths |
| `required_producers` | Agents needed to restore stale PERSISTED paths |
| `persisted_paths` | Subset of stale_paths with PathKind.PERSISTED |
| `execution_only_paths` | Subset with PathKind.EXECUTION_ONLY |
| `external_dependencies` | EXTERNAL paths that triggered changes |
| `reasoning` | path → human-readable explanation of why it is stale |
| `confidence` | HIGH (known specific paths) / MEDIUM (container expansion) / LOW (unknown paths) |

## Algorithm

BFS through the consumption graph:
1. Expand container paths from StateChanges (e.g. `research_state` → all PERSISTED sub-paths)
2. For each changed path: find agents that consume it → their produces become stale
3. Repeat until no new stale paths
4. EXTERNAL paths propagate BFS but are not themselves stale (cannot be recomputed)

## CLI

```
staleness explain --session <path>   # analyze all StateChanges in a session
staleness path    --path <path>      # synthetic analysis: what if this path changed?
staleness agent   --agent <name>     # synthetic analysis: what if this agent re-ran?
```

## Design constraints

- No execution. DependencyReasoner never calls agents.
- No incremental orchestration. Pipeline runs all 22 agents unchanged.
- No state diffing. Staleness is derived from declared dependencies, not observed state.
- Conservative: false positives acceptable; false negatives are not.

# Execution Planner Architecture (J13.3)

J13.3 introduces a deterministic Execution Planner that converts a StalenessPlan
into an ExecutionPlan.  It is analysis-only — the pipeline is unchanged.

## ExecutionPlan

Output of `ExecutionPlanner.plan(staleness_plan)`:

| Field | Description |
|---|---|
| `required_agents` | Agents needed to restore all stale PERSISTED paths, including EXECUTION_ONLY prerequisites |
| `optional_agents` | Stale agents producing only EXECUTION_ONLY paths not needed for PERSISTED restoration |
| `blocked_agents` | Agents whose consumed EXECUTION_ONLY path has no known producer (defensive; should be empty) |
| `execution_order` | Topological ordering of all planned agents (required + optional) |
| `execution_groups` | Agents grouped by topological level; agents within a group may run in parallel |
| `estimated_steps` | Number of sequential steps (len(execution_groups)) |
| `reasoning` | agent_name → why it appears in the plan |

## Algorithm

1. **required_agents** — start from `StalenessPlan.required_producers` (agents
   producing stale PERSISTED paths).  For each agent, find all EXECUTION_ONLY
   consumed paths and add their producers to the required set.  Repeat until
   fixpoint.  EXECUTION_ONLY paths are never in ResearchState, so their producers
   must always run even when the path is stale.

2. **optional_agents** — `stale_agents` not in `required_agents`; produce only
   EXECUTION_ONLY stale paths not needed by any required_agent.

3. **execution_groups** — Kahn's topological leveling over the planned subgraph.
   Only edges between agents within the plan are considered; dependencies on
   out-of-plan agents (PERSISTED paths read from ResearchState) are ignored.

## Key invariants

- `required_agents` is always a superset of `StalenessPlan.required_producers`.
- Every agent in `execution_order` appears exactly once.
- No intra-group dependency edges exist within a single execution_group.
- `estimated_steps = len(execution_groups)`.

## CLI

```
execution plan --session <path>             # compact execution plan summary
execution plan --session <path> --verbose   # full plan with per-agent reasoning
```

## Design constraints

- No execution. ExecutionPlanner never calls agents.
- No incremental orchestration. Pipeline runs all 22 agents unchanged.
- Conservative: if an agent needs an EXECUTION_ONLY input, its producer is required.

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