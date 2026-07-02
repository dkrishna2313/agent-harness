# Agent Execution Fixtures (PH2.0)

Fixtures are serialized `AgentContext` snapshots that let you run **one**
Functional Agent in isolation via the agent-level execution harness
(`functional_agents/run_agent.py`), without running the full pipeline.

```bash
python -m functional_agents.run_agent --agent planner --fixture fixtures/planner_start.json --no-llm
```

The harness does not modify any architectural contract — it is an additional
execution path. See the milestone: **PH2.0 — Agent-Level Execution Harness**.

## Fixture format

A fixture is a JSON object whose keys are `AgentContext` fields (the durable
ones). Unknown keys are ignored (with a warning); omitted fields fall back to
`AgentContext` defaults. Each fixture holds the **minimum valid upstream state**
the target agent needs.

| Fixture | Target agent | Minimum state it provides |
|---|---|---|
| `planner_start.json` | planner | `question`, `profiles`, `decision_model` |
| `evidence_start.json` | evidence | + `plan` (subquestions / investigation_areas) |
| `hypothesis_start.json` | hypothesis | + `evidence_notes[0].evidence_items` |
| `recommendation_start.json` | recommendation | + `hypotheses`, `surviving_hypotheses`, `hypothesis_challenges` |
| `report_start.json` | report | + `recommendations`, `qa` (memo is synthesized by the harness) |

`fixtures/sources/` holds a small source document used by the evidence agent's
legacy extraction path when no Knowledge Store is available.

## Preconditions / postconditions

Each agent declares preconditions (required upstream state) and postconditions
(expected outputs). Failures are deterministic and actionable, e.g.:

```
ERROR: Precondition failed for agent: evidence_notes present
```

| Agent | Precondition | Postcondition |
|---|---|---|
| planner | question or reasoning target present | `plan.research_type` produced |
| evidence | `plan` present | `evidence_notes` produced |
| hypothesis | `evidence_notes` present | `hypotheses` produced |
| strategic_synthesis | hypotheses / domain_hypotheses present | `strategic_synthesis` produced |
| recommendation | `hypotheses` present | `recommendations` produced |
| report | `question` present | report artifact produced |

## Output

- `--output ctx.json` — writes the updated context (JSON-safe; trace scratch and
  `_`-prefixed keys are stripped).
- `--trace mini.trace.json` — writes the mini trace: execution time, LLM call
  count, normalization result, validation, objects produced, warnings.
- Stdout always prints a summary + **context diff** (added / modified / unchanged
  fields).

`--no-llm` forces the deterministic mock client (also used automatically when
`ANTHROPIC_API_KEY` is unset).

## Creating a new fixture

Two ways:

1. **Hand-author** (as the shipped fixtures are): start from the table above and
   add the minimum upstream fields for your agent. Validate by running the
   harness — a precondition failure tells you exactly what's missing.

2. **Snapshot from a real run**: run the mock pipeline
   (`python3 -m functional_agents run "<question>" --mock --out /tmp/r.md`),
   then take `outputs/latest_research_object.json` (or the run's context) and
   trim it to the fields your target agent consumes. Save under `fixtures/`.

Fixtures should stay minimal — include only what the agent's preconditions
require, so the fixture documents the agent's true input contract.
