# Strategic Consulting Platform

An agentic pipeline that transforms research questions and strategic engagement briefs
into board-ready consulting deliverables — hypotheses, recommendations, strategic options,
scenarios, and an executive report — grounded in evidence from your document corpus or
knowledge store.

## Install

```bash
python3 -m pip install -e ".[dev]"
```

## Quick Start

Run with a structured Strategic Engagement file:

```bash
python3 -m functional_agents.cli run \
  --engagement engagements/hyperscaler_ai_strategy.yaml \
  --profiles ai_data_centers,transmission \
  --mock
```

Or with a direct research question:

```bash
python3 -m functional_agents.cli run \
  "What are the power and cooling requirements for AI data centers?" \
  --profiles ai_data_centers \
  --sources ./sources \
  --mock
```

Each run automatically creates a timestamped output directory:

```
outputs/runs/RUN-20260705-103422/
    report.md
    pipeline.trace.json
    research_object.json
    engagement.json
```

## CLI Reference

```
python3 -m functional_agents.cli run [OPTIONS] [QUESTION]
```

**Input (choose one):**

| Option | Description |
|---|---|
| `QUESTION` | Research question (positional argument) |
| `--goal TEXT` | Business goal — ProblemFramingAgent derives questions automatically |
| `--engagement PATH` | Strategic Engagement YAML/JSON file |

**Key options:**

| Option | Default | Description |
|---|---|---|
| `--profiles TEXT` | `ai_data_centers` | Comma-separated domain profile names |
| `--sources DIR` | `sources/` | Source documents directory |
| `--out PATH` | auto (run dir) | Report output path |
| `--mock` | off | Use deterministic mock client (no API key required) |
| `--knowledge-store DIR` | auto-detect | Knowledge Store directory (hybrid retrieval) |
| `--rerank` | off | Apply LLM reranking to retrieved evidence |
| `--log-level LEVEL` | `PROGRESS` | `DEBUG`, `INFO`, `PROGRESS`, `WARNING`, `ERROR` |
| `--model TEXT` | — | Override Anthropic model |

## Configuration

Set `ANTHROPIC_API_KEY` to run with Claude:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python3 -m functional_agents.cli run \
  --engagement engagements/hyperscaler_ai_strategy.yaml \
  --profiles ai_data_centers,transmission
```

The default model is `claude-sonnet-4-6`. Override with `--model` or `ANTHROPIC_MODEL`.

For deterministic local testing without an API key:

```bash
python3 -m functional_agents.cli run \
  --engagement engagements/hyperscaler_ai_strategy.yaml \
  --profiles ai_data_centers,transmission \
  --mock
```

## Knowledge Store

When a `knowledge_store/` directory is present (or specified via `--knowledge-store`),
the platform uses hybrid semantic + lexical retrieval with optional LLM reranking instead
of legacy document extraction. Build a knowledge store with:

```bash
python3 -m knowledge.cli ingest \
  --sources ./sources \
  --profiles ai_data_centers \
  --out ./knowledge_store
```

## Pipeline Architecture

The platform runs 18 functional agents in sequence:

```
ProblemFramingAgent → ResearchStrategyAgent → PlannerAgent → EvidenceAgent
→ HypothesisAgent → StrategicSynthesisAgent → ChallengeAgent
→ AssumptionAgent → RiskAgent → OpportunityAgent
→ RecommendationAgent → StrategicOptionAgent → DecisionAnalysisAgent
→ ExecutiveConfidenceAgent → ScenarioAgent → QAAgent
→ RecommendationImprovementAgent → ReportAgent
```

Each agent writes its outputs to `AgentContext` (a shared in-memory state object).
`ReportAgent` assembles everything into a Markdown deliverable.

## Run Summary

After each run the CLI prints:

```
Run:      3f7a2c9d1e8b
Mode:     Strategic Engagement
Profiles: ai_data_centers, transmission
Status:   SUCCESS
Agents:   18 run
Elapsed:  1.4s

Deliverables:
  [OK] Report             outputs/runs/RUN-20260705-103422/report.md
  [OK] Pipeline trace     outputs/runs/RUN-20260705-103422/pipeline.trace.json
  [OK] Research object    outputs/runs/RUN-20260705-103422/research_object.json
  [OK] Engagement         outputs/runs/RUN-20260705-103422/engagement.json

Output:   outputs/runs/RUN-20260705-103422/
```

## Logging

Use `--log-level PROGRESS` (default) to see agent progress messages during the run.
Use `--log-level DEBUG` for full diagnostic output.
Use `--log-level WARNING` for silent production mode.

## Tests

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_ph43_ux.py -v
```

## Engagement Files

Strategic Engagement files define the client context and research objectives:

```yaml
engagement:
  title: "Hyperscaler AI Infrastructure Strategy"
  client: "Hyperscaler Inc."
  industry: "AI Infrastructure"
  current_situation: "Planning a 500 MW AI data center campus."
  objectives:
    - "Identify power acquisition strategies"
    - "Determine cooling architecture for GB300 deployments"
  constraints:
    - "24-month deployment window"
    - "Net-zero commitment by 2030"
  stakeholders:
    - "CIO"
    - "Energy Procurement"
  decision_horizon: "24 months"
```

Example files are in `engagements/`.
