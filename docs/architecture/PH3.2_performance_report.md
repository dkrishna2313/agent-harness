# Platform Performance Report (PH3.1 / PH3.2)

**Source:** `outputs/ph32_mock_pipeline.trace.json`

## Pipeline totals

| Metric | Value |
|---|--:|
| Pipeline wall time | 18,232 ms |
| LLM generation | 0 ms (0.0%) |
| Non-LLM | 18,232 ms (100.0%) |
| LLM calls | 0 |
| Total tokens | 0 (prompt 0 / completion 0) |
| Active agents | 16 |

## Per-agent timing (by wall time)

| Agent | Wall | LLM | Non-LLM | LLM % | Calls | Tokens |
|---|--:|--:|--:|--:|--:|--:|
| EvidenceAgent | 18,061 | 0 | 18,061 | 0.0 | 0 | 0 |
| ReportAgent | 35 | 0 | 35 | 0.0 | 0 | 0 |
| DecisionAnalysisAgent | 27 | 0 | 27 | 0.0 | 0 | 0 |
| ExecutiveConfidenceAgent | 24 | 0 | 24 | 0.0 | 0 | 0 |
| AssumptionAgent | 22 | 0 | 22 | 0.0 | 0 | 0 |
| OpportunityAgent | 21 | 0 | 21 | 0.0 | 0 | 0 |
| RiskAgent | 20 | 0 | 20 | 0.0 | 0 | 0 |
| StrategicOptionAgent | 16 | 0 | 16 | 0.0 | 0 | 0 |
| PlannerAgent | 3 | 0 | 3 | 0.0 | 0 | 0 |
| HypothesisAgent | 2 | 0 | 2 | 0.0 | 0 | 0 |
| RecommendationAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| MultiProfileAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| QAAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| RecommendationImprovementAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| ChallengeAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| ScenarioAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |

## Non-LLM time by stage

| Stage | Duration |
|---|--:|
| normalization | 0 ms |
| validation | 0 ms |
| report_generation | 0 ms |
| _unattributed_ | 18,231 ms |

## Largest bottlenecks

**By agent (wall time):**

1. EvidenceAgent — 18,061 ms (LLM 0 / non-LLM 18,061)
2. ReportAgent — 35 ms (LLM 0 / non-LLM 35)
3. DecisionAnalysisAgent — 27 ms (LLM 0 / non-LLM 27)
4. ExecutiveConfidenceAgent — 24 ms (LLM 0 / non-LLM 24)
5. AssumptionAgent — 22 ms (LLM 0 / non-LLM 22)

**By non-LLM stage:**

1. normalization — 0 ms
2. validation — 0 ms
3. report_generation — 0 ms

## Prompt Efficiency (PH3.2 — measured, not yet applied)

Context-compaction opportunity measured for 5 agent(s) against their documented context profile. This is diagnostic only — no prompt sent to the LLM has changed.

| Agent | Original Context Size | Compacted Context Size | Reduction % | Est. Tokens Saved |
|---|--:|--:|--:|--:|
| RecommendationAgent | 397,610 tok | 32,860 tok | 91.7% | 364,750 tok |
| HypothesisAgent | 395,280 tok | 31,017 tok | 92.2% | 364,263 tok |
| ReportAgent | 401,926 tok | 401,847 tok | 0.0% | 79 tok |
| PlannerAgent | 14 tok | 14 tok | 0.0% | 0 tok |
| EvidenceAgent | 151 tok | 151 tok | 0.0% | 0 tok |
| **Total** | **1,194,981 tok** | **465,889 tok** | **61.0%** | **729,092 tok** |
