# Platform Performance Report (PH3.1 / PH3.2 / PH3.3)

**Source:** `outputs/ph32_mock_pipeline.trace.json`

## Pipeline totals

| Metric | Value |
|---|--:|
| Pipeline wall time | 17,056 ms |
| LLM generation | 0 ms (0.0%) |
| Non-LLM | 17,056 ms (100.0%) |
| LLM calls | 0 |
| Total tokens | 0 (prompt 0 / completion 0) |
| Active agents | 17 |

## Per-agent timing (by wall time)

| Agent | Wall | LLM | Non-LLM | LLM % | Calls | Tokens |
|---|--:|--:|--:|--:|--:|--:|
| EvidenceAgent | 16,900 | 0 | 16,900 | 0.0 | 0 | 0 |
| ReportAgent | 31 | 0 | 31 | 0.0 | 0 | 0 |
| ExecutiveConfidenceAgent | 23 | 0 | 23 | 0.0 | 0 | 0 |
| AssumptionAgent | 22 | 0 | 22 | 0.0 | 0 | 0 |
| DecisionAnalysisAgent | 20 | 0 | 20 | 0.0 | 0 | 0 |
| OpportunityAgent | 18 | 0 | 18 | 0.0 | 0 | 0 |
| RiskAgent | 17 | 0 | 17 | 0.0 | 0 | 0 |
| StrategicOptionAgent | 15 | 0 | 15 | 0.0 | 0 | 0 |
| PlannerAgent | 6 | 0 | 6 | 0.0 | 0 | 0 |
| RecommendationAgent | 2 | 0 | 2 | 0.0 | 0 | 0 |
| HypothesisAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| StrategicSynthesisAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| QAAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| RecommendationImprovementAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| ChallengeAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| MultiProfileAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |
| ScenarioAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |

## Non-LLM time by stage

| Stage | Duration |
|---|--:|
| normalization | 0 ms |
| validation | 0 ms |
| report_generation | 0 ms |
| _unattributed_ | 17,056 ms |

## Largest bottlenecks

**By agent (wall time):**

1. EvidenceAgent — 16,900 ms (LLM 0 / non-LLM 16,900)
2. ReportAgent — 31 ms (LLM 0 / non-LLM 31)
3. ExecutiveConfidenceAgent — 23 ms (LLM 0 / non-LLM 23)
4. AssumptionAgent — 22 ms (LLM 0 / non-LLM 22)
5. DecisionAnalysisAgent — 20 ms (LLM 0 / non-LLM 20)

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

## Applied Prompt Slicing (PH3.3 — actually sent to the LLM call)

Real byte-size reduction applied to 4 agent(s)' LLM call inputs. Every excluded field was verified unread by both the live prompt builder and MockClaudeClient, so this cannot change prompt wording, reasoning, or output.

| Agent | Original Size | Sliced Size | Reduction % | Bytes Saved | Fields Excluded |
|---|--:|--:|--:|--:|--:|
| StrategicSynthesisAgent | 123,967 B | 52 B | 100.0% | 123,915 B | 1 |
| PlannerAgent | 47 B | 47 B | 0.0% | 0 B | 0 |
| HypothesisAgent | 47 B | 47 B | 0.0% | 0 B | 0 |
| RecommendationAgent | 47 B | 47 B | 0.0% | 0 B | 0 |
| **Total** | **124,108 B** | **193 B** | **99.8%** | **123,915 B** | |
