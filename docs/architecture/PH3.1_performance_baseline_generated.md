# Platform Performance Report (PH3.1)

**Source:** `outputs/ph1_stability_3.trace.json`

## Pipeline totals

| Metric | Value |
|---|--:|
| Pipeline wall time | 588,979 ms |
| LLM generation | 565,950 ms (96.1%) |
| Non-LLM | 23,029 ms (3.9%) |
| LLM calls | 13 |
| Total tokens | 72,321 (prompt 42,709 / completion 29,612) |
| Active agents | 18 |

## Per-agent timing (by wall time)

| Agent | Wall | LLM | Non-LLM | LLM % | Calls | Tokens |
|---|--:|--:|--:|--:|--:|--:|
| ProblemFramingAgent | 67,187 | 67,120 | 67 | 99.9 | 2 | 8,498 |
| StrategicOptionAgent | 63,782 | 63,707 | 75 | 99.9 | 1 | 6,932 |
| ChallengeAgent | 62,471 | 62,462 | 8 | 100.0 | 1 | 6,778 |
| DecisionAnalysisAgent | 61,439 | 61,376 | 63 | 99.9 | 1 | 8,542 |
| HypothesisAgent | 57,509 | 57,507 | 2 | 100.0 | 1 | 6,225 |
| OpportunityAgent | 53,963 | 53,861 | 102 | 99.8 | 1 | 5,942 |
| RecommendationAgent | 49,892 | 49,891 | 2 | 100.0 | 1 | 6,454 |
| RiskAgent | 49,386 | 49,300 | 86 | 99.8 | 1 | 5,542 |
| ExecutiveConfidenceAgent | 44,957 | 44,879 | 78 | 99.8 | 1 | 5,845 |
| AssumptionAgent | 29,880 | 29,826 | 54 | 99.8 | 1 | 4,871 |
| EvidenceAgent | 22,484 | 0 | 22,484 | 0.0 | 0 | 0 |
| PlannerAgent | 14,105 | 14,100 | 4 | 100.0 | 1 | 3,410 |
| ResearchStrategyAgent | 11,921 | 11,920 | 1 | 100.0 | 1 | 3,282 |
| QAAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| MultiProfileAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| ScenarioAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| RecommendationImprovementAgent | 1 | 0 | 1 | 0.0 | 0 | 0 |
| RecommendationSynthesisAgent | 0 | 0 | 0 | 0.0 | 0 | 0 |

## Non-LLM time by stage

| Stage | Duration |
|---|--:|
| _unattributed_ | 23,029 ms |

## Largest bottlenecks

**By agent (wall time):**

1. ProblemFramingAgent — 67,187 ms (LLM 67,120 / non-LLM 67)
2. StrategicOptionAgent — 63,782 ms (LLM 63,707 / non-LLM 75)
3. ChallengeAgent — 62,471 ms (LLM 62,462 / non-LLM 8)
4. DecisionAnalysisAgent — 61,439 ms (LLM 61,376 / non-LLM 63)
5. HypothesisAgent — 57,509 ms (LLM 57,507 / non-LLM 2)
