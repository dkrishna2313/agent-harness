# PH3.1 — Platform Observability & Performance Baseline

**Status:** Complete. Instrumentation is additive; no contract, prompt, reasoning,
or business behaviour changed. 2245 tests pass (12 pre-existing `numpy`-missing
failures in `test_hybrid_retrieval.py` are environmental, unrelated to PH3.1).

**Date:** 2026-07-03
**Baseline source:** `outputs/ph1_stability_3.trace.json` — a full **live**
functional-pipeline run (goal/research mode, single question, `mock_mode: false`).
Auto-generated tables: `PH3.1_performance_baseline_generated.md` /
`PH3.1_performance_baseline.json`.

---

## 1. What shipped (instrumentation)

All additive, measurement-only:

- **Universal stage timers** (`functional_agents/performance.py`). The former
  EvidenceAgent-only sub-phase mechanism is generalized into categorized stages:
  `retrieval | normalization | validation | business_logic | serialization |
  report_generation | other`. New helpers: `stage_timer(context, name, category)`
  (context manager, no-op without a tracker), `record_stage(...)`, and
  `record_boundary_stages(...)`.
- **Boundary stage timing** (`functional_agents/boundary_framework.py`).
  `run_boundary` now times normalization and validation and records `duration_ms`
  additively inside the existing per-stage diagnostics. The top-level diagnostics
  schema is unchanged (PH2.5 contract preserved).
- **Agent wiring.** The five hardened agents (Planner, Evidence, Hypothesis,
  Recommendation, Report) record their boundary normalization/validation timings;
  EvidenceAgent's retrieval sub-phases are now categorized; ReportAgent times the
  executive-report render (`report_generation`).
- **Enriched summary.** `PerformanceTracker.summary()` now reports `llm_pct`,
  `non_llm_pct`, `stage_totals`, `unattributed_ms`, and ranked `bottlenecks`.
- **Report generator** (`functional_agents/performance_report.py` + CLI):
  `python3 -m functional_agents.performance_report --trace <trace.json>
  [--md out.md] [--json out.json]`. Reads any pipeline trace (handles pre-PH3.1
  traces by backfilling derived fields).
- **Tests:** `tests/test_performance.py` (20 tests).

Because every timer is a no-op when no tracker is attached, and stage durations
are additive keys, valid-run behaviour is byte-identical.

---

## 2. Performance report (baseline run)

### Pipeline totals
| Metric | Value |
|---|--:|
| Pipeline wall time | **588,979 ms (~9m 49s)** |
| LLM generation | 565,950 ms (**96.1%**) |
| Non-LLM | 23,029 ms (**3.9%**) |
| LLM calls | 13 (avg ~43.5 s/call) |
| Total tokens | 72,321 (prompt 42,709 / completion 29,612) |
| Active agents | 18 |

**Execution is fully sequential:** the sum of per-agent wall times equals the
pipeline wall (588,976 ≈ 588,979 ms) — zero overlap between agents.

### Per-agent timing (by wall time)
| Agent | Wall ms | LLM ms | Non-LLM ms | LLM % | Calls | Tokens |
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
| **EvidenceAgent** | **22,484** | **0** | **22,484** | 0.0 | 0 | 0 |
| PlannerAgent | 14,105 | 14,100 | 4 | 100.0 | 1 | 3,410 |
| ResearchStrategyAgent | 11,921 | 11,920 | 1 | 100.0 | 1 | 3,282 |
| MultiProfile / Scenario / QA / RecImprovement / RecSynthesis | ~0 | 0 | ~0 | 0 | 0 |

### Non-LLM time by stage
The baseline trace predates fine-grained stage tagging, so its non-LLM time is
reported as `unattributed` (23,029 ms). From the run's sub-phase data we already
know the composition: **EvidenceAgent reranking = 20,190 ms (87.7% of all
non-LLM time)**, then subquestion_expansion 1,139 ms and retrieval_query
1,129 ms. Future runs will populate `retrieval / normalization / validation /
report_generation` automatically via the new timers.

### Largest bottlenecks
1. **Sequential LLM chain** — 13 calls × ~43.5 s, back-to-back. The dominant
   wall-clock driver.
2. **Per-call LLM latency (~44 s)** — large reasoning outputs; ProblemFraming and
   DecisionAnalysis carry the heaviest token loads (~8.5k).
3. **EvidenceAgent reranking (20.2 s)** — the only material non-LLM hotspot.

---

## 3. Engineering recommendations (measure-first; NOT implemented)

The evidence is unambiguous: the platform is **~96% LLM-bound and fully
sequential**. Optimization effort belongs on LLM calls and scheduling, not on
harness plumbing (which accounts for <4% of wall time, almost all in reranking).

**Low-risk**
- **Prompt caching for the shared prefix.** ~10 downstream synthesis agents
  re-send largely identical context (evidence + decision model). Caching the
  stable prefix cuts prompt tokens and per-call latency across many calls with no
  reasoning change.
- **Cap/optimize reranking.** Reduce reranked candidate count or cache
  embeddings — directly targets the 20.2 s hotspot (~88% of non-LLM time).

**Medium-risk**
- **Model tiering.** Route lightweight agents (Assumption, Planner,
  ResearchStrategy) to a faster model; reserve the top tier for heavy reasoning
  (Challenge, DecisionAnalysis, StrategicOption). Attacks the ~44 s/call average.
- **Bounded `max_tokens`** where output is structurally capped (extends J9.1b) —
  trims completion latency on the larger calls.

**Architectural**
- **Parallel DAG execution.** The decision graph has independent branches (e.g.
  Risk ∥ Opportunity, both derived from Assumptions). Topological scheduling of
  independent agents could remove a large fraction of the 589 s serial wall — the
  single highest-leverage change available.
- **Reduce sequential LLM hops.** 10 sequential synthesis calls dominate;
  consolidating tightly-coupled reasoning stages cuts latency and token overhead.

---

## 4. Acceptance criteria

| Criterion | Status |
|---|---|
| No behavioural changes | ✅ additive-only; 2245 tests pass (12 pre-existing env failures) |
| No contract changes | ✅ AgentContext / Functional Agent Contract / RO / DM / Knowledge Layer untouched |
| Instrumentation available | ✅ universal stage timers + boundary timing + tracker summary |
| Engineering performance report produced | ✅ generator + CLI + this baseline |
| Top optimization opportunities identified | ✅ §3 |

**PH3.2 begins after architectural review.** Recommended first target (per the
data): parallel DAG execution + prompt caching — together they address the two
dominant costs (sequential chain, repeated context).
