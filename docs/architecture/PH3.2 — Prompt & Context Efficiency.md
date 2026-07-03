# PH3.2 — Prompt & Context Efficiency

**Status:** Complete. Additive-only: 2281 tests pass (12 pre-existing `numpy`-missing
`test_hybrid_retrieval.py` failures are environmental, same as PH3.1 — unrelated
to this milestone). No contract, prompt wording, or reasoning change; mock-pipeline
output is **byte-for-byte identical** before/after this milestone (verified by diff).

**Date:** 2026-07-03

---

## 1. Scope decision: measurement, not cutover

The spec's acceptance criteria — *no prompt wording changes, no reasoning
changes, same outputs on replay* — can only be satisfied with certainty if the
actual text sent to the LLM in every call is untouched. PH3.2 therefore builds
and wires the compaction/profiling machinery in **measurement mode**, exactly
as PH3.1 built performance instrumentation without applying any of the
optimizations it identified. The compactor computes, for each agent's real
context, what compaction *would* achieve — and records it as diagnostics —
without altering what any agent reads or what any prompt contains. Acting on
the measured savings (actually scoping what's sent per call) is the natural
PH3.3 follow-up, gated on reviewing this data, matching the project's
established "measure before optimizing" pattern.

## 2. Shared Context Compactor

`functional_agents/context_compactor.py`:

- **`estimate_tokens(value)`** — heuristic ~4 chars/token, matching the
  existing instrumentation-only convention already in
  `research_strategy_agent.py` (`_CHARS_PER_TOKEN = 4`). Not a live tokenizer.
- **`compact_context(sections, profile)`** — pure, deterministic function:
  1. Drops sections not named in the target profile (`removed_unused`).
  2. Among the survivors, drops any section whose content is byte-identical
     (via canonical `json.dumps(sort_keys=True)`) to an earlier section
     (`removed_duplicate`) — the information still exists under the first
     section's name, so nothing is lost.
  3. Never rewrites, truncates, or summarizes a kept section's content —
     compaction is whole-section only, so semantics are always preserved.
- **`build_context_sections(context)`** — read-only, deterministic snapshot of
  an `AgentContext` into 22 canonical named sections (`SECTION_ORDER`), in a
  fixed order independent of any one agent's profile.
- **`compact_context_for_agent(context, agent_name)`** — composes the two;
  returns `None` for agents without a documented profile.
- **`measure_and_record(context, agent_name)`** — computes the above and
  records it onto the active `PerformanceTracker` (no-op without one).

## 3. Agent Context Profiles — derived from code, not guessed

Each profile was established by reading the exact call sites, not assumed:

| Agent | Profile (verified source) |
|---|---|
| **PlannerAgent** | `question`, `profiles`, `decision_model`, `research_strategy` — `planner_agent.py` passes these into `plan_research_question_raw()`. |
| **EvidenceAgent** | `question`, `plan`, `profiles` — KB path uses `context.plan` + `context.question` for retrieval and `context.profiles` for attribution; legacy path calls `agent.analyze(question, documents)`. **`decision_model` is NOT read**, despite being a common assumption. |
| **HypothesisAgent** | `evidence_notes`, `profiles`, `contradictions`, `decision_model`, `research_strategy` — all passed into `_generate_hypotheses()`. |
| **RecommendationAgent** | `hypotheses`, `surviving_hypotheses`, `hypothesis_challenges`, `evidence_notes`, `decision_model`, `research_strategy`, `contradictions`, `strategic_synthesis` — the richest profile of the five, matching `_execute()`'s full argument list. |
| **ReportAgent** | 18 of 22 sections — the executive report legitimately reads nearly the entire decision graph. **This is itself a finding**: Report has almost no compaction opportunity (confirmed below: 0.0% measured reduction). |

Divergence from the spec's illustrative examples (e.g. Evidence was assumed to
need `decision_model`) was expected and is the point of "derive, don't guess."

## 4. Prompt Metrics (performance.py extensions)

- `AgentPerfRecord.context_compaction: dict | None` — one agent's measured
  `CompactionResult.to_dict()`.
- `PerformanceTracker.record_context_compaction()` / `flush_context_compaction()`
  — buffered per-agent, mirrors the existing sub-phase pattern.
- `PerformanceTracker.summary()["prompt_efficiency"]` — aggregate totals
  (`original_tokens`, `compacted_tokens`, `tokens_saved`, `reduction_pct`,
  `agents_measured`) plus the per-agent list.
- Wired at a **single seam**: `FunctionalAgent.run()` in `base.py` calls
  `measure_and_record(context, self.name)` immediately before `_execute()` —
  one line, applies automatically to any future agent added to
  `AGENT_CONTEXT_PROFILES`, and is a pure no-op for agents without a profile.

## 5. Reporting

`functional_agents/performance_report.py` gained a **Prompt Efficiency**
section: per-agent Original Context Size / Compacted Context Size /
Reduction % / Estimated Tokens Saved, plus a totals row. Renders "no agents
measured" gracefully for pre-PH3.2 traces (verified against the PH3.1 baseline
trace, which correctly shows no data).

## 6. Measured result (mock pipeline, confirms mechanism correctness)

A full mock pipeline run (`functional_agents.cli run ... --mock`) produced:

| Agent | Original | Compacted | Reduction | 
|---|--:|--:|--:|
| PlannerAgent | 14 tok | 14 tok | 0.0% |
| EvidenceAgent | 151 tok | 151 tok | 0.0% |
| **HypothesisAgent** | 395,280 tok | 31,017 tok | **92.2%** |
| **RecommendationAgent** | 397,610 tok | 32,860 tok | **91.7%** |
| ReportAgent | 401,926 tok | 401,847 tok | 0.0% |
| **Total** | **1,194,981 tok** | **465,889 tok** | **61.0%** |

This is a **mock run** — MockClaudeClient generates large deterministic
decision-model/research-strategy payloads, so absolute token counts aren't a
live-prompt estimate. What it *does* prove: the mechanism correctly
distinguishes agents with large unused-section overhead (Hypothesis,
Recommendation — both receive the full `decision_model`/`research_strategy`
bundle but use only fragments) from agents that already need nearly
everything (Report). **A live run against `ph1_stability_3.trace.json`-style
data is the natural next step** to get production-representative savings
estimates before any PH3.3 cutover decision.

## 7. Acceptance criteria

| Criterion | Status |
|---|---|
| No prompt wording changes | ✅ zero LLM prompt-building code touched |
| No reasoning changes | ✅ no agent reads different data than before |
| No contract changes | ✅ AgentContext / Functional Agent Contract unchanged |
| Same outputs on replay | ✅ verified — mock-pipeline memo output is byte-identical before/after |
| Performance report includes prompt reduction statistics | ✅ §5 |
| Existing tests pass | ✅ 2281 pass (12 pre-existing env failures, unrelated) |
| New tests verify deterministic context compaction | ✅ `tests/test_context_compactor.py` (31 tests) + `tests/test_performance.py` additions (5 tests) |

**Recommended next step (PH3.3, pending review):** run the compactor
measurement against a **live** trace to get production token-savings
estimates, then decide whether/how to cut the measured savings into actual
prompt construction for Hypothesis and Recommendation — the two agents with
the largest confirmed unused-context overhead.
