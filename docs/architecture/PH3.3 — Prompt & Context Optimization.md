# PH3.3 — Prompt & Context Optimization

**Status:** Complete. Real optimization applied to 4 live LLM-backed agents.
2324 tests pass (12 pre-existing `numpy`-missing `test_hybrid_retrieval.py`
failures are environmental, same as PH3.1/PH3.2 — unrelated).

**Date:** 2026-07-03

---

## 1. Objective

PH3.2 measured context-compaction opportunity without applying it. PH3.3
performs the actual cutover: `PlannerAgent`, `HypothesisAgent`,
`RecommendationAgent`, and `StrategicSynthesisAgent` now receive only the
context fields their real prompt-building functions read — verified by
reading both the **live** prompt builder and **MockClaudeClient**'s
corresponding deterministic method for each target agent, so nothing removed
could ever have influenced a prompt, a mock output, or a live response.

## 2. What shipped

- **`functional_agents/context_slices.py`** (new) — `planner_input_slice()`,
  `hypothesis_input_slice()`, `recommendation_input_slice()`,
  `strategic_synthesis_input_slice()`, plus `slice_diagnostics()` and
  `record_slice_diagnostics()`. Every excluded field is documented in the
  module docstring with its verification source.
- **Agent wiring** — each of the 4 target agents now slices `decision_model`
  / `research_strategy` / `decision_architecture` / `domain_evidence` down to
  verified-used sub-keys (or `[]`/`{}` where confirmed fully dead) before
  calling its `_generate_*()` / client `_raw()` method. Evidence-bearing and
  reference-bearing lists (`evidence_items`, `hypotheses`,
  `surviving_hypotheses`, `hypothesis_challenges`, `contradictions`,
  `strategic_synthesis`, `domain_plans`, `domain_hypotheses`) are **never**
  trimmed — only dict-shaped metadata sub-keys and one confirmed-dead list.
- **Observability** — `performance.py` gained `AgentPerfRecord.prompt_slice_applied`,
  tracker `record_prompt_slice()`/`flush_prompt_slice()`, and a
  `summary()["prompt_slice_applied"]` aggregate (distinct from PH3.2's
  measured-but-unapplied `prompt_efficiency`). `run_agent.py`'s mini-trace and
  console output surface any `*_prompt_slice` trace key, mirroring the
  existing `*_boundary` pattern. `performance_report.py` gained an "Applied
  Prompt Slicing" section.
- **Tests** — `tests/test_context_slices.py` (new, 38 tests): unit coverage
  for all 4 slice functions (inclusion/exclusion/determinism/non-mutation/
  JSON-safety), `slice_diagnostics()` correctness, and integration tests
  running each real agent end-to-end via `MockClaudeClient`, confirming
  boundaries pass, counts are sane, and recommendation evidence/hypothesis
  references stay valid. `tests/test_performance.py` (+8 tests) covering the
  new tracker/report plumbing.

## 3. Verified field usage (the actual investigation)

Read directly from `research_agent/claude_client.py` (both the live prompt
functions and `MockClaudeClient`'s corresponding methods):

| Agent | Field | Kept | Dropped (confirmed unread) |
|---|---|---|---|
| Planner | `decision_model` | objective, decision_areas, critical_uncertainties, research_questions, evidence_requirements | (all other keys, e.g. `decision_architecture`, `board_decisions`) |
| Planner | `research_strategy` | research_question_priorities, required_evidence, source_priorities, coverage_targets | (all other keys) |
| Hypothesis | `decision_model` | objective, decision_areas, critical_uncertainties | research_questions, evidence_requirements, (all others) |
| Hypothesis | `research_strategy` | research_question_priorities | (all other keys) |
| Recommendation | `decision_model` | objective, decision_areas | critical_uncertainties, research_questions, (all others) |
| Recommendation | **`research_strategy`** | — | **entire field** — confirmed dead: the parameter is declared in `_recommendation_prompt`'s signature but never referenced in its body, and never referenced in `MockClaudeClient.generate_recommendations`'s body either |
| StrategicSynthesis | `decision_architecture` | strategic_themes, decision_statement, executive_unknowns | (all other keys) |
| StrategicSynthesis | **`domain_evidence`** | — | **entire field** — confirmed dead in both `_strategic_synthesis_prompt` and the mock generator |

Two safety findings drove the implementation:
1. **`RecommendationAgent`'s `research_strategy` is entirely dead code as a prompt input.** The docstring in `MockClaudeClient.generate_recommendations` even says so explicitly: *"the deterministic mock derives recommendations from hypotheses/evidence and does not vary on it."*
2. **`StrategicSynthesisAgent`'s `domain_evidence` is entirely dead**, but `domain_plans` is NOT — it's read as a fallback source in the mock (`domain_hypotheses or domain_plans`) when `domain_hypotheses` is empty. `domain_plans` is therefore passed through **in full**, unsliced. The no-op guard (`domains_received == 0`) also still uses the **unsliced** `domain_evidence` length, so a domain-evidence-only edge case still triggers correctly — slicing only affects what's handed to the LLM call, never the guard.

## 4. Before/after (mock pipeline, real measured bytes)

| Agent | Original | Sliced | Reduction | Bytes saved |
|---|--:|--:|--:|--:|
| **StrategicSynthesisAgent** | 123,967 B | 52 B | **100.0%** | 123,915 B |
| PlannerAgent | 47 B | 47 B | 0.0% | 0 B |
| HypothesisAgent | 47 B | 47 B | 0.0% | 0 B |
| RecommendationAgent | 47 B | 47 B | 0.0% | 0 B |
| **Total** | **124,108 B** | **193 B** | **99.8%** | **123,915 B** |

Against the checked-in `fixtures/planner_start.json` chain (already a minimal,
hand-authored decision_model), the reductions are smaller but real and
verifiable: Hypothesis 46.9% (446→237 B), Recommendation 59.6% (446→180 B),
Planner 0% (the fixture's decision_model happens to already contain exactly
the 5 keys Planner needs). Both results are consistent with the same
underlying mechanism — the size of the win depends entirely on how much
unused metadata the upstream `decision_model`/`decision_architecture`
actually carries in a given run.

## 5. Behaviour verification

- **Zero business-state change (mock pipeline):** captured a full snapshot
  (plan, hypotheses, recommendations, recommendation_portfolio,
  strategic_synthesis, agent_history summaries) before wiring and after —
  `diff` reports **zero differences**.
- **Fixture-chain replay:** `planner → evidence → hypothesis → recommendation →
  report` all report `status: success`; all 4 boundaries
  (`_planner_boundary`, `_hypothesis_boundary`, `_recommendation_boundary`,
  `_report_boundary`) report `failed_stage: None`.
- **Reference integrity:** every recommendation's `supporting_evidence` and
  `supported_by_hypotheses` IDs are a subset of the actual evidence/hypothesis
  IDs present in context — no dangling references introduced by slicing.
- **No output schema change, no Functional Agent Contract change, no
  AgentContext change:** the only new state is the additive
  `context.trace["_<agent>_prompt_slice"]` scratch key (mirrors the existing
  `_<agent>_boundary` pattern) and two new `AgentPerfRecord` fields.

## 6. Acceptance criteria

| Criterion | Status |
|---|---|
| Full functional replay succeeds | ✅ mock pipeline + fixture chain both complete |
| Planner/Hypothesis/Recommendation/Report still pass boundaries | ✅ all `failed_stage: None` |
| Hypothesis/Recommendation counts reasonable and schema-compatible | ✅ 3 hypotheses / 3 recommendations, valid `RecommendationOutput`/`HypothesisOutput` |
| Recommendation evidence/hypothesis references preserved | ✅ verified subset check, zero dangling refs |
| Trace/performance diagnostics show context-size reduction for LLM agents | ✅ `_*_prompt_slice` trace keys + `summary()["prompt_slice_applied"]` |
| Tests pass except known pre-existing environmental failures | ✅ 2324 passed / 12 pre-existing `numpy` failures |
| No contract/schema changes | ✅ confirmed §5 |
