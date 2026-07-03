# PH3.4 — Canonical Pipeline Trace Consolidation

**Status:** Complete. Architectural cleanup only — no prompts, reasoning,
Functional Agent Contracts, outputs, report generation, boundary behavior, or
performance instrumentation logic changed. 2366 tests pass (12 pre-existing
`numpy`-missing `test_hybrid_retrieval.py` failures are environmental, same
as PH3.1–3.3).

**Date:** 2026-07-03

---

## 1. Objective

Before PH3.4, a full pipeline run scattered its diagnostics across several
incompatible shapes: `run_agent.py`'s isolated single-agent mini-trace,
`ReportAgent`'s own artifact-path bookkeeping, and the in-memory
`context.trace["_performance"]` / `_<agent>_boundary` / `_<agent>_prompt_slice`
keys that were never packaged into one file. `functional_agents.performance_report`
had to guess which shape it was handed. PH3.4 introduces **one canonical
pipeline trace** that a complete pipeline run writes, and makes the
performance report consume only that shape.

## 2. What shipped

- **`functional_agents/pipeline_trace.py`** (new) — `build_canonical_trace(context)`
  assembles the canonical schema **read-only** from `context.trace` /
  `context.agent_history` (no new instrumentation, no new computation — pure
  packaging of data every prior milestone already produced). `is_canonical_trace()`
  / `require_canonical_trace()` validate the shape; `write_canonical_trace()`
  writes it to `<out_dir>/pipeline.trace.json`.
- **Schema (v1):**
  ```json
  {
    "schema_version": "ph3.4-canonical-v1",
    "pipeline": {run_id, question, goal, run_mode, profiles, execution_profile},
    "agents": {"planner": {...}, "evidence": {...}, ...},
    "boundaries": {"planner": {...}, "hypothesis": {...}, ...},
    "performance": {...} | null,
    "prompt_slices": {"planner": {...}, "hypothesis": {...}, ...},
    "contracts": {"functional_agent_contract": "...", "agents_conforming": [...]},
    "summary": {agents_run, agents_succeeded, agents_warning, agents_failed,
                boundaries_passed, boundaries_failed, prompt_slices_applied,
                pipeline_status}
  }
  ```
  Per-agent keys (`planner`, `hypothesis`, `strategic_synthesis`, ...) are a
  canonical snake_case derivation of the agent's class name
  (`PlannerAgent` → `planner`) — the **exact same convention** every agent
  already hardcodes for its own `_<key>_boundary` / `_<key>_prompt_slice`
  trace keys (PH2.x / PH3.3). This groups by an existing convention; it does
  not invent a new one.
- **Orchestrator wiring** — one new block in `functional_agents/orchestrator.py`
  `_run_internal()`, immediately after the existing PH3.1 performance-summary
  block, right before `return result_ctx`:
  ```python
  try:
      trace_path = write_canonical_trace(result_ctx, Path(self._out_path).parent)
      print(f"Pipeline trace → {trace_path}")
  except Exception as exc:
      LOGGER.warning(...)
  ```
  Best-effort: a write failure can never fail the pipeline run itself. The
  file is colocated with wherever `--out` pointed (default: `outputs/`), so
  `python3 -m functional_agents.cli run ... --out outputs/x.md` produces
  `outputs/pipeline.trace.json` alongside it.
- **`functional_agents/performance_report.py`** — `extract_performance()` no
  longer guesses between `_performance`/`performance`/nested-`trace`/bare-summary
  shapes. It now calls `require_canonical_trace()` and raises `CanonicalTraceError`
  — with an actionable message naming the source file and what's missing —
  for anything else. `build_report()` itself is unchanged (it still consumes
  a plain `PerformanceTracker.summary()`-shaped dict, which is exactly what
  the canonical trace's `"performance"` section contains).

## 3. Trace ownership (unchanged, as specified)

- **`outputs/pipeline.trace.json`** is the one authoritative trace for a
  *complete* pipeline run (`functional_agents.cli run` / `Orchestrator`).
- **`run_agent.py` mini-traces** continue to exist, untouched —
  `git diff HEAD -- functional_agents/run_agent.py` is empty. They serve a
  different purpose (validating one agent in isolation) and were never meant
  to carry pipeline-wide performance data.
- **`ReportAgent`'s own artifact-path bookkeeping** (`context.artifacts["trace_path"]`)
  continues to exist, untouched — `git diff HEAD -- functional_agents/report_agent.py`
  is empty. The report markdown output is byte-for-byte identical to before
  PH3.4 (ReportAgent runs and writes its file *before* the orchestrator's new
  canonical-trace block executes).

## 4. Verification

- **Full pipeline (mock) run** produces a valid canonical trace with all
  sections populated: 18 `agents`, 5 `boundaries` (planner, evidence,
  hypothesis, recommendation, report — all `failed_stage: null`), 4
  `prompt_slices` (planner, hypothesis, strategic_synthesis, recommendation),
  real `performance` data, and a coherent `summary`
  (`agents_run: 18, agents_succeeded: 17, agents_warning: 1, pipeline_status: "partial"`).
- **`performance_report` on the canonical trace** renders correctly (Pipeline
  totals / Per-agent timing / Prompt Efficiency / Applied Prompt Slicing —
  all PH3.1–3.3 sections intact, unmodified).
- **`performance_report` on a legacy shape** (a `run_agent.py` mini-trace, or
  the PH3.2/PH3.3-era ad-hoc trace format) now fails with a **clear, specific
  error** instead of a generic "no performance data found":
  ```
  error: outputs/ph33_planner.trace.json: not a canonical pipeline trace
  (schema_version=None, expected 'ph3.4-canonical-v1'; missing keys: [...]).
  Generate one with a full pipeline run (functional_agents.cli run ...) —
  this file looks like a run_agent.py mini-trace, a legacy ReportAgent trace,
  or another non-canonical format.
  ```
- **Real CLI entrypoint** (`functional_agents.cli run ... --mock`) prints
  `Pipeline trace → <path>` immediately after the performance summary, as
  specified.
- **Backward compatibility:** `run_agent.py`'s fixture-chain harness
  (planner → evidence → hypothesis → recommendation → report) still runs
  identically; its mini-traces are unaffected by this milestone.

## 5. Tests

- `tests/test_pipeline_trace.py` (new, 41 tests): canonical agent-key
  derivation (matches every existing `_<key>_boundary`/`_<key>_prompt_slice`
  string), full schema assembly from a synthetic context, boundary/prompt-slice/
  performance grouping correctness, summary-count correctness, graceful
  degradation on an empty/partial context, non-mutation, JSON-safety,
  `is_canonical_trace`/`require_canonical_trace` validation (valid + 6
  malformed shapes), and `write_canonical_trace` file I/O.
- `tests/test_performance.py`: updated `test_extract_performance_*` to
  reflect the new strict, canonical-only contract (previously tested the
  guessing behavior this milestone deliberately removes).

## 6. Acceptance criteria

| Criterion | Status |
|---|---|
| One canonical pipeline trace exists | ✅ `outputs/pipeline.trace.json`, schema v1 |
| Performance Report consumes only that trace | ✅ `require_canonical_trace()` enforced |
| No CLI ambiguity | ✅ legacy shapes now raise a clear, specific error |
| No duplicate authoritative traces | ✅ mini-traces/legacy traces are explicitly non-authoritative, unchanged |
| Existing report output unchanged | ✅ `report_agent.py` untouched (empty diff) |
| Existing Functional Agent Contracts unchanged | ✅ `context.py`/`base.py` untouched (empty diff) |
| Existing boundary diagnostics preserved | ✅ passed through verbatim into `boundaries` section |
| Existing prompt-slice diagnostics preserved | ✅ passed through verbatim into `prompt_slices` section |
| Existing performance instrumentation preserved | ✅ passed through verbatim into `performance` section |
| All tests pass except known environmental failures | ✅ 2366 passed / 12 pre-existing `numpy` failures |
