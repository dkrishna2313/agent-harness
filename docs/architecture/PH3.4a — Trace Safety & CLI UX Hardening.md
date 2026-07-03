# PH3.4a — Trace Safety & CLI UX Hardening

**Status:** Complete. Strictly additive engineering UX hardening — no prompts,
reasoning, Functional Agent Contracts, AgentContext, Research Object, Decision
Model, report outputs, boundary schemas, performance instrumentation, or
canonical pipeline trace schema changed. 2396 tests pass (12 pre-existing
`numpy`-missing `test_hybrid_retrieval.py` failures are environmental, same
as PH3.1–3.4).

**Date:** 2026-07-03

---

## 1. Motivation

While validating PH3.4, a real user ran `run_agent.py --trace outputs/pipeline.trace.json`
(or a similarly-named file), which silently overwrote the file the canonical
pipeline trace lives at with a single-agent mini-trace. `performance_report`
then failed with a correct but unhelpful "not a canonical pipeline trace"
error, and there was no protection preventing the same mistake from
recurring. PH3.4a closes that gap.

## 2. What shipped

### Part A — Reserve canonical trace filenames

`functional_agents/run_agent.py`'s `main()` now checks `--trace` **before any
execution**: if it targets a reserved filename (`pipeline.trace.json`, or
anything ending in `_pipeline.trace.json`, in any directory), it prints
`CanonicalTraceReservedError` with an actionable message and exits 2 —
nothing runs, nothing is written, the canonical trace (if one already exists
at that path) is never touched.

### Part B — Clearer `performance_report` errors

`pipeline_trace.require_canonical_trace()` now diagnoses *what kind* of
non-canonical file it was handed — a run_agent.py mini-trace, a legacy
ReportAgent trace, a bare `PerformanceTracker.summary()` dict, or something
unrecognized — and explains specifically why that shape can never be a
canonical trace. When `schema_version` is present but wrong, it reports both
the found and expected values side by side.

### Part C — Centralized reserved trace paths

New `functional_agents/trace_paths.py` — the single source of truth for:
- `CANONICAL_PIPELINE_TRACE_FILENAME` (`"pipeline.trace.json"`)
- `is_reserved_pipeline_trace(path)` — basename-only check
- `default_pipeline_trace(out_dir)` — canonical trace's default location
- `require_not_reserved(path, tool_name=...)` — raise-before-acting guard
- `CanonicalTraceReservedError`

`functional_agents/pipeline_trace.py` now sources `DEFAULT_TRACE_FILENAME`
and its default-path computation from `trace_paths.py` instead of a second
hardcoded literal — a pure refactor, verified to produce byte-identical
output (same filename, same schema, same content).

### Part D — Documentation

Added a **"Trace Types"** section to
`docs/architecture/PH3.4 — Canonical Pipeline Trace Consolidation.md`
describing Mini Trace vs. Canonical Pipeline Trace (producer, purpose,
expected filename, consumer) for future reference.

## 3. Design decisions

- **Fail before execution, not after.** The guard runs immediately after
  argument parsing, before `run_agent()` is even called — matching the spec's
  "never silently overwrite" requirement literally: if the check fires, zero
  side effects have occurred.
- **Basename-only matching.** `is_reserved_pipeline_trace()` checks only
  `Path(path).name`, not the full path — `outputs/pipeline.trace.json` and
  `/any/dir/pipeline.trace.json` are both reserved; a directory that happens
  to be *named* `pipeline.trace.json` is not (edge case, tested).
- **Suffix pattern, not prefix.** `*_pipeline.trace.json` catches
  run-specific names like `ph34_pipeline.trace.json` or
  `mock_pipeline.trace.json` (exactly the naming pattern used during PH3.2–3.4
  validation) without over-matching unrelated names like
  `pipeline_summary.trace.json`.
- **`--output` (context snapshot) is untouched.** Only `--trace` (mini-trace
  output) is guarded — that's the literal risk (trace-shape confusion), and
  the spec's own example is scoped to `--trace`.
- **Diagnosis is best-effort and cosmetic.** `_diagnose_trace_shape()` only
  improves the error *message*; `is_canonical_trace()` remains the single
  source of truth for whether a trace is accepted. A trace that happens not
  to match any known "looks like X" pattern still gets a correct (if generic)
  rejection.

## 4. Verification

- Reserved-filename guard fires for both exact (`pipeline.trace.json`) and
  suffix (`ph34_pipeline.trace.json`) matches, in any directory; exit code 2;
  file never written.
- A pre-existing canonical trace at a reserved path is provably untouched
  after a blocked `run_agent.py` invocation (byte-identical sentinel content
  before/after).
- A normal (non-reserved) `--trace` filename works exactly as before PH3.4a —
  confirmed identical console output.
- `performance_report` against a real mini-trace now names the file, explains
  it's a run_agent.py mini-trace, and points at the fix.
- `performance_report` against the canonical trace still succeeds and renders
  identically — confirmed schema/keys/summary unchanged after the Part C
  refactor.
- `git diff <PH3.4-end>..HEAD -- functional_agents/context.py functional_agents/base.py functional_agents/report_agent.py` → 0 lines (no contract/output changes).

## 5. Acceptance criteria

| Criterion | Status |
|---|---|
| run_agent refuses reserved canonical filenames | ✅ exact + suffix match, exit 2, pre-execution |
| Helpful deterministic error | ✅ `CanonicalTraceReservedError` message, always identical for the same input |
| Canonical pipeline trace unchanged | ✅ schema/content verified identical after Part C refactor |
| performance_report clearly explains incorrect trace usage | ✅ shape-specific diagnosis + schema-version reporting |
| No Functional Agent Contract changes | ✅ empty diff |
| No schema changes | ✅ empty diff on boundary/report/context schemas |
| No prompt changes | ✅ no prompt-building code touched |
| No reasoning changes | ✅ no agent logic touched |
| Existing behaviour unchanged | ✅ non-reserved mini-trace path behaviourally identical |
