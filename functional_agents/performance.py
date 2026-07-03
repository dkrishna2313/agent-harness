"""Performance instrumentation for the functional agent pipeline (J8.8a; PH3.1).

Tracks per-agent wall-clock time, LLM call time, token usage, and named
per-stage sub-phase breakdowns.  All data is measurement-only; no behaviour
is modified.

PH3.1 (Platform Observability) generalizes the original EvidenceAgent-only
sub-phase mechanism into universal, categorized *stages* so every functional
agent can record where its non-LLM time is spent:

    retrieval | normalization | validation | business_logic
             | serialization | report_generation | other

Instrumentation is strictly additive — contracts, prompts, and reasoning are
unchanged.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:  # avoid import cycle; only used for typing
    from .context import AgentContext

# Canonical, ordered stage categories (PH3.1). "llm_generation" and "total"
# are derived by the tracker; the categories below are recorded via stage timers.
STAGE_RETRIEVAL = "retrieval"
STAGE_NORMALIZATION = "normalization"
STAGE_VALIDATION = "validation"
STAGE_BUSINESS_LOGIC = "business_logic"
STAGE_SERIALIZATION = "serialization"
STAGE_REPORT_GENERATION = "report_generation"
STAGE_OTHER = "other"

STAGE_CATEGORIES: tuple[str, ...] = (
    STAGE_RETRIEVAL,
    STAGE_NORMALIZATION,
    STAGE_VALIDATION,
    STAGE_BUSINESS_LOGIC,
    STAGE_SERIALIZATION,
    STAGE_REPORT_GENERATION,
    STAGE_OTHER,
)


@dataclass
class LLMCallRecord:
    """Metrics for one LLM API call."""

    operation: str
    model: str
    duration_ms: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    success: bool
    error: str | None = None


@dataclass
class SubPhaseRecord:
    """Timing for a named sub-phase within an agent (e.g. EvidenceAgent stages).

    PH3.1: ``category`` classifies the sub-phase into one of STAGE_CATEGORIES so
    the pipeline can aggregate non-LLM time by stage across all agents.
    """

    name: str
    duration_ms: float
    category: str = STAGE_OTHER
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerfRecord:
    """Complete performance record for one agent execution."""

    agent_name: str
    wall_ms: float                            # monotonic elapsed for _execute()
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    sub_phases: list[SubPhaseRecord] = field(default_factory=list)

    @property
    def llm_total_ms(self) -> float:
        return sum(c.duration_ms for c in self.llm_calls)

    @property
    def prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.llm_calls)

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def measured_stage_ms(self) -> float:
        """Total non-LLM time explicitly attributed to timed stages."""
        return sum(sp.duration_ms for sp in self.sub_phases)

    def stage_breakdown(self) -> dict[str, float]:
        """Non-LLM time grouped by canonical stage category (PH3.1)."""
        breakdown: dict[str, float] = {}
        for sp in self.sub_phases:
            cat = sp.category if sp.category in STAGE_CATEGORIES else STAGE_OTHER
            breakdown[cat] = breakdown.get(cat, 0.0) + sp.duration_ms
        return breakdown

    @property
    def unattributed_ms(self) -> float:
        """Wall time not accounted for by LLM calls or timed stages.

        Represents business-logic / serialization / overhead that was not
        wrapped in an explicit stage timer.  Floored at 0 because sub-phase
        timers and LLM slices are independent measurements that can, in edge
        cases, marginally exceed wall time.
        """
        return max(0.0, self.wall_ms - self.llm_total_ms - self.measured_stage_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "wall_ms": round(self.wall_ms, 1),
            "llm_total_ms": round(self.llm_total_ms, 1),
            "llm_call_count": self.llm_call_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": [
                {
                    "operation": c.operation,
                    "model": c.model,
                    "duration_ms": round(c.duration_ms, 1),
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "total_tokens": c.total_tokens,
                    "success": c.success,
                    **({"error": c.error} if c.error else {}),
                }
                for c in self.llm_calls
            ],
            "llm_generation_ms": round(self.llm_total_ms, 1),
            "non_llm_ms": round(self.wall_ms - self.llm_total_ms, 1),
            "stage_breakdown": {k: round(v, 1) for k, v in self.stage_breakdown().items()},
            "unattributed_ms": round(self.unattributed_ms, 1),
            "sub_phases": [
                {
                    "name": sp.name,
                    "duration_ms": round(sp.duration_ms, 1),
                    "category": sp.category,
                    **({k: v for k, v in sp.metadata.items()} if sp.metadata else {}),
                }
                for sp in self.sub_phases
            ],
        }


class PerformanceTracker:
    """Accumulates AgentPerfRecords across a full pipeline run.

    Usage pattern in Orchestrator:
        tracker = PerformanceTracker()
        context.trace["_perf_tracker"] = tracker
        # base class FunctionalAgent.run() records each agent automatically
        summary = tracker.summary()
        context.trace["_performance"] = summary
    """

    def __init__(self) -> None:
        self._records: list[AgentPerfRecord] = []
        self._pending_sub_phases: list[SubPhaseRecord] = []  # written by EvidenceAgent

    def record(self, rec: AgentPerfRecord) -> None:
        self._records.append(rec)

    def record_stage(
        self,
        name: str,
        category: str,
        duration_ms: float,
        **metadata: Any,
    ) -> None:
        """Register a categorized stage timing for the current agent (PH3.1).

        The stage is buffered and flushed onto the active agent's perf record
        when its ``run()`` completes.
        """
        cat = category if category in STAGE_CATEGORIES else STAGE_OTHER
        self._pending_sub_phases.append(
            SubPhaseRecord(name=name, duration_ms=duration_ms, category=cat, metadata=metadata)
        )

    def add_sub_phase(
        self,
        name: str,
        duration_ms: float,
        category: str = STAGE_OTHER,
        **metadata: Any,
    ) -> None:
        """Backward-compatible alias for :meth:`record_stage` (EvidenceAgent)."""
        self.record_stage(name, category, duration_ms, **metadata)

    def flush_sub_phases(self) -> list[SubPhaseRecord]:
        """Drain pending sub-phases (called by base class after EvidenceAgent completes)."""
        phases = list(self._pending_sub_phases)
        self._pending_sub_phases.clear()
        return phases

    def summary(self) -> dict[str, Any]:
        """Return a structured performance summary suitable for trace JSON."""
        agents = [r.to_dict() for r in self._records]
        total_wall_ms = sum(r.wall_ms for r in self._records)
        total_llm_ms = sum(r.llm_total_ms for r in self._records)
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)
        total_tokens = sum(r.total_tokens for r in self._records)
        total_llm_calls = sum(r.llm_call_count for r in self._records)

        # PH3.1 — aggregate non-LLM time by canonical stage category
        stage_totals: dict[str, float] = {}
        for r in self._records:
            for cat, ms in r.stage_breakdown().items():
                stage_totals[cat] = stage_totals.get(cat, 0.0) + ms
        total_unattributed = sum(r.unattributed_ms for r in self._records)

        non_llm_ms = total_wall_ms - total_llm_ms
        llm_pct = round(100.0 * total_llm_ms / total_wall_ms, 1) if total_wall_ms else 0.0

        # Bottlenecks: agents ranked by wall time, and stages by aggregate time
        agent_bottlenecks = [
            {"agent": r.agent_name, "wall_ms": round(r.wall_ms, 1),
             "llm_ms": round(r.llm_total_ms, 1), "non_llm_ms": round(r.wall_ms - r.llm_total_ms, 1)}
            for r in sorted(self._records, key=lambda x: x.wall_ms, reverse=True)
            if r.wall_ms > 0
        ]
        stage_bottlenecks = [
            {"stage": cat, "duration_ms": round(ms, 1)}
            for cat, ms in sorted(stage_totals.items(), key=lambda kv: kv[1], reverse=True)
        ]

        return {
            "totals": {
                "pipeline_wall_ms": round(total_wall_ms, 1),
                "llm_total_ms": round(total_llm_ms, 1),
                "llm_overhead_ms": round(non_llm_ms, 1),
                "non_llm_ms": round(non_llm_ms, 1),
                "llm_pct": llm_pct,
                "non_llm_pct": round(100.0 - llm_pct, 1) if total_wall_ms else 0.0,
                "llm_call_count": total_llm_calls,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
            },
            "stage_totals": {k: round(v, 1) for k, v in stage_totals.items()},
            "unattributed_ms": round(total_unattributed, 1),
            "bottlenecks": {
                "agents": agent_bottlenecks[:5],
                "stages": stage_bottlenecks,
            },
            "agents": agents,
        }

    def print_summary(self) -> None:
        """Print a human-readable performance summary to stdout."""
        s = self.summary()
        t = s["totals"]
        print(
            f"\n{'='*70}\n"
            f"  PERFORMANCE SUMMARY\n"
            f"{'='*70}\n"
            f"  Pipeline wall time : {t['pipeline_wall_ms']:>8.0f} ms\n"
            f"  LLM total time     : {t['llm_total_ms']:>8.0f} ms  ({t.get('llm_pct', 0):.1f}%)\n"
            f"  LLM overhead       : {t['llm_overhead_ms']:>8.0f} ms  (non-LLM work, {t.get('non_llm_pct', 0):.1f}%)\n"
            f"  LLM calls          : {t['llm_call_count']:>8d}\n"
            f"  Prompt tokens      : {t['prompt_tokens']:>8,d}\n"
            f"  Completion tokens  : {t['completion_tokens']:>8,d}\n"
            f"  Total tokens       : {t['total_tokens']:>8,d}\n"
            f"{'='*70}"
        )
        print(f"\n  {'Agent':<35} {'Wall':>7}  {'LLM':>7}  {'Calls':>5}  {'Tokens':>8}")
        print(f"  {'─'*35} {'─'*7}  {'─'*7}  {'─'*5}  {'─'*8}")
        for a in s["agents"]:
            print(
                f"  {a['agent']:<35} {a['wall_ms']:>7.0f}  "
                f"{a['llm_total_ms']:>7.0f}  {a['llm_call_count']:>5d}  "
                f"{a['total_tokens']:>8,d}"
            )
            if a.get("sub_phases"):
                for sp in a["sub_phases"]:
                    print(f"    {'└─ ' + sp['name']:<33} {sp['duration_ms']:>7.0f}")
        print()


# ---------------------------------------------------------------------------
# Module-level helpers (PH3.1) — used by agents to record stage timings.
# All are no-ops when no PerformanceTracker is attached to the context, so
# they are safe to call unconditionally and never alter behaviour.
# ---------------------------------------------------------------------------

def _resolve_tracker(context: "AgentContext | None") -> "PerformanceTracker | None":
    """Return the PerformanceTracker attached to the context, if any."""
    if context is None:
        return None
    trace = getattr(context, "trace", None)
    if not isinstance(trace, dict):
        return None
    tracker = trace.get("_perf_tracker")
    return tracker if isinstance(tracker, PerformanceTracker) else None


@contextmanager
def stage_timer(
    context: "AgentContext | None",
    name: str,
    category: str = STAGE_OTHER,
    **metadata: Any,
) -> Iterator[None]:
    """Time a named stage and record it on the active agent's perf record.

    Usage::

        with stage_timer(context, "retrieval_query", STAGE_RETRIEVAL):
            candidates = retriever.search(...)

    A no-op (still runs the wrapped block) when no tracker is attached, so it is
    safe to wrap any code path unconditionally.
    """
    tracker = _resolve_tracker(context)
    if tracker is None:
        yield
        return
    t0 = time.monotonic()
    try:
        yield
    finally:
        duration_ms = (time.monotonic() - t0) * 1000
        tracker.record_stage(name, category, duration_ms, **metadata)


def record_stage(
    context: "AgentContext | None",
    name: str,
    category: str,
    duration_ms: float,
    **metadata: Any,
) -> None:
    """Record an already-measured stage duration (no-op without a tracker)."""
    tracker = _resolve_tracker(context)
    if tracker is not None:
        tracker.record_stage(name, category, duration_ms, **metadata)


# Keys the boundary framework writes durations under (see boundary_framework).
_BOUNDARY_STAGE_MAP = {
    "normalization": STAGE_NORMALIZATION,
    "validation": STAGE_VALIDATION,
}


def record_boundary_stages(
    context: "AgentContext | None",
    diagnostics: dict[str, Any] | None,
    *,
    prefix: str = "",
) -> None:
    """Record normalization/validation stage timings from boundary diagnostics.

    Reads the per-stage ``duration_ms`` that :func:`run_boundary` records under
    ``diagnostics[stage]["duration_ms"]`` and registers them as timed stages.
    Silently does nothing when a tracker, diagnostics, or the durations are
    absent — so callers need no guards.
    """
    tracker = _resolve_tracker(context)
    if tracker is None or not isinstance(diagnostics, dict):
        return
    for stage_key, category in _BOUNDARY_STAGE_MAP.items():
        stage_diag = diagnostics.get(stage_key)
        if isinstance(stage_diag, dict) and "duration_ms" in stage_diag:
            name = f"{prefix}{stage_key}" if prefix else stage_key
            tracker.record_stage(name, category, float(stage_diag["duration_ms"]))
