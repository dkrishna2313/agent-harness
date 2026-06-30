"""Performance instrumentation for the functional agent pipeline (J8.8a).

Tracks per-agent wall-clock time, LLM call time, token usage, and
EvidenceAgent sub-phase breakdowns.  All data is measurement-only;
no behaviour is modified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


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
    """Timing for a named sub-phase within an agent (e.g. EvidenceAgent stages)."""

    name: str
    duration_ms: float
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
            "sub_phases": [
                {
                    "name": sp.name,
                    "duration_ms": round(sp.duration_ms, 1),
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

    def add_sub_phase(self, name: str, duration_ms: float, **metadata: Any) -> None:
        """Called by EvidenceAgent to register a sub-phase timing."""
        self._pending_sub_phases.append(SubPhaseRecord(name=name, duration_ms=duration_ms, metadata=metadata))

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
        return {
            "totals": {
                "pipeline_wall_ms": round(total_wall_ms, 1),
                "llm_total_ms": round(total_llm_ms, 1),
                "llm_overhead_ms": round(total_wall_ms - total_llm_ms, 1),
                "llm_call_count": total_llm_calls,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens,
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
            f"  LLM total time     : {t['llm_total_ms']:>8.0f} ms\n"
            f"  LLM overhead       : {t['llm_overhead_ms']:>8.0f} ms  (non-LLM work)\n"
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
