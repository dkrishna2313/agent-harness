"""Canonical Pipeline Trace (PH3.4).

The functional pipeline has accumulated several distinct trace shapes as it
grew: PH3.1's `_performance` summary, PH2.x's per-agent `_<agent>_boundary`
diagnostics, PH3.3's `_<agent>_prompt_slice` diagnostics, `run_agent.py`'s
isolated single-agent mini-trace, and `ReportAgent`'s own artifact-path
bookkeeping. Each is real and useful, but nothing packages them into one
authoritative file for a *complete* pipeline run — so a Performance Report
consumer has to guess which shape it was handed.

This module assembles those ALREADY-COMPUTED diagnostics — read-only, off
`AgentContext.trace` / `AgentContext.agent_history` — into one canonical,
versioned trace. It introduces no new instrumentation and changes no
prompts, reasoning, Functional Agent Contracts, boundary computation, or
performance instrumentation: it is packaging only.

Schema (v1)::

    {
      "schema_version": "ph3.4-canonical-v1",
      "pipeline": {run_id, question, goal, run_mode, profiles, execution_profile},
      "agents": {<agent_key>: {status, summary, ..., agent_class}},
      "boundaries": {<agent_key>: {...boundary diagnostics, unmodified}},
      "performance": {...} | null,
      "prompt_slices": {<agent_key>: {...slice diagnostics, unmodified}},
      "deliverables": [{"type": "markdown", "status": "generated"}, ...],
      "contracts": {"functional_agent_contract": "...", "agents_conforming": [...]},
      "summary": {agents_run, agents_succeeded, agents_warning, agents_failed,
                  boundaries_passed, boundaries_failed, prompt_slices_applied,
                  pipeline_status},
    }

``<agent_key>`` is a canonical snake_case label derived from the agent's
class name (``PlannerAgent`` -> ``planner``, ``StrategicSynthesisAgent`` ->
``strategic_synthesis``) — the exact same convention every agent already uses
for its own ``_<key>_boundary`` / ``_<key>_prompt_slice`` trace keys, so this
module simply groups by that existing naming, it does not invent a new one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .trace_paths import CANONICAL_PIPELINE_TRACE_FILENAME, default_pipeline_trace

SCHEMA_VERSION = "ph3.4-canonical-v1"

_REQUIRED_TOP_LEVEL_KEYS = (
    "schema_version", "pipeline", "agents", "boundaries",
    "performance", "prompt_slices", "contracts", "summary",
)

# PH3.4a — sourced from trace_paths.py (single source of truth for the
# literal filename); value is unchanged from PH3.4.
DEFAULT_TRACE_FILENAME = CANONICAL_PIPELINE_TRACE_FILENAME


class CanonicalTraceError(Exception):
    """Raised when a file expected to be a canonical pipeline trace is not."""


def _canonical_agent_key(agent_class_name: str) -> str:
    """``PlannerAgent`` -> ``planner``; ``StrategicSynthesisAgent`` -> ``strategic_synthesis``."""
    name = agent_class_name
    if name.endswith("Agent") and len(name) > len("Agent"):
        name = name[: -len("Agent")]
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
    key = s2.lower()
    return key or agent_class_name.lower()


def _infer_run_mode(context: Any) -> str:
    if getattr(context, "engagement", None):
        return "engagement"
    if getattr(context, "goal", None):
        return "goal"
    return "question"


def build_canonical_trace(context: Any) -> dict[str, Any]:
    """Assemble the canonical pipeline trace from a completed AgentContext.

    Read-only: never mutates ``context``. Safe to call on a partially-run
    pipeline (e.g. after an early failure) — every section degrades to an
    empty/None value rather than raising.
    """
    trace = getattr(context, "trace", None) or {}
    agent_history = getattr(context, "agent_history", None) or []

    agents: dict[str, Any] = {}
    for entry in agent_history:
        if not isinstance(entry, dict):
            continue
        cls = entry.get("agent", "")
        key = _canonical_agent_key(cls) if cls else "unknown"
        # Last entry for a given agent wins (e.g. a looping/multi-domain agent
        # that records more than once) — matches the agent's own final state.
        agents[key] = {**{k: v for k, v in entry.items() if k != "agent"}, "agent_class": cls}

    boundaries: dict[str, Any] = {}
    prompt_slices: dict[str, Any] = {}
    for k, v in trace.items():
        if not (isinstance(k, str) and k.startswith("_")):
            continue
        if k.endswith("_boundary"):
            boundaries[k[1:-len("_boundary")]] = v
        elif k.endswith("_prompt_slice"):
            prompt_slices[k[1:-len("_prompt_slice")]] = v

    performance = trace.get("_performance")

    # J11.0 — Strategic Deliverables Framework: DeliverableArtifact records
    # produced this run (already dicts via DeliverableArtifact.to_dict()).
    # Additive only — not in _REQUIRED_TOP_LEVEL_KEYS, so pre-J11.0 traces
    # remain valid canonical traces.
    deliverables = list(getattr(context, "deliverables", None) or [])

    contracts = {
        "functional_agent_contract": "FunctionalAgent.run(context: AgentContext) -> AgentResult",
        "agents_conforming": sorted(agents.keys()),
    }

    agents_succeeded = sum(1 for a in agents.values() if a.get("status") == "success")
    agents_warning = sum(1 for a in agents.values() if a.get("status") == "warning")
    agents_failed = sum(1 for a in agents.values() if a.get("status") == "error")
    boundaries_failed = sum(
        1 for b in boundaries.values()
        if isinstance(b, dict) and b.get("failed_stage") is not None
    )
    boundaries_passed = len(boundaries) - boundaries_failed
    prompt_slices_applied = sum(
        1 for p in prompt_slices.values()
        if isinstance(p, dict) and p.get("bytes_saved", 0) > 0
    )
    pipeline_status = "failed" if agents_failed else ("partial" if agents_warning else "success")

    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline": {
            "run_id": getattr(context, "run_id", None),
            "question": getattr(context, "question", None),
            "goal": getattr(context, "goal", None),
            "run_mode": _infer_run_mode(context),
            "profiles": list(getattr(context, "profiles", None) or []),
            "execution_profile": getattr(context, "execution_profile", None),
        },
        "agents": agents,
        "boundaries": boundaries,
        "performance": performance,
        "prompt_slices": prompt_slices,
        "deliverables": deliverables,
        "contracts": contracts,
        "summary": {
            "agents_run": len(agents),
            "agents_succeeded": agents_succeeded,
            "agents_warning": agents_warning,
            "agents_failed": agents_failed,
            "boundaries_passed": boundaries_passed,
            "boundaries_failed": boundaries_failed,
            "prompt_slices_applied": prompt_slices_applied,
            "pipeline_status": pipeline_status,
        },
    }


def is_canonical_trace(data: Any) -> bool:
    """Structural check: does ``data`` look like a canonical pipeline trace?"""
    return (
        isinstance(data, dict)
        and data.get("schema_version") == SCHEMA_VERSION
        and all(key in data for key in _REQUIRED_TOP_LEVEL_KEYS)
    )


def _diagnose_trace_shape(data: dict[str, Any]) -> str:
    """Best-effort, human-readable description of a non-canonical trace's shape.

    Used only to make the PH3.4a error message specific; never affects
    validation logic (``is_canonical_trace`` is the only source of truth for
    whether a trace is accepted).
    """
    if {"agent_class", "execution_time_ms", "llm_mode"} <= set(data):
        return (
            "a run_agent.py MINI TRACE — a record of one isolated agent's "
            "pre/postconditions and boundary result. run_agent.py never "
            "attaches a PerformanceTracker or assembles pipeline-wide "
            "diagnostics, so it can never produce a canonical pipeline trace."
        )
    if "question_topics_detected" in data or {"timestamp", "documents_loaded"} <= set(data):
        return (
            "a legacy ReportAgent / research-memo trace — per-question "
            "extraction and synthesis diagnostics from the legacy research_agent "
            "path, not a functional-pipeline run."
        )
    if {"totals", "agents"} <= set(data) and "schema_version" not in data:
        return (
            "a bare PerformanceTracker.summary() dict — this is the "
            "canonical trace's \"performance\" section on its own, not the "
            "full canonical trace envelope around it."
        )
    return "an unrecognized trace format"


def require_canonical_trace(data: Any, *, source: str = "<trace>") -> dict[str, Any]:
    """Return ``data`` if it is a canonical pipeline trace; otherwise raise.

    Used by ``performance_report.py`` so it stops guessing between the
    several legacy trace shapes. The error names the source file, describes
    what kind of trace it actually looks like, reports the schema-version
    mismatch (if any), and points at the command that produces a canonical
    trace.
    """
    if is_canonical_trace(data):
        return data
    if not isinstance(data, dict):
        raise CanonicalTraceError(
            f"{source}: not a JSON object (got {type(data).__name__}).\n"
            "Generate a canonical pipeline trace with:\n"
            "    python3 -m functional_agents.cli run ..."
        )

    got_version = data.get("schema_version")
    shape = _diagnose_trace_shape(data)
    missing = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in data]

    lines = [f"{source}: not a canonical pipeline trace."]
    lines.append(f"  This file looks like {shape}")
    if "schema_version" in data:
        lines.append(f"  schema_version found:    {got_version!r}")
        lines.append(f"  schema_version expected: {SCHEMA_VERSION!r}")
    if missing:
        lines.append(f"  missing keys: {missing}")
    lines.append("")
    lines.append("Generate a canonical pipeline trace with:")
    lines.append("    python3 -m functional_agents.cli run ...")
    raise CanonicalTraceError("\n".join(lines))


def write_canonical_trace(context: Any, out_dir: str | Path) -> Path:
    """Build and write the canonical trace next to a run's other outputs.

    Returns the path written. Callers should treat write failures as
    non-fatal (this is diagnostic tooling, not pipeline output) — this
    function itself does not swallow errors so callers can choose their own
    fallback behaviour.
    """
    trace = build_canonical_trace(context)
    out_path = default_pipeline_trace(out_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(trace, indent=2, default=str))
    return out_path
