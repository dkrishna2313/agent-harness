"""PH3.1 / PH3.4 — Performance report generator.

Reads the canonical pipeline trace (PH3.4, ``functional_agents.pipeline_trace``)
and produces a structured engineering performance report: per-agent timing,
stage breakdown, LLM vs non-LLM share, total pipeline timing, ranked
bottlenecks, measured prompt-efficiency (PH3.2), and applied prompt-slice
savings (PH3.3).

This is a *reporting* tool — it consumes measurement data already captured by
``functional_agents.performance``.  It changes no platform behaviour.

PH3.4 note: this module used to guess between several trace shapes (a raw
``_performance``/``performance`` key, a nested ``trace`` wrapper, a bare
summary dict). It no longer guesses — the CLI now requires a canonical
pipeline trace (``functional_agents.pipeline_trace.build_canonical_trace``)
and raises a clear ``CanonicalTraceError`` otherwise. ``build_report()``
itself still operates on a plain ``PerformanceTracker.summary()``-shaped
dict (the canonical trace's ``"performance"`` section *is* exactly that
shape), so it composes cleanly with both the CLI and direct unit tests.

CLI::

    python3 -m functional_agents.performance_report --trace outputs/pipeline.trace.json
    python3 -m functional_agents.performance_report --trace pipeline.trace.json --md perf.md --json perf.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .performance import STAGE_CATEGORIES
from .pipeline_trace import CanonicalTraceError, require_canonical_trace


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_performance(data: Any, *, source: str = "<trace>") -> dict:
    """Extract the performance section from a canonical pipeline trace (PH3.4).

    Requires ``data`` to be a canonical pipeline trace (schema_version
    ``pipeline_trace.SCHEMA_VERSION``, produced by a full pipeline run).
    Raises ``CanonicalTraceError`` — with an actionable message — for any
    other shape (a ``run_agent.py`` mini-trace, a legacy ReportAgent trace,
    a bare performance summary, etc.). This function no longer guesses.
    """
    canonical = require_canonical_trace(data, source=source)
    perf = canonical.get("performance")
    if not isinstance(perf, dict) or "agents" not in perf:
        raise CanonicalTraceError(
            f"{source}: canonical pipeline trace has no performance data "
            "(no PerformanceTracker was attached to this pipeline run)."
        )
    return perf


# ---------------------------------------------------------------------------
# Report model
# ---------------------------------------------------------------------------

def build_report(perf: dict) -> dict[str, Any]:
    """Build a structured performance report from a performance summary dict."""
    totals = dict(perf.get("totals", {}))
    wall = totals.get("pipeline_wall_ms", 0.0) or 0.0
    llm = totals.get("llm_total_ms", 0.0) or 0.0
    non_llm = round(wall - llm, 1)

    # Backfill derived fields older summaries (pre-PH3.1 traces) omitted.
    totals["non_llm_ms"] = totals.get("non_llm_ms", non_llm)
    if wall:
        totals.setdefault("llm_pct", round(100.0 * llm / wall, 1))
        totals.setdefault("non_llm_pct", round(100.0 - totals["llm_pct"], 1))

    agents = perf.get("agents", [])
    active_agents = [a for a in agents if (a.get("wall_ms") or 0) > 0]

    # Per-agent rows with derived shares.
    agent_rows = []
    for a in sorted(active_agents, key=lambda x: x.get("wall_ms", 0), reverse=True):
        a_wall = a.get("wall_ms", 0.0) or 0.0
        a_llm = a.get("llm_total_ms", 0.0) or 0.0
        agent_rows.append({
            "agent": a.get("agent", "?"),
            "wall_ms": round(a_wall, 1),
            "llm_ms": round(a_llm, 1),
            "non_llm_ms": round(a_wall - a_llm, 1),
            "llm_pct": round(100.0 * a_llm / a_wall, 1) if a_wall else 0.0,
            "llm_calls": a.get("llm_call_count", 0),
            "tokens": a.get("total_tokens", 0),
            "stage_breakdown": a.get("stage_breakdown", {}),
            "sub_phases": a.get("sub_phases", []),
        })

    # Stage totals: prefer summary-level, else aggregate from agents.
    stage_totals = dict(perf.get("stage_totals", {}))
    if not stage_totals:
        for a in active_agents:
            for cat, ms in (a.get("stage_breakdown") or {}).items():
                stage_totals[cat] = stage_totals.get(cat, 0.0) + ms
    stage_totals = {k: round(v, 1) for k, v in stage_totals.items()}

    unattributed = perf.get("unattributed_ms")
    if unattributed is None:
        measured = sum(stage_totals.values())
        unattributed = round(max(0.0, non_llm - measured), 1)

    bottlenecks = perf.get("bottlenecks") or {
        "agents": [
            {"agent": r["agent"], "wall_ms": r["wall_ms"],
             "llm_ms": r["llm_ms"], "non_llm_ms": r["non_llm_ms"]}
            for r in agent_rows[:5]
        ],
        "stages": [
            {"stage": k, "duration_ms": v}
            for k, v in sorted(stage_totals.items(), key=lambda kv: kv[1], reverse=True)
        ],
    }

    # PH3.2 — Prompt Efficiency: prefer the tracker's own aggregate; otherwise
    # derive it from each agent's context_compaction field (e.g. a summary
    # built directly from records rather than via PerformanceTracker.summary()).
    prompt_efficiency = perf.get("prompt_efficiency")
    if not prompt_efficiency:
        measured = [
            {"agent": a.get("agent", "?"), **a["context_compaction"]}
            for a in agents if a.get("context_compaction")
        ]
        orig = sum(m["original_tokens"] for m in measured)
        comp = sum(m["compacted_tokens"] for m in measured)
        saved = sum(m["tokens_saved"] for m in measured)
        prompt_efficiency = {
            "totals": {
                "original_tokens": orig, "compacted_tokens": comp, "tokens_saved": saved,
                "reduction_pct": round(100.0 * saved / orig, 1) if orig else 0.0,
                "agents_measured": len(measured),
            },
            "agents": measured,
        }

    # PH3.3 — Applied Prompt Slicing: bytes actually removed from real LLM
    # calls (distinct from PH3.2's measured-but-not-applied token estimate).
    prompt_slice_applied = perf.get("prompt_slice_applied")
    if not prompt_slice_applied:
        applied = [
            {"agent": a.get("agent", "?"), **a["prompt_slice_applied"]}
            for a in agents if a.get("prompt_slice_applied")
        ]
        orig_b = sum(m["original_bytes"] for m in applied)
        sliced_b = sum(m["sliced_bytes"] for m in applied)
        saved_b = sum(m["bytes_saved"] for m in applied)
        prompt_slice_applied = {
            "totals": {
                "original_bytes": orig_b, "sliced_bytes": sliced_b, "bytes_saved": saved_b,
                "reduction_pct": round(100.0 * saved_b / orig_b, 1) if orig_b else 0.0,
                "agents_applied": len(applied),
            },
            "agents": applied,
        }

    return {
        "totals": totals,
        "agent_count": len(active_agents),
        "agents": agent_rows,
        "stage_totals": stage_totals,
        "unattributed_ms": unattributed,
        "bottlenecks": bottlenecks,
        "prompt_efficiency": prompt_efficiency,
        "prompt_slice_applied": prompt_slice_applied,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _ms(v: float) -> str:
    return f"{v:,.0f} ms"


def render_markdown(report: dict, *, source: str | None = None) -> str:
    t = report["totals"]
    wall = t.get("pipeline_wall_ms", 0.0)
    lines: list[str] = ["# Platform Performance Report (PH3.1 / PH3.2 / PH3.3 / PH3.4)", ""]
    if source:
        lines += [f"**Source:** `{source}`", ""]

    lines += [
        "## Pipeline totals",
        "",
        "| Metric | Value |",
        "|---|--:|",
        f"| Pipeline wall time | {_ms(wall)} |",
        f"| LLM generation | {_ms(t.get('llm_total_ms', 0))} ({t.get('llm_pct', 0)}%) |",
        f"| Non-LLM | {_ms(t.get('non_llm_ms', 0))} ({t.get('non_llm_pct', 0)}%) |",
        f"| LLM calls | {t.get('llm_call_count', 0)} |",
        f"| Total tokens | {t.get('total_tokens', 0):,} "
        f"(prompt {t.get('prompt_tokens', 0):,} / completion {t.get('completion_tokens', 0):,}) |",
        f"| Active agents | {report['agent_count']} |",
        "",
        "## Per-agent timing (by wall time)",
        "",
        "| Agent | Wall | LLM | Non-LLM | LLM % | Calls | Tokens |",
        "|---|--:|--:|--:|--:|--:|--:|",
    ]
    for a in report["agents"]:
        lines.append(
            f"| {a['agent']} | {a['wall_ms']:,.0f} | {a['llm_ms']:,.0f} | "
            f"{a['non_llm_ms']:,.0f} | {a['llm_pct']} | {a['llm_calls']} | {a['tokens']:,} |"
        )

    lines += ["", "## Non-LLM time by stage", "", "| Stage | Duration |", "|---|--:|"]
    stage_totals = report["stage_totals"]
    for cat in STAGE_CATEGORIES:
        if cat in stage_totals:
            lines.append(f"| {cat} | {_ms(stage_totals[cat])} |")
    lines.append(f"| _unattributed_ | {_ms(report['unattributed_ms'])} |")

    lines += ["", "## Largest bottlenecks", "", "**By agent (wall time):**", ""]
    for i, a in enumerate(report["bottlenecks"]["agents"], 1):
        lines.append(f"{i}. {a['agent']} — {a['wall_ms']:,.0f} ms "
                     f"(LLM {a['llm_ms']:,.0f} / non-LLM {a['non_llm_ms']:,.0f})")
    if report["bottlenecks"]["stages"]:
        lines += ["", "**By non-LLM stage:**", ""]
        for i, s in enumerate(report["bottlenecks"]["stages"], 1):
            lines.append(f"{i}. {s['stage']} — {s['duration_ms']:,.0f} ms")

    pe = report.get("prompt_efficiency", {})
    pe_totals = pe.get("totals", {})
    lines += ["", "## Prompt Efficiency (PH3.2 — measured, not yet applied)", ""]
    if pe_totals.get("agents_measured"):
        lines += [
            f"Context-compaction opportunity measured for {pe_totals['agents_measured']} "
            "agent(s) against their documented context profile. This is diagnostic only — "
            "no prompt sent to the LLM has changed.",
            "",
            "| Agent | Original Context Size | Compacted Context Size | Reduction % | Est. Tokens Saved |",
            "|---|--:|--:|--:|--:|",
        ]
        for a in sorted(pe.get("agents", []), key=lambda x: x.get("tokens_saved", 0), reverse=True):
            lines.append(
                f"| {a['agent']} | {a['original_tokens']:,} tok | {a['compacted_tokens']:,} tok | "
                f"{a['reduction_pct']}% | {a['tokens_saved']:,} tok |"
            )
        lines.append(
            f"| **Total** | **{pe_totals['original_tokens']:,} tok** | "
            f"**{pe_totals['compacted_tokens']:,} tok** | **{pe_totals['reduction_pct']}%** | "
            f"**{pe_totals['tokens_saved']:,} tok** |"
        )
    else:
        lines.append("_No agents measured in this trace (pre-PH3.2 run, or no profiled agents executed)._")

    psa = report.get("prompt_slice_applied", {})
    psa_totals = psa.get("totals", {})
    lines += ["", "## Applied Prompt Slicing (PH3.3 — actually sent to the LLM call)", ""]
    if psa_totals.get("agents_applied"):
        lines += [
            f"Real byte-size reduction applied to {psa_totals['agents_applied']} agent(s)' "
            "LLM call inputs. Every excluded field was verified unread by both the live "
            "prompt builder and MockClaudeClient, so this cannot change prompt wording, "
            "reasoning, or output.",
            "",
            "| Agent | Original Size | Sliced Size | Reduction % | Bytes Saved | Fields Excluded |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        for a in sorted(psa.get("agents", []), key=lambda x: x.get("bytes_saved", 0), reverse=True):
            lines.append(
                f"| {a['agent']} | {a['original_bytes']:,} B | {a['sliced_bytes']:,} B | "
                f"{a['reduction_pct']}% | {a['bytes_saved']:,} B | {a['fields_excluded_count']} |"
            )
        lines.append(
            f"| **Total** | **{psa_totals['original_bytes']:,} B** | "
            f"**{psa_totals['sliced_bytes']:,} B** | **{psa_totals['reduction_pct']}%** | "
            f"**{psa_totals['bytes_saved']:,} B** | |"
        )
    else:
        lines.append("_No agents applied slicing in this trace (pre-PH3.3 run, or no wired agents executed)._")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m functional_agents.performance_report",
        description="Generate a performance report from a canonical pipeline trace (PH3.4).",
    )
    parser.add_argument("--trace", required=True, type=Path, help="Path to a pipeline trace JSON.")
    parser.add_argument("--md", type=Path, default=None, help="Write markdown report to this path.")
    parser.add_argument("--json", dest="json_out", type=Path, default=None,
                        help="Write the structured report JSON to this path.")
    args = parser.parse_args(argv)

    try:
        trace = json.loads(args.trace.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"error: could not read trace {args.trace}: {exc}", file=sys.stderr)
        return 2

    try:
        perf = extract_performance(trace, source=str(args.trace))
    except CanonicalTraceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    report = build_report(perf)
    md = render_markdown(report, source=str(args.trace))

    if args.md:
        args.md.write_text(md)
        print(f"wrote markdown report → {args.md}")
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2))
        print(f"wrote json report → {args.json_out}")
    if not args.md and not args.json_out:
        print(md)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
