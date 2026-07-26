"""Centralized reserved trace path helpers (PH3.4a).

Single source of truth for the canonical pipeline trace's filename, and for
detecting when a caller is about to write to a path that *looks like* it
holds the canonical pipeline trace (e.g. via ``run_agent.py --trace``) even
though only a full pipeline run (``functional_agents.cli run`` →
``functional_agents.pipeline_trace.write_canonical_trace``) can actually
produce one.

This module owns the filename convention; it has no dependency on
``pipeline_trace.py`` or any agent module, so it can be imported from
anywhere (``run_agent.py``, ``pipeline_trace.py``, ``performance_report.py``)
without a circular import.
"""

from __future__ import annotations

from pathlib import Path

# The exact, fixed filename functional_agents.pipeline_trace writes for a
# complete pipeline run. This is the same literal PH3.4 established —
# centralizing it here (PH3.4a) is a pure refactor, not a filename change.
CANONICAL_PIPELINE_TRACE_FILENAME = "pipeline.trace.json"

# The canonical pipeline trace location — single source of truth (HH2).
# All CLI, pipeline, and test code must reference this constant rather than
# hard-coding the path string.
CANONICAL_PIPELINE_TRACE = Path("outputs") / CANONICAL_PIPELINE_TRACE_FILENAME

# PH11.1 — StrategyTrace standalone artifact filename and default path.
STRATEGY_TRACE_FILENAME = "strategy.trace.json"
STRATEGY_TRACE = Path("outputs") / STRATEGY_TRACE_FILENAME

# Any basename ending in this suffix is also reserved: a run-specific prefix
# (e.g. "ph34_pipeline.trace.json") still reads as "the pipeline trace for
# this run" to a human, so it's just as misleading coming out of run_agent.py.
_RESERVED_SUFFIX = "_pipeline.trace.json"


class CanonicalTraceReservedError(Exception):
    """Raised when a caller tries to write a non-canonical trace to a
    filename reserved for the canonical pipeline trace."""


def is_reserved_pipeline_trace(path: str | Path) -> bool:
    """True if ``path``'s basename is reserved for the canonical pipeline trace.

    Directory-independent (checks only the basename), matching:
      - exactly ``"pipeline.trace.json"``
      - anything ending in ``"_pipeline.trace.json"`` (e.g. ``"ph34_pipeline.trace.json"``)
    """
    name = Path(path).name
    return name == CANONICAL_PIPELINE_TRACE_FILENAME or name.endswith(_RESERVED_SUFFIX)


def default_pipeline_trace(out_dir: str | Path) -> Path:
    """The canonical pipeline trace's default location for a given output dir."""
    return Path(out_dir) / CANONICAL_PIPELINE_TRACE_FILENAME


def require_not_reserved(path: str | Path, *, tool_name: str = "run_agent") -> None:
    """Raise ``CanonicalTraceReservedError`` if ``path`` is a reserved filename.

    Call this BEFORE doing any work — the whole point is to fail fast,
    deterministically, and never silently overwrite a canonical pipeline trace.
    """
    if not is_reserved_pipeline_trace(path):
        return
    raise CanonicalTraceReservedError(
        f"{Path(path).name} is reserved for the full Functional Pipeline.\n"
        f"\n"
        f"{tool_name} produces mini traces only.\n"
        f"\n"
        f"Generate a canonical pipeline trace with:\n"
        f"\n"
        f"    python3 -m functional_agents.cli run\n"
    )
