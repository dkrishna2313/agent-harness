"""Tests for the PH3.4a centralized reserved trace path helpers."""

from __future__ import annotations

import pytest

from functional_agents.trace_paths import (
    CANONICAL_PIPELINE_TRACE_FILENAME,
    CanonicalTraceReservedError,
    default_pipeline_trace,
    is_reserved_pipeline_trace,
    require_not_reserved,
)


# ---------------------------------------------------------------------------
# is_reserved_pipeline_trace
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [
    "pipeline.trace.json",
    "outputs/pipeline.trace.json",
    "/abs/path/to/pipeline.trace.json",
    "ph34_pipeline.trace.json",
    "outputs/ph34_pipeline.trace.json",
    "mock_pipeline.trace.json",
    "a/b/c/x_y_z_pipeline.trace.json",
])
def test_reserved_names_detected(path):
    assert is_reserved_pipeline_trace(path) is True


@pytest.mark.parametrize("path", [
    "planner.trace.json",
    "outputs/hypothesis.trace.json",
    "outputs/pipeline_summary.trace.json",  # prefix, not suffix — not reserved
    "outputs/harness_report.trace.json",
    "pipeline.json",  # wrong extension
    "not_a_trace.txt",
])
def test_non_reserved_names_allowed(path):
    assert is_reserved_pipeline_trace(path) is False


def test_reserved_check_is_basename_only():
    # A directory literally named "pipeline.trace.json" around a normal file
    # is not what we're guarding against — only the basename matters.
    assert is_reserved_pipeline_trace("pipeline.trace.json/notes.txt") is False


# ---------------------------------------------------------------------------
# default_pipeline_trace
# ---------------------------------------------------------------------------

def test_default_pipeline_trace_path():
    result = default_pipeline_trace("outputs")
    assert str(result) == f"outputs/{CANONICAL_PIPELINE_TRACE_FILENAME}"
    assert result.name == "pipeline.trace.json"


def test_default_pipeline_trace_accepts_path_object():
    from pathlib import Path
    result = default_pipeline_trace(Path("/tmp/run1"))
    assert result == Path("/tmp/run1/pipeline.trace.json")


# ---------------------------------------------------------------------------
# require_not_reserved
# ---------------------------------------------------------------------------

def test_require_not_reserved_passes_for_normal_filename():
    require_not_reserved("outputs/planner.trace.json")  # must not raise


def test_require_not_reserved_raises_for_exact_match():
    with pytest.raises(CanonicalTraceReservedError) as ei:
        require_not_reserved("outputs/pipeline.trace.json")
    msg = str(ei.value)
    assert "pipeline.trace.json" in msg
    assert "reserved" in msg.lower()
    assert "python3 -m functional_agents.cli run" in msg


def test_require_not_reserved_raises_for_suffix_match():
    with pytest.raises(CanonicalTraceReservedError):
        require_not_reserved("outputs/ph34_pipeline.trace.json")


def test_require_not_reserved_message_names_tool():
    with pytest.raises(CanonicalTraceReservedError) as ei:
        require_not_reserved("pipeline.trace.json", tool_name="run_agent")
    assert "run_agent" in str(ei.value)
