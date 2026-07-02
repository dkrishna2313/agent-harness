"""AgentContext snapshot (de)serialization — shared harness utilities (PH2.0).

Extracted from ``run_agent.py`` so the same serialize↔load round-trip can be
reused by the agent execution harness today and by orchestrator-side snapshot
capture in future. Behavior is identical to the original harness implementation.

Round-trip contract:
    ``load_context(path)`` -> AgentContext -> ``context_to_jsonable(ctx)`` -> JSON

Serialization is JSON-safe and strips inter-agent scratch (``_``-prefixed trace
keys such as ``_client`` / ``_perf_tracker``) and anything not serializable.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

from .context import AgentContext

LOGGER = logging.getLogger(__name__)


class HarnessError(Exception):
    """Raised for deterministic, user-facing harness / snapshot failures."""


CONTEXT_FIELDS = {f.name for f in dataclasses.fields(AgentContext)}


def _jsonify(obj: Any) -> Any:
    """Recursively convert to JSON-safe values; drop what cannot serialize."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("_"):
                continue  # scratch keys (_client, _perf_tracker, _memo, …)
            try:
                out[k] = _jsonify(v)
            except Exception:
                continue
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if hasattr(obj, "model_dump"):  # pydantic
        try:
            return _jsonify(obj.model_dump())
        except Exception:
            return str(obj)
    if dataclasses.is_dataclass(obj):
        return _jsonify(dataclasses.asdict(obj))
    return str(obj)


def context_to_jsonable(ctx: AgentContext) -> dict[str, Any]:
    """Serialize the durable AgentContext fields (trace scratch filtered)."""
    result: dict[str, Any] = {}
    for name in CONTEXT_FIELDS:
        result[name] = _jsonify(getattr(ctx, name))
    return result


def load_context(path: str | Path) -> AgentContext:
    """Load an AgentContext from a snapshot/fixture JSON file (unknown keys ignored)."""
    p = Path(path)
    if not p.exists():
        raise HarnessError(f"Fixture not found: {p}")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(f"Invalid JSON in fixture {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise HarnessError(f"Fixture must be a JSON object, got {type(raw).__name__}: {p}")

    unknown = sorted(set(raw) - CONTEXT_FIELDS)
    data = {k: v for k, v in raw.items() if k in CONTEXT_FIELDS}
    try:
        ctx = AgentContext(**data)
    except TypeError as exc:
        raise HarnessError(f"Fixture does not match AgentContext schema: {exc}") from exc
    if unknown:
        LOGGER.warning("[harness] ignored unknown fixture keys: %s", ", ".join(unknown))
    return ctx


def context_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, list[str]]:
    """Field-level diff of two serialized contexts."""
    added, modified, unchanged = [], [], []
    for name in sorted(CONTEXT_FIELDS):
        b, a = before.get(name), after.get(name)
        b_empty = not b
        a_empty = not a
        if b_empty and not a_empty:
            added.append(name)
        elif b == a:
            unchanged.append(name)
        else:
            modified.append(name)
    return {"added": added, "modified": modified, "unchanged": unchanged}
