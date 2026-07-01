"""LLM output boundary normalization — Platform Hardening PH1.

Platform rule:

    LLM → Normalize → Validate → Typed Object → Business Logic

Never let typed business logic consume raw LLM output directly. LLMs
intermittently emit structurally-valid-but-wrong shapes (a list of strings where
a list of objects was declared, missing required fields, nulls). When typed code
assumes the declared schema and calls ``.get()`` / indexes those items, it throws
at runtime — the class of defect found in the J10 reranker investigation.

This module provides one small, dependency-free helper that every structured
LLM boundary can route its raw output through. It:

  * accepts only a list (anything else → empty result, recorded);
  * coerces bare strings into ``{coerce_str_key: value}`` when a key is given
    (the common "the model returned just the id" case);
  * drops items that are not dicts after coercion;
  * drops dicts missing any ``required_fields`` (present and non-None);
  * returns the surviving items plus lightweight diagnostics.

It never raises on malformed input — malformed items are dropped so the caller's
existing fallback path (e.g. retrieval-order) can take over.

``research_agent`` is the lowest shared layer (``knowledge`` and
``functional_agents`` both import from it), so this helper lives here to be
importable everywhere without a cycle.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def normalize_llm_items(
    raw: Any,
    *,
    required_fields: tuple[str, ...] = (),
    coerce_str_key: str | None = None,
    component: str = "",
) -> tuple[list[dict], dict]:
    """Normalize a raw LLM list into well-formed dict items.

    Parameters
    ----------
    raw:
        The raw value produced by the LLM (expected: a list of dicts). Any
        non-list value yields an empty result rather than an exception.
    required_fields:
        Field names each surviving item must contain (present and not None).
    coerce_str_key:
        When set, a bare string item ``s`` is coerced to ``{coerce_str_key: s}``
        before validation. Handles the frequent "model returned the id string
        instead of an object" case.
    component:
        Label for diagnostics (e.g. ``"reranker"``).

    Returns
    -------
    (items, diagnostics)
        ``items`` is the list of surviving dicts. ``diagnostics`` is a dict:
        ``{component, items_received, items_valid, items_dropped, fallback_used}``
        (``fallback_used`` defaults to False; callers set it when the empty
        result triggers their fallback path).
    """
    if not isinstance(raw, list):
        received = 0 if raw is None else 1
        diag = {
            "component": component,
            "items_received": received,
            "items_valid": 0,
            "items_dropped": received,
            "fallback_used": False,
        }
        if received:
            LOGGER.warning(
                "llm_normalize[%s]: expected a list, got %s — dropping all.",
                component, type(raw).__name__,
            )
        return [], diag

    received = len(raw)
    valid: list[dict] = []
    for item in raw:
        if isinstance(item, str) and coerce_str_key is not None:
            item = {coerce_str_key: item}
        if not isinstance(item, dict):
            continue
        if required_fields and not all(
            (f in item and item[f] is not None) for f in required_fields
        ):
            continue
        valid.append(item)

    dropped = received - len(valid)
    if dropped:
        LOGGER.warning(
            "llm_normalize[%s]: dropped %d/%d malformed items "
            "(non-dict or missing required fields %s).",
            component, dropped, received, list(required_fields),
        )

    return valid, {
        "component": component,
        "items_received": received,
        "items_valid": len(valid),
        "items_dropped": dropped,
        "fallback_used": False,
    }
