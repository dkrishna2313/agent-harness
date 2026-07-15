"""Disk cache for planning-layer LLM outputs (PH4.1).

Cache layout: .cache/planning/<16-char-sha256>.json
Key: sha256(operation + canonical JSON of inputs).

Guarantees that identical inputs to frame_problem and frame_executive_decision
always return identical payloads across separate runs, eliminating residual
floating-point non-determinism that persists even at temperature=0.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = pathlib.Path(".cache/planning")


class PlanningCache:
    """Disk-backed cache for planning-layer LLM payloads.

    The cache key is a SHA256 over the canonicalised input dict so identical
    inputs always map to the same entry regardless of call order.

    Parameters
    ----------
    cache_dir:
        Directory to store cache files. Defaults to ``.cache/planning``
        relative to the current working directory. Created on demand.
    """

    def __init__(self, cache_dir: str | pathlib.Path = _DEFAULT_CACHE_DIR) -> None:
        self._dir = pathlib.Path(cache_dir)

    def _cache_key(self, operation: str, inputs: dict[str, Any]) -> str:
        raw = operation + "|" + json.dumps(inputs, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _path_for(self, key: str) -> pathlib.Path:
        return self._dir / f"{key}.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, operation: str, inputs: dict[str, Any]) -> dict[str, Any] | None:
        """Return the cached payload dict, or ``None`` on a miss."""
        key = self._cache_key(operation, inputs)
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            LOGGER.debug("planning_cache: hit  op=%s  key=%s", operation, key)
            return payload
        except Exception as exc:
            LOGGER.warning("planning_cache: corrupt entry %s — %s", path, exc)
            return None

    def put(self, operation: str, inputs: dict[str, Any], payload: dict[str, Any]) -> None:
        """Persist *payload* for the given operation + inputs. Silently ignores write errors."""
        key = self._cache_key(operation, inputs)
        try:
            self._ensure_dir()
            path = self._path_for(key)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            LOGGER.debug("planning_cache: stored  op=%s  key=%s", operation, key)
        except Exception as exc:
            LOGGER.warning("planning_cache: failed to write op=%s key=%s — %s", operation, key, exc)
