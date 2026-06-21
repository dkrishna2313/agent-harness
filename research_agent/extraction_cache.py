"""Disk cache for evidence extraction results.

Cache layout: .cache/extraction/<16-char-sha256>.json
Key: sha256(question + sorted_chunk_ids).
Each file stores a JSON-serialised list of EvidenceItem dicts.

A cache hit on the same question + same set of chunks returns the previously
extracted items instantly, eliminating the largest per-question LLM call on
repeat runs (regressions, re-benchmarks with the same web cache).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import Chunk, EvidenceItem

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = pathlib.Path(".cache/extraction")


class ExtractionCache:
    """Disk-backed cache for evidence extraction results.

    The cache key is derived from the question text and the sorted set of
    chunk IDs, so identical inputs always hit the same entry regardless of
    how the chunks were assembled.

    Parameters
    ----------
    cache_dir:
        Directory to store cache files. Defaults to ``.cache/extraction``
        relative to the current working directory. Created on demand.
    """

    def __init__(self, cache_dir: str | pathlib.Path = _DEFAULT_CACHE_DIR) -> None:
        self._dir = pathlib.Path(cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, question: str, chunks: "list[Chunk]") -> str:
        chunk_ids = sorted(getattr(c, "chunk_id", str(i)) for i, c in enumerate(chunks))
        raw = question + "|" + ",".join(chunk_ids)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path_for(self, key: str) -> pathlib.Path:
        return self._dir / f"{key}.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        question: str,
        chunks: "list[Chunk]",
    ) -> "list[EvidenceItem] | None":
        """Return cached items, or ``None`` on a miss."""
        from .schemas import EvidenceItem

        key = self._cache_key(question, chunks)
        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = [EvidenceItem.model_validate(d) for d in raw]
            LOGGER.debug("extraction_cache: hit  key=%s  items=%d", key, len(items))
            return items
        except Exception as exc:
            LOGGER.warning("extraction_cache: corrupt entry %s — %s", path, exc)
            return None

    def put(
        self,
        question: str,
        chunks: "list[Chunk]",
        items: "list[EvidenceItem]",
    ) -> None:
        """Persist *items* for the given question + chunks. Silently ignores write errors."""
        key = self._cache_key(question, chunks)
        try:
            self._ensure_dir()
            path = self._path_for(key)
            data = [item.model_dump() for item in items]
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            LOGGER.debug("extraction_cache: stored  key=%s  items=%d", key, len(items))
        except Exception as exc:
            LOGGER.warning("extraction_cache: failed to write key=%s — %s", key, exc)

    def invalidate(self, question: str, chunks: "list[Chunk]") -> bool:
        """Delete the cache entry. Returns True if a file was removed."""
        key = self._cache_key(question, chunks)
        path = self._path_for(key)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as exc:
                LOGGER.warning("extraction_cache: failed to delete %s — %s", path, exc)
        return False
