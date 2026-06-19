"""K1.0 – Disk cache for downloaded web pages.

Cache layout: .cache/web/<first-16-chars-of-sha256-hex>.json
Each file stores the JSON-encoded dict that web_retrieve() passes to cache.set().
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib

LOGGER = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = pathlib.Path(".cache/web")


class WebPageCache:
    """Simple disk-backed key-value store for raw web-page content.

    Parameters
    ----------
    cache_dir:
        Directory to store cache files.  Defaults to ``.cache/web`` relative
        to the current working directory.  Created on demand.
    """

    def __init__(self, cache_dir: str | pathlib.Path = _DEFAULT_CACHE_DIR) -> None:
        self._dir = pathlib.Path(cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path_for(self, url: str) -> pathlib.Path:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._dir / f"{key}.json"

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, url: str) -> dict | None:
        """Return cached data for *url*, or ``None`` on a cache miss."""
        path = self._path_for(url)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Failed to read cache file %s: %s", path, exc)
            return None

    def set(self, url: str, data: dict) -> None:
        """Persist *data* for *url*.  Silently ignores write errors."""
        try:
            self._ensure_dir()
            path = self._path_for(url)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Failed to write cache file for %s: %s", url, exc)

    def invalidate(self, url: str) -> bool:
        """Delete the cache entry for *url*.  Returns True if a file was removed."""
        path = self._path_for(url)
        if path.exists():
            try:
                path.unlink()
                return True
            except Exception as exc:
                LOGGER.warning("Failed to delete cache file %s: %s", path, exc)
        return False
