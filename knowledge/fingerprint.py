"""Source fingerprinting utilities."""

from __future__ import annotations

import hashlib


def compute_text_fingerprint(text: str) -> str:
    """SHA-256 of canonical text. Determines source_id via Source.compute_source_id()."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


