"""Source fingerprinting utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_text_fingerprint(text: str) -> str:
    """SHA-256 of canonical text. Determines source_id via Source.compute_source_id()."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_fingerprint(path: Path) -> str:
    """SHA-256 of raw file bytes — fast check before text extraction."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprints_match(a: str, b: str) -> bool:
    return a == b
