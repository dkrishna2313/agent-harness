"""Presentation Specification loader — P2.5.

Reads ``presentation_spec.yaml`` and exposes typed accessors used by the
executive report renderer.  The spec is loaded once at import time; the
module-level ``_SPEC`` singleton is the entry point for all consumers.

Architecture note
-----------------
This module sits between the YAML file (policy owner) and ``report_agent``
(policy consumer).  It must not import from any reasoning or communication
layer.  Its only dependency is the standard library plus PyYAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_SPEC_PATH = Path(__file__).parent / "presentation_spec.yaml"


class PresentationSpec:
    """Typed view over the presentation_spec.yaml configuration."""

    def __init__(self, path: Path = _SPEC_PATH) -> None:
        with open(path, encoding="utf-8") as fh:
            raw: dict[str, Any] = yaml.safe_load(fh)
        self._p: dict[str, Any] = raw["presentation"]

    # ------------------------------------------------------------------
    # Conventions
    # ------------------------------------------------------------------

    @property
    def conventions(self) -> dict[str, Any]:
        return self._p.get("conventions", {})

    @property
    def missing_value(self) -> str:
        """Fallback text for absent optional fields in tables."""
        return self.conventions.get("missing_value", "Not specified")

    @property
    def bullet_style(self) -> str:
        """List-item prefix character (default: '-')."""
        return self.conventions.get("bullet_style", "-")

    # ------------------------------------------------------------------
    # Terminology
    # ------------------------------------------------------------------

    @property
    def terminology(self) -> dict[str, dict[str, str]]:
        return self._p.get("terminology", {})

    @property
    def acronym_expansions(self) -> dict[str, str]:
        """Map of acronym → full expansion for first-use expansion."""
        return {
            term: entry["expand_first"]
            for term, entry in self.terminology.items()
            if "expand_first" in entry
        }

    @property
    def glossary_definitions(self) -> dict[str, str]:
        """Map of term → one-sentence definition for the executive glossary."""
        return {
            term: entry["glossary"]
            for term, entry in self.terminology.items()
            if "glossary" in entry
        }

    # ------------------------------------------------------------------
    # Components (metadata only — no active dispatch in P2.5)
    # ------------------------------------------------------------------

    @property
    def components(self) -> dict[str, Any]:
        return self._p.get("components", {})


_SPEC = PresentationSpec()
