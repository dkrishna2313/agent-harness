"""Strategic Engagement input model — J9.1.

A *Strategic Engagement* is the consulting-style brief a client hands to the
Functional Agent pipeline.  It replaces the single ``--goal`` string with a
structured context (situation, objectives, constraints, stakeholders, …) so that
``ProblemFramingAgent`` can derive sharper research questions.

This module defines:

    EngagementSpec          – the validated input contract
    load_engagement_spec()  – YAML / JSON loader with clear error handling

The model is deliberately separate from ``research_agent.engagement``'s
``StrategicEngagement`` (the internal, auto-generated persistence object that
links Decision Models and Research Objects).  ``EngagementSpec`` is the *input*;
``StrategicEngagement`` is the *runtime record*.  ``to_strategic_engagement()``
converts one into the other so existing engagement linkage keeps working.

Backwards compatibility
-----------------------
``--goal`` runs do not touch this module.  Engagement Mode is purely additive:
it synthesises a rich framing brief (``to_framing_brief()``) that is fed to the
existing goal-driven pipeline entry point.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

# Fields that materially shape the research. If absent we surface the gap
# explicitly (see ``missing_important_fields``) rather than silently inventing it.
_IMPORTANT_FIELDS = ("title", "current_situation", "objectives")


class EngagementError(ValueError):
    """Raised when an engagement file cannot be loaded or validated."""


class EngagementSpec(BaseModel):
    """Structured consulting engagement brief (J9.1).

    All fields are optional so partial engagements load cleanly, but the
    important ones (title, current_situation, objectives) are reported via
    ``missing_important_fields()`` instead of being fabricated.
    """

    title: str = ""
    client: str = ""
    industry: str = ""

    current_situation: str = ""

    objectives: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)

    decision_horizon: str = ""

    priorities: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    known_unknowns: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def missing_important_fields(self) -> list[str]:
        """Return the names of important fields that are empty.

        Used to record gaps in the trace explicitly rather than inventing
        content the client never supplied.
        """
        missing: list[str] = []
        for name in _IMPORTANT_FIELDS:
            value = getattr(self, name)
            if not value:
                missing.append(name)
        return missing

    def is_effectively_empty(self) -> bool:
        """True when no field carries content (a useless engagement)."""
        return not any(
            getattr(self, f.lower())
            for f in (
                "title", "client", "industry", "current_situation",
                "objectives", "constraints", "stakeholders", "assumptions",
                "success_criteria", "decision_horizon", "priorities",
                "risks", "known_unknowns",
            )
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_framing_brief(self) -> str:
        """Render a structured multi-section brief for ProblemFramingAgent.

        This is the richer replacement for a one-line ``--goal`` string.  Only
        sections with content are emitted, so a sparse engagement produces a
        compact brief and a rich one produces a detailed brief.  Missing
        important fields are noted inline so the framing model knows the gap is
        real rather than assuming the omitted context.
        """
        parts: list[str] = []

        headline = self.title or "Strategic engagement"
        if self.client:
            headline += f" for {self.client}"
        if self.industry:
            headline += f" ({self.industry})"
        parts.append(headline.strip())

        def _section(label: str, value: str) -> None:
            if value:
                parts.append(f"{label}: {value}")

        def _list_section(label: str, items: list[str]) -> None:
            if items:
                rendered = "; ".join(str(i) for i in items if str(i).strip())
                if rendered:
                    parts.append(f"{label}: {rendered}")

        _section("Current situation", self.current_situation)
        _list_section("Objectives", self.objectives)
        _list_section("Priorities", self.priorities)
        _section("Decision horizon", self.decision_horizon)
        _list_section("Constraints", self.constraints)
        _list_section("Stakeholders", self.stakeholders)
        _list_section("Stated assumptions", self.assumptions)
        _list_section("Known unknowns", self.known_unknowns)
        _list_section("Known risks", self.risks)
        _list_section("Success criteria", self.success_criteria)

        missing = self.missing_important_fields()
        if missing:
            parts.append(
                "Note — the client did not specify: "
                + ", ".join(name.replace("_", " ") for name in missing)
                + ". Do not invent these; frame research questions to surface them."
            )

        return "\n".join(parts)

    def to_trace_metadata(self) -> dict[str, Any]:
        """Compact engagement summary for the execution trace."""
        return {
            "title": self.title,
            "client": self.client,
            "industry": self.industry,
            "decision_horizon": self.decision_horizon,
            "objective_count": len(self.objectives),
            "constraint_count": len(self.constraints),
            "stakeholder_count": len(self.stakeholders),
            "risk_count": len(self.risks),
            "known_unknown_count": len(self.known_unknowns),
            "success_criteria_count": len(self.success_criteria),
            "missing_important_fields": self.missing_important_fields(),
        }


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _coerce_payload(raw: Any, *, source: str) -> dict[str, Any]:
    """Unwrap an optional top-level ``engagement:`` key and validate shape."""
    if raw is None:
        raise EngagementError(f"Engagement file is empty: {source}")
    if not isinstance(raw, dict):
        raise EngagementError(
            f"Engagement file must contain a mapping, got {type(raw).__name__}: {source}"
        )
    # Support both `engagement: {...}` wrapping and a bare top-level mapping.
    if "engagement" in raw and isinstance(raw["engagement"], dict):
        return raw["engagement"]
    return raw


def load_engagement_spec(path: str | Path) -> EngagementSpec:
    """Load and validate an engagement from a YAML or JSON file.

    Format is chosen by extension (.yaml/.yml → YAML, .json → JSON); files with
    any other extension are parsed as YAML (a superset of JSON).  Raises
    ``EngagementError`` with a clear, user-facing message on any failure.
    """
    p = Path(path)
    if not p.exists():
        raise EngagementError(f"Engagement file not found: {p}")
    if not p.is_file():
        raise EngagementError(f"Engagement path is not a file: {p}")

    text = p.read_text(encoding="utf-8")
    suffix = p.suffix.lower()

    try:
        if suffix == ".json":
            raw = json.loads(text)
        else:
            import yaml
            raw = yaml.safe_load(text)
    except json.JSONDecodeError as exc:
        raise EngagementError(f"Invalid JSON in {p}: {exc}") from exc
    except Exception as exc:  # yaml.YAMLError and friends
        raise EngagementError(f"Could not parse engagement file {p}: {exc}") from exc

    payload = _coerce_payload(raw, source=str(p))

    try:
        spec = EngagementSpec.model_validate(payload)
    except ValidationError as exc:
        raise EngagementError(f"Engagement validation failed for {p}:\n{exc}") from exc

    if spec.is_effectively_empty():
        raise EngagementError(
            f"Engagement file {p} contains no usable fields. "
            "Provide at least a title, current_situation, or objectives."
        )

    return spec
