"""Report boundary (PH2.5 Part B) — Universal LLM Boundary for ReportAgent.

ReportAgent does not call an LLM to *generate* its content — it assembles the
report from already-typed context produced by the (now hardened) upstream agents.
Its boundary therefore validates the report INPUTS before business logic (report
rendering) runs:

    Report inputs → Normalize → Validate → Typed ReportInput → business logic → Report artifact

This formalizes ReportAgent's existing readiness checks (a memo must be present)
into the shared boundary pattern, emitting the same diagnostics schema as the
other hardened agents. Report output is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .boundary_framework import BoundaryError, run_boundary


# ---------------------------------------------------------------------------
# Typed input (holds live objects — a dataclass, not a serialized model)
# ---------------------------------------------------------------------------

@dataclass
class ReportInput:
    """The normalized, validated inputs ReportAgent's business logic consumes."""

    question: str
    memo: Any
    documents: list = field(default_factory=list)
    plan: dict = field(default_factory=dict)
    evidence_note: dict = field(default_factory=dict)
    recommendation_count: int = 0


# ---------------------------------------------------------------------------
# Typed boundary errors
# ---------------------------------------------------------------------------

class ReportBoundaryError(BoundaryError):
    """Base for Report boundary failures; carries stage diagnostics."""


class ReportGenerationError(ReportBoundaryError):
    stage = "generation"


class ReportNormalizationError(ReportBoundaryError):
    stage = "normalization"


class ReportValidationError(ReportBoundaryError):
    stage = "validation"


# ---------------------------------------------------------------------------
# Normalization (safe structural coercion only)
# ---------------------------------------------------------------------------

def normalize_report_input(raw: Any) -> tuple[dict, dict]:
    """Coerce raw report inputs into a clean dict; return (dict, diagnostics)."""
    if not isinstance(raw, dict):
        raise ReportNormalizationError(
            "report inputs are not a mapping",
            {"failed_stage": "normalization"},
        )
    normalized = {
        "question": raw.get("question") if isinstance(raw.get("question"), str) else (raw.get("question") or ""),
        "memo": raw.get("memo"),
        "documents": raw.get("documents") if isinstance(raw.get("documents"), list) else [],
        "plan": raw.get("plan") if isinstance(raw.get("plan"), dict) else {},
        "evidence_note": raw.get("evidence_note") if isinstance(raw.get("evidence_note"), dict) else {},
        "recommendation_count": int(raw.get("recommendation_count") or 0),
    }
    return normalized, {"repairs": [], "has_memo": normalized["memo"] is not None}


# ---------------------------------------------------------------------------
# Validation (readiness gate)
# ---------------------------------------------------------------------------

def validate_report_input(normalized: dict) -> tuple[ReportInput, dict]:
    """Validate report readiness; return (ReportInput, diagnostics).

    Raises ReportValidationError when a required input is missing (mirrors the
    prior "no memo → cannot write report" guard).
    """
    errors: list[str] = []
    if normalized.get("memo") is None:
        errors.append("no memo available; report cannot be rendered")
    if not (normalized.get("question") or "").strip():
        errors.append("no question available")

    if errors:
        raise ReportValidationError(
            "; ".join(errors),
            {"failed_stage": "validation", "errors": errors},
        )

    report_input = ReportInput(
        question=normalized["question"],
        memo=normalized["memo"],
        documents=normalized["documents"],
        plan=normalized["plan"],
        evidence_note=normalized["evidence_note"],
        recommendation_count=normalized["recommendation_count"],
    )
    diag = {
        "passed": True,
        "errors": [],
        "documents": len(report_input.documents),
        "recommendations": report_input.recommendation_count,
    }
    return report_input, diag


# ---------------------------------------------------------------------------
# Boundary orchestration
# ---------------------------------------------------------------------------

def finalize_report_input(raw: Any) -> tuple[ReportInput, dict]:
    """Run the full boundary: normalize → validate → typed ReportInput (PH2.5)."""
    return run_boundary(
        raw,
        normalize=normalize_report_input,
        validate=validate_report_input,
        error_base=ReportBoundaryError,
    )
