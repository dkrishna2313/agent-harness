"""Planner LLM boundary (PH2.1) — the first Universal LLM Boundary.

Enforces one path for all Planner reasoning:

    Raw LLM output → Normalize → Validate → Typed PlannerOutput → business logic

Business logic must never consume raw model output. Normalization repairs safe
formatting issues (never invents/changes semantics); validation checks required
fields, the research_type enum, collection types, and planner invariants
(non-empty subquestions / investigation areas). Each stage raises a *distinct*
typed error carrying diagnostics, so the failing stage is reported precisely.

This module reuses the shared PH1 helper (`normalize_llm_object`) for the
object-shape step and adds Planner-specific list/enum normalization.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_agent.llm_normalize import normalize_llm_object
from .boundary_framework import BoundaryError, run_boundary

VALID_RESEARCH_TYPES = {"FACT_LOOKUP", "COMPARISON", "EXPLANATION", "RESEARCH"}


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

class PlannerOutput(BaseModel):
    """The normalized, validated Planner response consumed by business logic."""

    research_type: str
    subquestions: list[str] = Field(default_factory=list)
    investigation_areas: list[str] = Field(default_factory=list)
    profiles_used: list[str] = Field(default_factory=list)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Typed boundary errors (distinct per stage)
# ---------------------------------------------------------------------------

class PlannerBoundaryError(BoundaryError):
    """Base for Planner boundary failures; carries stage diagnostics."""


class PlannerGenerationError(PlannerBoundaryError):
    """The LLM call itself failed (network, truncation, tool error)."""
    stage = "generation"


class PlannerNormalizationError(PlannerBoundaryError):
    """Raw output is not structurally recoverable into the expected shape."""
    stage = "normalization"


class PlannerValidationError(PlannerBoundaryError):
    """Normalized output violates required fields / enum / invariants."""
    stage = "validation"


# ---------------------------------------------------------------------------
# Normalization (safe repair only — no semantic invention)
# ---------------------------------------------------------------------------

def _norm_str_list(value: Any) -> tuple[list[str], list[str]]:
    """Coerce a value into a clean list[str]; return (list, repair notes)."""
    repairs: list[str] = []
    if value is None:
        return [], repairs
    if isinstance(value, str):
        s = value.strip()
        return ([s], ["wrapped scalar string into a list"]) if s else ([], repairs)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        dropped = 0
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
                else:
                    dropped += 1
            else:
                dropped += 1
        if dropped:
            repairs.append(f"dropped {dropped} non-string/empty item(s)")
        return out, repairs
    return [], ["dropped non-list value"]


def normalize_planner_payload(raw: Any) -> tuple[dict, dict]:
    """Normalize raw planner output into a clean dict; return (dict, diagnostics).

    Raises PlannerNormalizationError when the payload is not a recoverable object.
    """
    obj, obj_diag = normalize_llm_object(raw, component="planner")
    if obj is None:
        raise PlannerNormalizationError(
            "planner payload is not a JSON object",
            {"failed_stage": "normalization", "object": obj_diag},
        )

    repairs: list[str] = []

    rt_in = obj.get("research_type", "")
    rt = rt_in.strip().upper() if isinstance(rt_in, str) else str(rt_in).strip().upper()
    if rt != rt_in:
        repairs.append("normalized research_type casing/whitespace")

    subq, r1 = _norm_str_list(obj.get("subquestions"))
    areas, r2 = _norm_str_list(obj.get("investigation_areas"))
    profs, r3 = _norm_str_list(obj.get("profiles_used"))
    repairs += [f"subquestions: {r}" for r in r1]
    repairs += [f"investigation_areas: {r}" for r in r2]
    repairs += [f"profiles_used: {r}" for r in r3]

    reasoning_in = obj.get("reasoning", "")
    reasoning = reasoning_in if isinstance(reasoning_in, str) else str(reasoning_in)

    normalized = {
        "research_type": rt,
        "subquestions": subq,
        "investigation_areas": areas,
        "profiles_used": profs,
        "reasoning": reasoning,
    }
    diag = {
        "object": obj_diag,
        "repairs": repairs,
        "recovered": bool(repairs) or obj_diag.get("items_valid") == 1 and obj_diag.get("items_received") == 1,
    }
    return normalized, diag


# ---------------------------------------------------------------------------
# Validation (deterministic, actionable)
# ---------------------------------------------------------------------------

def validate_planner_output(normalized: dict) -> tuple[PlannerOutput, dict]:
    """Validate a normalized planner dict; return (PlannerOutput, diagnostics).

    Raises PlannerValidationError with the full list of violations.
    """
    errors: list[str] = []

    rt = normalized.get("research_type", "")
    if rt not in VALID_RESEARCH_TYPES:
        errors.append(
            f"research_type {rt!r} not in {sorted(VALID_RESEARCH_TYPES)}"
        )
    if not normalized.get("subquestions"):
        errors.append("subquestions must be non-empty")
    if not normalized.get("investigation_areas"):
        errors.append("investigation_areas must be non-empty")

    if errors:
        raise PlannerValidationError(
            "; ".join(errors),
            {"failed_stage": "validation", "errors": errors},
        )

    try:
        output = PlannerOutput(**normalized)
    except Exception as exc:  # pydantic type errors → validation failure
        raise PlannerValidationError(
            f"planner output failed schema validation: {exc}",
            {"failed_stage": "validation", "errors": [str(exc)]},
        ) from exc

    return output, {"passed": True, "errors": []}


# ---------------------------------------------------------------------------
# Boundary orchestration
# ---------------------------------------------------------------------------

def plan_from_raw(raw: Any) -> tuple[PlannerOutput, dict]:
    """Run the full boundary: normalize → validate → typed PlannerOutput (PH2.5)."""
    return run_boundary(
        raw,
        normalize=normalize_planner_payload,
        validate=validate_planner_output,
        error_base=PlannerBoundaryError,
    )
