"""Hypothesis LLM boundary (PH2.3) — Universal LLM Boundary for HypothesisAgent.

Enforces the path established by Planner (PH2.1) and Evidence (PH2.2):

    Raw LLM output → Normalize → Validate → Typed HypothesisOutput → business logic

Business logic consumes only a validated ``HypothesisOutput``; raw model output is
never consumed directly. Normalization performs safe structural repair only
(stringified JSON, scalar→list, confidence casing, whitespace, dropping malformed
hypotheses) and never invents hypotheses, rewrites reasoning, modifies confidence,
or alters evidence references. Validation verifies required fields, hypothesis ids,
the confidence enum, and reference/collection structure.

The typed output wraps ``HypothesisItem`` (the canonical hypothesis schema
consumed downstream), so ``as_dicts()`` reproduces the exact per-hypothesis dicts
the agent produced before — behaviour and evidence references are preserved for
valid inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_agent.claude_client import HypothesisItem
from research_agent.llm_normalize import normalize_llm_object, normalize_llm_items
from .boundary_framework import BoundaryError, run_boundary

VALID_CONFIDENCE = {"high", "medium", "low"}

# Hypothesis fields that must be list[str].
_LIST_FIELDS = (
    "supporting_evidence", "contradicting_evidence", "evidence_gaps",
    "decision_implications", "disconfirming_evidence_needed",
)
# Fields required for a hypothesis to be structurally usable.
_REQUIRED_FIELDS = ("id", "title", "summary")


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

class HypothesisOutput(BaseModel):
    """The normalized, validated HypothesisAgent response consumed downstream."""

    hypotheses: list[HypothesisItem] = Field(default_factory=list)
    synthesis_note: str = ""

    def as_dicts(self) -> list[dict]:
        """Per-hypothesis dicts, identical to the pre-boundary agent output."""
        return [h.model_dump() for h in self.hypotheses]


# ---------------------------------------------------------------------------
# Typed boundary errors (distinct per stage)
# ---------------------------------------------------------------------------

class HypothesisBoundaryError(BoundaryError):
    """Base for Hypothesis boundary failures; carries stage diagnostics."""


class HypothesisGenerationError(HypothesisBoundaryError):
    stage = "generation"


class HypothesisNormalizationError(HypothesisBoundaryError):
    stage = "normalization"


class HypothesisValidationError(HypothesisBoundaryError):
    stage = "validation"


# ---------------------------------------------------------------------------
# Normalization (safe repair only)
# ---------------------------------------------------------------------------

def _norm_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


def normalize_hypothesis_payload(raw: Any) -> tuple[dict, dict]:
    """Normalize a raw hypothesis payload; return (dict, diagnostics).

    Raises HypothesisNormalizationError when the payload is not recoverable.
    """
    obj, obj_diag = normalize_llm_object(raw, component="hypothesis")
    if obj is None:
        raise HypothesisNormalizationError(
            "hypothesis payload is not a JSON object",
            {"failed_stage": "normalization", "object": obj_diag},
        )

    repairs: list[str] = []

    # Drop malformed hypotheses (non-dict / missing required fields).
    items, item_diag = normalize_llm_items(
        obj.get("hypotheses"), required_fields=_REQUIRED_FIELDS, component="hypothesis"
    )
    if item_diag.get("items_dropped"):
        repairs.append(f"dropped {item_diag['items_dropped']} malformed hypothesis(es)")

    normalized_hyps: list[dict] = []
    for h in items:
        nh = dict(h)  # preserve all fields; repair the known ones
        conf_in = nh.get("confidence", "medium")
        conf = conf_in.strip().lower() if isinstance(conf_in, str) else str(conf_in).strip().lower()
        if conf != conf_in:
            repairs.append(f"{nh.get('id','?')}: confidence casing")
        nh["confidence"] = conf or "medium"
        for f in _LIST_FIELDS:
            if f in nh:
                nh[f] = _norm_str_list(nh.get(f))
        for f in ("title", "summary", "type", "confidence_rationale"):
            if f in nh and not isinstance(nh[f], str):
                nh[f] = str(nh[f])
        normalized_hyps.append(nh)

    note_in = obj.get("synthesis_note", "")
    synthesis_note = note_in if isinstance(note_in, str) else str(note_in)

    normalized = {"hypotheses": normalized_hyps, "synthesis_note": synthesis_note}
    diag = {
        "object": obj_diag,
        "repairs": repairs,
        "hypotheses_received": item_diag.get("items_received", 0),
        "hypotheses_kept": len(normalized_hyps),
    }
    return normalized, diag


# ---------------------------------------------------------------------------
# Validation (deterministic, actionable)
# ---------------------------------------------------------------------------

def validate_hypothesis_output(normalized: dict) -> tuple[HypothesisOutput, dict]:
    """Validate a normalized hypothesis dict; return (HypothesisOutput, diagnostics).

    Raises HypothesisValidationError on required-field / enum / structure violations.
    """
    errors: list[str] = []
    hyps = normalized.get("hypotheses", [])
    if not isinstance(hyps, list):
        errors.append("hypotheses must be a list")
        hyps = []

    for i, h in enumerate(hyps):
        hid = h.get("id") if isinstance(h, dict) else None
        label = hid or f"index {i}"
        if not hid:
            errors.append(f"hypothesis {label} missing id")
        if not h.get("title"):
            errors.append(f"hypothesis {label} missing title")
        conf = h.get("confidence", "medium")
        if conf not in VALID_CONFIDENCE:
            errors.append(f"hypothesis {label} invalid confidence {conf!r}")
        for f in _LIST_FIELDS:
            if f in h and not isinstance(h[f], list):
                errors.append(f"hypothesis {label} field {f} must be a list")

    if errors:
        raise HypothesisValidationError(
            "; ".join(errors[:8]),
            {"failed_stage": "validation", "errors": errors},
        )

    try:
        output = HypothesisOutput(
            hypotheses=[HypothesisItem(**h) for h in hyps],
            synthesis_note=normalized.get("synthesis_note", ""),
        )
    except Exception as exc:  # pydantic type errors → validation failure
        raise HypothesisValidationError(
            f"hypothesis output failed schema validation: {exc}",
            {"failed_stage": "validation", "errors": [str(exc)]},
        ) from exc

    evidence_refs = sum(len(h.supporting_evidence) + len(h.contradicting_evidence)
                        for h in output.hypotheses)
    diag = {
        "passed": True,
        "errors": [],
        "hypotheses": len(output.hypotheses),
        "evidence_references": evidence_refs,
    }
    return output, diag


# ---------------------------------------------------------------------------
# Boundary orchestration
# ---------------------------------------------------------------------------

def finalize_hypotheses(raw: Any) -> tuple[HypothesisOutput, dict]:
    """Run the full boundary: normalize → validate → typed HypothesisOutput (PH2.5)."""
    return run_boundary(
        raw,
        normalize=normalize_hypothesis_payload,
        validate=validate_hypothesis_output,
        error_base=HypothesisBoundaryError,
    )
