"""Recommendation LLM boundary (PH2.4) — Universal LLM Boundary for RecommendationAgent.

Enforces the path established by Planner (PH2.1), Evidence (PH2.2), and
Hypothesis (PH2.3):

    Raw LLM output → Normalize → Validate → Typed RecommendationOutput → business logic

Business logic consumes only a validated ``RecommendationOutput``; raw model
output is never consumed directly. Normalization performs safe structural repair
only (stringified JSON, scalar→list, enum casing, whitespace, dropping malformed
recommendations) and never invents recommendations, rewrites rationale, or alters
priority / confidence / evidence references / supported hypotheses. Validation
verifies required fields, ids, the priority / confidence / time-horizon enums, and
reference/collection structure.

The typed output wraps ``RecommendationItem`` and ``RecommendationPortfolio`` (the
canonical schemas consumed downstream), so ``as_dicts()`` / ``portfolio_dict()``
reproduce the exact structures the agent produced before — preserving behaviour,
prioritization, and the portfolio for valid inputs.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_agent.claude_client import RecommendationItem, RecommendationPortfolio
from research_agent.llm_normalize import normalize_llm_object, normalize_llm_items

VALID_PRIORITY = {"high", "medium", "low"}
VALID_CONFIDENCE = {"high", "medium", "low"}
VALID_TIME_HORIZON = {"near_term", "medium_term", "long_term"}

_ENUM_FIELDS = {
    "priority": VALID_PRIORITY,
    "confidence": VALID_CONFIDENCE,
    "time_horizon": VALID_TIME_HORIZON,
}
_LIST_FIELDS = (
    "supported_assumption_ids", "supported_by_hypotheses", "supporting_evidence",
    "key_risks", "trigger_conditions",
)
_REQUIRED_FIELDS = ("id", "title", "summary")


# ---------------------------------------------------------------------------
# Typed output
# ---------------------------------------------------------------------------

class RecommendationOutput(BaseModel):
    """The normalized, validated RecommendationAgent response consumed downstream."""

    recommendations: list[RecommendationItem] = Field(default_factory=list)
    recommendation_portfolio: RecommendationPortfolio = Field(default_factory=RecommendationPortfolio)
    synthesis_note: str = ""

    def as_dicts(self) -> list[dict]:
        """Per-recommendation dicts, identical to the pre-boundary agent output."""
        return [r.model_dump() for r in self.recommendations]

    def portfolio_dict(self) -> dict:
        return self.recommendation_portfolio.model_dump()


# ---------------------------------------------------------------------------
# Typed boundary errors (distinct per stage)
# ---------------------------------------------------------------------------

class RecommendationBoundaryError(Exception):
    """Base for Recommendation boundary failures; carries stage diagnostics."""

    stage = "boundary"

    def __init__(self, message: str, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {"failed_stage": self.stage}


class RecommendationGenerationError(RecommendationBoundaryError):
    stage = "generation"


class RecommendationNormalizationError(RecommendationBoundaryError):
    stage = "normalization"


class RecommendationValidationError(RecommendationBoundaryError):
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


def _coerce_portfolio(value: Any) -> dict:
    v = value if isinstance(value, dict) else {}
    return {
        "near_term": _norm_str_list(v.get("near_term")),
        "medium_term": _norm_str_list(v.get("medium_term")),
        "long_term": _norm_str_list(v.get("long_term")),
    }


def normalize_recommendation_payload(raw: Any) -> tuple[dict, dict]:
    """Normalize a raw recommendation payload; return (dict, diagnostics).

    Raises RecommendationNormalizationError when the payload is not recoverable.
    """
    obj, obj_diag = normalize_llm_object(raw, component="recommendation")
    if obj is None:
        raise RecommendationNormalizationError(
            "recommendation payload is not a JSON object",
            {"failed_stage": "normalization", "object": obj_diag},
        )

    repairs: list[str] = []

    items, item_diag = normalize_llm_items(
        obj.get("recommendations"), required_fields=_REQUIRED_FIELDS, component="recommendation"
    )
    if item_diag.get("items_dropped"):
        repairs.append(f"dropped {item_diag['items_dropped']} malformed recommendation(s)")

    normalized_recs: list[dict] = []
    for r in items:
        nr = dict(r)
        for f, _ in _ENUM_FIELDS.items():
            if f in nr and isinstance(nr[f], str):
                low = nr[f].strip().lower()
                if low != nr[f]:
                    repairs.append(f"{nr.get('id','?')}: {f} casing")
                nr[f] = low
        for f in _LIST_FIELDS:
            if f in nr:
                nr[f] = _norm_str_list(nr.get(f))
        for f in ("title", "summary", "confidence_rationale", "recommendation_id"):
            if f in nr and not isinstance(nr[f], str):
                nr[f] = str(nr[f])
        normalized_recs.append(nr)

    note_in = obj.get("synthesis_note", "")
    synthesis_note = note_in if isinstance(note_in, str) else str(note_in)

    normalized = {
        "recommendations": normalized_recs,
        "recommendation_portfolio": _coerce_portfolio(obj.get("recommendation_portfolio")),
        "synthesis_note": synthesis_note,
    }
    diag = {
        "object": obj_diag,
        "repairs": repairs,
        "recommendations_received": item_diag.get("items_received", 0),
        "recommendations_kept": len(normalized_recs),
    }
    return normalized, diag


# ---------------------------------------------------------------------------
# Validation (deterministic, actionable)
# ---------------------------------------------------------------------------

def validate_recommendation_output(normalized: dict) -> tuple[RecommendationOutput, dict]:
    """Validate a normalized recommendation dict; return (RecommendationOutput, diagnostics).

    Raises RecommendationValidationError on required-field / enum / structure failure.
    """
    errors: list[str] = []
    recs = normalized.get("recommendations", [])
    if not isinstance(recs, list):
        errors.append("recommendations must be a list")
        recs = []

    for i, r in enumerate(recs):
        rid = r.get("id") if isinstance(r, dict) else None
        label = rid or f"index {i}"
        if not rid:
            errors.append(f"recommendation {label} missing id")
        if not r.get("title"):
            errors.append(f"recommendation {label} missing title")
        for f, valid in _ENUM_FIELDS.items():
            val = r.get(f)
            if val is not None and val not in valid:
                errors.append(f"recommendation {label} invalid {f} {val!r}")
        for f in _LIST_FIELDS:
            if f in r and not isinstance(r[f], list):
                errors.append(f"recommendation {label} field {f} must be a list")

    if errors:
        raise RecommendationValidationError(
            "; ".join(errors[:8]),
            {"failed_stage": "validation", "errors": errors},
        )

    try:
        output = RecommendationOutput(
            recommendations=[RecommendationItem(**r) for r in recs],
            recommendation_portfolio=RecommendationPortfolio(
                **normalized.get("recommendation_portfolio", {})
            ),
            synthesis_note=normalized.get("synthesis_note", ""),
        )
    except Exception as exc:  # pydantic type errors → validation failure
        raise RecommendationValidationError(
            f"recommendation output failed schema validation: {exc}",
            {"failed_stage": "validation", "errors": [str(exc)]},
        ) from exc

    diag = {
        "passed": True,
        "errors": [],
        "recommendations": len(output.recommendations),
        "evidence_references": sum(len(r.supporting_evidence) for r in output.recommendations),
        "hypothesis_references": sum(len(r.supported_by_hypotheses) for r in output.recommendations),
    }
    return output, diag


# ---------------------------------------------------------------------------
# Boundary orchestration
# ---------------------------------------------------------------------------

def finalize_recommendations(raw: Any) -> tuple[RecommendationOutput, dict]:
    """Run the full boundary: normalize → validate → typed RecommendationOutput.

    Returns (RecommendationOutput, boundary_diagnostics). Raises a distinct
    RecommendationBoundaryError subclass (with .diagnostics) on any stage failure.
    """
    stages = {"generation": "ok", "normalization": "pending", "validation": "pending"}
    try:
        normalized, norm_diag = normalize_recommendation_payload(raw)
        stages["normalization"] = "ok"
        output, val_diag = validate_recommendation_output(normalized)
        stages["validation"] = "ok"
    except RecommendationBoundaryError as exc:
        stages[exc.stage] = "failed"
        exc.diagnostics = {**exc.diagnostics, "stages": stages, "failed_stage": exc.stage}
        raise

    return output, {
        "stages": stages,
        "failed_stage": None,
        "normalization": norm_diag,
        "validation": val_diag,
    }
