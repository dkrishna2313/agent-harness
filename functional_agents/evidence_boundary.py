"""Evidence LLM boundary (PH2.2) — Universal LLM Boundary for EvidenceAgent.

Enforces the same explicit path Planner established (PH2.1):

    Raw evidence output → Normalize → Validate → Typed EvidenceOutput → business logic

The EvidenceAgent's evidence originates from Knowledge-Layer retrieval / legacy
extraction (the reranker LLM boundary is already hardened in PH1). This boundary
sits at the agent's OUTPUT: the assembled evidence note is normalized and
validated into a typed ``EvidenceOutput`` before it becomes ``context.evidence_notes``
and reaches downstream agents (Hypothesis, QA, Report).

Normalization performs safe structural repair only (never invents evidence,
citations, or mappings): drops malformed items (via the PH1 helper), coerces
mapping collections to ``list[str]``, and removes mapping references to dropped
items. Validation verifies structural integrity (required fields, mapping
integrity), and records citation-coverage and planner-alignment diagnostics.

For a well-formed note, normalization is a no-op and ``EvidenceOutput.as_note()``
reproduces the exact note dict — so existing evidence behavior and citation
integrity are preserved.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from research_agent.llm_normalize import normalize_llm_object, normalize_llm_items


# ---------------------------------------------------------------------------
# Typed output — mirrors the evidence_note dict shape (field order = note order)
# ---------------------------------------------------------------------------

class EvidenceOutput(BaseModel):
    """The normalized, validated EvidenceAgent response consumed downstream."""

    evidence_items: list[dict] = Field(default_factory=list)
    evidence_by_subquestion: dict[str, list[str]] = Field(default_factory=dict)
    evidence_by_area: dict[str, list[str]] = Field(default_factory=dict)
    coverage_by_subquestion: dict[str, Any] = Field(default_factory=dict)
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    profile_coverage_by_profile: dict[str, Any] = Field(default_factory=dict)
    profiles_requested: list[str] = Field(default_factory=list)
    profiles_contributing: list[str] = Field(default_factory=list)
    profiles_missing: list[str] = Field(default_factory=list)

    def as_note(self) -> dict:
        """Return the evidence-note dict downstream agents expect."""
        return self.model_dump()


# ---------------------------------------------------------------------------
# Typed boundary errors (distinct per stage)
# ---------------------------------------------------------------------------

class EvidenceBoundaryError(Exception):
    """Base for Evidence boundary failures; carries stage diagnostics."""

    stage = "boundary"

    def __init__(self, message: str, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {"failed_stage": self.stage}


class EvidenceGenerationError(EvidenceBoundaryError):
    """Retrieval / extraction (evidence generation) failed."""
    stage = "generation"


class EvidenceNormalizationError(EvidenceBoundaryError):
    """The evidence note is not structurally recoverable."""
    stage = "normalization"


class EvidenceValidationError(EvidenceBoundaryError):
    """Normalized evidence violates required fields / mapping integrity."""
    stage = "validation"


# ---------------------------------------------------------------------------
# Normalization (safe repair only)
# ---------------------------------------------------------------------------

def _coerce_id_map(value: Any) -> dict[str, list[str]]:
    """Coerce a {key: [ids]} mapping to clean dict[str, list[str]]."""
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in value.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, str):
            out[k] = [v.strip()] if v.strip() else []
        elif isinstance(v, (list, tuple)):
            out[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            out[k] = []
    return out


def normalize_evidence_note(raw: Any) -> tuple[dict, dict]:
    """Normalize a raw evidence note into a clean dict; return (dict, diagnostics).

    Raises EvidenceNormalizationError when the note is not a recoverable object.
    """
    obj, obj_diag = normalize_llm_object(raw, component="evidence")
    if obj is None:
        raise EvidenceNormalizationError(
            "evidence note is not a JSON object",
            {"failed_stage": "normalization", "object": obj_diag},
        )

    repairs: list[str] = []

    # Evidence items — drop malformed (non-dict / missing evidence_id) via PH1 helper.
    items, item_diag = normalize_llm_items(
        obj.get("evidence_items"), required_fields=("evidence_id",), component="evidence"
    )
    if item_diag.get("items_dropped"):
        repairs.append(f"dropped {item_diag['items_dropped']} malformed evidence item(s)")
    valid_ids = {str(it.get("evidence_id")) for it in items}

    # Mapping collections — coerce to dict[str, list[str]], drop refs to dropped items.
    eb_sub = _coerce_id_map(obj.get("evidence_by_subquestion"))
    eb_area = _coerce_id_map(obj.get("evidence_by_area"))
    dangling = 0
    for mapping in (eb_sub, eb_area):
        for key, ids in list(mapping.items()):
            kept = [i for i in ids if i in valid_ids]
            dangling += len(ids) - len(kept)
            mapping[key] = kept
    if dangling:
        repairs.append(f"dropped {dangling} mapping reference(s) to missing items")

    def _as_dict(v: Any) -> dict:
        return v if isinstance(v, dict) else {}

    def _as_str_list(v: Any) -> list[str]:
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
        if isinstance(v, (list, tuple)):
            return [str(x).strip() for x in v if str(x).strip()]
        return []

    normalized = {
        "evidence_items": items,
        "evidence_by_subquestion": eb_sub,
        "evidence_by_area": eb_area,
        "coverage_by_subquestion": _as_dict(obj.get("coverage_by_subquestion")),
        "evidence_summary": _as_dict(obj.get("evidence_summary")),
        "profile_coverage_by_profile": _as_dict(obj.get("profile_coverage_by_profile")),
        "profiles_requested": _as_str_list(obj.get("profiles_requested")),
        "profiles_contributing": _as_str_list(obj.get("profiles_contributing")),
        "profiles_missing": _as_str_list(obj.get("profiles_missing")),
    }
    return normalized, {"object": obj_diag, "repairs": repairs, "items_valid": len(items)}


# ---------------------------------------------------------------------------
# Validation (deterministic; records citation + planner-alignment diagnostics)
# ---------------------------------------------------------------------------

def validate_evidence_output(normalized: dict, *, plan: dict | None = None) -> tuple[EvidenceOutput, dict]:
    """Validate a normalized evidence note; return (EvidenceOutput, diagnostics).

    Raises EvidenceValidationError on structural-integrity violations.
    """
    errors: list[str] = []

    items = normalized.get("evidence_items", [])
    if not isinstance(items, list):
        errors.append("evidence_items must be a list")
        items = []

    # Required field: every item carries an evidence_id (citation anchor).
    valid_ids = set()
    missing_id = 0
    for it in items:
        eid = it.get("evidence_id") if isinstance(it, dict) else None
        if eid:
            valid_ids.add(str(eid))
        else:
            missing_id += 1
    if missing_id:
        errors.append(f"{missing_id} evidence item(s) missing evidence_id")

    # Mapping integrity: no references to unknown evidence_ids.
    dangling = []
    for field in ("evidence_by_subquestion", "evidence_by_area"):
        for key, ids in (normalized.get(field) or {}).items():
            for i in ids:
                if i not in valid_ids:
                    dangling.append(f"{field}[{key}] → {i}")
    if dangling:
        errors.append(f"dangling mapping references: {dangling[:5]}")

    if errors:
        raise EvidenceValidationError(
            "; ".join(errors),
            {"failed_stage": "validation", "errors": errors},
        )

    # Citation integrity (non-fatal diagnostic): items with a source attribution.
    cited = sum(1 for it in items if it.get("source_document"))
    # Planner alignment (non-fatal diagnostic): subquestion keys vs plan.
    plan_subqs = set((plan or {}).get("subquestions", []) or [])
    mapped_subqs = set(normalized.get("evidence_by_subquestion", {}).keys()) - {"_unmapped"}
    aligned = (not plan_subqs) or mapped_subqs.issubset(plan_subqs)

    try:
        output = EvidenceOutput(**normalized)
    except Exception as exc:  # pydantic type errors → validation failure
        raise EvidenceValidationError(
            f"evidence output failed schema validation: {exc}",
            {"failed_stage": "validation", "errors": [str(exc)]},
        ) from exc

    diag = {
        "passed": True,
        "errors": [],
        "evidence_items": len(items),
        "citations_present": cited,
        "citations_missing": len(items) - cited,
        "planner_aligned": aligned,
    }
    return output, diag


# ---------------------------------------------------------------------------
# Boundary orchestration
# ---------------------------------------------------------------------------

def finalize_evidence(raw_note: Any, *, plan: dict | None = None) -> tuple[EvidenceOutput, dict]:
    """Run the full boundary: normalize → validate → typed EvidenceOutput.

    Returns (EvidenceOutput, boundary_diagnostics). Raises a distinct
    EvidenceBoundaryError subclass (with .diagnostics) on any stage failure.
    Generation (retrieval/extraction) is assumed complete before this call.
    """
    stages = {"generation": "ok", "normalization": "pending", "validation": "pending"}
    try:
        normalized, norm_diag = normalize_evidence_note(raw_note)
        stages["normalization"] = "ok"
        output, val_diag = validate_evidence_output(normalized, plan=plan)
        stages["validation"] = "ok"
    except EvidenceBoundaryError as exc:
        stages[exc.stage] = "failed"
        exc.diagnostics = {**exc.diagnostics, "stages": stages, "failed_stage": exc.stage}
        raise

    return output, {
        "stages": stages,
        "failed_stage": None,
        "normalization": norm_diag,
        "validation": val_diag,
    }
