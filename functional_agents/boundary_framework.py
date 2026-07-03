"""Universal LLM Boundary framework (PH2.5).

The Planner, Evidence, Hypothesis, and Recommendation boundaries (PH2.1–PH2.4)
independently implemented the same shape:

    Raw → Normalize → Validate → Typed Object → (business logic) + Diagnostics

This module consolidates the shared parts so each boundary module only defines
its **typed object**, **normalization rules**, and **validation rules**:

- ``BoundaryError`` — base exception carrying ``.stage`` and ``.diagnostics``.
- ``run_boundary()`` — the stage orchestration + uniform diagnostics.

Every hardened boundary therefore emits an identical diagnostics schema:

    {"stages": {"generation", "normalization", "validation"},
     "failed_stage": None | "<stage>",
     "normalization": {...}, "validation": {...}}

On failure the raised ``BoundaryError`` subclass carries ``.diagnostics`` with
``stages`` and ``failed_stage`` populated.
"""

from __future__ import annotations

from typing import Any, Callable


class BoundaryError(Exception):
    """Base for all LLM-boundary failures; carries per-stage diagnostics.

    Subclasses set a ``stage`` class attribute ("generation" | "normalization" |
    "validation"); ``run_boundary`` uses it to mark the failing stage.
    """

    stage: str = "boundary"

    def __init__(self, message: str, diagnostics: dict | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {"failed_stage": self.stage}


def run_boundary(
    raw: Any,
    *,
    normalize: Callable[[Any], tuple[Any, dict]],
    validate: Callable[[Any], tuple[Any, dict]],
    error_base: type[BoundaryError],
) -> tuple[Any, dict]:
    """Run normalize → validate, returning (typed_output, boundary_diagnostics).

    ``normalize(raw) -> (normalized, norm_diag)`` and
    ``validate(normalized) -> (output, val_diag)`` each raise an ``error_base``
    subclass (with a ``stage``) on failure. Generation is assumed complete before
    this call (callers raise a generation-stage error upstream when it is not).
    """
    stages = {"generation": "ok", "normalization": "pending", "validation": "pending"}
    try:
        normalized, norm_diag = normalize(raw)
        stages["normalization"] = "ok"
        output, val_diag = validate(normalized)
        stages["validation"] = "ok"
    except error_base as exc:
        failed = getattr(exc, "stage", "boundary")
        stages[failed] = "failed"
        exc.diagnostics = {**exc.diagnostics, "stages": stages, "failed_stage": failed}
        raise

    return output, {
        "stages": stages,
        "failed_stage": None,
        "normalization": norm_diag,
        "validation": val_diag,
    }
