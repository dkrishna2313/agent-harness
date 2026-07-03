"""Tests for the shared Universal Boundary framework + Report boundary (PH2.5)."""

from __future__ import annotations

import pytest

from functional_agents.boundary_framework import BoundaryError, run_boundary


# ---------------------------------------------------------------------------
# Shared framework
# ---------------------------------------------------------------------------

class _MyError(BoundaryError):
    pass


class _NormErr(_MyError):
    stage = "normalization"


class _ValErr(_MyError):
    stage = "validation"


def test_run_boundary_success_diagnostics():
    out, diag = run_boundary(
        {"x": 1},
        normalize=lambda r: (r, {"repairs": []}),
        validate=lambda n: ({"typed": n}, {"passed": True}),
        error_base=_MyError,
    )
    assert out == {"typed": {"x": 1}}
    assert diag["stages"] == {"generation": "ok", "normalization": "ok", "validation": "ok"}
    assert diag["failed_stage"] is None
    assert set(diag) == {"stages", "failed_stage", "normalization", "validation"}


def test_run_boundary_normalization_failure():
    def _norm(_):
        raise _NormErr("bad shape")
    with pytest.raises(_NormErr) as ei:
        run_boundary({}, normalize=_norm, validate=lambda n: (n, {}), error_base=_MyError)
    d = ei.value.diagnostics
    assert d["failed_stage"] == "normalization"
    assert d["stages"]["normalization"] == "failed"
    assert d["stages"]["validation"] == "pending"


def test_run_boundary_validation_failure():
    def _val(_):
        raise _ValErr("bad values")
    with pytest.raises(_ValErr) as ei:
        run_boundary({}, normalize=lambda r: (r, {}), validate=_val, error_base=_MyError)
    d = ei.value.diagnostics
    assert d["failed_stage"] == "validation"
    assert d["stages"]["normalization"] == "ok"
    assert d["stages"]["validation"] == "failed"


def test_boundary_error_default_diagnostics():
    e = _ValErr("x")
    assert e.diagnostics["failed_stage"] == "validation"


# ---------------------------------------------------------------------------
# Part D — every hardened boundary derives from the shared base + identical schema
# ---------------------------------------------------------------------------

def test_all_boundary_bases_share_framework():
    from functional_agents.planner_boundary import PlannerBoundaryError
    from functional_agents.evidence_boundary import EvidenceBoundaryError
    from functional_agents.hypothesis_boundary import HypothesisBoundaryError
    from functional_agents.recommendation_boundary import RecommendationBoundaryError
    from functional_agents.report_boundary import ReportBoundaryError
    for cls in (PlannerBoundaryError, EvidenceBoundaryError, HypothesisBoundaryError,
                RecommendationBoundaryError, ReportBoundaryError):
        assert issubclass(cls, BoundaryError)


def test_diagnostics_schema_identical_across_agents():
    """Part C: all boundaries emit the same diagnostics keys and stage names."""
    from functional_agents.planner_boundary import plan_from_raw
    from functional_agents.hypothesis_boundary import finalize_hypotheses
    from functional_agents.recommendation_boundary import finalize_recommendations
    from functional_agents.evidence_boundary import finalize_evidence
    from functional_agents.report_boundary import finalize_report_input

    _, d_plan = plan_from_raw({"research_type": "RESEARCH", "subquestions": ["q"],
                               "investigation_areas": ["A"]})
    _, d_hyp = finalize_hypotheses({"hypotheses": [{"id": "H1", "title": "t", "summary": "s"}]})
    _, d_rec = finalize_recommendations({"recommendations": [{"id": "R1", "title": "t", "summary": "s"}]})
    _, d_ev = finalize_evidence({"evidence_items": [{"evidence_id": "E1"}]})
    _, d_rep = finalize_report_input({"question": "q", "memo": object()})

    expected_keys = {"stages", "failed_stage", "normalization", "validation"}
    expected_stages = {"generation", "normalization", "validation"}
    for d in (d_plan, d_hyp, d_rec, d_ev, d_rep):
        assert set(d) == expected_keys
        assert set(d["stages"]) == expected_stages
        assert d["failed_stage"] is None
