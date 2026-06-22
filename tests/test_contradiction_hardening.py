"""Tests for J6.5a Contradiction Hardening.

Covers:
- New scope pairs (rack↔facility, rack↔cluster, cluster↔facility)
- Temporal progression gate (current vs target/future)
- compute_suppression_metrics helper
- EvidenceAgent propagation to context
- QA contradiction_validation block
- ReportAgent contradiction_hardening trace block
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

sys.modules.setdefault("yaml", MagicMock())

from research_agent.contradiction import (
    _scopes_compatible,
    _check_numeric_conflict,
    compute_suppression_metrics,
    detect_contradictions,
    _extract_numeric_kind_from_text,
)
from research_agent.schemas import EvidenceItem, SuppressedComparison


# ---------------------------------------------------------------------------
# Scope pair fixtures
# ---------------------------------------------------------------------------

def _evidence(claim: str, scope: str = "", entity: str = "") -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"E_{abs(hash(claim)) % 10000:04d}",
        claim=claim,
        category="power",
        source_document="test_doc.txt",
        evidence_snippet=claim[:60],
        relevance="direct",
        confidence="medium",
        scope=scope,
        entity=entity,
    )


# ---------------------------------------------------------------------------
# New scope pairs — _scopes_compatible returns False
# ---------------------------------------------------------------------------

def test_rack_facility_incompatible():
    assert _scopes_compatible("rack", "facility") is False


def test_rack_cluster_incompatible():
    assert _scopes_compatible("rack", "cluster") is False


def test_cluster_facility_incompatible():
    assert _scopes_compatible("cluster", "facility") is False


def test_component_rack_still_incompatible():
    assert _scopes_compatible("component", "rack") is False


def test_rack_rack_compatible():
    assert _scopes_compatible("rack", "rack") is True


def test_unknown_scope_passes():
    assert _scopes_compatible("rack", "unknown") is True
    assert _scopes_compatible("unknown", "facility") is True


def test_cluster_cluster_compatible():
    assert _scopes_compatible("cluster", "cluster") is True


# ---------------------------------------------------------------------------
# Rack vs facility scope suppression in numeric check
# ---------------------------------------------------------------------------

def test_rack_mw_vs_facility_mw_suppressed():
    """10 MW rack vs 100 MW facility: same unit, different scopes → scope_mismatch suppressed."""
    a = _evidence("Each rack system uses 10 mw of power", scope="rack")
    b = _evidence("The campus facility requires 100 mw total", scope="facility")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    assert result is None
    scope_sups = [s for s in suppressed if s.reason == "scope_mismatch"]
    assert len(scope_sups) > 0


def test_rack_kw_vs_facility_mw_no_false_positive():
    """132 kW rack vs 100 MW campus: different unit keywords → no comparison attempted."""
    a = _evidence("Each rack requires 132 kW of power", scope="rack")
    b = _evidence("The campus facility uses 100 MW total", scope="facility")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    # Units differ so no comparison even attempted — no false positive
    assert result is None


def test_rack_mw_vs_cluster_mw_suppressed():
    a = _evidence("Each rack operates at 5 mw", scope="rack")
    b = _evidence("The cluster uses 50 mw total", scope="cluster")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    assert result is None
    assert any(s.reason == "scope_mismatch" for s in suppressed)


def test_same_scope_rack_numeric_not_suppressed():
    """Two rack-level conflicting power claims should NOT be suppressed."""
    a = _evidence("Each rack requires 100 kW", scope="rack")
    b = _evidence("Each rack requires 200 kW", scope="rack")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    # Not suppressed — should be a real contradiction
    assert result is not None


# ---------------------------------------------------------------------------
# Temporal kind extractor
# ---------------------------------------------------------------------------

def test_current_kind():
    assert _extract_numeric_kind_from_text("current rack power is 100 kW") == "current"


def test_existing_kind():
    assert _extract_numeric_kind_from_text("existing installations use 120 kW") == "current"


def test_target_kind():
    assert _extract_numeric_kind_from_text("target capacity of 200 kW by 2030") == "target"


def test_future_projection_kind():
    assert _extract_numeric_kind_from_text("projected to reach 150 kW in future deployments") == "target"


def test_rate_kind():
    assert _extract_numeric_kind_from_text("build rate of 20 units per year") == "rate"


def test_unknown_kind():
    assert _extract_numeric_kind_from_text("the system uses 100 kW") == "unknown"


# ---------------------------------------------------------------------------
# Temporal progression gate in numeric check
# ---------------------------------------------------------------------------

def test_current_vs_target_suppressed():
    """Current state vs future target should be suppressed as temporal_progression."""
    a = _evidence("Current rack power is 100 kW", scope="rack")
    b = _evidence("Target rack power will be 200 kW by 2030", scope="rack")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    assert result is None
    temporal_sups = [s for s in suppressed if s.reason == "temporal_progression"]
    assert len(temporal_sups) > 0


def test_current_vs_rate_suppressed():
    """Current state vs build rate is temporal — should be suppressed."""
    a = _evidence("Existing data centres use 100 kW per rack", scope="rack")
    b = _evidence("Build rate targeting 200 kW per rack per year", scope="rack")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    assert result is None
    assert any(s.reason == "temporal_progression" for s in suppressed)


def test_target_vs_target_not_temporal():
    """Two different future targets for the same thing ARE a contradiction."""
    a = _evidence("Target rack power goal 100 kW by 2035", scope="rack")
    b = _evidence("Target rack power goal 200 kW by 2035", scope="rack")
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    # Both are 'target', so temporal gate should not suppress
    temporal_sups = [s for s in suppressed if s.reason == "temporal_progression"]
    assert len(temporal_sups) == 0


def test_unknown_kind_vs_current_not_temporal():
    """If one kind is unknown, the temporal gate should not suppress."""
    a = _evidence("The system uses 100 kW", scope="rack")  # unknown kind
    b = _evidence("Current system uses 200 kW", scope="rack")  # current kind
    suppressed: list[SuppressedComparison] = []
    result = _check_numeric_conflict(a, b, suppressed)
    temporal_sups = [s for s in suppressed if s.reason == "temporal_progression"]
    assert len(temporal_sups) == 0


# ---------------------------------------------------------------------------
# compute_suppression_metrics
# ---------------------------------------------------------------------------

def _make_suppressed(reason: str) -> SuppressedComparison:
    return SuppressedComparison(
        evidence_a_id="E001",
        evidence_b_id="E002",
        evidence_a_claim="claim a",
        evidence_b_claim="claim b",
        reason=reason,
        scope_a="rack",
        scope_b="facility",
        detail="test detail",
    )


def test_suppression_metrics_empty():
    metrics = compute_suppression_metrics([], final_count=3)
    assert metrics["candidate_count"] == 3
    assert metrics["suppressed_count"] == 0
    assert metrics["final_count"] == 3
    assert metrics["by_reason"] == {}
    assert metrics["scope_filtering_present"] is False
    assert metrics["entity_filtering_present"] is False
    assert metrics["temporal_filtering_present"] is False


def test_suppression_metrics_by_reason():
    suppressed = [
        _make_suppressed("scope_mismatch"),
        _make_suppressed("scope_mismatch"),
        _make_suppressed("entity_mismatch"),
        _make_suppressed("temporal_progression"),
    ]
    metrics = compute_suppression_metrics(suppressed, final_count=2)
    assert metrics["suppressed_count"] == 4
    assert metrics["final_count"] == 2
    assert metrics["candidate_count"] == 6
    assert metrics["by_reason"]["scope_mismatch"] == 2
    assert metrics["by_reason"]["entity_mismatch"] == 1
    assert metrics["by_reason"]["temporal_progression"] == 1
    assert metrics["scope_filtering_present"] is True
    assert metrics["entity_filtering_present"] is True
    assert metrics["temporal_filtering_present"] is True


def test_suppression_metrics_metric_scope_mismatch_counts():
    suppressed = [_make_suppressed("metric_scope_mismatch")]
    metrics = compute_suppression_metrics(suppressed, final_count=1)
    assert metrics["scope_filtering_present"] is True


# ---------------------------------------------------------------------------
# detect_contradictions integration — new scope pairs reduce count
# ---------------------------------------------------------------------------

def test_rack_facility_pair_not_contradicted():
    """Evidence pairs with rack vs facility scope should not appear as contradictions."""
    items = [
        _evidence("Each rack consumes 132 kW of power", scope="rack"),
        _evidence("The 100 MW campus facility serves all racks", scope="facility"),
    ]
    suppressed: list = []
    contradictions = detect_contradictions(items, out_suppressed=suppressed)
    # Should have NO contradictions — these are different scopes
    contra_claims = [
        (c.evidence_a_claim, c.evidence_b_claim) for c in contradictions
    ]
    rack_vs_facility = [
        p for p in contra_claims
        if ("132 kW" in p[0] or "132 kW" in p[1])
        and ("100 MW" in p[0] or "100 MW" in p[1])
    ]
    assert len(rack_vs_facility) == 0


# ---------------------------------------------------------------------------
# QA validation helper
# ---------------------------------------------------------------------------

def test_validate_contradiction_hardening_no_metrics():
    from functional_agents.qa_agent import _validate_contradiction_hardening
    from functional_agents.context import AgentContext
    ctx = AgentContext(
        question="test", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_CHH_001"}, run_id="chh001",
    )
    result = _validate_contradiction_hardening(ctx)
    assert "scope_filtering_present" in result
    assert "entity_filtering_present" in result
    assert "temporal_filtering_present" in result
    assert "suppressed_count" in result
    assert "final_count" in result
    assert "issues" in result
    assert len(result["issues"]) > 0  # no metrics → issue flagged


def test_validate_contradiction_hardening_with_metrics():
    from functional_agents.qa_agent import _validate_contradiction_hardening
    from functional_agents.context import AgentContext
    ctx = AgentContext(
        question="test", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_CHH_002"}, run_id="chh002",
    )
    ctx.contradiction_metrics = {
        "candidate_count": 10,
        "suppressed_count": 4,
        "final_count": 6,
        "by_reason": {
            "scope_mismatch": 2,
            "entity_mismatch": 1,
            "temporal_progression": 1,
        },
        "scope_filtering_present": True,
        "entity_filtering_present": True,
        "temporal_filtering_present": True,
    }
    result = _validate_contradiction_hardening(ctx)
    assert result["scope_filtering_present"] is True
    assert result["entity_filtering_present"] is True
    assert result["temporal_filtering_present"] is True
    assert result["suppressed_count"] == 4
    assert result["final_count"] == 6
    assert result["issues"] == []


def test_qa_agent_writes_contradiction_validation():
    """QAAgent._execute() must write contradiction_validation to context.qa."""
    from functional_agents.qa_agent import QAAgent
    from functional_agents.context import AgentContext
    ctx = AgentContext(
        question="test", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_CHH_003"}, run_id="chh003",
    )
    ctx.contradiction_metrics = {
        "candidate_count": 5,
        "suppressed_count": 2,
        "final_count": 3,
        "by_reason": {"scope_mismatch": 2},
        "scope_filtering_present": True,
        "entity_filtering_present": False,
        "temporal_filtering_present": False,
    }
    QAAgent().run(ctx)
    assert "contradiction_validation" in ctx.qa
    cv = ctx.qa["contradiction_validation"]
    assert cv["scope_filtering_present"] is True
    assert cv["suppressed_count"] == 2


# ---------------------------------------------------------------------------
# ReportAgent contradiction_hardening trace block
# ---------------------------------------------------------------------------

def _make_report_ctx(run_id: str, research_id: str) -> "AgentContext":
    """Build a minimal AgentContext with a stub memo for ReportAgent tests."""
    from functional_agents.context import AgentContext
    from research_agent.schemas import ResearchMemo
    ctx = AgentContext(
        question="test contradiction hardening", profiles=["p"], execution_profile="p",
        research_object={"research_id": research_id}, run_id=run_id,
    )
    ctx.trace["_memo"] = ResearchMemo(
        question="test", answer="stub", title="Stub", executive_summary="stub"
    )
    return ctx


def test_report_agent_writes_contradiction_hardening_trace():
    from functional_agents.report_agent import ReportAgent
    from pathlib import Path
    import tempfile, json
    ctx = _make_report_ctx("chh004", "R-TEST_CHH_004")
    ctx.contradiction_metrics = {
        "candidate_count": 8,
        "suppressed_count": 3,
        "final_count": 5,
        "by_reason": {"scope_mismatch": 2, "temporal_progression": 1},
        "scope_filtering_present": True,
        "entity_filtering_present": False,
        "temporal_filtering_present": True,
    }
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "report.md"
        ReportAgent(out_path=out_path).run(ctx)
        trace_path = out_path.with_suffix(".trace.json")
        trace = json.loads(trace_path.read_text())
    assert "contradiction_hardening" in trace
    ch = trace["contradiction_hardening"]
    assert ch["suppressed_count"] == 3
    assert ch["final_count"] == 5
    assert ch["scope_filtering_present"] is True
    assert ch["temporal_filtering_present"] is True


def test_report_agent_skips_contradiction_hardening_when_no_metrics():
    from functional_agents.report_agent import ReportAgent
    from pathlib import Path
    import tempfile, json
    ctx = _make_report_ctx("chh005", "R-TEST_CHH_005")
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "report.md"
        ReportAgent(out_path=out_path).run(ctx)
        trace_path = out_path.with_suffix(".trace.json")
        trace = json.loads(trace_path.read_text())
    # No metrics → block should not be present
    assert "contradiction_hardening" not in trace


# ---------------------------------------------------------------------------
# EvidenceAgent propagation (unit test with mock memo)
# ---------------------------------------------------------------------------

def test_evidence_agent_propagates_contradiction_metrics():
    """EvidenceAgent must copy contradiction_metrics from memo.metadata to context."""
    from functional_agents.evidence_agent import EvidenceAgent
    from functional_agents.context import AgentContext
    from unittest.mock import patch, MagicMock

    ctx = AgentContext(
        question="test propagation", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_CHH_006"}, run_id="chh006",
    )

    # Build a fake memo with contradiction metrics in metadata
    fake_memo = MagicMock()
    fake_memo.source_notes = []
    fake_memo.evidence = []
    fake_memo.confirmed_facts = []
    fake_memo.evaluation_warnings = []
    fake_memo.metadata = {
        "contradictions": [],
        "suppressed_comparisons": [],
        "contradiction_metrics": {
            "candidate_count": 3,
            "suppressed_count": 1,
            "final_count": 2,
            "by_reason": {"scope_mismatch": 1},
            "scope_filtering_present": True,
            "entity_filtering_present": False,
            "temporal_filtering_present": False,
        },
    }

    fake_agent = MagicMock()
    fake_agent.analyze.return_value = fake_memo

    fake_collection = MagicMock()
    fake_collection.errors = []
    fake_collection.documents = []

    with patch("research_agent.loaders.load_sources", return_value=fake_collection), \
         patch("research_agent.agent.DcPowerAgent", return_value=fake_agent):
        EvidenceAgent().run(ctx)

    assert ctx.contradiction_metrics["suppressed_count"] == 1
    assert ctx.contradiction_metrics["final_count"] == 2
    assert ctx.research_object.get("contradiction_metrics", {})["scope_filtering_present"] is True


def test_evidence_agent_propagates_validated_contradictions():
    """EvidenceAgent must populate context.validated_contradictions from memo."""
    from functional_agents.evidence_agent import EvidenceAgent
    from functional_agents.context import AgentContext
    from unittest.mock import patch, MagicMock

    ctx = AgentContext(
        question="test validated", profiles=["p"], execution_profile="p",
        research_object={"research_id": "R-TEST_CHH_007"}, run_id="chh007",
    )

    fake_contradiction = {
        "contradiction_id": "C001",
        "topic": "rack power",
        "severity": "high",
        "evidence_a_claim": "100 kW",
        "evidence_b_claim": "200 kW",
    }
    fake_memo = MagicMock()
    fake_memo.source_notes = []
    fake_memo.evidence = []
    fake_memo.confirmed_facts = []
    fake_memo.evaluation_warnings = []
    fake_memo.metadata = {
        "contradictions": [fake_contradiction],
        "suppressed_comparisons": [],
        "contradiction_metrics": {
            "candidate_count": 1,
            "suppressed_count": 0,
            "final_count": 1,
            "by_reason": {},
            "scope_filtering_present": False,
            "entity_filtering_present": False,
            "temporal_filtering_present": False,
        },
    }

    fake_agent = MagicMock()
    fake_agent.analyze.return_value = fake_memo

    fake_collection = MagicMock()
    fake_collection.errors = []
    fake_collection.documents = []

    with patch("research_agent.loaders.load_sources", return_value=fake_collection), \
         patch("research_agent.agent.DcPowerAgent", return_value=fake_agent):
        EvidenceAgent().run(ctx)

    assert len(ctx.validated_contradictions) == 1
    assert ctx.validated_contradictions[0]["contradiction_id"] == "C001"
    assert ctx.research_object.get("validated_contradictions", [{}])[0]["contradiction_id"] == "C001"
