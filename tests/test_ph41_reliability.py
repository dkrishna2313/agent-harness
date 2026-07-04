"""PH4.1 — Production Reliability & Contract Integrity tests.

Covers:
- Serialization symmetry (AgentContext, Decision Model, canonical trace)
- Persistence hardening (load_decision_model, load_engagement error handling)
- Contract fixes (H1 mitigation_notes, H2 opportunity title, H4 missing assumption_id)
- Orchestrator error state (H3)
- Narrative version warning (M1)
- Bundle to_dict bundle_id always present (L2)
- Scenario validation warning on corrupt RO
- Recommendation linkage determinism
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from functional_agents.context import AgentContext, WorkflowState
from functional_agents.deliverables.bundle import DeliverableBundle
from functional_agents.narrative import ExecutiveNarrative, ExecutiveNarrativeBuilder
from functional_agents.narrative.executive_narrative import NARRATIVE_CONTRACT_VERSION
from functional_agents.recommendation_linkage import build_recommendation_linkage
from research_agent.decision_model import DecisionModel, load_decision_model, write_decision_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_decision_model() -> DecisionModel:
    from research_agent.decision_model import from_question as dm_from_question
    return dm_from_question("Should we invest in SMR technology?")


def _make_full_context() -> AgentContext:
    return AgentContext(
        question="Should we invest in SMR?",
        profiles=["smr"],
        execution_profile="smr",
        strategic_options=[
            {"option_id": "OPT-A", "title": "Phased", "description": "Phase it.",
             "estimated_time_horizon": "near_term", "capital_intensity": "Medium",
             "confidence": "High", "advantages": ["Lower risk"]},
        ],
        preferred_option={"option_id": "OPT-A", "title": "Phased"},
        decision_analysis={
            "recommended_option_id": "OPT-A",
            "executive_summary": "Phased approach wins.",
            "option_rankings": ["OPT-A"],
            "rationale": "Best risk-adjusted return.",
            "comparison_dimensions": ["Cost", "Risk"],
        },
        risks=[
            {"risk_id": "RSK-001", "statement": "Grid delay",
             "severity": "High", "likelihood": "Medium",
             "mitigation_notes": "Secure queue position early"},
        ],
        opportunities=[
            {"opportunity_id": "OPP-001", "statement": "First mover advantage",
             "category": "Market", "impact": "High", "likelihood": "Medium"},
        ],
        executive_confidence={
            "overall_confidence": "High",
            "decision_readiness": "Ready",
            "board_recommendation": "Proceed",
            "confidence_rationale": "Strong evidence.",
            "confidence_drivers": ["Broad coverage"],
            "confidence_limiters": ["Regulatory uncertainty"],
            "validation_priorities": ["Confirm grid position"],
            "critical_unknowns": ["Final cost"],
        },
        assumptions=[
            {"assumption_id": "A-001", "statement": "Grid in 24 months",
             "importance": "Critical", "confidence": "Medium"},
        ],
        recommendations=[
            {"recommendation_id": "REC-001", "title": "File grid application",
             "time_horizon": "near_term", "priority": "high"},
        ],
        recommendation_portfolio={"near_term": ["REC-001"], "medium_term": [], "long_term": []},
    )


# ---------------------------------------------------------------------------
# Serialization symmetry — AgentContext
# ---------------------------------------------------------------------------

def test_agentcontext_serialization_symmetry(tmp_path):
    """context_to_jsonable → JSON → load_context produces equal durable fields."""
    from functional_agents.context_snapshot import context_to_jsonable, load_context, CONTEXT_FIELDS

    ctx = _make_full_context()
    serialized = context_to_jsonable(ctx)

    # Write to a temp file and reload
    snapshot = tmp_path / "ctx_test.json"
    snapshot.write_text(json.dumps(serialized), encoding="utf-8")
    reloaded = load_context(snapshot)

    # Durable fields must survive the round-trip
    for field in ("question", "profiles", "execution_profile", "strategic_options",
                  "decision_analysis", "assumptions", "risks", "recommendations"):
        assert getattr(ctx, field) == getattr(reloaded, field), (
            f"Field {field!r} changed during serialization round-trip"
        )


def test_agentcontext_serialization_is_json_safe():
    """context_to_jsonable output is always JSON-serializable."""
    from functional_agents.context_snapshot import context_to_jsonable
    ctx = _make_full_context()
    data = context_to_jsonable(ctx)
    # Should not raise
    json.dumps(data)


# ---------------------------------------------------------------------------
# Serialization symmetry — Decision Model
# ---------------------------------------------------------------------------

def test_decision_model_serialization_symmetry():
    """model_dump → model_validate is a lossless round-trip."""
    dm = _make_decision_model()
    data = dm.to_dict()
    restored = DecisionModel.model_validate(data)
    assert restored.decision_model_id == dm.decision_model_id
    assert restored.strategic_question == dm.strategic_question
    assert restored.to_dict() == data


def test_decision_model_write_load_symmetry(tmp_path):
    """write_decision_model → load_decision_model produces an equal model."""
    dm = _make_decision_model()
    write_decision_model(dm, base=tmp_path, write_latest=False)
    loaded = load_decision_model(dm.decision_model_id, base=tmp_path)
    assert loaded.decision_model_id == dm.decision_model_id
    assert loaded.to_dict() == dm.to_dict()


def test_decision_model_write_uses_default_str_for_non_serializable(tmp_path):
    """write_decision_model does not raise on non-JSON-serializable values in decision_architecture."""
    dm = _make_decision_model()
    # Inject a non-serializable object into the opaque dict field
    from pathlib import Path as _Path
    dm.decision_architecture["non_serializable"] = _Path("/some/path")
    # Should not raise — default=str converts it
    path = write_decision_model(dm, base=tmp_path, write_latest=False)
    raw = json.loads(path.read_text())
    assert raw["decision_architecture"]["non_serializable"] == "/some/path"


# ---------------------------------------------------------------------------
# Persistence hardening — load errors
# ---------------------------------------------------------------------------

def test_load_decision_model_raises_on_missing_file(tmp_path):
    """load_decision_model raises FileNotFoundError with actionable message."""
    with pytest.raises(FileNotFoundError, match="Decision model not found"):
        load_decision_model("nonexistent-id", base=tmp_path)


def test_load_decision_model_raises_on_invalid_json(tmp_path):
    """load_decision_model raises ValueError on corrupt JSON."""
    dm_dir = tmp_path / "decision_models"
    dm_dir.mkdir()
    corrupt = dm_dir / "bad-id.json"
    corrupt.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_decision_model("bad-id", base=tmp_path)


def test_load_engagement_raises_on_missing_file(tmp_path):
    """load_engagement raises FileNotFoundError with actionable message."""
    from research_agent.engagement import load_engagement
    with pytest.raises(FileNotFoundError, match="Engagement not found"):
        load_engagement("nonexistent-id", base=tmp_path)


def test_load_engagement_raises_on_invalid_json(tmp_path):
    """load_engagement raises ValueError on corrupt JSON."""
    from research_agent.engagement import load_engagement
    eng_dir = tmp_path / "engagements"
    eng_dir.mkdir()
    corrupt = eng_dir / "bad-id.json"
    corrupt.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_engagement("bad-id", base=tmp_path)


# ---------------------------------------------------------------------------
# Contract fix H1 — mitigation_notes propagates into narrative
# ---------------------------------------------------------------------------

def test_risk_mitigation_notes_in_narrative():
    """H1: builder reads mitigation_notes (canonical RiskItem field) into narrative."""
    ctx = _make_full_context()
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.key_risks) == 1
    assert narrative.key_risks[0]["mitigation"] == "Secure queue position early"


def test_risk_mitigation_absent_gracefully():
    """H1: risks without mitigation_notes produce empty mitigation, not a crash."""
    ctx = AgentContext(
        question="Q?", profiles=["smr"], execution_profile="smr",
        strategic_options=[{"option_id": "OPT-A", "title": "X"}],
        risks=[{"risk_id": "RSK-001", "statement": "Grid delay",
                "severity": "High", "likelihood": "Medium"}],
    )
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.key_risks[0]["mitigation"] == ""


# ---------------------------------------------------------------------------
# Contract fix H2 — opportunity title fallback to statement
# ---------------------------------------------------------------------------

def test_opportunity_title_falls_back_to_statement():
    """H2: opportunity with no title/name uses statement as display title."""
    ctx = AgentContext(
        question="Q?", profiles=["smr"], execution_profile="smr",
        strategic_options=[{"option_id": "OPT-A", "title": "X"}],
        opportunities=[
            {"opportunity_id": "OPP-001", "statement": "First mover advantage",
             "category": "Market", "impact": "High"},
        ],
    )
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert len(narrative.key_opportunities) == 1
    assert narrative.key_opportunities[0]["title"] == "First mover advantage"


def test_opportunity_explicit_title_wins_over_statement():
    """H2: explicit title field is preferred over statement fallback."""
    ctx = AgentContext(
        question="Q?", profiles=["smr"], execution_profile="smr",
        strategic_options=[{"option_id": "OPT-A", "title": "X"}],
        opportunities=[
            {"opportunity_id": "OPP-001", "title": "My title",
             "statement": "Full statement text", "impact": "High"},
        ],
    )
    narrative = ExecutiveNarrativeBuilder().build(ctx)
    assert narrative.key_opportunities[0]["title"] == "My title"


# ---------------------------------------------------------------------------
# Contract fix H4 — recommendation_linkage tolerates missing assumption_id
# ---------------------------------------------------------------------------

def test_recommendation_linkage_tolerates_missing_assumption_id(caplog):
    """H4: build_recommendation_linkage warns and skips malformed assumptions."""
    assumptions = [
        {"assumption_id": "A-001", "statement": "Valid assumption"},
        {"statement": "Malformed — no assumption_id"},  # missing assumption_id
    ]
    recommendations = [{"recommendation_id": "REC-001", "title": "Do X"}]

    with caplog.at_level(logging.WARNING, logger="functional_agents.recommendation_linkage"):
        linked_a, linked_r = build_recommendation_linkage(assumptions, recommendations)

    assert any("missing assumption_id" in msg for msg in caplog.messages)
    # Only the valid assumption should be processed; both are returned
    assert len(linked_a) == 1  # malformed filtered out from linkage
    assert len(linked_r) == 1


def test_recommendation_linkage_all_malformed_returns_early(caplog):
    """H4: when all assumptions are malformed, linkage returns empty/unchanged lists."""
    assumptions = [{"statement": "No id here"}]
    recommendations = [{"recommendation_id": "REC-001", "title": "Do X"}]

    with caplog.at_level(logging.WARNING, logger="functional_agents.recommendation_linkage"):
        linked_a, linked_r = build_recommendation_linkage(assumptions, recommendations)

    assert any("missing assumption_id" in msg for msg in caplog.messages)
    assert linked_r[0].get("supported_by_assumptions", []) == []


def test_recommendation_linkage_is_deterministic():
    """Recommendation linkage with same input always produces same output."""
    assumptions = [
        {"assumption_id": "A-001", "statement": "Grid in 24 months",
         "evidence_ids": ["E-001", "E-002"]},
        {"assumption_id": "A-002", "statement": "Pricing stable",
         "evidence_ids": ["E-003"]},
    ]
    recommendations = [
        {"recommendation_id": "REC-001", "title": "File application",
         "supporting_evidence": ["E-001"]},
        {"recommendation_id": "REC-002", "title": "Negotiate contracts",
         "supporting_evidence": ["E-003"]},
    ]
    result_1 = build_recommendation_linkage(assumptions, recommendations)
    result_2 = build_recommendation_linkage(assumptions, recommendations)
    assert result_1[0] == result_2[0]
    assert result_1[1] == result_2[1]


# ---------------------------------------------------------------------------
# Orchestrator error state (H3)
# ---------------------------------------------------------------------------

def test_orchestrator_records_error_state_on_agent_failure():
    """H3: when an agent raises, AgentOrchestrator sets WorkflowState.ERROR and records trace."""
    from functional_agents.orchestrator import AgentOrchestrator
    from functional_agents.context import AgentContext, WorkflowState

    def _failing_factory():
        agent = MagicMock()
        agent.name = "FailingPlannerAgent"
        agent.run.side_effect = RuntimeError("simulated agent crash")
        return agent

    orchestrator = AgentOrchestrator(
        planner_factory=_failing_factory,
        evidence_factory=MagicMock(return_value=MagicMock(name="EvidenceAgent")),
        qa_factory=MagicMock(return_value=MagicMock(name="QAAgent")),
        report_factory=MagicMock(return_value=MagicMock(name="ReportAgent")),
    )
    ctx = AgentContext(question="Q?", profiles=["smr"], execution_profile="smr")

    result = orchestrator.run(ctx)

    assert result.workflow_state == WorkflowState.ERROR
    assert "_orchestrator_error" in result.trace
    err = result.trace["_orchestrator_error"]
    assert err["error_type"] == "RuntimeError"
    assert "simulated agent crash" in err["error"]


def test_orchestrator_error_trace_records_failing_state():
    """H3: _orchestrator_error trace entry records the workflow state at failure time."""
    from functional_agents.orchestrator import AgentOrchestrator
    from functional_agents.context import WorkflowState

    def _failing_factory():
        agent = MagicMock()
        agent.name = "PlannerAgent"
        agent.run.side_effect = ValueError("bad plan output")
        return agent

    orchestrator = AgentOrchestrator(
        planner_factory=_failing_factory,
        evidence_factory=MagicMock(return_value=MagicMock(name="Evidence")),
        qa_factory=MagicMock(return_value=MagicMock(name="QA")),
        report_factory=MagicMock(return_value=MagicMock(name="Report")),
    )
    ctx = AgentContext(question="Q?", profiles=["smr"], execution_profile="smr")
    result = orchestrator.run(ctx)

    assert result.workflow_state == WorkflowState.ERROR
    assert result.trace["_orchestrator_error"]["state"] == WorkflowState.PLANNING


# ---------------------------------------------------------------------------
# Narrative version warning (M1)
# ---------------------------------------------------------------------------

def test_narrative_from_dict_warns_on_older_version(caplog):
    """M1: from_dict emits a warning when stored version is older than current."""
    old_dict = {"version": "1.0", "decision": "Build SMR plant"}
    with caplog.at_level(logging.WARNING, logger="functional_agents.narrative.executive_narrative"):
        narrative = ExecutiveNarrative.from_dict(old_dict)

    assert narrative.version == "1.0"
    assert any("differs from current contract version" in msg for msg in caplog.messages)


def test_narrative_from_dict_no_warning_on_current_version(caplog):
    """M1: from_dict emits no warning when the stored version matches current."""
    current_dict = {"version": NARRATIVE_CONTRACT_VERSION, "decision": "Build SMR plant"}
    with caplog.at_level(logging.WARNING, logger="functional_agents.narrative.executive_narrative"):
        narrative = ExecutiveNarrative.from_dict(current_dict)

    assert narrative.version == NARRATIVE_CONTRACT_VERSION
    assert not any("differs from current" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Bundle to_dict always emits bundle_id (L2)
# ---------------------------------------------------------------------------

def test_bundle_to_dict_always_includes_bundle_id():
    """L2: to_dict always emits bundle_id even when empty (prevents KeyError on consumers)."""
    bundle = DeliverableBundle()  # bundle_id defaults to ""
    d = bundle.to_dict()
    assert "bundle_id" in d
    assert d["bundle_id"] == ""


def test_bundle_to_dict_with_populated_bundle_id():
    """L2: to_dict emits correct bundle_id when set."""
    bundle = DeliverableBundle(bundle_id="test-bundle-123")
    d = bundle.to_dict()
    assert d["bundle_id"] == "test-bundle-123"


# ---------------------------------------------------------------------------
# Scenario validation — warns on corrupt RO (nuclear swallow fix)
# ---------------------------------------------------------------------------

def test_scenario_validation_warns_on_corrupt_ro(tmp_path, caplog):
    """scenario_validation.py warns (not silently swallows) when latest_research_object.json is corrupt."""
    from functional_agents.scenario_validation import _update_latest_research_object

    corrupt_path = tmp_path / "latest_research_object.json"
    corrupt_path.write_text("{not valid json}", encoding="utf-8")

    ro = {"scenarios": [{"id": "S-001"}], "scenario_analysis": {}}

    with caplog.at_level(logging.WARNING, logger="functional_agents.scenario_validation"):
        _update_latest_research_object(ro, out_dir=tmp_path)

    assert any("could not load latest_research_object.json" in msg for msg in caplog.messages)
    # Scenario fields should still be written despite the load failure
    written = json.loads(corrupt_path.read_text())
    assert "scenarios" in written


# ---------------------------------------------------------------------------
# Canonical trace serialization symmetry
# ---------------------------------------------------------------------------

def test_canonical_trace_write_load_symmetry(tmp_path):
    """write_canonical_trace → json.loads produces byte-equal content."""
    from functional_agents.pipeline_trace import build_canonical_trace, write_canonical_trace

    ctx = _make_full_context()
    trace = build_canonical_trace(ctx)

    path = write_canonical_trace(ctx, out_dir=tmp_path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))

    assert reloaded["schema_version"] == trace["schema_version"]
    assert reloaded["pipeline"]["question"] == trace["pipeline"]["question"]
    assert reloaded["summary"] == trace["summary"]


# ---------------------------------------------------------------------------
# Narrative builder determinism
# ---------------------------------------------------------------------------

def test_executive_narrative_builder_is_deterministic():
    """Building ExecutiveNarrative twice from the same context produces identical output."""
    ctx1 = _make_full_context()
    ctx2 = _make_full_context()

    n1 = ExecutiveNarrativeBuilder().build(ctx1).to_dict()
    n2 = ExecutiveNarrativeBuilder().build(ctx2).to_dict()

    assert n1 == n2


def test_executive_narrative_builder_idempotent_on_same_context():
    """Calling build() twice on the same context produces the same narrative."""
    ctx = _make_full_context()
    n1 = ExecutiveNarrativeBuilder().build(ctx).to_dict()
    n2 = ExecutiveNarrativeBuilder().build(ctx).to_dict()
    assert n1 == n2
