"""Tests for J13.2 — PathKind, StalenessPlan, and DependencyReasoner.

Coverage:
- PathKind classification
- Container path expansion
- StalenessPlan creation and serialization
- DependencyReasoner: single path change
- DependencyReasoner: container path (research_state)
- DependencyReasoner: external dependency (knowledge_store / EXTERNAL_EVIDENCE_ADDED)
- DependencyReasoner: unknown path
- DependencyReasoner: empty state_changes
- DependencyReasoner: confidence levels (HIGH / MEDIUM / LOW)
- stale_paths vs external_dependencies separation
- stale_agents determination
- required_producers determination
- execution_only_paths classification
- persisted_paths classification
- reasoning trace
- determinism
- serialization roundtrip
- CLI smoke tests
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_path_kind_importable(self):
        from functional_agents.staleness import PathKind  # noqa: F401

    def test_staleness_plan_importable(self):
        from functional_agents.staleness import StalenessPlan  # noqa: F401

    def test_dependency_reasoner_importable(self):
        from functional_agents.staleness import DependencyReasoner  # noqa: F401

    def test_classify_path_importable(self):
        from functional_agents.staleness import classify_path  # noqa: F401

    def test_expand_path_importable(self):
        from functional_agents.staleness import expand_path  # noqa: F401

    def test_all_exports_in_all(self):
        import functional_agents.staleness as pkg
        for name in ["PathKind", "StalenessPlan", "DependencyReasoner",
                     "classify_path", "expand_path", "is_container_path",
                     "PATH_CLASSIFICATION"]:
            assert name in pkg.__all__, f"{name!r} not in __all__"


# ---------------------------------------------------------------------------
# PathKind classification
# ---------------------------------------------------------------------------

class TestPathKindConstants:
    def test_persisted_constant(self):
        from functional_agents.staleness import PathKind
        assert PathKind.PERSISTED == "PERSISTED"

    def test_execution_only_constant(self):
        from functional_agents.staleness import PathKind
        assert PathKind.EXECUTION_ONLY == "EXECUTION_ONLY"

    def test_external_constant(self):
        from functional_agents.staleness import PathKind
        assert PathKind.EXTERNAL == "EXTERNAL"


class TestClassifyPath:
    def test_persisted_root_fields(self):
        from functional_agents.staleness import classify_path, PathKind
        for path in ["engagement", "research_object", "decision_model",
                     "research_gap_analysis", "executive_confidence", "iteration_plan"]:
            assert classify_path(path) == PathKind.PERSISTED, f"{path!r} should be PERSISTED"

    def test_persisted_sub_fields(self):
        from functional_agents.staleness import classify_path, PathKind
        for path in [
            "research_object.evidence",
            "research_object.hypotheses",
            "decision_model.assumptions",
            "decision_model.recommendations",
            "decision_model.risks",
            "decision_model.opportunities",
            "decision_model.strategic_options",
            "decision_model.decision_analysis",
        ]:
            assert classify_path(path) == PathKind.PERSISTED, f"{path!r} should be PERSISTED"

    def test_execution_only_paths(self):
        from functional_agents.staleness import classify_path, PathKind
        for path in [
            "decision_architecture", "research_strategy", "planner",
            "strategic_synthesis", "challenge_results", "multi_profile_analysis",
            "scenario_analysis", "qa", "recommendation_improvement",
            "recommendation_synthesis", "report",
        ]:
            assert classify_path(path) == PathKind.EXECUTION_ONLY, (
                f"{path!r} should be EXECUTION_ONLY"
            )

    def test_external_paths(self):
        from functional_agents.staleness import classify_path, PathKind
        assert classify_path("knowledge_store") == PathKind.EXTERNAL

    def test_unknown_path_defaults_to_execution_only(self):
        from functional_agents.staleness import classify_path, PathKind
        assert classify_path("some.unknown.path") == PathKind.EXECUTION_ONLY


class TestContainerExpansion:
    def test_research_state_expands(self):
        from functional_agents.staleness import expand_path, is_container_path
        assert is_container_path("research_state")
        expanded = expand_path("research_state")
        assert "engagement" in expanded
        assert "research_object.evidence" in expanded
        assert "decision_model.assumptions" in expanded
        assert "iteration_plan" in expanded

    def test_research_object_expands(self):
        from functional_agents.staleness import expand_path
        expanded = expand_path("research_object")
        assert set(expanded) == {"research_object.evidence", "research_object.hypotheses"}

    def test_decision_model_expands(self):
        from functional_agents.staleness import expand_path
        expanded = expand_path("decision_model")
        expected = {
            "decision_model.assumptions", "decision_model.recommendations",
            "decision_model.risks", "decision_model.opportunities",
            "decision_model.strategic_options", "decision_model.decision_analysis",
        }
        assert set(expanded) == expected

    def test_leaf_path_returns_itself(self):
        from functional_agents.staleness import expand_path, is_container_path
        assert not is_container_path("research_object.evidence")
        assert expand_path("research_object.evidence") == ["research_object.evidence"]

    def test_unknown_path_returns_itself(self):
        from functional_agents.staleness import expand_path
        assert expand_path("some.path") == ["some.path"]


# ---------------------------------------------------------------------------
# StalenessPlan model
# ---------------------------------------------------------------------------

class TestStalenessPlanCreation:
    def _make(self, **kwargs):
        from functional_agents.staleness import StalenessPlan
        defaults = dict(
            source_changes=["SC-001"],
            changed_paths=["research_object.evidence"],
            stale_paths=["research_object.hypotheses"],
            stale_agents=["HypothesisAgent"],
            required_producers=["HypothesisAgent"],
            persisted_paths=["research_object.hypotheses"],
            execution_only_paths=[],
            external_dependencies=[],
            reasoning={"research_object.hypotheses": "consumed by HypothesisAgent"},
            confidence="HIGH",
        )
        defaults.update(kwargs)
        return StalenessPlan.create(**defaults)

    def test_create_generates_plan_id(self):
        plan = self._make()
        assert plan.plan_id.startswith("SP-")
        parts = plan.plan_id.split("-")
        assert len(parts) == 4

    def test_create_sets_created_at(self):
        plan = self._make()
        assert "T" in plan.created_at

    def test_fields_preserved(self):
        plan = self._make()
        assert plan.source_changes == ["SC-001"]
        assert plan.changed_paths == ["research_object.evidence"]
        assert plan.stale_paths == ["research_object.hypotheses"]
        assert plan.confidence == "HIGH"

    def test_inputs_are_copied(self):
        original = ["research_object.evidence"]
        plan = self._make(changed_paths=original)
        original.append("extra")
        assert plan.changed_paths == ["research_object.evidence"]


class TestStalenessPlanSerialization:
    def _make(self):
        from functional_agents.staleness import StalenessPlan
        return StalenessPlan.create(
            source_changes=["SC-abc"],
            changed_paths=["research_object.evidence"],
            stale_paths=["research_object.hypotheses", "research_gap_analysis"],
            stale_agents=["HypothesisAgent", "ResearchGapAgent"],
            required_producers=["HypothesisAgent", "ResearchGapAgent"],
            persisted_paths=["research_object.hypotheses", "research_gap_analysis"],
            execution_only_paths=[],
            external_dependencies=[],
            reasoning={"research_object.hypotheses": "consumed by HypothesisAgent"},
            confidence="HIGH",
        )

    def test_to_dict_has_expected_keys(self):
        plan = self._make()
        d = plan.to_dict()
        expected_keys = {
            "plan_id", "created_at", "source_changes", "changed_paths",
            "stale_paths", "stale_agents", "required_producers",
            "persisted_paths", "execution_only_paths", "external_dependencies",
            "reasoning", "confidence",
        }
        assert set(d.keys()) == expected_keys

    def test_roundtrip(self):
        from functional_agents.staleness import StalenessPlan
        plan = self._make()
        restored = StalenessPlan.from_dict(plan.to_dict())
        assert restored.plan_id == plan.plan_id
        assert restored.changed_paths == plan.changed_paths
        assert restored.stale_paths == plan.stale_paths
        assert restored.confidence == plan.confidence
        assert restored.reasoning == plan.reasoning

    def test_json_roundtrip(self):
        from functional_agents.staleness import StalenessPlan
        plan = self._make()
        restored = StalenessPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
        assert restored.plan_id == plan.plan_id

    def test_from_dict_empty(self):
        from functional_agents.staleness import StalenessPlan
        plan = StalenessPlan.from_dict({})
        assert plan.stale_paths == []
        assert plan.reasoning == {}
        assert plan.confidence == "LOW"


# ---------------------------------------------------------------------------
# DependencyReasoner — core scenarios
# ---------------------------------------------------------------------------

def _make_state_change(affected_paths, change_type="REPLACE", source="orchestrator"):
    from functional_agents.session.state_change import StateChange
    return StateChange.create(
        source=source,
        change_type=change_type,
        affected_paths=affected_paths,
        description="test change",
    )


class TestDependencyReasonerEmptyInput:
    def test_no_state_changes_returns_empty_plan(self):
        from functional_agents.staleness import DependencyReasoner
        plan = DependencyReasoner().analyze(None, [])
        assert plan.changed_paths == []
        assert plan.stale_paths == []
        assert plan.stale_agents == []
        assert plan.required_producers == []
        assert plan.confidence == "LOW"

    def test_state_change_with_no_affected_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change([])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.changed_paths == []
        assert plan.stale_paths == []


class TestDependencyReasonerSinglePath:
    def test_evidence_change_marks_hypotheses_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.hypotheses" in plan.stale_paths

    def test_evidence_change_marks_research_gap_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_gap_analysis" in plan.stale_paths

    def test_evidence_change_marks_report_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "report" in plan.stale_paths

    def test_evidence_change_high_confidence(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.confidence == "HIGH"

    def test_evidence_change_stale_includes_evidence(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" in plan.stale_paths

    def test_evidence_change_not_in_external_deps(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" not in plan.external_dependencies

    def test_evidence_change_evidence_in_persisted_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" in plan.persisted_paths

    def test_evidence_change_changed_path_in_source_changes(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert sc.change_id in plan.source_changes

    def test_planner_change_marks_evidence_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["planner"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" in plan.stale_paths

    def test_planner_change_is_execution_only(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["planner"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "planner" in plan.execution_only_paths

    def test_engagement_change_marks_decision_model_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["engagement"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "decision_model" in plan.stale_paths


class TestDependencyReasonerContainerExpansion:
    def test_research_state_expands_to_persisted(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_state"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "engagement" in plan.changed_paths
        assert "research_object.evidence" in plan.changed_paths
        assert "iteration_plan" in plan.changed_paths

    def test_research_state_confidence_medium(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_state"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.confidence == "MEDIUM"

    def test_research_state_makes_everything_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_state"])
        plan = DependencyReasoner().analyze(None, [sc])
        # All EXECUTION_ONLY intermediate artifacts should be stale
        for path in ["research_strategy", "planner", "strategic_synthesis",
                     "challenge_results", "recommendation_synthesis", "report"]:
            assert path in plan.stale_paths, f"{path!r} should be stale"

    def test_research_object_expands_correctly(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" in plan.changed_paths
        assert "research_object.hypotheses" in plan.changed_paths


class TestDependencyReasonerExternalDependency:
    def test_knowledge_store_in_external_deps(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert "knowledge_store" in plan.external_dependencies

    def test_knowledge_store_not_in_stale_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert "knowledge_store" not in plan.stale_paths

    def test_knowledge_store_change_marks_evidence_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert "research_object.evidence" in plan.stale_paths

    def test_knowledge_store_change_marks_report_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert "report" in plan.stale_paths

    def test_knowledge_store_confidence_high(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.confidence == "HIGH"

    def test_knowledge_store_in_changed_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        assert "knowledge_store" in plan.changed_paths


class TestDependencyReasonerUnknownPath:
    def test_unknown_path_confidence_low(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["some.completely.unknown.path"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.confidence == "LOW"

    def test_unknown_path_in_changed_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["some.unknown.path"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "some.unknown.path" in plan.changed_paths


# ---------------------------------------------------------------------------
# stale_agents and required_producers
# ---------------------------------------------------------------------------

class TestStaleAgentsAndProducers:
    def test_evidence_change_includes_hypothesis_agent_in_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "HypothesisAgent" in plan.stale_agents

    def test_evidence_change_includes_report_agent_in_stale(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "ReportAgent" in plan.stale_agents

    def test_required_producers_subset_of_stale_agents(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        stale_set = set(plan.stale_agents)
        for producer in plan.required_producers:
            assert producer in stale_set

    def test_required_producers_produce_persisted_paths(self):
        from functional_agents.staleness import DependencyReasoner
        from functional_agents.dependencies import DependencyRegistry
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        persisted_set = set(plan.persisted_paths)
        for producer in plan.required_producers:
            dep = DependencyRegistry.get_dependency(producer)
            assert any(p in persisted_set for p in dep.produces), (
                f"{producer} in required_producers but produces no persisted stale path"
            )

    def test_report_agent_not_in_required_producers(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        # ReportAgent produces 'report' which is EXECUTION_ONLY
        assert "ReportAgent" not in plan.required_producers

    def test_stale_agents_in_declaration_order(self):
        from functional_agents.staleness import DependencyReasoner
        from functional_agents.dependencies import DependencyRegistry
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        all_names = [d.agent_name for d in DependencyRegistry.list_dependencies()]
        filtered = [a for a in all_names if a in plan.stale_agents]
        assert filtered == plan.stale_agents


# ---------------------------------------------------------------------------
# Execution-only and persisted classification
# ---------------------------------------------------------------------------

class TestPathClassificationInPlan:
    def test_planner_is_execution_only_in_plan(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        # planner is EXECUTION_ONLY (consumed but not stale here)
        # research_strategy is also EXECUTION_ONLY
        # check that qa is execution_only when stale
        if "qa" in plan.stale_paths:
            assert "qa" in plan.execution_only_paths

    def test_persisted_and_execution_only_are_disjoint(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        persisted_set = set(plan.persisted_paths)
        execution_set = set(plan.execution_only_paths)
        assert persisted_set.isdisjoint(execution_set)

    def test_all_stale_paths_are_classified(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        classified = set(plan.persisted_paths) | set(plan.execution_only_paths)
        for path in plan.stale_paths:
            assert path in classified, (
                f"stale path {path!r} not in persisted_paths or execution_only_paths"
            )

    def test_external_deps_not_in_stale_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        external_set = set(plan.external_dependencies)
        stale_set = set(plan.stale_paths)
        assert external_set.isdisjoint(stale_set)


# ---------------------------------------------------------------------------
# Reasoning trace
# ---------------------------------------------------------------------------

class TestReasoningTrace:
    def test_reasoning_populated_for_stale_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        for path in plan.stale_paths:
            assert path in plan.reasoning, f"Missing reasoning for stale path {path!r}"

    def test_reasoning_references_change_type(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"], change_type="EXTERNAL_EVIDENCE_ADDED")
        plan = DependencyReasoner().analyze(None, [sc])
        # The changed path itself should have a reason mentioning the change type
        assert "EXTERNAL_EVIDENCE_ADDED" in plan.reasoning.get("research_object.evidence", "")

    def test_downstream_reasoning_references_upstream_path(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        hyp_reason = plan.reasoning.get("research_object.hypotheses", "")
        assert "research_object.evidence" in hyp_reason

    def test_reasoning_contains_consumer_agent_name(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        hyp_reason = plan.reasoning.get("research_object.hypotheses", "")
        assert "HypothesisAgent" in hyp_reason

    def test_external_dependency_has_reasoning(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["knowledge_store"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert "knowledge_store" in plan.reasoning


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_output_stale_paths(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan_a = DependencyReasoner().analyze(None, [sc])
        plan_b = DependencyReasoner().analyze(None, [sc])
        # stale_paths are sorted so they must be identical
        assert plan_a.stale_paths == plan_b.stale_paths

    def test_same_input_same_stale_agents(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan_a = DependencyReasoner().analyze(None, [sc])
        plan_b = DependencyReasoner().analyze(None, [sc])
        assert plan_a.stale_agents == plan_b.stale_agents

    def test_same_input_same_confidence(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan_a = DependencyReasoner().analyze(None, [sc])
        plan_b = DependencyReasoner().analyze(None, [sc])
        assert plan_a.confidence == plan_b.confidence

    def test_stale_paths_are_sorted(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan.stale_paths == sorted(plan.stale_paths)


# ---------------------------------------------------------------------------
# Research state parameter (accepted but not required)
# ---------------------------------------------------------------------------

class TestResearchStateParameter:
    def test_none_research_state_accepted(self):
        from functional_agents.staleness import DependencyReasoner
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(None, [sc])
        assert plan is not None

    def test_real_research_state_accepted(self):
        from functional_agents.staleness import DependencyReasoner
        from functional_agents.session import ResearchState
        state = ResearchState(
            engagement={"id": "E-001"},
            research_object={"research_id": "R-001"},
            decision_model={},
            research_gap_analysis={},
            executive_confidence={},
            iteration_plan={},
            updated_at="2026-07-09T00:00:00+00:00",
        )
        sc = _make_state_change(["research_object.evidence"])
        plan = DependencyReasoner().analyze(state, [sc])
        assert plan is not None
        assert "research_object.hypotheses" in plan.stale_paths


# ---------------------------------------------------------------------------
# Registry change — knowledge_store now in EvidenceAgent.consumes
# ---------------------------------------------------------------------------

class TestRegistryKnowledgeStore:
    def test_evidence_agent_consumes_knowledge_store(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("EvidenceAgent")
        assert "knowledge_store" in dep.consumes

    def test_agents_consuming_knowledge_store_includes_evidence(self):
        from functional_agents.dependencies import DependencyRegistry
        consumers = DependencyRegistry.agents_consuming("knowledge_store")
        assert "EvidenceAgent" in consumers


# ---------------------------------------------------------------------------
# CLI smoke tests
# ---------------------------------------------------------------------------

class TestStalenessCLI:
    def _cli(self, *args: str) -> tuple[int, str]:
        result = subprocess.run(
            [sys.executable, "-m", "functional_agents.cli", *args],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    def _make_session_file(self) -> Path:
        """Write a minimal session file with one StateChange to a temp file."""
        import uuid
        from functional_agents.session import (
            ResearchSession,
            ResearchState,
            StateChange,
        )
        state = ResearchState(
            engagement={"engagement_id": "E-001"},
            research_object={},
            decision_model={},
            research_gap_analysis={},
            executive_confidence={},
            iteration_plan={},
            updated_at="2026-07-09T00:00:00+00:00",
        )
        session = ResearchSession.create(
            metadata={"run_id": "RUN-CLI-TEST"},
            research_state=state,
        )
        session.record_state_change(StateChange.create(
            source="orchestrator",
            change_type="REPLACE",
            affected_paths=["research_state"],
            description="test",
        ))
        tmp = Path(tempfile.mkdtemp()) / "test_session.json"
        import json
        tmp.write_text(json.dumps(session.to_dict()), encoding="utf-8")
        return tmp

    def test_staleness_explain_exits_zero(self):
        session_path = self._make_session_file()
        rc, out = self._cli("staleness", "explain", "--session", str(session_path))
        assert rc == 0, f"Non-zero exit: {out}"

    def test_staleness_explain_shows_plan_id(self):
        session_path = self._make_session_file()
        rc, out = self._cli("staleness", "explain", "--session", str(session_path))
        assert rc == 0
        assert "SP-" in out

    def test_staleness_explain_shows_confidence(self):
        session_path = self._make_session_file()
        rc, out = self._cli("staleness", "explain", "--session", str(session_path))
        assert rc == 0
        assert "Confidence" in out

    def test_staleness_path_exits_zero(self):
        rc, out = self._cli("staleness", "path", "--path", "research_object.evidence")
        assert rc == 0, f"Non-zero exit: {out}"

    def test_staleness_path_shows_plan(self):
        rc, out = self._cli("staleness", "path", "--path", "research_object.evidence")
        assert rc == 0
        assert "SP-" in out
        assert "research_object.evidence" in out

    def test_staleness_agent_exits_zero(self):
        rc, out = self._cli("staleness", "agent", "--agent", "EvidenceAgent")
        assert rc == 0, f"Non-zero exit: {out}"

    def test_staleness_agent_shows_downstream(self):
        rc, out = self._cli("staleness", "agent", "--agent", "EvidenceAgent")
        assert rc == 0
        assert "research_object.hypotheses" in out

    def test_staleness_agent_unknown_exits_nonzero(self):
        rc, _out = self._cli("staleness", "agent", "--agent", "NonExistentAgent")
        assert rc != 0

    def test_staleness_explain_missing_session_exits_nonzero(self):
        rc, _out = self._cli("staleness", "explain", "--session", "/nonexistent/path.json")
        assert rc != 0
