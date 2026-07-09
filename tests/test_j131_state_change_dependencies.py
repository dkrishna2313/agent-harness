"""Tests for J13.1 — StateChange model, AgentDependency model, and DependencyRegistry.

Coverage:
- StateChange creation, id format, serialization roundtrip
- ChangeType constants
- AgentDependency creation, serialization roundtrip
- DependencyRegistry: completeness, all 22 agents, path lookups
- ResearchSession state_changes field persistence
- CLI dependency commands (smoke test via subprocess)
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALL_AGENTS = [
    "ProblemFramingAgent",
    "ResearchStrategyAgent",
    "PlannerAgent",
    "EvidenceAgent",
    "HypothesisAgent",
    "ResearchGapAgent",
    "StrategicSynthesisAgent",
    "ChallengeAgent",
    "AssumptionAgent",
    "RecommendationAgent",
    "RiskAgent",
    "OpportunityAgent",
    "StrategicOptionAgent",
    "DecisionAnalysisAgent",
    "ExecutiveConfidenceAgent",
    "IterationPlanAgent",
    "MultiProfileAgent",
    "ScenarioAgent",
    "QAAgent",
    "RecommendationImprovementAgent",
    "RecommendationSynthesisAgent",
    "ReportAgent",
]


# ---------------------------------------------------------------------------
# Package import
# ---------------------------------------------------------------------------

class TestPackageImports:
    def test_state_change_importable(self):
        from functional_agents.session import StateChange  # noqa: F401

    def test_change_type_importable(self):
        from functional_agents.session import ChangeType  # noqa: F401

    def test_agent_dependency_importable(self):
        from functional_agents.dependencies import AgentDependency  # noqa: F401

    def test_dependency_registry_importable(self):
        from functional_agents.dependencies import DependencyRegistry  # noqa: F401

    def test_session_package_exports_state_change(self):
        import functional_agents.session as pkg
        assert "StateChange" in pkg.__all__
        assert "ChangeType" in pkg.__all__

    def test_dependencies_package_exports(self):
        import functional_agents.dependencies as pkg
        assert "AgentDependency" in pkg.__all__
        assert "DependencyRegistry" in pkg.__all__


# ---------------------------------------------------------------------------
# ChangeType constants
# ---------------------------------------------------------------------------

class TestChangeType:
    def test_all_expected_constants_present(self):
        from functional_agents.session import ChangeType
        assert ChangeType.CREATE == "CREATE"
        assert ChangeType.UPDATE == "UPDATE"
        assert ChangeType.DELETE == "DELETE"
        assert ChangeType.REPLACE == "REPLACE"
        assert ChangeType.APPEND == "APPEND"
        assert ChangeType.EXTERNAL_EVIDENCE_ADDED == "EXTERNAL_EVIDENCE_ADDED"
        assert ChangeType.SESSION_CONTINUED == "SESSION_CONTINUED"
        assert ChangeType.MANUAL_OVERRIDE == "MANUAL_OVERRIDE"

    def test_all_set_has_eight_members(self):
        from functional_agents.session import ChangeType
        assert len(ChangeType._ALL) == 8


# ---------------------------------------------------------------------------
# StateChange model
# ---------------------------------------------------------------------------

class TestStateChangeCreation:
    def test_create_returns_state_change(self):
        from functional_agents.session import StateChange, ChangeType
        sc = StateChange.create(
            source="orchestrator",
            change_type=ChangeType.REPLACE,
            affected_paths=["research_state"],
            description="Test change",
        )
        assert sc.source == "orchestrator"
        assert sc.change_type == "REPLACE"
        assert sc.affected_paths == ["research_state"]
        assert sc.description == "Test change"

    def test_create_generates_change_id(self):
        from functional_agents.session import StateChange
        sc = StateChange.create(
            source="cli",
            change_type="UPDATE",
            affected_paths=["decision_model"],
            description="d",
        )
        assert sc.change_id.startswith("SC-")
        parts = sc.change_id.split("-")
        # SC-YYYYMMDD-HHMMSS-hex6 → ["SC", "YYYYMMDD", "HHMMSS", "hex6"]
        assert len(parts) == 4

    def test_create_sets_timestamp(self):
        from functional_agents.session import StateChange
        sc = StateChange.create(
            source="s", change_type="CREATE", affected_paths=[], description="d"
        )
        assert "T" in sc.timestamp  # ISO 8601

    def test_create_with_metadata(self):
        from functional_agents.session import StateChange
        sc = StateChange.create(
            source="s",
            change_type="APPEND",
            affected_paths=["research_object.evidence"],
            description="d",
            metadata={"run_id": "R-123"},
        )
        assert sc.metadata == {"run_id": "R-123"}

    def test_create_metadata_defaults_empty(self):
        from functional_agents.session import StateChange
        sc = StateChange.create(
            source="s", change_type="CREATE", affected_paths=[], description="d"
        )
        assert sc.metadata == {}

    def test_affected_paths_is_copy(self):
        from functional_agents.session import StateChange
        original = ["research_state"]
        sc = StateChange.create(
            source="s", change_type="REPLACE", affected_paths=original, description="d"
        )
        original.append("extra")
        assert sc.affected_paths == ["research_state"]


class TestStateChangeSerialization:
    def _make(self):
        from functional_agents.session import StateChange
        return StateChange.create(
            source="orchestrator",
            change_type="REPLACE",
            affected_paths=["research_state"],
            description="Full pipeline execution replaced ResearchState",
            metadata={"run_id": "RUN-123"},
        )

    def test_to_dict_has_expected_keys(self):
        sc = self._make()
        d = sc.to_dict()
        assert set(d.keys()) == {
            "change_id", "timestamp", "source", "change_type",
            "affected_paths", "description", "metadata",
        }

    def test_roundtrip(self):
        from functional_agents.session import StateChange
        sc = self._make()
        restored = StateChange.from_dict(sc.to_dict())
        assert restored.change_id == sc.change_id
        assert restored.timestamp == sc.timestamp
        assert restored.source == sc.source
        assert restored.change_type == sc.change_type
        assert restored.affected_paths == sc.affected_paths
        assert restored.description == sc.description
        assert restored.metadata == sc.metadata

    def test_json_roundtrip(self):
        from functional_agents.session import StateChange
        sc = self._make()
        restored = StateChange.from_dict(json.loads(json.dumps(sc.to_dict())))
        assert restored.change_id == sc.change_id

    def test_from_dict_empty_dict(self):
        from functional_agents.session import StateChange
        sc = StateChange.from_dict({})
        assert sc.change_id == ""
        assert sc.affected_paths == []
        assert sc.metadata == {}


# ---------------------------------------------------------------------------
# AgentDependency model
# ---------------------------------------------------------------------------

class TestAgentDependency:
    def test_creation(self):
        from functional_agents.dependencies import AgentDependency
        dep = AgentDependency(
            agent_name="EvidenceAgent",
            consumes=["planner", "research_strategy"],
            produces=["research_object.evidence"],
            invalidates=["research_object.hypotheses"],
        )
        assert dep.agent_name == "EvidenceAgent"
        assert dep.consumes == ["planner", "research_strategy"]
        assert dep.produces == ["research_object.evidence"]
        assert dep.invalidates == ["research_object.hypotheses"]

    def test_default_empty_lists(self):
        from functional_agents.dependencies import AgentDependency
        dep = AgentDependency(agent_name="X")
        assert dep.consumes == []
        assert dep.produces == []
        assert dep.invalidates == []

    def test_roundtrip(self):
        from functional_agents.dependencies import AgentDependency
        dep = AgentDependency(
            agent_name="EvidenceAgent",
            consumes=["planner"],
            produces=["research_object.evidence"],
            invalidates=["research_object.hypotheses"],
        )
        restored = AgentDependency.from_dict(dep.to_dict())
        assert restored.agent_name == dep.agent_name
        assert restored.consumes == dep.consumes
        assert restored.produces == dep.produces
        assert restored.invalidates == dep.invalidates

    def test_from_dict_empty(self):
        from functional_agents.dependencies import AgentDependency
        dep = AgentDependency.from_dict({})
        assert dep.agent_name == ""
        assert dep.consumes == []


# ---------------------------------------------------------------------------
# DependencyRegistry — completeness
# ---------------------------------------------------------------------------

class TestDependencyRegistryCompleteness:
    def test_all_22_agents_registered(self):
        from functional_agents.dependencies import DependencyRegistry
        registered = {d.agent_name for d in DependencyRegistry.list_dependencies()}
        missing = set(_ALL_AGENTS) - registered
        assert missing == set(), f"Missing declarations: {sorted(missing)}"

    def test_exactly_22_declarations(self):
        from functional_agents.dependencies import DependencyRegistry
        assert len(DependencyRegistry.list_dependencies()) == 22

    def test_all_agents_have_non_empty_agent_name(self):
        from functional_agents.dependencies import DependencyRegistry
        for dep in DependencyRegistry.list_dependencies():
            assert dep.agent_name, f"Empty agent_name found: {dep}"

    def test_all_path_values_are_non_empty_strings(self):
        from functional_agents.dependencies import DependencyRegistry
        for dep in DependencyRegistry.list_dependencies():
            for path in dep.consumes + dep.produces + dep.invalidates:
                assert isinstance(path, str) and path, (
                    f"Empty/non-string path in {dep.agent_name}: {path!r}"
                )

    def test_all_agents_produce_at_least_one_path(self):
        from functional_agents.dependencies import DependencyRegistry
        for dep in DependencyRegistry.list_dependencies():
            assert dep.produces, f"{dep.agent_name} has no produces"

    def test_all_agents_consume_at_least_one_path(self):
        from functional_agents.dependencies import DependencyRegistry
        for dep in DependencyRegistry.list_dependencies():
            assert dep.consumes, f"{dep.agent_name} has no consumes"

    def test_report_agent_has_no_invalidates(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("ReportAgent")
        assert dep.invalidates == [], "ReportAgent is terminal — must have no invalidates"

    @pytest.mark.parametrize("agent_name", _ALL_AGENTS)
    def test_each_agent_registered(self, agent_name):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency(agent_name)
        assert dep.agent_name == agent_name


# ---------------------------------------------------------------------------
# DependencyRegistry — lookups
# ---------------------------------------------------------------------------

class TestDependencyRegistryLookups:
    def test_get_dependency_returns_correct_agent(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("EvidenceAgent")
        assert dep.agent_name == "EvidenceAgent"
        assert "research_object.evidence" in dep.produces

    def test_get_dependency_raises_key_error_for_unknown(self):
        from functional_agents.dependencies import DependencyRegistry
        with pytest.raises(KeyError, match="NonExistentAgent"):
            DependencyRegistry.get_dependency("NonExistentAgent")

    def test_agents_producing_evidence(self):
        from functional_agents.dependencies import DependencyRegistry
        producers = DependencyRegistry.agents_producing("research_object.evidence")
        assert "EvidenceAgent" in producers

    def test_agents_consuming_evidence(self):
        from functional_agents.dependencies import DependencyRegistry
        consumers = DependencyRegistry.agents_consuming("research_object.evidence")
        assert "HypothesisAgent" in consumers
        assert "ResearchGapAgent" in consumers
        assert "StrategicSynthesisAgent" in consumers
        assert "ChallengeAgent" in consumers

    def test_agents_consuming_engagement(self):
        from functional_agents.dependencies import DependencyRegistry
        consumers = DependencyRegistry.agents_consuming("engagement")
        assert "ProblemFramingAgent" in consumers
        assert "ResearchStrategyAgent" in consumers
        assert "PlannerAgent" in consumers
        assert "StrategicSynthesisAgent" in consumers

    def test_agents_producing_report(self):
        from functional_agents.dependencies import DependencyRegistry
        producers = DependencyRegistry.agents_producing("report")
        assert producers == ["ReportAgent"]

    def test_agents_invalidated_by_evidence(self):
        from functional_agents.dependencies import DependencyRegistry
        # agents_invalidated_by("research_object.evidence") returns agents whose
        # invalidates list contains that path — i.e. upstream agents that, when
        # their output changes, make evidence stale.
        # Only ProblemFramingAgent is upstream enough to directly invalidate evidence.
        inv = DependencyRegistry.agents_invalidated_by("research_object.evidence")
        assert "ProblemFramingAgent" in inv

    def test_agents_consuming_report_is_empty(self):
        from functional_agents.dependencies import DependencyRegistry
        # report is the terminal artifact; no agent consumes it as input.
        consumers = DependencyRegistry.agents_consuming("report")
        assert consumers == []

    def test_agents_consuming_unknown_path_returns_empty(self):
        from functional_agents.dependencies import DependencyRegistry
        assert DependencyRegistry.agents_consuming("nonexistent.path") == []

    def test_agents_producing_unknown_path_returns_empty(self):
        from functional_agents.dependencies import DependencyRegistry
        assert DependencyRegistry.agents_producing("nonexistent.path") == []

    def test_agents_invalidated_by_unknown_path_returns_empty(self):
        from functional_agents.dependencies import DependencyRegistry
        assert DependencyRegistry.agents_invalidated_by("nonexistent.path") == []

    def test_list_dependencies_is_deterministic(self):
        from functional_agents.dependencies import DependencyRegistry
        names_a = [d.agent_name for d in DependencyRegistry.list_dependencies()]
        names_b = [d.agent_name for d in DependencyRegistry.list_dependencies()]
        assert names_a == names_b

    def test_planner_agent_invalidates_evidence(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("PlannerAgent")
        assert "research_object.evidence" in dep.invalidates

    def test_evidence_agent_invalidates_hypotheses(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("EvidenceAgent")
        assert "research_object.hypotheses" in dep.invalidates

    def test_problem_framing_invalidates_all_downstream(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("ProblemFramingAgent")
        for path in ["research_strategy", "research_object.evidence", "report"]:
            assert path in dep.invalidates, (
                f"ProblemFramingAgent should invalidate {path!r}"
            )

    def test_executive_confidence_invalidates_iteration_plan(self):
        from functional_agents.dependencies import DependencyRegistry
        dep = DependencyRegistry.get_dependency("ExecutiveConfidenceAgent")
        assert "iteration_plan" in dep.invalidates
        assert "report" in dep.invalidates


# ---------------------------------------------------------------------------
# ResearchSession state_changes integration
# ---------------------------------------------------------------------------

def _make_session_for_test():
    from functional_agents.session import ResearchSession, ResearchState
    state = ResearchState(
        engagement={"engagement_id": "E-001"},
        research_object={},
        decision_model={},
        research_gap_analysis={},
        executive_confidence={},
        iteration_plan={},
        updated_at="2026-07-09T00:00:00+00:00",
    )
    return ResearchSession.create(
        metadata={"run_id": "RUN-TEST"},
        research_state=state,
    )


class TestResearchSessionStateChanges:
    def _make_session(self):
        return _make_session_for_test()

    def test_new_session_has_empty_state_changes(self):
        session = self._make_session()
        assert session.state_changes == []

    def test_record_state_change_appends(self):
        from functional_agents.session import StateChange
        session = self._make_session()
        sc = StateChange.create(
            source="orchestrator",
            change_type="REPLACE",
            affected_paths=["research_state"],
            description="Test",
        )
        session.record_state_change(sc)
        assert len(session.state_changes) == 1
        assert session.state_changes[0].change_id == sc.change_id

    def test_record_state_change_updates_updated_at(self):
        from functional_agents.session import StateChange
        session = self._make_session()
        before = session.updated_at
        sc = StateChange.create(
            source="s", change_type="UPDATE", affected_paths=[], description="d"
        )
        session.record_state_change(sc)
        # updated_at should be >= before (may be equal in fast tests)
        assert session.updated_at >= before

    def test_state_changes_serialized_in_to_dict(self):
        from functional_agents.session import StateChange
        session = self._make_session()
        sc = StateChange.create(
            source="orchestrator",
            change_type="REPLACE",
            affected_paths=["research_state"],
            description="d",
        )
        session.record_state_change(sc)
        d = session.to_dict()
        assert "state_changes" in d
        assert len(d["state_changes"]) == 1
        assert d["state_changes"][0]["change_id"] == sc.change_id

    def test_state_changes_deserialized_from_dict(self):
        from functional_agents.session import ResearchSession, StateChange
        session = self._make_session()
        sc = StateChange.create(
            source="orchestrator",
            change_type="REPLACE",
            affected_paths=["research_state"],
            description="d",
        )
        session.record_state_change(sc)
        restored = ResearchSession.from_dict(session.to_dict())
        assert len(restored.state_changes) == 1
        assert restored.state_changes[0].change_id == sc.change_id

    def test_from_dict_missing_state_changes_defaults_empty(self):
        from functional_agents.session import ResearchSession
        session = self._make_session()
        d = session.to_dict()
        del d["state_changes"]
        restored = ResearchSession.from_dict(d)
        assert restored.state_changes == []

    def test_json_roundtrip_with_state_changes(self):
        from functional_agents.session import ResearchSession, StateChange
        session = self._make_session()
        session.record_state_change(StateChange.create(
            source="cli", change_type="SESSION_CONTINUED",
            affected_paths=["research_state"],
            description="Session resumed",
        ))
        raw = json.dumps(session.to_dict())
        restored = ResearchSession.from_dict(json.loads(raw))
        assert len(restored.state_changes) == 1
        assert restored.state_changes[0].source == "cli"
        assert restored.state_changes[0].change_type == "SESSION_CONTINUED"

    def test_to_dict_keys_includes_state_changes(self):
        session = self._make_session()
        d = session.to_dict()
        assert "state_changes" in d


# ---------------------------------------------------------------------------
# CLI dependency commands — smoke tests
# ---------------------------------------------------------------------------

class TestDependencyCLI:
    def _cli(self, *args: str) -> tuple[int, str]:
        result = subprocess.run(
            [sys.executable, "-m", "functional_agents.cli", *args],
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout + result.stderr

    def test_dependencies_list_exits_zero(self):
        rc, out = self._cli("dependencies", "list")
        assert rc == 0, f"Non-zero exit: {out}"

    def test_dependencies_list_shows_22_agents(self):
        rc, out = self._cli("dependencies", "list")
        assert rc == 0
        assert "22" in out

    def test_dependencies_list_shows_evidence_agent(self):
        rc, out = self._cli("dependencies", "list")
        assert rc == 0
        assert "EvidenceAgent" in out

    def test_dependencies_show_evidence_agent(self):
        rc, out = self._cli("dependencies", "show", "--agent", "EvidenceAgent")
        assert rc == 0, f"Non-zero exit: {out}"
        assert "EvidenceAgent" in out
        assert "research_object.evidence" in out

    def test_dependencies_show_unknown_agent_exits_nonzero(self):
        rc, _out = self._cli("dependencies", "show", "--agent", "NonExistentAgent")
        assert rc != 0

    def test_dependencies_affected_evidence(self):
        rc, out = self._cli("dependencies", "affected", "--path", "research_object.evidence")
        assert rc == 0, f"Non-zero exit: {out}"
        assert "EvidenceAgent" in out

    def test_dependencies_affected_unknown_path(self):
        rc, out = self._cli("dependencies", "affected", "--path", "totally.unknown.path")
        assert rc == 0
        assert "(none)" in out
