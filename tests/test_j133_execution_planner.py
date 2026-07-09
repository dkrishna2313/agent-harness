"""Tests for J13.3 — ExecutionPlanner and ExecutionPlan."""

from __future__ import annotations

import pytest

from functional_agents.dependencies import DependencyRegistry
from functional_agents.planning import ExecutionPlan, ExecutionPlanner
from functional_agents.session.state_change import ChangeType, StateChange
from functional_agents.staleness import DependencyReasoner, PathKind, classify_path
from functional_agents.staleness.staleness_plan import StalenessPlan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_staleness_plan(changed_paths: list[str]) -> StalenessPlan:
    """Create a StalenessPlan from a list of changed paths."""
    changes = [
        StateChange.create(
            source="test",
            change_type=ChangeType.UPDATE,
            affected_paths=[p],
            description=f"test change to {p}",
        )
        for p in changed_paths
    ]
    return DependencyReasoner().analyze(None, changes)


def _plan_for(changed_paths: list[str]) -> ExecutionPlan:
    staleness = _make_staleness_plan(changed_paths)
    return ExecutionPlanner().plan(staleness)


def _has_intra_group_dependency(group: list[str]) -> bool:
    """Return True if any pair in the group has a within-group dependency."""
    group_set = set(group)
    for agent_name in group:
        dep = DependencyRegistry.get_dependency(agent_name)
        for consumed in dep.consumes:
            for producer in DependencyRegistry.agents_producing(consumed):
                if producer in group_set and producer != agent_name:
                    return True
    return False


def _execution_order_is_valid(plan: ExecutionPlan) -> bool:
    """Return True if execution_order respects all dependency edges within the plan."""
    pos = {a: i for i, a in enumerate(plan.execution_order)}
    plan_set = set(plan.execution_order)
    for agent_name in plan.execution_order:
        dep = DependencyRegistry.get_dependency(agent_name)
        for consumed in dep.consumes:
            for producer in DependencyRegistry.agents_producing(consumed):
                if producer in plan_set:
                    if pos[producer] >= pos[agent_name]:
                        return False
    return True


# ---------------------------------------------------------------------------
# Empty / trivial plans
# ---------------------------------------------------------------------------

class TestEmptyPlan:
    def test_empty_plan_no_state_changes(self):
        """StalenessPlan with no stale agents produces an empty ExecutionPlan."""
        # Construct a minimal StalenessPlan with no stale agents
        sp = StalenessPlan.create(
            source_changes=[],
            changed_paths=[],
            stale_paths=[],
            stale_agents=[],
            required_producers=[],
            persisted_paths=[],
            execution_only_paths=[],
            external_dependencies=[],
            reasoning={},
            confidence="LOW",
        )
        plan = ExecutionPlanner().plan(sp)
        assert plan.required_agents == []
        assert plan.optional_agents == []
        assert plan.execution_order == []
        assert plan.execution_groups == []
        assert plan.estimated_steps == 0
        assert plan.blocked_agents == []


# ---------------------------------------------------------------------------
# ExecutionPlan identity and metadata
# ---------------------------------------------------------------------------

class TestPlanIdentity:
    def test_plan_id_format(self):
        plan = _plan_for(["research_object.evidence"])
        assert plan.plan_id.startswith("EP-")

    def test_staleness_plan_id_propagated(self):
        sp = _make_staleness_plan(["research_object.evidence"])
        plan = ExecutionPlanner().plan(sp)
        assert plan.staleness_plan_id == sp.plan_id

    def test_confidence_propagated(self):
        sp = _make_staleness_plan(["research_object.evidence"])
        plan = ExecutionPlanner().plan(sp)
        assert plan.confidence == sp.confidence

    def test_triggering_state_changes_propagated(self):
        sp = _make_staleness_plan(["research_object.evidence"])
        plan = ExecutionPlanner().plan(sp)
        assert plan.triggering_state_changes == sp.source_changes


# ---------------------------------------------------------------------------
# required_agents expansion
# ---------------------------------------------------------------------------

class TestRequiredAgentsExpansion:
    def test_evidence_changed_includes_planner_prerequisites(self):
        """When evidence changes, the planner computes PlannerAgent et al. as prerequisites."""
        plan = _plan_for(["research_object.evidence"])
        # EvidenceAgent consumes planner (EXECUTION_ONLY) → PlannerAgent must be required
        assert "PlannerAgent" in plan.required_agents
        # PlannerAgent consumes research_strategy → ResearchStrategyAgent must be required
        assert "ResearchStrategyAgent" in plan.required_agents
        # ResearchStrategyAgent consumes decision_architecture → ProblemFramingAgent required
        assert "ProblemFramingAgent" in plan.required_agents

    def test_evidence_changed_includes_persisted_producers(self):
        plan = _plan_for(["research_object.evidence"])
        # Agents producing stale PERSISTED paths must all be present
        for agent in [
            "EvidenceAgent", "HypothesisAgent", "ResearchGapAgent",
            "AssumptionAgent", "RecommendationAgent", "RiskAgent",
            "OpportunityAgent", "StrategicOptionAgent", "DecisionAnalysisAgent",
            "ExecutiveConfidenceAgent", "IterationPlanAgent",
        ]:
            assert agent in plan.required_agents, f"{agent} should be required"

    def test_evidence_changed_includes_eo_prerequisites_of_persisted_producers(self):
        plan = _plan_for(["research_object.evidence"])
        # AssumptionAgent consumes challenge_results (EO) → ChallengeAgent required
        assert "ChallengeAgent" in plan.required_agents
        # AssumptionAgent also consumes strategic_synthesis (EO) → StrategicSynthesisAgent required
        assert "StrategicSynthesisAgent" in plan.required_agents

    def test_evidence_changed_optional_agents_are_execution_only_producers(self):
        plan = _plan_for(["research_object.evidence"])
        # MultiProfileAgent, ScenarioAgent, etc. produce only EXECUTION_ONLY paths
        optional_set = set(plan.optional_agents)
        for agent_name in optional_set:
            dep = DependencyRegistry.get_dependency(agent_name)
            assert all(
                classify_path(p) == PathKind.EXECUTION_ONLY for p in dep.produces
            ), f"{agent_name} produces a PERSISTED path but is optional"

    def test_evidence_changed_report_agent_is_optional(self):
        plan = _plan_for(["research_object.evidence"])
        assert "ReportAgent" in plan.optional_agents

    def test_late_stage_override_only_requires_that_agent(self):
        """MANUAL_OVERRIDE on iteration_plan: only IterationPlanAgent needed (no EO inputs)."""
        plan = _plan_for(["iteration_plan"])
        assert "IterationPlanAgent" in plan.required_agents
        # iteration_plan consumes all PERSISTED paths → no EO prerequisites
        assert "EvidenceAgent" not in plan.required_agents

    def test_manual_override_research_gap_expands_to_planner_chain(self):
        """MANUAL_OVERRIDE on research_gap_analysis pulls in PlannerAgent chain."""
        plan = _plan_for(["research_gap_analysis"])
        # ResearchGapAgent consumes planner (EO, not stale) → needs PlannerAgent
        assert "PlannerAgent" in plan.required_agents
        assert "ResearchStrategyAgent" in plan.required_agents
        assert "ProblemFramingAgent" in plan.required_agents
        # Does NOT need EvidenceAgent (research_object.evidence is PERSISTED, not stale)
        assert "EvidenceAgent" not in plan.required_agents

    def test_required_agents_subset_does_not_include_eo_only_nonprerequisites(self):
        """MultiProfileAgent should not appear in required when only executive_confidence changes."""
        plan = _plan_for(["executive_confidence"])
        # Only ExecutiveConfidenceAgent produces executive_confidence (PERSISTED)
        # ExecutiveConfidenceAgent consumes decision_model.decision_analysis (PERSISTED),
        # decision_model.assumptions (PERSISTED), research_gap_analysis (PERSISTED) — all PERSISTED
        # So only ExecutiveConfidenceAgent (and downstream IterationPlanAgent?) should be required
        assert "MultiProfileAgent" not in plan.required_agents


# ---------------------------------------------------------------------------
# Execution groups — topological ordering
# ---------------------------------------------------------------------------

class TestExecutionGroups:
    def test_problem_framing_is_first_group(self):
        plan = _plan_for(["research_object.evidence"])
        # ProblemFramingAgent has no predecessors in the plan → first group
        assert plan.execution_groups[0] == ["ProblemFramingAgent"]

    def test_research_strategy_follows_problem_framing(self):
        plan = _plan_for(["research_object.evidence"])
        order = plan.execution_order
        pf_idx = order.index("ProblemFramingAgent")
        rs_idx = order.index("ResearchStrategyAgent")
        assert pf_idx < rs_idx

    def test_planner_follows_research_strategy(self):
        plan = _plan_for(["research_object.evidence"])
        order = plan.execution_order
        assert order.index("ResearchStrategyAgent") < order.index("PlannerAgent")

    def test_evidence_follows_planner(self):
        plan = _plan_for(["research_object.evidence"])
        order = plan.execution_order
        assert order.index("PlannerAgent") < order.index("EvidenceAgent")

    def test_hypothesis_follows_evidence(self):
        plan = _plan_for(["research_object.evidence"])
        order = plan.execution_order
        assert order.index("EvidenceAgent") < order.index("HypothesisAgent")

    def test_report_agent_is_last(self):
        plan = _plan_for(["research_object.evidence"])
        assert plan.execution_order[-1] == "ReportAgent"

    def test_no_intra_group_dependencies(self):
        """No agent in a group may depend on another agent in the same group."""
        plan = _plan_for(["research_object.evidence"])
        for group in plan.execution_groups:
            assert not _has_intra_group_dependency(group), (
                f"Group has intra-group dependency: {group}"
            )

    def test_execution_order_is_topologically_valid(self):
        plan = _plan_for(["research_object.evidence"])
        assert _execution_order_is_valid(plan)

    def test_parallel_group_challenge_strategic_synthesis(self):
        """ChallengeAgent and StrategicSynthesisAgent should be in the same group."""
        plan = _plan_for(["research_object.evidence"])
        pos = {a: i for i, a in enumerate(plan.execution_order)}
        # Both come after HypothesisAgent; neither depends on the other
        hy_pos = pos["HypothesisAgent"]
        ch_pos = pos["ChallengeAgent"]
        ss_pos = pos["StrategicSynthesisAgent"]
        rg_pos = pos["ResearchGapAgent"]
        assert hy_pos < ch_pos
        assert hy_pos < ss_pos
        assert hy_pos < rg_pos
        # All three in the same group
        for group in plan.execution_groups:
            group_set = set(group)
            if "ChallengeAgent" in group_set:
                assert "StrategicSynthesisAgent" in group_set
                assert "ResearchGapAgent" in group_set
                break

    def test_risk_and_opportunity_in_same_group(self):
        plan = _plan_for(["research_object.evidence"])
        for group in plan.execution_groups:
            group_set = set(group)
            if "RiskAgent" in group_set:
                assert "OpportunityAgent" in group_set
                break

    def test_minimal_plan_order(self):
        """MANUAL_OVERRIDE on research_gap_analysis: 6 agents in sequential order."""
        plan = _plan_for(["research_gap_analysis"])
        assert _execution_order_is_valid(plan)
        order = plan.execution_order
        pf = order.index("ProblemFramingAgent")
        rs = order.index("ResearchStrategyAgent")
        pl = order.index("PlannerAgent")
        rg = order.index("ResearchGapAgent")
        ec = order.index("ExecutiveConfidenceAgent")
        ip = order.index("IterationPlanAgent")
        assert pf < rs < pl < rg < ec < ip

    def test_execution_groups_cover_all_planned_agents(self):
        """execution_order must contain exactly required + optional agents."""
        plan = _plan_for(["research_object.evidence"])
        expected = set(plan.required_agents) | set(plan.optional_agents)
        actual = set(plan.execution_order)
        assert expected == actual

    def test_estimated_steps_equals_group_count(self):
        plan = _plan_for(["research_object.evidence"])
        assert plan.estimated_steps == len(plan.execution_groups)


# ---------------------------------------------------------------------------
# Blocked agents
# ---------------------------------------------------------------------------

class TestBlockedAgents:
    def test_no_blocked_agents_in_normal_operation(self):
        """With a complete registry, no agent should be blocked."""
        plan = _plan_for(["research_object.evidence"])
        assert plan.blocked_agents == []
        assert plan.blocked_reasons == {}

    def test_no_blocked_agents_for_any_persisted_path(self):
        persisted = [
            "engagement", "research_object.evidence", "research_object.hypotheses",
            "decision_model", "research_gap_analysis", "executive_confidence",
            "iteration_plan",
        ]
        for path in persisted:
            plan = _plan_for([path])
            assert plan.blocked_agents == [], f"Unexpected blocked agents for path={path}"


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------

class TestReasoning:
    def test_required_producers_labeled_required(self):
        plan = _plan_for(["research_object.evidence"])
        assert "EvidenceAgent" in plan.reasoning
        assert "required" in plan.reasoning["EvidenceAgent"]

    def test_prerequisites_labeled_prerequisite(self):
        plan = _plan_for(["research_object.evidence"])
        assert "ProblemFramingAgent" in plan.reasoning
        assert "prerequisite" in plan.reasoning["ProblemFramingAgent"]

    def test_optional_agents_labeled_optional(self):
        plan = _plan_for(["research_object.evidence"])
        assert "ReportAgent" in plan.reasoning
        assert "optional" in plan.reasoning["ReportAgent"]

    def test_all_planned_agents_have_reasoning(self):
        plan = _plan_for(["research_object.evidence"])
        for agent_name in plan.execution_order:
            assert agent_name in plan.reasoning, f"No reasoning for {agent_name}"


# ---------------------------------------------------------------------------
# Serialization roundtrip
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        plan = _plan_for(["research_object.evidence"])
        d = plan.to_dict()
        restored = ExecutionPlan.from_dict(d)
        assert restored.plan_id == plan.plan_id
        assert restored.staleness_plan_id == plan.staleness_plan_id
        assert restored.required_agents == plan.required_agents
        assert restored.optional_agents == plan.optional_agents
        assert restored.execution_order == plan.execution_order
        assert restored.execution_groups == plan.execution_groups
        assert restored.estimated_steps == plan.estimated_steps
        assert restored.confidence == plan.confidence
        assert restored.blocked_agents == plan.blocked_agents
        assert restored.blocked_reasons == plan.blocked_reasons

    def test_reasoning_roundtrip(self):
        plan = _plan_for(["research_object.evidence"])
        restored = ExecutionPlan.from_dict(plan.to_dict())
        assert restored.reasoning == plan.reasoning

    def test_empty_plan_roundtrip(self):
        sp = StalenessPlan.create(
            source_changes=[], changed_paths=[], stale_paths=[],
            stale_agents=[], required_producers=[], persisted_paths=[],
            execution_only_paths=[], external_dependencies=[],
            reasoning={}, confidence="LOW",
        )
        plan = ExecutionPlanner().plan(sp)
        restored = ExecutionPlan.from_dict(plan.to_dict())
        assert restored.execution_order == []
        assert restored.execution_groups == []
        assert restored.estimated_steps == 0
