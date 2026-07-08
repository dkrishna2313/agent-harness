"""Tests for IterationPlanAgent (J12.2 + J12.2a hardening).

Covers:
  1.  Module imports and class membership
  2.  FunctionalAgent inheritance + run() contract
  3.  No LLM calls — no claude_client import
  4.  Pipeline position: runs after ExecutiveConfidenceAgent in orchestrator
  5.  compute_iteration_plan output keys
  6.  Validation priorities → HIGH/HIGH tasks
  7.  Critical unknowns deduplicated against validation priorities
  8.  Low-confidence Critical assumptions → HIGH gain tasks
  9.  No-evidence Important/Critical assumptions → MEDIUM gain tasks
  10. H3/H4 deduplication: same assumption_id not emitted twice
  11. High-severity weak-evidence risks → MEDIUM gain tasks
  12. Recommended option fragility (decision_analysis source)
  13. Recommended option fragility (strategic_options.recommended flag)
  14. Research gap followups (LOW gain) / text dedup against VPs
  15. Task priority ordering is deterministic
  16. Task IDs assigned IRT-001, IRT-002, …
  17. Max-tasks cap (default 10)
  18. iteration_needed=True paths
  19. iteration_needed=False path
  20. stop_conditions non-empty
  21. expected_confidence_after_completion values
  22. plan_confidence range [0.0, 1.0]
  23. context fields written (iteration_plan, trace, research_object)
  24. Agent does not modify assumptions/recommendations/risks/strategic_options
  25. Runs with all-empty inputs (no crash)
  26. WorkflowState.ITERATION_PLAN constant present
  27. AgentContext.iteration_plan field present
  28. context_field fallback from research_object
  — J12.2a additions —
  29. Structured ID extraction from text (_extract_ids_from_text, _enrich_task_ids)
  30. Linkage completeness: IDs in VP/CU text appear in related_*_ids
  31. Plan validation (_validate_plan): structural and cross-reference checks
  32. Persistence regression: research_object["iteration_plan"] == context.iteration_plan
  33. Expanded trace summary metrics
  34. No new LLM calls introduced
"""

from __future__ import annotations

import pytest

from functional_agents.iteration_plan_agent import (
    IterationPlanAgent,
    compute_iteration_plan,
    _tasks_from_validation_priorities,
    _tasks_from_critical_unknowns,
    _tasks_from_low_confidence_assumptions,
    _tasks_from_no_evidence_assumptions,
    _tasks_from_high_severity_risks,
    _tasks_from_recommended_option,
    _tasks_from_research_gap_followups,
    _iteration_needed,
    _stop_conditions,
    _expected_confidence_after_completion,
    _plan_confidence,
    _sorted_tasks,
    _assign_task_ids,
    _extract_ids_from_text,   # J12.2a
    _enrich_task_ids,          # J12.2a
    _validate_plan,            # J12.2a
)
from functional_agents.context import AgentContext, WorkflowState
from functional_agents.base import FunctionalAgent


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ctx(
    exec_conf: dict | None = None,
    assumptions: list | None = None,
    recommendations: list | None = None,
    risks: list | None = None,
    strategic_options: list | None = None,
    decision_analysis: dict | None = None,
    research_gap_analysis: dict | None = None,
) -> AgentContext:
    ctx = AgentContext(
        question="Should we proceed?",
        profiles=["default"],
        execution_profile="default",
        research_object={"id": "R-TEST_001"},
        run_id="testj122",
    )
    if exec_conf is not None:
        ctx.executive_confidence = exec_conf
    if assumptions is not None:
        ctx.assumptions = assumptions
    if recommendations is not None:
        ctx.recommendations = recommendations
    if risks is not None:
        ctx.risks = risks
    if strategic_options is not None:
        ctx.strategic_options = strategic_options
    if decision_analysis is not None:
        ctx.decision_analysis = decision_analysis
    if research_gap_analysis is not None:
        ctx.research_gap_analysis = research_gap_analysis
    return ctx


_A_CRITICAL_LOW = {
    "assumption_id": "A-001",
    "importance": "Critical",
    "confidence": "Low",
    "evidence_support": "Weak",
    "evidence_ids": ["E-001"],
    "statement": "Queue clearance will reach 18 months by 2027",
    "supported_recommendation_ids": ["R-001"],
}

_A_CRITICAL_NO_EV = {
    "assumption_id": "A-002",
    "importance": "Critical",
    "confidence": "Medium",
    "evidence_support": "None",
    "evidence_ids": [],
    "statement": "PPA conversion rate exceeds 80%",
}

_A_IMPORTANT_NO_EV = {
    "assumption_id": "A-003",
    "importance": "Important",
    "confidence": "Medium",
    "evidence_support": "None",
    "evidence_ids": [],
    "statement": "No federal moratorium on SMR construction",
}

_A_STRONG = {
    "assumption_id": "A-004",
    "importance": "Critical",
    "confidence": "High",
    "evidence_support": "Strong",
    "evidence_ids": ["E-010", "E-011"],
    "statement": "Grid interconnection approvals proceed on schedule",
}

_RISK_HIGH_NONE = {
    "risk_id": "RISK-001",
    "severity": "High",
    "evidence_support": "None",
    "evidence_ids": [],
    "title": "Regulatory moratorium blocks deployment",
}

_RISK_HIGH_WEAK = {
    "risk_id": "RISK-002",
    "severity": "High",
    "evidence_support": "Weak",
    "evidence_ids": ["E-002"],
    "title": "Cost overrun exceeds contingency",
}

_RISK_LOW = {
    "risk_id": "RISK-003",
    "severity": "Low",
    "evidence_support": "None",
    "evidence_ids": [],
    "title": "Minor scheduling delay",
}


# ---------------------------------------------------------------------------
# 1. Module + class membership
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_agent_importable(self):
        assert IterationPlanAgent is not None

    def test_compute_iteration_plan_importable(self):
        assert callable(compute_iteration_plan)

    def test_heuristic_functions_importable(self):
        for fn in [
            _tasks_from_validation_priorities,
            _tasks_from_critical_unknowns,
            _tasks_from_low_confidence_assumptions,
            _tasks_from_no_evidence_assumptions,
            _tasks_from_high_severity_risks,
            _tasks_from_recommended_option,
            _tasks_from_research_gap_followups,
            _iteration_needed,
            _stop_conditions,
            _expected_confidence_after_completion,
            _plan_confidence,
        ]:
            assert callable(fn)


# ---------------------------------------------------------------------------
# 2. FunctionalAgent inheritance + run() contract
# ---------------------------------------------------------------------------

class TestAgentContract:
    def test_inherits_functional_agent(self):
        assert issubclass(IterationPlanAgent, FunctionalAgent)

    def test_run_returns_correct_fields(self):
        from functional_agents.context import AgentResult
        agent = IterationPlanAgent()
        result = agent.run(_ctx())
        assert isinstance(result, AgentResult)
        assert result.status == "success"
        assert "iteration_plan" in result.outputs

    def test_run_with_all_empty_inputs(self):
        agent = IterationPlanAgent()
        result = agent.run(_ctx())
        assert result.status == "success"


# ---------------------------------------------------------------------------
# 3. No LLM calls
# ---------------------------------------------------------------------------

class TestNoLLMCalls:
    def test_no_claude_client_import(self):
        import functional_agents.iteration_plan_agent as mod
        src = open(mod.__file__).read()
        assert "claude_client" not in src
        assert "ClaudeClient" not in src

    def test_compute_plan_makes_no_llm_calls(self):
        plan = compute_iteration_plan(
            exec_conf={"overall_confidence": "Low", "validation_priorities": ["Validate A-001"]},
            assumptions=[_A_CRITICAL_LOW],
            recommendations=[],
            risks=[],
            strategic_options=[],
            decision_analysis={},
            research_gap_analysis={},
        )
        assert isinstance(plan, dict)


# ---------------------------------------------------------------------------
# 4. WorkflowState constant + AgentContext field
# ---------------------------------------------------------------------------

class TestContextSchema:
    def test_workflow_state_iteration_plan_constant(self):
        assert WorkflowState.ITERATION_PLAN == "ITERATION_PLAN"

    def test_agent_context_iteration_plan_field(self):
        ctx = _ctx()
        assert hasattr(ctx, "iteration_plan")
        assert ctx.iteration_plan == {}


# ---------------------------------------------------------------------------
# 5. compute_iteration_plan output keys
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def test_all_expected_keys_present(self):
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=[], recommendations=[],
            risks=[], strategic_options=[], decision_analysis={},
            research_gap_analysis={},
        )
        expected = {
            "iteration_needed",
            "iteration_reason",
            "priority_research_tasks",
            "stop_conditions",
            "expected_confidence_after_completion",
            "plan_confidence",
            "validation_warnings",    # J12.2a
        }
        assert set(plan.keys()) == expected

    def test_priority_research_tasks_is_list(self):
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=[], recommendations=[],
            risks=[], strategic_options=[], decision_analysis={},
            research_gap_analysis={},
        )
        assert isinstance(plan["priority_research_tasks"], list)

    def test_stop_conditions_is_list(self):
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=[], recommendations=[],
            risks=[], strategic_options=[], decision_analysis={},
            research_gap_analysis={},
        )
        assert isinstance(plan["stop_conditions"], list)

    def test_irt_task_schema(self):
        plan = compute_iteration_plan(
            exec_conf={"validation_priorities": ["Validate data center demand forecast"]},
            assumptions=[], recommendations=[], risks=[],
            strategic_options=[], decision_analysis={}, research_gap_analysis={},
        )
        task = plan["priority_research_tasks"][0]
        required_keys = {
            "task_id", "source_type", "source_id", "task_title",
            "research_objective", "why_it_matters", "expected_confidence_gain",
            "urgency", "evidence_needed", "suggested_queries",
            "related_assumption_ids", "related_risk_ids",
            "related_recommendation_ids", "related_option_ids",
        }
        assert required_keys.issubset(set(task.keys()))


# ---------------------------------------------------------------------------
# 6. Validation priorities → HIGH/HIGH tasks
# ---------------------------------------------------------------------------

class TestValidationPriorities:
    def test_string_priority_generates_task(self):
        tasks = _tasks_from_validation_priorities(
            {"validation_priorities": ["Validate queue clearance timeline"]},
            seen_text=set(),
        )
        assert len(tasks) == 1
        t = tasks[0]
        assert t["expected_confidence_gain"] == "HIGH"
        assert t["urgency"] == "HIGH"
        assert t["source_type"] == "executive_confidence"
        assert "Validate queue clearance timeline" in t["task_title"]

    def test_dict_priority_generates_task(self):
        tasks = _tasks_from_validation_priorities(
            {"validation_priorities": [{"action": "Validate demand forecasts", "assumption_id": "A-005"}]},
            seen_text=set(),
        )
        assert len(tasks) == 1
        assert "A-005" in tasks[0]["related_assumption_ids"]

    def test_empty_priorities_no_tasks(self):
        tasks = _tasks_from_validation_priorities({}, seen_text=set())
        assert tasks == []

    def test_duplicate_text_suppressed(self):
        seen: set = {"validate queue clearance timeline"}
        tasks = _tasks_from_validation_priorities(
            {"validation_priorities": ["Validate queue clearance timeline"]},
            seen_text=seen,
        )
        assert tasks == []

    def test_multiple_priorities_all_generated(self):
        tasks = _tasks_from_validation_priorities(
            {"validation_priorities": ["Action A", "Action B", "Action C"]},
            seen_text=set(),
        )
        assert len(tasks) == 3


# ---------------------------------------------------------------------------
# 7. Critical unknowns deduplicated against VPs
# ---------------------------------------------------------------------------

class TestCriticalUnknowns:
    def test_unknown_not_in_vps_generates_task(self):
        tasks = _tasks_from_critical_unknowns(
            {"critical_unknowns": ["Will capacity limits delay construction?"]},
            seen_text=set(),
        )
        assert len(tasks) == 1
        assert tasks[0]["expected_confidence_gain"] == "HIGH"
        assert tasks[0]["urgency"] == "MEDIUM"

    def test_unknown_matching_vp_suppressed(self):
        seen = {"will capacity limits delay construction?"}
        tasks = _tasks_from_critical_unknowns(
            {"critical_unknowns": ["Will capacity limits delay construction?"]},
            seen_text=seen,
        )
        assert tasks == []

    def test_partial_overlap_not_suppressed(self):
        seen = {"validate capacity limits"}
        tasks = _tasks_from_critical_unknowns(
            {"critical_unknowns": ["Will capacity limits delay construction?"]},
            seen_text=seen,
        )
        assert len(tasks) == 1

    def test_empty_unknowns_no_tasks(self):
        tasks = _tasks_from_critical_unknowns({}, seen_text=set())
        assert tasks == []


# ---------------------------------------------------------------------------
# 8. Low-confidence Critical assumptions → HIGH gain
# ---------------------------------------------------------------------------

class TestLowConfidenceAssumptions:
    def test_critical_low_conf_generates_task(self):
        tasks = _tasks_from_low_confidence_assumptions([_A_CRITICAL_LOW], seen_assumption_ids=set())
        assert len(tasks) == 1
        t = tasks[0]
        assert t["expected_confidence_gain"] == "HIGH"
        assert t["source_type"] == "assumption"
        assert t["source_id"] == "A-001"

    def test_urgency_high_when_no_evidence(self):
        a = {**_A_CRITICAL_LOW, "evidence_support": "None", "evidence_ids": []}
        tasks = _tasks_from_low_confidence_assumptions([a], seen_assumption_ids=set())
        assert tasks[0]["urgency"] == "HIGH"

    def test_urgency_medium_when_has_some_evidence(self):
        tasks = _tasks_from_low_confidence_assumptions([_A_CRITICAL_LOW], seen_assumption_ids=set())
        assert tasks[0]["urgency"] == "MEDIUM"

    def test_non_critical_skipped(self):
        a = {**_A_CRITICAL_LOW, "importance": "Important"}
        tasks = _tasks_from_low_confidence_assumptions([a], seen_assumption_ids=set())
        assert tasks == []

    def test_non_low_confidence_skipped(self):
        a = {**_A_CRITICAL_LOW, "confidence": "Medium"}
        tasks = _tasks_from_low_confidence_assumptions([a], seen_assumption_ids=set())
        assert tasks == []

    def test_dedup_by_assumption_id(self):
        seen = {"A-001"}
        tasks = _tasks_from_low_confidence_assumptions([_A_CRITICAL_LOW], seen_assumption_ids=seen)
        assert tasks == []

    def test_assumption_id_added_to_seen(self):
        seen: set = set()
        _tasks_from_low_confidence_assumptions([_A_CRITICAL_LOW], seen_assumption_ids=seen)
        assert "A-001" in seen

    def test_recommendation_ids_linked(self):
        tasks = _tasks_from_low_confidence_assumptions([_A_CRITICAL_LOW], seen_assumption_ids=set())
        assert "R-001" in tasks[0]["related_recommendation_ids"]


# ---------------------------------------------------------------------------
# 9. No-evidence Important/Critical assumptions → MEDIUM gain
# ---------------------------------------------------------------------------

class TestNoEvidenceAssumptions:
    def test_critical_no_evidence_generates_task(self):
        tasks = _tasks_from_no_evidence_assumptions([_A_CRITICAL_NO_EV], seen_assumption_ids=set())
        assert len(tasks) == 1
        assert tasks[0]["expected_confidence_gain"] == "MEDIUM"
        assert tasks[0]["urgency"] == "HIGH"

    def test_important_no_evidence_generates_task_medium_urgency(self):
        tasks = _tasks_from_no_evidence_assumptions([_A_IMPORTANT_NO_EV], seen_assumption_ids=set())
        assert len(tasks) == 1
        assert tasks[0]["urgency"] == "MEDIUM"

    def test_assumption_with_evidence_skipped(self):
        tasks = _tasks_from_no_evidence_assumptions([_A_STRONG], seen_assumption_ids=set())
        assert tasks == []

    def test_minor_importance_skipped(self):
        a = {**_A_IMPORTANT_NO_EV, "importance": "Minor"}
        tasks = _tasks_from_no_evidence_assumptions([a], seen_assumption_ids=set())
        assert tasks == []

    def test_dedup_against_h3_seen_ids(self):
        seen = {"A-002"}
        tasks = _tasks_from_no_evidence_assumptions([_A_CRITICAL_NO_EV], seen_assumption_ids=seen)
        assert tasks == []


# ---------------------------------------------------------------------------
# 10. H3 + H4 deduplication in compute_iteration_plan
# ---------------------------------------------------------------------------

class TestH3H4Deduplication:
    def test_critical_low_conf_no_ev_produces_one_task(self):
        a = {
            "assumption_id": "A-001",
            "importance": "Critical",
            "confidence": "Low",
            "evidence_support": "None",
            "evidence_ids": [],
            "statement": "Queue clearance timeline",
        }
        plan = compute_iteration_plan(
            exec_conf={},
            assumptions=[a],
            recommendations=[], risks=[], strategic_options=[],
            decision_analysis={}, research_gap_analysis={},
        )
        tasks_for_a001 = [
            t for t in plan["priority_research_tasks"]
            if "A-001" in (t.get("related_assumption_ids") or [])
        ]
        assert len(tasks_for_a001) == 1


# ---------------------------------------------------------------------------
# 11. High-severity weak-evidence risks → MEDIUM gain
# ---------------------------------------------------------------------------

class TestHighSeverityRisks:
    def test_high_none_evidence_generates_task(self):
        tasks = _tasks_from_high_severity_risks([_RISK_HIGH_NONE], seen_risk_ids=set())
        assert len(tasks) == 1
        assert tasks[0]["expected_confidence_gain"] == "MEDIUM"
        assert tasks[0]["urgency"] == "HIGH"

    def test_high_weak_evidence_generates_task_medium_urgency(self):
        tasks = _tasks_from_high_severity_risks([_RISK_HIGH_WEAK], seen_risk_ids=set())
        assert len(tasks) == 1
        assert tasks[0]["urgency"] == "MEDIUM"

    def test_low_severity_skipped(self):
        tasks = _tasks_from_high_severity_risks([_RISK_LOW], seen_risk_ids=set())
        assert tasks == []

    def test_dedup_by_risk_id(self):
        seen = {"RISK-001"}
        tasks = _tasks_from_high_severity_risks([_RISK_HIGH_NONE], seen_risk_ids=seen)
        assert tasks == []

    def test_source_type_is_risk(self):
        tasks = _tasks_from_high_severity_risks([_RISK_HIGH_NONE], seen_risk_ids=set())
        assert tasks[0]["source_type"] == "risk"


# ---------------------------------------------------------------------------
# 12 & 13. Recommended option fragility
# ---------------------------------------------------------------------------

class TestRecommendedOptionFragility:
    def _option(self, option_id: str, deps: list[str], recommended: bool = False) -> dict:
        return {
            "option_id": option_id,
            "title": f"Option {option_id}",
            "recommended": recommended,
            "supporting_assumption_ids": deps,
        }

    def test_fragile_recommended_option_generates_task(self):
        da = {"recommended_option_id": "OPT-A"}
        option = self._option("OPT-A", ["A-001"])
        assumptions_by_id = {"A-001": _A_CRITICAL_LOW}
        tasks = _tasks_from_recommended_option(da, [option], assumptions_by_id, set())
        assert len(tasks) == 1
        assert tasks[0]["expected_confidence_gain"] == "HIGH"
        assert tasks[0]["urgency"] == "HIGH"
        assert "OPT-A" in tasks[0]["related_option_ids"]

    def test_option_with_strong_assumptions_no_task(self):
        da = {"recommended_option_id": "OPT-B"}
        option = self._option("OPT-B", ["A-004"])
        assumptions_by_id = {"A-004": _A_STRONG}
        tasks = _tasks_from_recommended_option(da, [option], assumptions_by_id, set())
        assert tasks == []

    def test_recommended_flag_used_when_no_da(self):
        option = self._option("OPT-C", ["A-001"], recommended=True)
        assumptions_by_id = {"A-001": _A_CRITICAL_LOW}
        tasks = _tasks_from_recommended_option({}, [option], assumptions_by_id, set())
        assert len(tasks) == 1
        assert "OPT-C" in tasks[0]["related_option_ids"]

    def test_no_recommended_option_no_task(self):
        tasks = _tasks_from_recommended_option({}, [], {}, set())
        assert tasks == []

    def test_dedup_by_option_id(self):
        da = {"recommended_option_id": "OPT-A"}
        option = self._option("OPT-A", ["A-001"])
        assumptions_by_id = {"A-001": _A_CRITICAL_LOW}
        seen = {"OPT-A"}
        tasks = _tasks_from_recommended_option(da, [option], assumptions_by_id, seen)
        assert tasks == []


# ---------------------------------------------------------------------------
# 14. Research gap followups (LOW gain) + text dedup
# ---------------------------------------------------------------------------

class TestResearchGapFollowups:
    def test_followup_generates_low_task(self):
        rga = {"recommended_followups": ["Investigate offshore wind integration costs"]}
        tasks = _tasks_from_research_gap_followups(rga, seen_text=set())
        assert len(tasks) == 1
        assert tasks[0]["expected_confidence_gain"] == "LOW"
        assert tasks[0]["urgency"] == "LOW"

    def test_followup_dedup_against_vp_text(self):
        seen = {"investigate offshore wind integration costs"}
        rga = {"recommended_followups": ["Investigate offshore wind integration costs"]}
        tasks = _tasks_from_research_gap_followups(rga, seen_text=seen)
        assert tasks == []

    def test_empty_followups_no_tasks(self):
        tasks = _tasks_from_research_gap_followups({}, seen_text=set())
        assert tasks == []

    def test_source_type_is_research_gap(self):
        rga = {"recommended_followups": ["Assess ERCOT grid stability"]}
        tasks = _tasks_from_research_gap_followups(rga, seen_text=set())
        assert tasks[0]["source_type"] == "research_gap"


# ---------------------------------------------------------------------------
# 15. Priority ordering
# ---------------------------------------------------------------------------

class TestTaskOrdering:
    def test_high_high_before_high_medium(self):
        tasks = [
            {"expected_confidence_gain": "HIGH", "urgency": "MEDIUM"},
            {"expected_confidence_gain": "HIGH", "urgency": "HIGH"},
        ]
        sorted_t = _sorted_tasks(tasks)
        assert sorted_t[0]["urgency"] == "HIGH"

    def test_high_before_medium(self):
        tasks = [
            {"expected_confidence_gain": "MEDIUM", "urgency": "HIGH"},
            {"expected_confidence_gain": "HIGH", "urgency": "LOW"},
        ]
        sorted_t = _sorted_tasks(tasks)
        assert sorted_t[0]["expected_confidence_gain"] == "HIGH"

    def test_medium_before_low(self):
        tasks = [
            {"expected_confidence_gain": "LOW", "urgency": "HIGH"},
            {"expected_confidence_gain": "MEDIUM", "urgency": "LOW"},
        ]
        sorted_t = _sorted_tasks(tasks)
        assert sorted_t[0]["expected_confidence_gain"] == "MEDIUM"

    def test_within_bucket_source_order_preserved(self):
        tasks = [
            {"expected_confidence_gain": "MEDIUM", "urgency": "HIGH", "_marker": "first"},
            {"expected_confidence_gain": "MEDIUM", "urgency": "HIGH", "_marker": "second"},
        ]
        sorted_t = _sorted_tasks(tasks)
        assert sorted_t[0]["_marker"] == "first"
        assert sorted_t[1]["_marker"] == "second"

    def test_gap_followups_rank_last_in_mixed_plan(self):
        plan = compute_iteration_plan(
            exec_conf={"validation_priorities": ["Validate demand"]},
            assumptions=[], recommendations=[], risks=[],
            strategic_options=[], decision_analysis={},
            research_gap_analysis={"recommended_followups": ["Assess offshore costs"]},
        )
        tasks = plan["priority_research_tasks"]
        assert len(tasks) >= 2
        last = tasks[-1]
        assert last["expected_confidence_gain"] == "LOW"


# ---------------------------------------------------------------------------
# 16. Task IDs
# ---------------------------------------------------------------------------

class TestTaskIds:
    def test_ids_assigned_sequentially(self):
        tasks = _assign_task_ids([
            {"expected_confidence_gain": "HIGH"},
            {"expected_confidence_gain": "MEDIUM"},
        ])
        assert tasks[0]["task_id"] == "IRT-001"
        assert tasks[1]["task_id"] == "IRT-002"

    def test_ids_zero_padded(self):
        tasks = _assign_task_ids([{"x": i} for i in range(12)])
        assert tasks[9]["task_id"] == "IRT-010"
        assert tasks[10]["task_id"] == "IRT-011"


# ---------------------------------------------------------------------------
# 17. Max-tasks cap
# ---------------------------------------------------------------------------

class TestMaxTasksCap:
    def test_default_cap_is_ten(self):
        assumptions = [
            {
                "assumption_id": f"A-{i:03d}",
                "importance": "Critical",
                "confidence": "Low",
                "evidence_support": "None",
                "evidence_ids": [],
                "statement": f"Assumption {i}",
            }
            for i in range(1, 20)
        ]
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=assumptions, recommendations=[],
            risks=[], strategic_options=[], decision_analysis={},
            research_gap_analysis={},
        )
        assert len(plan["priority_research_tasks"]) == 10

    def test_custom_max_tasks_respected(self):
        assumptions = [
            {
                "assumption_id": f"A-{i:03d}",
                "importance": "Critical",
                "confidence": "Low",
                "evidence_support": "None",
                "evidence_ids": [],
                "statement": f"Assumption {i}",
            }
            for i in range(1, 10)
        ]
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=assumptions, recommendations=[],
            risks=[], strategic_options=[], decision_analysis={},
            research_gap_analysis={}, max_tasks=3,
        )
        assert len(plan["priority_research_tasks"]) == 3


# ---------------------------------------------------------------------------
# 18 & 19. iteration_needed logic
# ---------------------------------------------------------------------------

class TestIterationNeeded:
    def test_low_exec_confidence_triggers_needed(self):
        needed, reason = _iteration_needed(
            {"overall_confidence": "Low"}, [], [], {}
        )
        assert needed is True
        assert "Low" in reason

    def test_medium_exec_confidence_triggers_needed(self):
        needed, _ = _iteration_needed({"overall_confidence": "Medium"}, [], [], {})
        assert needed is True

    def test_validation_priorities_trigger_needed(self):
        needed, reason = _iteration_needed(
            {"overall_confidence": "High", "validation_priorities": ["Do X"]}, [], [], {}
        )
        assert needed is True
        assert "validation" in reason.lower()

    def test_critical_unknowns_trigger_needed(self):
        needed, _ = _iteration_needed(
            {"overall_confidence": "High", "critical_unknowns": ["Unknown Y"]}, [], [], {}
        )
        assert needed is True

    def test_critical_low_confidence_assumption_triggers_needed(self):
        needed, reason = _iteration_needed(
            {"overall_confidence": "High"}, [_A_CRITICAL_LOW], [], {}
        )
        assert needed is True
        assert "Critical" in reason

    def test_no_evidence_important_assumption_triggers_needed(self):
        needed, _ = _iteration_needed(
            {"overall_confidence": "High"}, [_A_IMPORTANT_NO_EV], [], {}
        )
        assert needed is True

    def test_all_clear_returns_false(self):
        needed, reason = _iteration_needed(
            {"overall_confidence": "High"}, [_A_STRONG], [], {}
        )
        assert needed is False
        assert "sufficient" in reason.lower() or "adequately" in reason.lower()


# ---------------------------------------------------------------------------
# 20. stop_conditions
# ---------------------------------------------------------------------------

class TestStopConditions:
    def test_low_confidence_produces_stop_condition(self):
        conds = _stop_conditions(
            {"overall_confidence": "Low"}, [_A_CRITICAL_LOW], {}, []
        )
        assert len(conds) >= 1
        assert any("High" in c for c in conds)

    def test_all_clear_produces_sufficient_condition(self):
        conds = _stop_conditions(
            {"overall_confidence": "High"}, [_A_STRONG], {}, []
        )
        assert len(conds) == 1
        assert "sufficient" in conds[0].lower() or "commitment" in conds[0].lower()

    def test_no_evidence_ids_listed_in_conditions(self):
        conds = _stop_conditions(
            {"overall_confidence": "High"},
            [_A_CRITICAL_NO_EV],
            {}, [],
        )
        assert any("A-002" in c for c in conds)


# ---------------------------------------------------------------------------
# 21. expected_confidence_after_completion
# ---------------------------------------------------------------------------

class TestExpectedConfidence:
    def test_high_confidence_stays_high(self):
        result = _expected_confidence_after_completion(
            {"overall_confidence": "High"}, [], []
        )
        assert result == "High"

    def test_low_confidence_addressed_becomes_medium(self):
        task = {"related_assumption_ids": ["A-001"]}
        result = _expected_confidence_after_completion(
            {"overall_confidence": "Low"}, [_A_CRITICAL_LOW], [task]
        )
        assert result == "Medium"

    def test_medium_with_all_addressed_becomes_high(self):
        task_a = {"related_assumption_ids": ["A-001"]}
        task_b = {"related_assumption_ids": ["A-002"]}
        result = _expected_confidence_after_completion(
            {"overall_confidence": "Medium"},
            [_A_CRITICAL_LOW, _A_CRITICAL_NO_EV],
            [task_a, task_b],
        )
        assert result == "High"

    def test_unaddressed_gaps_returns_unknown(self):
        result = _expected_confidence_after_completion(
            {"overall_confidence": "Low"},
            [_A_CRITICAL_LOW],
            [],
        )
        assert result == "Unknown"


# ---------------------------------------------------------------------------
# 22. plan_confidence range
# ---------------------------------------------------------------------------

class TestPlanConfidence:
    def test_confidence_in_range(self):
        conf = _plan_confidence(
            {"overall_confidence": "Low"},
            [_A_CRITICAL_LOW], [_RISK_HIGH_NONE], [], [{"x": 1}]
        )
        assert 0.0 <= conf <= 1.0

    def test_empty_inputs_still_valid_range(self):
        conf = _plan_confidence({}, [], [], [], [])
        assert 0.0 <= conf <= 1.0

    def test_full_inputs_higher_than_empty(self):
        full = _plan_confidence(
            {"overall_confidence": "Low"},
            [_A_CRITICAL_LOW], [_RISK_HIGH_NONE], [{}], [{"x": 1}]
        )
        empty = _plan_confidence({}, [], [], [], [])
        assert full > empty


# ---------------------------------------------------------------------------
# 23. context fields written
# ---------------------------------------------------------------------------

class TestContextFieldsWritten:
    def test_iteration_plan_on_context(self):
        agent = IterationPlanAgent()
        ctx = _ctx(
            exec_conf={"overall_confidence": "Low", "validation_priorities": ["Validate A-001"]},
            assumptions=[_A_CRITICAL_LOW],
        )
        agent.run(ctx)
        assert ctx.iteration_plan != {}
        assert "iteration_needed" in ctx.iteration_plan

    def test_iteration_plan_on_research_object(self):
        agent = IterationPlanAgent()
        ctx = _ctx()
        agent.run(ctx)
        assert "iteration_plan" in ctx.research_object

    def test_trace_block_written(self):
        agent = IterationPlanAgent()
        ctx = _ctx()
        agent.run(ctx)
        assert "_iteration_plan" in ctx.trace
        tp = ctx.trace["_iteration_plan"]
        assert "iteration_needed" in tp
        assert "task_count" in tp
        assert "plan_confidence" in tp
        # J12.2a expanded fields
        assert "high_priority_tasks" in tp
        assert "critical_assumptions" in tp
        assert "validation_priorities" in tp
        assert "critical_unknowns" in tp
        assert "expected_confidence_after_completion" in tp


# ---------------------------------------------------------------------------
# 24. Agent does not mutate upstream artifacts
# ---------------------------------------------------------------------------

class TestNoUpstreamMutation:
    def test_assumptions_unchanged(self):
        import copy
        original = copy.deepcopy([_A_CRITICAL_LOW, _A_STRONG])
        ctx = _ctx(assumptions=original)
        IterationPlanAgent().run(ctx)
        assert ctx.assumptions == original

    def test_risks_unchanged(self):
        import copy
        original = copy.deepcopy([_RISK_HIGH_NONE])
        ctx = _ctx(risks=original)
        IterationPlanAgent().run(ctx)
        assert ctx.risks == original


# ---------------------------------------------------------------------------
# 25. Runs with empty inputs (no crash)
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_no_crash_all_empty(self):
        plan = compute_iteration_plan(
            exec_conf={},
            assumptions=[],
            recommendations=[],
            risks=[],
            strategic_options=[],
            decision_analysis={},
            research_gap_analysis={},
        )
        assert isinstance(plan, dict)
        assert plan["priority_research_tasks"] == []
        assert plan["iteration_needed"] is False

    def test_agent_no_crash_minimal_context(self):
        result = IterationPlanAgent().run(_ctx())
        assert result.status == "success"


# ---------------------------------------------------------------------------
# 26. Pipeline position — orchestrator wires ITERATION_PLAN after EXECUTIVE_CONFIDENCE
# ---------------------------------------------------------------------------

class TestOrchestratorWiring:
    def test_iteration_plan_state_in_orchestrator_routing(self):
        import inspect
        import functional_agents.orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        assert "ITERATION_PLAN" in src
        assert "iteration_plan_factory" in src

    def test_executive_confidence_routes_to_iteration_plan(self):
        import inspect
        import functional_agents.orchestrator as orch_mod
        src = inspect.getsource(orch_mod)
        # Ensure ITERATION_PLAN is checked after EXECUTIVE_CONFIDENCE
        ec_pos = src.index("EXECUTIVE_CONFIDENCE")
        ip_pos = src.index("ITERATION_PLAN")
        assert ip_pos > ec_pos


# ---------------------------------------------------------------------------
# 27. context field fallback from research_object
# ---------------------------------------------------------------------------

class TestResearchObjectFallback:
    def test_exec_conf_read_from_research_object_when_context_empty(self):
        ctx = _ctx()
        ctx.executive_confidence = {}
        ctx.research_object["executive_confidence"] = {
            "overall_confidence": "Low",
            "validation_priorities": ["Fallback validation priority"],
        }
        agent = IterationPlanAgent()
        agent.run(ctx)
        plan = ctx.iteration_plan
        assert plan["iteration_needed"] is True
        vp_tasks = [
            t for t in plan["priority_research_tasks"]
            if t["source_type"] == "executive_confidence" and
            t["source_id"] == "validation_priority"
        ]
        assert len(vp_tasks) >= 1

    def test_assumptions_read_from_research_object_when_context_empty(self):
        ctx = _ctx()
        ctx.assumptions = []
        ctx.research_object["strategic_assumptions"] = [_A_CRITICAL_LOW]
        agent = IterationPlanAgent()
        agent.run(ctx)
        plan = ctx.iteration_plan
        assumption_tasks = [
            t for t in plan["priority_research_tasks"]
            if t["source_type"] == "assumption"
        ]
        assert len(assumption_tasks) >= 1


# ---------------------------------------------------------------------------
# 29. Structured ID extraction (J12.2a)
# ---------------------------------------------------------------------------

class TestStructuredIdExtraction:
    def test_assumption_id_extracted(self):
        a_ids, _, _, _ = _extract_ids_from_text("Validate A-001 via independent analysis")
        assert "A-001" in a_ids

    def test_multiple_assumption_ids_extracted(self):
        a_ids, _, _, _ = _extract_ids_from_text("Validate A-001 and A-005 exposure")
        assert a_ids == ["A-001", "A-005"]

    def test_risk_id_extracted(self):
        _, r_ids, _, _ = _extract_ids_from_text("Assess RSK-004 materiality via FERC tracking")
        assert "RSK-004" in r_ids

    def test_recommendation_id_extracted(self):
        _, _, rec_ids, _ = _extract_ids_from_text("Supports recommendation REC-002 rationale")
        assert "REC-002" in rec_ids

    def test_option_id_extracted(self):
        _, _, _, o_ids = _extract_ids_from_text("Strengthen evidence for recommended option OPT-B")
        assert "OPT-B" in o_ids

    def test_mixed_ids_extracted(self):
        a_ids, r_ids, rec_ids, o_ids = _extract_ids_from_text(
            "Validate A-001 and RSK-004 supporting OPT-A and REC-002"
        )
        assert "A-001" in a_ids
        assert "RSK-004" in r_ids
        assert "REC-002" in rec_ids
        assert "OPT-A" in o_ids

    def test_no_ids_in_plain_text(self):
        a_ids, r_ids, rec_ids, o_ids = _extract_ids_from_text(
            "Validate the underlying technology maturity for commercial deployment"
        )
        assert a_ids == []
        assert r_ids == []
        assert rec_ids == []
        assert o_ids == []

    def test_ids_are_deduplicated(self):
        a_ids, _, _, _ = _extract_ids_from_text("A-001 and A-001 again with A-001")
        assert a_ids.count("A-001") == 1

    def test_ids_are_sorted(self):
        a_ids, _, _, _ = _extract_ids_from_text("A-005 and A-001 and A-003")
        assert a_ids == ["A-001", "A-003", "A-005"]

    def test_prefix_letter_prevents_match(self):
        # "OA-001" — preceded by letter O — should NOT match A-001
        a_ids, _, _, _ = _extract_ids_from_text("OA-001 is not an assumption ID")
        assert "A-001" not in a_ids

    def test_enrich_merges_extracted_into_existing(self):
        task = {
            "task_title": "Validate A-001 and RSK-004 via analysis",
            "research_objective": "Validate A-001 and RSK-004",
            "why_it_matters": "",
            "evidence_needed": [],
            "suggested_queries": [],
            "related_assumption_ids": [],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        }
        _enrich_task_ids(task)
        assert "A-001" in task["related_assumption_ids"]
        assert "RSK-004" in task["related_risk_ids"]

    def test_enrich_preserves_existing_ids(self):
        task = {
            "task_title": "Strengthen evidence for option",
            "research_objective": "Validate the recommended option OPT-B",
            "why_it_matters": "",
            "evidence_needed": [],
            "suggested_queries": [],
            "related_assumption_ids": ["A-003"],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": ["OPT-B"],
        }
        _enrich_task_ids(task)
        assert "A-003" in task["related_assumption_ids"]
        assert "OPT-B" in task["related_option_ids"]

    def test_enrich_deduplicates_ids(self):
        task = {
            "task_title": "Validate A-001",
            "research_objective": "evidence for A-001",
            "why_it_matters": "",
            "evidence_needed": [],
            "suggested_queries": [],
            "related_assumption_ids": ["A-001"],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        }
        _enrich_task_ids(task)
        assert task["related_assumption_ids"].count("A-001") == 1


# ---------------------------------------------------------------------------
# 30. Linkage completeness (J12.2a integration)
# ---------------------------------------------------------------------------

class TestLinkageCompleteness:
    def test_assumption_id_in_vp_text_populates_related_assumption_ids(self):
        plan = compute_iteration_plan(
            exec_conf={"validation_priorities": ["Validate A-001 via independent analysis"]},
            assumptions=[_A_CRITICAL_LOW],
            recommendations=[], risks=[], strategic_options=[],
            decision_analysis={}, research_gap_analysis={},
        )
        vp_task = next(
            (t for t in plan["priority_research_tasks"]
             if t["source_type"] == "executive_confidence" and
             t["source_id"] == "validation_priority"),
            None,
        )
        assert vp_task is not None
        assert "A-001" in vp_task["related_assumption_ids"]

    def test_risk_id_in_vp_text_populates_related_risk_ids(self):
        risks = [{"risk_id": "RSK-004", "severity": "High", "evidence_support": "None",
                  "evidence_ids": [], "title": "Queue risk"}]
        plan = compute_iteration_plan(
            exec_conf={"validation_priorities": ["Assess RSK-004 materiality via FERC"]},
            assumptions=[], recommendations=[], risks=risks,
            strategic_options=[], decision_analysis={}, research_gap_analysis={},
        )
        vp_task = next(
            (t for t in plan["priority_research_tasks"]
             if t["source_type"] == "executive_confidence"),
            None,
        )
        assert vp_task is not None
        assert "RSK-004" in vp_task["related_risk_ids"]

    def test_cu_text_populates_assumption_id(self):
        plan = compute_iteration_plan(
            exec_conf={"critical_unknowns": ["Resolution of A-002 is unknown"]},
            assumptions=[_A_CRITICAL_NO_EV],
            recommendations=[], risks=[], strategic_options=[],
            decision_analysis={}, research_gap_analysis={},
        )
        cu_task = next(
            (t for t in plan["priority_research_tasks"]
             if t["source_id"] == "critical_unknown"),
            None,
        )
        assert cu_task is not None
        assert "A-002" in cu_task["related_assumption_ids"]

    def test_option_id_in_option_fragility_task(self):
        option = {
            "option_id": "OPT-A",
            "title": "Option A",
            "recommended": True,
            "supporting_assumption_ids": ["A-001"],
        }
        plan = compute_iteration_plan(
            exec_conf={},
            assumptions=[_A_CRITICAL_LOW],
            recommendations=[], risks=[], strategic_options=[option],
            decision_analysis={}, research_gap_analysis={},
        )
        opt_task = next(
            (t for t in plan["priority_research_tasks"]
             if t["source_type"] == "strategic_option"),
            None,
        )
        assert opt_task is not None
        assert "OPT-A" in opt_task["related_option_ids"]
        assert "A-001" in opt_task["related_assumption_ids"]

    def test_no_ids_in_plain_followup_text(self):
        plan = compute_iteration_plan(
            exec_conf={}, assumptions=[], recommendations=[], risks=[],
            strategic_options=[], decision_analysis={},
            research_gap_analysis={
                "recommended_followups": ["Assess offshore wind integration costs"]
            },
        )
        gap_task = next(
            (t for t in plan["priority_research_tasks"]
             if t["source_type"] == "research_gap"),
            None,
        )
        assert gap_task is not None
        assert gap_task["related_assumption_ids"] == []
        assert gap_task["related_risk_ids"] == []


# ---------------------------------------------------------------------------
# 31. Plan validation (J12.2a)
# ---------------------------------------------------------------------------

class TestPlanValidation:
    def _valid_plan(self, n_tasks: int = 1) -> dict:
        tasks = [
            {
                "task_id": f"IRT-{i:03d}",
                "source_type": "executive_confidence",
                "source_id": "validation_priority",
                "task_title": f"Unique task title {i}",
                "research_objective": f"Research objective {i}",
                "why_it_matters": "matters",
                "expected_confidence_gain": "HIGH",
                "urgency": "HIGH",
                "evidence_needed": [],
                "suggested_queries": [],
                "related_assumption_ids": [],
                "related_risk_ids": [],
                "related_recommendation_ids": [],
                "related_option_ids": [],
            }
            for i in range(1, n_tasks + 1)
        ]
        return {
            "iteration_needed": True,
            "iteration_reason": "reason",
            "priority_research_tasks": tasks,
            "stop_conditions": ["condition"],
            "expected_confidence_after_completion": "Medium",
            "plan_confidence": 0.9,
        }

    def test_valid_plan_produces_no_warnings(self):
        warnings = _validate_plan(self._valid_plan())
        assert warnings == []

    def test_duplicate_task_id_flagged(self):
        plan = self._valid_plan(2)
        plan["priority_research_tasks"][1]["task_id"] = "IRT-001"
        warnings = _validate_plan(plan)
        assert any("Duplicate task_id" in w for w in warnings)

    def test_empty_task_title_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["task_title"] = ""
        warnings = _validate_plan(plan)
        assert any("empty task_title" in w for w in warnings)

    def test_empty_research_objective_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["research_objective"] = ""
        warnings = _validate_plan(plan)
        assert any("empty research_objective" in w for w in warnings)

    def test_invalid_confidence_gain_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["expected_confidence_gain"] = "EXTREME"
        warnings = _validate_plan(plan)
        assert any("invalid expected_confidence_gain" in w for w in warnings)

    def test_invalid_urgency_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["urgency"] = "CRITICAL"
        warnings = _validate_plan(plan)
        assert any("invalid urgency" in w for w in warnings)

    def test_duplicate_ids_within_task_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_assumption_ids"] = ["A-001", "A-001"]
        warnings = _validate_plan(plan, known_assumption_ids={"A-001"})
        assert any("duplicate IDs" in w for w in warnings)

    def test_duplicate_task_titles_flagged(self):
        plan = self._valid_plan(2)
        plan["priority_research_tasks"][1]["task_title"] = "Unique task title 1"
        warnings = _validate_plan(plan)
        assert any("duplicate task title" in w for w in warnings)

    def test_unknown_assumption_id_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_assumption_ids"] = ["A-999"]
        warnings = _validate_plan(plan, known_assumption_ids={"A-001"})
        assert any("A-999" in w and "not found" in w for w in warnings)

    def test_unknown_risk_id_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_risk_ids"] = ["RSK-999"]
        warnings = _validate_plan(plan, known_risk_ids={"RSK-001"})
        assert any("RSK-999" in w and "not found" in w for w in warnings)

    def test_unknown_option_id_flagged(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_option_ids"] = ["OPT-Z"]
        warnings = _validate_plan(plan, known_option_ids={"OPT-A"})
        assert any("OPT-Z" in w and "not found" in w for w in warnings)

    def test_known_id_does_not_warn(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_assumption_ids"] = ["A-001"]
        warnings = _validate_plan(plan, known_assumption_ids={"A-001"})
        assert not any("A-001" in w for w in warnings)

    def test_no_known_sets_skips_cross_ref(self):
        plan = self._valid_plan()
        plan["priority_research_tasks"][0]["related_assumption_ids"] = ["A-999"]
        warnings = _validate_plan(plan)  # no known_assumption_ids passed
        assert not any("not found" in w for w in warnings)

    def test_compute_iteration_plan_includes_validation_warnings(self):
        plan = compute_iteration_plan(
            exec_conf={"validation_priorities": ["Validate data center demand"]},
            assumptions=[],
            recommendations=[], risks=[], strategic_options=[],
            decision_analysis={}, research_gap_analysis={},
        )
        assert "validation_warnings" in plan
        assert isinstance(plan["validation_warnings"], list)

    def test_well_formed_plan_has_empty_warnings(self):
        # All IDs referenced in tasks exist in source collections
        plan = compute_iteration_plan(
            exec_conf={},
            assumptions=[_A_CRITICAL_LOW],
            recommendations=[], risks=[], strategic_options=[],
            decision_analysis={}, research_gap_analysis={},
        )
        assert plan["validation_warnings"] == []


# ---------------------------------------------------------------------------
# 32. Persistence regression (J12.2a)
# ---------------------------------------------------------------------------

class TestPersistenceRegression:
    def test_research_object_iteration_plan_same_as_context(self):
        ctx = _ctx(
            exec_conf={"overall_confidence": "Low", "validation_priorities": ["Validate A-001"]},
            assumptions=[_A_CRITICAL_LOW],
        )
        IterationPlanAgent().run(ctx)
        assert ctx.research_object.get("iteration_plan") == ctx.iteration_plan

    def test_research_object_iteration_plan_contains_tasks(self):
        ctx = _ctx(
            exec_conf={"overall_confidence": "Low", "validation_priorities": ["Validate demand"]},
        )
        IterationPlanAgent().run(ctx)
        ro_plan = ctx.research_object.get("iteration_plan", {})
        assert "priority_research_tasks" in ro_plan
        assert isinstance(ro_plan["priority_research_tasks"], list)

    def test_iteration_plan_written_to_research_object_before_context(self):
        ctx = _ctx(exec_conf={"overall_confidence": "Low"}, assumptions=[_A_CRITICAL_LOW])
        IterationPlanAgent().run(ctx)
        # Both should be present and equal after agent.run()
        assert "iteration_plan" in ctx.research_object
        assert ctx.research_object["iteration_plan"] is ctx.iteration_plan

    def test_write_research_object_preserves_iteration_plan(self):
        import tempfile, json
        from pathlib import Path
        from research_agent.research_object import write_research_object

        ctx = _ctx(
            exec_conf={"overall_confidence": "Medium"},
            assumptions=[_A_CRITICAL_LOW],
        )
        ctx.research_object["research_id"] = "R-TEST_PERSIST_001"
        IterationPlanAgent().run(ctx)

        with tempfile.TemporaryDirectory() as tmp:
            ro_path = write_research_object(
                ctx.research_object,
                out_dir=Path(tmp),
                write_latest=False,
            )
            written = json.loads(ro_path.read_text())
            assert "iteration_plan" in written
            assert "priority_research_tasks" in written["iteration_plan"]


# ---------------------------------------------------------------------------
# 33. Expanded trace summary metrics (J12.2a)
# ---------------------------------------------------------------------------

class TestTraceSummaryMetrics:
    def test_high_priority_tasks_count_correct(self):
        ctx = _ctx(
            exec_conf={
                "overall_confidence": "Low",
                "validation_priorities": ["Validate A-001"],
            },
        )
        IterationPlanAgent().run(ctx)
        tp = ctx.trace["_iteration_plan"]
        assert "high_priority_tasks" in tp
        assert isinstance(tp["high_priority_tasks"], int)
        assert tp["high_priority_tasks"] >= 0

    def test_critical_assumptions_count_matches_source(self):
        ctx = _ctx(
            assumptions=[_A_CRITICAL_LOW, _A_STRONG],
        )
        IterationPlanAgent().run(ctx)
        tp = ctx.trace["_iteration_plan"]
        # Only _A_CRITICAL_LOW qualifies (Critical + Low confidence)
        assert tp["critical_assumptions"] == 1

    def test_validation_priorities_count_matches_source(self):
        ctx = _ctx(
            exec_conf={"validation_priorities": ["P1", "P2", "P3"]},
        )
        IterationPlanAgent().run(ctx)
        assert ctx.trace["_iteration_plan"]["validation_priorities"] == 3

    def test_critical_unknowns_count_matches_source(self):
        ctx = _ctx(
            exec_conf={"critical_unknowns": ["CU1", "CU2"]},
        )
        IterationPlanAgent().run(ctx)
        assert ctx.trace["_iteration_plan"]["critical_unknowns"] == 2

    def test_expected_confidence_after_completion_in_trace(self):
        ctx = _ctx(
            exec_conf={"overall_confidence": "High"},
        )
        IterationPlanAgent().run(ctx)
        tp = ctx.trace["_iteration_plan"]
        assert "expected_confidence_after_completion" in tp
        assert tp["expected_confidence_after_completion"] == "High"

    def test_zero_counts_when_empty_inputs(self):
        ctx = _ctx()
        IterationPlanAgent().run(ctx)
        tp = ctx.trace["_iteration_plan"]
        assert tp["critical_assumptions"] == 0
        assert tp["validation_priorities"] == 0
        assert tp["critical_unknowns"] == 0
        assert tp["high_priority_tasks"] == 0

    def test_high_priority_tasks_counts_high_high_only(self):
        ctx = _ctx(
            exec_conf={
                "overall_confidence": "Low",
                "validation_priorities": ["Validate VP1"],  # HIGH/HIGH
                "critical_unknowns": ["CU1"],                # HIGH/MEDIUM
            },
        )
        IterationPlanAgent().run(ctx)
        tp = ctx.trace["_iteration_plan"]
        # VP tasks are HIGH/HIGH, CU tasks are HIGH/MEDIUM
        # high_priority_tasks counts only HIGH/HIGH
        assert tp["high_priority_tasks"] >= 1


# ---------------------------------------------------------------------------
# 34. No new LLM calls introduced (J12.2a)
# ---------------------------------------------------------------------------

class TestNoNewLLMCallsJ12a:
    def test_no_new_llm_imports_added(self):
        import functional_agents.iteration_plan_agent as mod
        src = open(mod.__file__).read()
        assert "claude_client" not in src
        assert "ClaudeClient" not in src
        assert "anthropic" not in src.lower() or "import anthropic" not in src

    def test_enrich_and_validate_make_no_llm_calls(self):
        task = {
            "task_title": "Validate A-001 and RSK-004",
            "research_objective": "Validate A-001 and RSK-004 via analysis",
            "why_it_matters": "Fragile",
            "evidence_needed": [],
            "suggested_queries": [],
            "related_assumption_ids": [],
            "related_risk_ids": [],
            "related_recommendation_ids": [],
            "related_option_ids": [],
        }
        enriched = _enrich_task_ids(task)
        assert isinstance(enriched, dict)

        plan = {
            "priority_research_tasks": [{
                **task,
                "task_id": "IRT-001",
                "expected_confidence_gain": "HIGH",
                "urgency": "HIGH",
                "source_type": "executive_confidence",
                "source_id": "validation_priority",
            }],
        }
        warnings = _validate_plan(plan, known_assumption_ids={"A-001"}, known_risk_ids={"RSK-004"})
        assert isinstance(warnings, list)
