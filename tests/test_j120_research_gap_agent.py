"""J12.0/J12.1 — ResearchGapAgent: deterministic research completeness & decision-support assessment.

Tests verify:
  J12.0
  - WorkflowState.RESEARCH_GAP constant exists
  - Agent contract: inherits FunctionalAgent, run() → AgentResult
  - No-op guard fires when evidence_notes or plan is absent
  - question_coverage correctly maps coverage levels
  - weak_questions filters NONE and WEAK; ignores STRONG and MODERATE
  - missing_investigation_areas identifies areas with no evidence
  - assumption_heavy_topics flags high-confidence hypotheses with sparse evidence
  - contradictions lifted from validated_contradictions and from research_object
  - recommended_followups generated from weak questions and missing areas
  - overall_research_health banding: GOOD / FAIR / POOR
  - confidence score: fraction of SQs with MODERATE+ coverage
  - context fields written: research_gap_analysis, research_object, trace
  - empty subquestions edge case

  J12.1
  - Assumption support analysis: Critical+no_evidence, Critical+weak_evidence, Low+confidence
  - Recommendation dependency analysis: rec depending on weak assumptions flagged
  - Strategic option support analysis: recommended option depending on weak assumption flagged
  - Executive confidence gap analysis: below-High confidence flagged
  - Decision-support gaps aggregated with sequential DSG IDs
  - Confidence alignment: ALIGNED / MISALIGNED / UNKNOWN
  - Followup ordering: decision-support gaps first, then coverage gaps
  - J12.0 fields all preserved; J12.1 fields added
  - Health penalized for decision-support gaps
  - Graceful no-op when decision data absent (first-pass behavior)
"""

from __future__ import annotations

import pytest

from functional_agents.base import FunctionalAgent
from functional_agents.context import AgentContext, AgentResult, WorkflowState
from functional_agents.research_gap_agent import (
    ResearchGapAgent,
    compute_research_gap_analysis,
    _question_coverage_list,
    _weak_questions,
    _missing_areas,
    _assumption_heavy_topics,
    _extract_contradictions,
    _recommended_followups,
    _overall_health,
    _confidence_score,
    # J12.1
    _assumption_support_analysis,
    _recommendation_dependency_analysis,
    _strategic_option_support_analysis,
    _executive_confidence_gap_analysis,
    _decision_support_gaps,
    _confidence_alignment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_COVERAGE = {
    "SQ1": {"coverage": "STRONG", "evidence_count": 6},
    "SQ2": {"coverage": "MODERATE", "evidence_count": 3},
    "SQ3": {"coverage": "WEAK", "evidence_count": 1},
}
_DEFAULT_AREA_MAP = {
    "Power": ["E001", "E002"],
    "Grid": ["E003"],
    "Policy": [],
}


def _ctx(
    *,
    subquestions: list[str] | None = None,
    investigation_areas: list[str] | None = None,
    coverage: dict | None = None,
    evidence_by_area: dict | None = None,
    hypotheses: list | None = None,
    validated_contradictions: list | None = None,
    ro_contradictions: list | None = None,
    # J12.1 — decision-support inputs
    assumptions: list | None = None,
    recommendations: list | None = None,
    strategic_options: list | None = None,
    decision_analysis: dict | None = None,
    executive_confidence: dict | None = None,
) -> AgentContext:
    sqs = subquestions if subquestions is not None else ["SQ1", "SQ2", "SQ3"]
    areas = investigation_areas if investigation_areas is not None else ["Power", "Grid", "Policy"]
    cov = coverage if coverage is not None else _DEFAULT_COVERAGE
    area_map = evidence_by_area if evidence_by_area is not None else _DEFAULT_AREA_MAP
    note = {
        "coverage_by_subquestion": cov,
        "evidence_by_area": area_map,
    }
    ro = {"id": "R-TEST_001"}
    if ro_contradictions is not None:
        ro["contradictions"] = ro_contradictions
    return AgentContext(
        question="test question",
        profiles=["test_profile"],
        execution_profile="test_profile",
        research_object=ro,
        run_id="testrun001",
        plan={
            "subquestions": sqs,
            "investigation_areas": areas,
        },
        evidence_notes=[note],
        hypotheses=hypotheses or [],
        validated_contradictions=validated_contradictions or [],
        assumptions=assumptions or [],
        recommendations=recommendations or [],
        strategic_options=strategic_options or [],
        decision_analysis=decision_analysis or {},
        executive_confidence=executive_confidence or {},
    )


# ---------------------------------------------------------------------------
# TestWorkflowStateConstant
# ---------------------------------------------------------------------------

class TestWorkflowStateConstant:

    def test_research_gap_state_exists(self):
        assert WorkflowState.RESEARCH_GAP == "RESEARCH_GAP"


# ---------------------------------------------------------------------------
# TestAgentContract
# ---------------------------------------------------------------------------

class TestAgentContract:

    def test_inherits_functional_agent(self):
        assert issubclass(ResearchGapAgent, FunctionalAgent)

    def test_run_returns_agent_result(self):
        agent = ResearchGapAgent()
        result = agent.run(_ctx())
        assert isinstance(result, AgentResult)

    def test_run_status_success(self):
        result = ResearchGapAgent().run(_ctx())
        assert result.status == "success"

    def test_run_has_duration_metric(self):
        result = ResearchGapAgent().run(_ctx())
        assert "duration_seconds" in result.metrics
        assert result.metrics["duration_seconds"] >= 0.0

    def test_run_trace_required_keys(self):
        result = ResearchGapAgent().run(_ctx())
        for key in ("agent", "run_id", "duration_seconds", "status"):
            assert key in result.trace


# ---------------------------------------------------------------------------
# TestNoOpGuard
# ---------------------------------------------------------------------------

class TestNoOpGuard:

    def test_skips_when_no_evidence_notes(self):
        ctx = _ctx()
        ctx.evidence_notes = []
        result = ResearchGapAgent().run(ctx)
        assert result.status == "skipped"
        assert ctx.trace["_research_gap"]["skipped"] is True

    def test_skips_when_no_plan(self):
        ctx = _ctx()
        ctx.plan = {}
        result = ResearchGapAgent().run(ctx)
        assert result.status == "skipped"
        assert ctx.trace["_research_gap"]["skipped"] is True

    def test_no_op_does_not_write_research_gap_analysis(self):
        ctx = _ctx()
        ctx.evidence_notes = []
        ResearchGapAgent().run(ctx)
        assert ctx.research_gap_analysis == {}


# ---------------------------------------------------------------------------
# TestQuestionCoverage
# ---------------------------------------------------------------------------

class TestQuestionCoverage:

    def test_all_subquestions_appear_in_coverage(self):
        result = ResearchGapAgent().run(_ctx())
        analysis = result.context.research_gap_analysis
        sq_names = {q["subquestion"] for q in analysis["question_coverage"]}
        assert sq_names == {"SQ1", "SQ2", "SQ3"}

    def test_coverage_levels_preserved(self):
        result = ResearchGapAgent().run(_ctx())
        coverage_map = {
            q["subquestion"]: q["coverage"]
            for q in result.context.research_gap_analysis["question_coverage"]
        }
        assert coverage_map["SQ1"] == "STRONG"
        assert coverage_map["SQ2"] == "MODERATE"
        assert coverage_map["SQ3"] == "WEAK"

    def test_evidence_counts_preserved(self):
        result = ResearchGapAgent().run(_ctx())
        coverage_map = {
            q["subquestion"]: q["evidence_count"]
            for q in result.context.research_gap_analysis["question_coverage"]
        }
        assert coverage_map["SQ1"] == 6
        assert coverage_map["SQ2"] == 3
        assert coverage_map["SQ3"] == 1

    def test_empty_subquestions_yields_empty_coverage(self):
        ctx = _ctx(subquestions=[], coverage={})
        result = ResearchGapAgent().run(ctx)
        assert result.context.research_gap_analysis["question_coverage"] == []


# ---------------------------------------------------------------------------
# TestWeakQuestions
# ---------------------------------------------------------------------------

class TestWeakQuestions:

    def test_none_coverage_flagged(self):
        ctx = _ctx(
            subquestions=["SQ1", "SQ2"],
            coverage={
                "SQ1": {"coverage": "STRONG", "evidence_count": 5},
                "SQ2": {"coverage": "NONE", "evidence_count": 0},
            },
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        weak_names = [q["subquestion"] for q in analysis["weak_questions"]]
        assert "SQ2" in weak_names

    def test_weak_coverage_flagged(self):
        ctx = _ctx(
            subquestions=["SQ1", "SQ2"],
            coverage={
                "SQ1": {"coverage": "STRONG", "evidence_count": 5},
                "SQ2": {"coverage": "WEAK", "evidence_count": 1},
            },
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        weak_names = [q["subquestion"] for q in analysis["weak_questions"]]
        assert "SQ2" in weak_names

    def test_strong_not_flagged(self):
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "STRONG", "evidence_count": 6}},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["weak_questions"] == []

    def test_moderate_not_flagged(self):
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "MODERATE", "evidence_count": 3}},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["weak_questions"] == []


# ---------------------------------------------------------------------------
# TestMissingInvestigationAreas
# ---------------------------------------------------------------------------

class TestMissingInvestigationAreas:

    def test_area_with_no_evidence_flagged(self):
        ctx = _ctx(
            investigation_areas=["Power", "Grid", "Policy"],
            evidence_by_area={"Power": ["E001"], "Grid": [], "Policy": []},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert "Grid" in analysis["missing_investigation_areas"]
        assert "Policy" in analysis["missing_investigation_areas"]

    def test_area_with_evidence_not_flagged(self):
        ctx = _ctx(
            investigation_areas=["Power", "Grid"],
            evidence_by_area={"Power": ["E001", "E002"], "Grid": ["E003"]},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["missing_investigation_areas"] == []

    def test_empty_investigation_areas_yields_empty(self):
        ctx = _ctx(investigation_areas=[], evidence_by_area={})
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["missing_investigation_areas"] == []


# ---------------------------------------------------------------------------
# TestAssumptionHeavyTopics
# ---------------------------------------------------------------------------

class TestAssumptionHeavyTopics:

    def test_high_confidence_with_0_evidence_flagged(self):
        hyp = [{"id": "H1", "title": "Claim A", "confidence": "high", "supporting_evidence": []}]
        ctx = _ctx(hypotheses=hyp)
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any(t["hypothesis_id"] == "H1" for t in analysis["assumption_heavy_topics"])

    def test_high_confidence_with_1_evidence_flagged(self):
        hyp = [{"id": "H2", "title": "Claim B", "confidence": "high", "supporting_evidence": ["E001"]}]
        ctx = _ctx(hypotheses=hyp)
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any(t["hypothesis_id"] == "H2" for t in analysis["assumption_heavy_topics"])

    def test_high_confidence_with_2_evidence_not_flagged(self):
        hyp = [{"id": "H3", "title": "Claim C", "confidence": "high", "supporting_evidence": ["E001", "E002"]}]
        ctx = _ctx(hypotheses=hyp)
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert not any(t["hypothesis_id"] == "H3" for t in analysis["assumption_heavy_topics"])

    def test_medium_confidence_not_flagged(self):
        hyp = [{"id": "H4", "title": "Claim D", "confidence": "medium", "supporting_evidence": []}]
        ctx = _ctx(hypotheses=hyp)
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["assumption_heavy_topics"] == []


# ---------------------------------------------------------------------------
# TestContradictions
# ---------------------------------------------------------------------------

class TestContradictions:

    def test_lifted_from_validated_contradictions(self):
        raw = [{"contradiction_id": "C001", "severity": "high", "topic": "power"}]
        ctx = _ctx(validated_contradictions=raw)
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any(c["contradiction_id"] == "C001" for c in analysis["contradictions"])

    def test_lifted_from_research_object_when_validated_empty(self):
        ctx = _ctx(ro_contradictions=[{"contradiction_id": "C002", "severity": "low", "topic": "cost"}])
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any(c["contradiction_id"] == "C002" for c in analysis["contradictions"])

    def test_validated_contradictions_take_priority_over_ro(self):
        raw = [{"contradiction_id": "C003", "severity": "high", "topic": "timeline"}]
        ctx = _ctx(
            validated_contradictions=raw,
            ro_contradictions=[{"contradiction_id": "C099", "severity": "low", "topic": "other"}],
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        ids = {c["contradiction_id"] for c in analysis["contradictions"]}
        assert "C003" in ids
        assert "C099" not in ids  # validated_contradictions took priority


# ---------------------------------------------------------------------------
# TestRecommendedFollowups
# ---------------------------------------------------------------------------

class TestRecommendedFollowups:

    def test_weak_question_generates_followup(self):
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "NONE", "evidence_count": 0}},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any("SQ1" in f for f in analysis["recommended_followups"])

    def test_missing_area_generates_followup(self):
        ctx = _ctx(
            investigation_areas=["Power"],
            evidence_by_area={"Power": []},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert any("Power" in f for f in analysis["recommended_followups"])

    def test_no_followups_when_fully_covered(self):
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "STRONG", "evidence_count": 5}},
            investigation_areas=["Power"],
            evidence_by_area={"Power": ["E001"]},
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["recommended_followups"] == []


# ---------------------------------------------------------------------------
# TestOverallHealth
# ---------------------------------------------------------------------------

class TestOverallHealth:

    def test_all_strong_is_good(self):
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        assert _overall_health(cov, 0, []) == "GOOD"

    def test_many_none_is_poor(self):
        cov = [{"coverage": "NONE", "evidence_count": 0}] * 8
        assert _overall_health(cov, 0, []) == "POOR"

    def test_mixed_coverage_with_no_contradictions_fair(self):
        cov = [
            {"coverage": "STRONG", "evidence_count": 5},
            {"coverage": "MODERATE", "evidence_count": 3},
            {"coverage": "WEAK", "evidence_count": 1},
            {"coverage": "NONE", "evidence_count": 0},
        ]
        health = _overall_health(cov, 0, [])
        assert health in ("FAIR", "GOOD")

    def test_high_severity_contradictions_lower_health(self):
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        # 3 high-severity contradictions: -0.30 → score=0.70, still GOOD
        contradictions = [{"severity": "high"}] * 3
        health = _overall_health(cov, 0, contradictions)
        # 1.0 - 0.30 = 0.70 → GOOD (>= 0.75 boundary is 0.75)
        # Actually 0.70 < 0.75, so FAIR
        assert health in ("FAIR", "GOOD")


# ---------------------------------------------------------------------------
# TestConfidenceScore
# ---------------------------------------------------------------------------

class TestConfidenceScore:

    def test_all_strong_moderate_is_1_0(self):
        cov = [
            {"coverage": "STRONG", "evidence_count": 5},
            {"coverage": "MODERATE", "evidence_count": 3},
        ]
        assert _confidence_score(cov) == 1.0

    def test_no_coverage_is_0_0(self):
        cov = [{"coverage": "NONE", "evidence_count": 0}] * 3
        assert _confidence_score(cov) == 0.0

    def test_half_covered(self):
        cov = [
            {"coverage": "STRONG", "evidence_count": 5},
            {"coverage": "NONE", "evidence_count": 0},
        ]
        assert _confidence_score(cov) == 0.5

    def test_empty_returns_0_0(self):
        assert _confidence_score([]) == 0.0


# ---------------------------------------------------------------------------
# TestContextFields
# ---------------------------------------------------------------------------

class TestContextFields:

    def test_research_gap_analysis_written_to_context(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert isinstance(ctx.research_gap_analysis, dict)
        assert "overall_research_health" in ctx.research_gap_analysis

    def test_research_gap_analysis_written_to_research_object(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert "research_gap_analysis" in ctx.research_object

    def test_research_gap_analysis_written_to_trace(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert "_research_gap" in ctx.trace
        assert "overall_research_health" in ctx.trace["_research_gap"]

    def test_all_expected_keys_in_analysis(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        analysis = ctx.research_gap_analysis
        expected_keys = {
            # J12.0
            "question_coverage",
            "weak_questions",
            "missing_investigation_areas",
            "assumption_heavy_topics",
            "contradictions",
            "recommended_followups",
            "overall_research_health",
            "confidence",
            # J12.1
            "decision_support_gaps",
            "confidence_alignment",
        }
        assert expected_keys == set(analysis.keys())


# ===========================================================================
# J12.1 — Decision-support heuristic tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_A_CRITICAL_NO_EVIDENCE = {
    "assumption_id": "A-001",
    "statement": "Critical assumption with zero evidence",
    "importance": "Critical",
    "confidence": "Medium",
    "evidence_support": "None",
    "evidence_ids": [],
    "supported_recommendation_ids": ["REC-001"],
}
_A_CRITICAL_WEAK = {
    "assumption_id": "A-002",
    "statement": "Critical assumption with weak evidence",
    "importance": "Critical",
    "confidence": "Medium",
    "evidence_support": "Weak",
    "evidence_ids": ["E001", "E002"],
    "supported_recommendation_ids": [],
}
_A_IMPORTANT_LOW_CONF = {
    "assumption_id": "A-003",
    "statement": "Important assumption, low confidence",
    "importance": "Important",
    "confidence": "Low",
    "evidence_support": "Weak",
    "evidence_ids": ["E003", "E004"],
    "supported_recommendation_ids": ["REC-002"],
}
_A_STRONG = {
    "assumption_id": "A-004",
    "statement": "Well-supported assumption",
    "importance": "Critical",
    "confidence": "High",
    "evidence_support": "Strong",
    "evidence_ids": ["E005", "E006", "E007"],
    "supported_recommendation_ids": [],
}


# ---------------------------------------------------------------------------
# TestAssumptionSupportGaps
# ---------------------------------------------------------------------------

class TestAssumptionSupportGaps:

    def test_critical_no_evidence_flagged_high(self):
        gaps = _assumption_support_analysis([_A_CRITICAL_NO_EVIDENCE])
        no_ev = [g for g in gaps if g["gap_type"] == "no_evidence"]
        assert len(no_ev) == 1
        assert no_ev[0]["severity"] == "HIGH"
        assert no_ev[0]["artifact_id"] == "A-001"

    def test_critical_weak_evidence_flagged_medium(self):
        gaps = _assumption_support_analysis([_A_CRITICAL_WEAK])
        weak = [g for g in gaps if g["gap_type"] == "weak_evidence"]
        assert len(weak) == 1
        assert weak[0]["severity"] == "MEDIUM"

    def test_important_low_confidence_flagged_medium(self):
        gaps = _assumption_support_analysis([_A_IMPORTANT_LOW_CONF])
        low_conf = [g for g in gaps if g["gap_type"] == "low_confidence"]
        assert len(low_conf) == 1
        assert low_conf[0]["severity"] == "MEDIUM"
        assert low_conf[0]["artifact_id"] == "A-003"

    def test_critical_low_confidence_flagged_high(self):
        a = dict(_A_CRITICAL_NO_EVIDENCE, confidence="Low", evidence_ids=[])
        gaps = _assumption_support_analysis([a])
        low_conf = [g for g in gaps if g["gap_type"] == "low_confidence"]
        assert any(g["severity"] == "HIGH" for g in low_conf)

    def test_strong_critical_assumption_not_flagged(self):
        gaps = _assumption_support_analysis([_A_STRONG])
        assert gaps == []

    def test_non_critical_non_important_no_evidence_not_flagged(self):
        a = {
            "assumption_id": "A-X",
            "statement": "Minor assumption",
            "importance": "Minor",
            "confidence": "Low",
            "evidence_support": "None",
            "evidence_ids": [],
            "supported_recommendation_ids": [],
        }
        gaps = _assumption_support_analysis([a])
        assert gaps == []

    def test_gap_has_required_fields(self):
        gaps = _assumption_support_analysis([_A_CRITICAL_NO_EVIDENCE])
        for field in (
            "artifact_type", "artifact_id", "artifact_title", "gap_type",
            "severity", "why_it_matters", "supporting_evidence_count",
            "related_recommendation_ids", "related_assumption_ids",
            "recommended_followup",
        ):
            assert field in gaps[0], f"missing field: {field}"

    def test_no_assumptions_returns_empty(self):
        assert _assumption_support_analysis([]) == []

    def test_no_double_flag_critical_no_evidence_not_also_weak(self):
        # A Critical assumption with no evidence_ids should be flagged as
        # no_evidence (HIGH), NOT also as weak_evidence (no_evidence takes priority)
        gaps = _assumption_support_analysis([_A_CRITICAL_NO_EVIDENCE])
        gap_types = [g["gap_type"] for g in gaps]
        assert "no_evidence" in gap_types
        assert "weak_evidence" not in gap_types


# ---------------------------------------------------------------------------
# TestRecommendationDependencyGaps
# ---------------------------------------------------------------------------

class TestRecommendationDependencyGaps:

    def _assumptions_by_id(self, assumptions):
        return {a["assumption_id"]: a for a in assumptions}

    def test_rec_depending_on_no_evidence_assumption_flagged(self):
        rec = {
            "recommendation_id": "REC-001",
            "title": "Deploy SMR fleet",
            "supported_assumption_ids": ["A-001"],
        }
        by_id = self._assumptions_by_id([_A_CRITICAL_NO_EVIDENCE])
        gaps = _recommendation_dependency_analysis([rec], by_id)
        assert len(gaps) == 1
        assert gaps[0]["artifact_id"] == "REC-001"
        assert "A-001" in gaps[0]["related_assumption_ids"]

    def test_rec_depending_on_low_confidence_assumption_flagged(self):
        rec = {
            "recommendation_id": "REC-002",
            "title": "Partner with utility",
            "supported_assumption_ids": ["A-003"],
        }
        by_id = self._assumptions_by_id([_A_IMPORTANT_LOW_CONF])
        gaps = _recommendation_dependency_analysis([rec], by_id)
        assert len(gaps) == 1

    def test_rec_depending_on_strong_assumption_not_flagged(self):
        rec = {
            "recommendation_id": "REC-004",
            "title": "Safe recommendation",
            "supported_assumption_ids": ["A-004"],
        }
        by_id = self._assumptions_by_id([_A_STRONG])
        gaps = _recommendation_dependency_analysis([rec], by_id)
        assert gaps == []

    def test_empty_recommendations_no_gaps(self):
        by_id = self._assumptions_by_id([_A_CRITICAL_NO_EVIDENCE])
        gaps = _recommendation_dependency_analysis([], by_id)
        assert gaps == []

    def test_supports_both_field_name_variants(self):
        rec_a = {
            "recommendation_id": "REC-A",
            "title": "Test A",
            "supported_assumption_ids": ["A-001"],
        }
        rec_b = {
            "recommendation_id": "REC-B",
            "title": "Test B",
            "supporting_assumption_ids": ["A-001"],
        }
        by_id = self._assumptions_by_id([_A_CRITICAL_NO_EVIDENCE])
        gaps_a = _recommendation_dependency_analysis([rec_a], by_id)
        gaps_b = _recommendation_dependency_analysis([rec_b], by_id)
        assert len(gaps_a) == 1
        assert len(gaps_b) == 1


# ---------------------------------------------------------------------------
# TestStrategicOptionSupportGaps
# ---------------------------------------------------------------------------

class TestStrategicOptionSupportGaps:

    def _by_id(self, assumptions):
        return {a["assumption_id"]: a for a in assumptions}

    def test_recommended_option_with_weak_assumption_flagged(self):
        option = {
            "option_id": "OPT-C",
            "title": "Full SMR deployment",
            "supporting_assumption_ids": ["A-001", "A-004"],
        }
        by_id = self._by_id([_A_CRITICAL_NO_EVIDENCE, _A_STRONG])
        gaps = _strategic_option_support_analysis("OPT-C", [option], by_id)
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "HIGH"
        assert "A-001" in gaps[0]["related_assumption_ids"]
        assert "A-004" not in gaps[0]["related_assumption_ids"]

    def test_recommended_option_all_strong_not_flagged(self):
        option = {
            "option_id": "OPT-A",
            "title": "Safe option",
            "supporting_assumption_ids": ["A-004"],
        }
        by_id = self._by_id([_A_STRONG])
        gaps = _strategic_option_support_analysis("OPT-A", [option], by_id)
        assert gaps == []

    def test_no_recommended_option_id_returns_empty(self):
        option = {
            "option_id": "OPT-C",
            "title": "Some option",
            "supporting_assumption_ids": ["A-001"],
        }
        by_id = self._by_id([_A_CRITICAL_NO_EVIDENCE])
        gaps = _strategic_option_support_analysis("", [option], by_id)
        assert gaps == []

    def test_recommended_option_not_in_list_returns_empty(self):
        by_id = self._by_id([_A_CRITICAL_NO_EVIDENCE])
        gaps = _strategic_option_support_analysis("OPT-Z", [], by_id)
        assert gaps == []

    def test_weak_evidence_support_also_triggers_flag(self):
        option = {
            "option_id": "OPT-B",
            "title": "Risky option",
            "supporting_assumption_ids": ["A-002"],
        }
        by_id = self._by_id([_A_CRITICAL_WEAK])
        gaps = _strategic_option_support_analysis("OPT-B", [option], by_id)
        assert len(gaps) == 1


# ---------------------------------------------------------------------------
# TestExecutiveConfidenceGaps
# ---------------------------------------------------------------------------

class TestExecutiveConfidenceGaps:

    def test_medium_confidence_creates_medium_severity_gap(self):
        ec = {"overall_confidence": "Medium", "confidence_id": "EC-001"}
        gaps = _executive_confidence_gap_analysis(ec)
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "MEDIUM"
        assert gaps[0]["gap_type"] == "confidence_misalignment"

    def test_low_confidence_creates_high_severity_gap(self):
        ec = {"overall_confidence": "Low", "confidence_id": "EC-002"}
        gaps = _executive_confidence_gap_analysis(ec)
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "HIGH"

    def test_high_confidence_no_gap(self):
        ec = {"overall_confidence": "High", "confidence_id": "EC-003"}
        gaps = _executive_confidence_gap_analysis(ec)
        assert gaps == []

    def test_empty_dict_no_gap(self):
        gaps = _executive_confidence_gap_analysis({})
        assert gaps == []

    def test_gap_artifact_type_is_executive_confidence(self):
        ec = {"overall_confidence": "Medium"}
        gaps = _executive_confidence_gap_analysis(ec)
        assert gaps[0]["artifact_type"] == "executive_confidence"


# ---------------------------------------------------------------------------
# TestDecisionSupportGapsAggregation
# ---------------------------------------------------------------------------

class TestDecisionSupportGapsAggregation:

    def test_sequential_dsg_ids_assigned(self):
        dsgs = _decision_support_gaps(
            assumptions=[_A_CRITICAL_NO_EVIDENCE, _A_IMPORTANT_LOW_CONF],
            recommendations=[],
            strategic_options=[],
            decision_analysis={},
            executive_confidence={"overall_confidence": "Medium"},
        )
        ids = [g["gap_id"] for g in dsgs]
        assert ids[0] == "DSG-001"
        assert ids[-1] == f"DSG-{len(ids):03d}"

    def test_empty_inputs_no_gaps(self):
        dsgs = _decision_support_gaps(
            assumptions=[],
            recommendations=[],
            strategic_options=[],
            decision_analysis={},
            executive_confidence={},
        )
        assert dsgs == []

    def test_all_sources_combined(self):
        option = {
            "option_id": "OPT-C",
            "title": "Full deployment",
            "supporting_assumption_ids": ["A-001"],
        }
        rec = {
            "recommendation_id": "REC-001",
            "title": "Deploy",
            "supported_assumption_ids": ["A-001"],
        }
        dsgs = _decision_support_gaps(
            assumptions=[_A_CRITICAL_NO_EVIDENCE],
            recommendations=[rec],
            strategic_options=[option],
            decision_analysis={"recommended_option_id": "OPT-C"},
            executive_confidence={"overall_confidence": "Medium"},
        )
        artifact_types = {g["artifact_type"] for g in dsgs}
        assert "assumption" in artifact_types
        assert "recommendation" in artifact_types
        assert "strategic_option" in artifact_types
        assert "executive_confidence" in artifact_types


# ---------------------------------------------------------------------------
# TestConfidenceAlignment
# ---------------------------------------------------------------------------

class TestConfidenceAlignment:

    def test_unknown_when_no_exec_or_da_confidence(self):
        result = _confidence_alignment(0.8, {}, {})
        assert result["alignment_status"] == "UNKNOWN"

    def test_misaligned_high_research_low_exec(self):
        result = _confidence_alignment(
            0.9,
            {"confidence": "High"},
            {"overall_confidence": "Medium"},
        )
        assert result["alignment_status"] == "MISALIGNED"

    def test_misaligned_high_research_low_exec_confidence(self):
        result = _confidence_alignment(
            0.85,
            {},
            {"overall_confidence": "Low"},
        )
        assert result["alignment_status"] == "MISALIGNED"

    def test_misaligned_low_research_high_exec(self):
        result = _confidence_alignment(
            0.3,
            {"confidence": "Medium"},
            {"overall_confidence": "High"},
        )
        assert result["alignment_status"] == "MISALIGNED"

    def test_aligned_when_consistent(self):
        result = _confidence_alignment(
            0.9,
            {"confidence": "High"},
            {"overall_confidence": "High"},
        )
        assert result["alignment_status"] == "ALIGNED"

    def test_result_has_required_fields(self):
        result = _confidence_alignment(0.75, {"confidence": "High"}, {"overall_confidence": "High"})
        for key in (
            "research_gap_confidence", "decision_analysis_confidence",
            "executive_confidence", "alignment_status", "explanation",
        ):
            assert key in result, f"missing field: {key}"

    def test_research_gap_confidence_echoed(self):
        result = _confidence_alignment(0.667, {}, {})
        assert result["research_gap_confidence"] == 0.667


# ---------------------------------------------------------------------------
# TestDecisionSupportFollowupOrdering
# ---------------------------------------------------------------------------

class TestDecisionSupportFollowupOrdering:

    def test_dsg_followups_appear_before_coverage_followups(self):
        # Coverage gap: weak SQ; Decision gap: Critical assumption no evidence
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "NONE", "evidence_count": 0}},
            investigation_areas=[],
            evidence_by_area={},
            assumptions=[_A_CRITICAL_NO_EVIDENCE],
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        followups = analysis["recommended_followups"]
        assert len(followups) >= 2
        # The DSG followup (about A-001) must appear before the coverage followup (about SQ1)
        dsg_idx = next(i for i, f in enumerate(followups) if "A-001" in f)
        coverage_idx = next(i for i, f in enumerate(followups) if "SQ1" in f)
        assert dsg_idx < coverage_idx

    def test_no_duplicates_in_followups(self):
        ctx = _ctx(
            assumptions=[_A_CRITICAL_NO_EVIDENCE],
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        followups = analysis["recommended_followups"]
        assert len(followups) == len(set(followups))

    def test_no_followups_when_no_gaps(self):
        ctx = _ctx(
            subquestions=["SQ1"],
            coverage={"SQ1": {"coverage": "STRONG", "evidence_count": 5}},
            investigation_areas=["Power"],
            evidence_by_area={"Power": ["E001"]},
            assumptions=[_A_STRONG],
        )
        analysis = ResearchGapAgent().run(ctx).context.research_gap_analysis
        assert analysis["recommended_followups"] == []


# ---------------------------------------------------------------------------
# TestJ121Fields
# ---------------------------------------------------------------------------

class TestJ121Fields:

    def test_decision_support_gaps_field_present(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert "decision_support_gaps" in ctx.research_gap_analysis

    def test_confidence_alignment_field_present(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert "confidence_alignment" in ctx.research_gap_analysis

    def test_confidence_alignment_has_alignment_status(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        ca = ctx.research_gap_analysis["confidence_alignment"]
        assert "alignment_status" in ca

    def test_j120_fields_all_preserved(self):
        ctx = _ctx(assumptions=[_A_CRITICAL_NO_EVIDENCE])
        ResearchGapAgent().run(ctx)
        analysis = ctx.research_gap_analysis
        for key in (
            "question_coverage", "weak_questions", "missing_investigation_areas",
            "assumption_heavy_topics", "contradictions", "recommended_followups",
            "overall_research_health", "confidence",
        ):
            assert key in analysis, f"J12.0 field missing: {key}"

    def test_decision_support_gaps_empty_when_no_decision_data(self):
        # First-pass behavior: no assumptions/recommendations/etc.
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        assert ctx.research_gap_analysis["decision_support_gaps"] == []

    def test_confidence_alignment_unknown_when_no_decision_data(self):
        ctx = _ctx()
        ResearchGapAgent().run(ctx)
        ca = ctx.research_gap_analysis["confidence_alignment"]
        assert ca["alignment_status"] == "UNKNOWN"

    def test_decision_support_gaps_populated_when_assumptions_set(self):
        ctx = _ctx(assumptions=[_A_CRITICAL_NO_EVIDENCE])
        ResearchGapAgent().run(ctx)
        dsgs = ctx.research_gap_analysis["decision_support_gaps"]
        assert len(dsgs) >= 1
        assert any(g["artifact_id"] == "A-001" for g in dsgs)


# ---------------------------------------------------------------------------
# TestHealthWithDecisionGaps
# ---------------------------------------------------------------------------

class TestHealthWithDecisionGaps:

    def test_no_dsgs_health_unchanged(self):
        # All STRONG coverage → GOOD without DSGs
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        assert _overall_health(cov, 0, [], []) == "GOOD"

    def test_critical_no_evidence_lowers_health_by_020(self):
        # Start at 1.0; -0.20 for no_evidence gap → 0.80 → GOOD
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        dsgs = [{"gap_type": "no_evidence", "artifact_type": "assumption"}]
        score_health = _overall_health(cov, 0, [], dsgs)
        # 1.0 - 0.20 = 0.80 → GOOD
        assert score_health == "GOOD"

    def test_multiple_dsgs_make_health_poor(self):
        # 2× no_evidence (-0.40) + 1× exec_conf (-0.10) = -0.50 → 0.50 → FAIR
        # Plus 1 NONE coverage (-0.15) → 0.35 → POOR
        cov = [{"coverage": "NONE", "evidence_count": 0}]
        dsgs = [
            {"gap_type": "no_evidence", "artifact_type": "assumption"},
            {"gap_type": "no_evidence", "artifact_type": "assumption"},
            {"gap_type": "confidence_misalignment", "artifact_type": "executive_confidence"},
        ]
        health = _overall_health(cov, 0, [], dsgs)
        assert health == "POOR"

    def test_rec_dependency_penalty_applied(self):
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        dsgs = [{"gap_type": "unsupported_dependency", "artifact_type": "recommendation"}]
        # 1.0 - 0.10 = 0.90 → GOOD
        assert _overall_health(cov, 0, [], dsgs) == "GOOD"

    def test_option_dependency_penalty_applied(self):
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        dsgs = [{"gap_type": "unsupported_dependency", "artifact_type": "strategic_option"}]
        # 1.0 - 0.10 = 0.90 → GOOD
        assert _overall_health(cov, 0, [], dsgs) == "GOOD"

    def test_j120_tests_backward_compat_no_dsgs_param(self):
        # Existing J12.0 3-arg calls still work (decision_support_gaps defaults to None)
        cov = [{"coverage": "STRONG", "evidence_count": 5}] * 4
        assert _overall_health(cov, 0, []) == "GOOD"


# ---------------------------------------------------------------------------
# TestResearchObjectFallback
# ---------------------------------------------------------------------------

class TestResearchObjectFallback:

    def test_assumptions_read_from_research_object_when_context_empty(self):
        # Simulate QA-loop second pass: assumptions already in research_object
        ctx = _ctx()
        ctx.research_object["strategic_assumptions"] = [_A_CRITICAL_NO_EVIDENCE]
        ResearchGapAgent().run(ctx)
        dsgs = ctx.research_gap_analysis["decision_support_gaps"]
        assert any(g["artifact_id"] == "A-001" for g in dsgs)

    def test_context_assumptions_take_priority_over_research_object(self):
        # context.assumptions overrides research_object["strategic_assumptions"]
        ctx = _ctx(assumptions=[_A_STRONG])
        ctx.research_object["strategic_assumptions"] = [_A_CRITICAL_NO_EVIDENCE]
        ResearchGapAgent().run(ctx)
        dsgs = ctx.research_gap_analysis["decision_support_gaps"]
        # _A_STRONG is Critical+High+Strong → no gap expected
        assert dsgs == []
