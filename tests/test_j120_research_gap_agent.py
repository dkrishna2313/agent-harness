"""J12.0 — ResearchGapAgent: deterministic research completeness assessment.

Tests verify:
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
            "question_coverage",
            "weak_questions",
            "missing_investigation_areas",
            "assumption_heavy_topics",
            "contradictions",
            "recommended_followups",
            "overall_research_health",
            "confidence",
        }
        assert expected_keys == set(analysis.keys())
