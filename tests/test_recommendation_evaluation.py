"""Tests for J6.6 / J6.6a Recommendation Evaluator.

Covers:
- score_single_recommendation() – all 5 dimension scorers
- evaluate_recommendations() – empty and multi-item
- build_recommendation_traceability()
- score_recommendations_from_memo() – proxy for benchmark pipeline
- AgentScores.recommendation_score populated in score_agents()
- QA agent wires recommendation_evaluation into context.qa
- J6.6a: missing_evidence_links, primary_penalty, aggregate_score alias
- J6.6a: recommendation_warnings, recommendation_summary
- J6.6a: score_recommendation_dimensions_from_memo() per-dimension proxy
- J6.6a: AgentScores dimension fields populated
- J6.6a: aggregate_agent_scores includes recommendation_dimension_summary
- J6.6a: EvaluationRun.recommendation_dimension_summary propagated
- J6.6a: benchmark JSON report includes recommendation_dimension_summary
- J6.6a: benchmark MD report has Recommendation Evaluation section
- J6.6a: QA validation block recommendation_evaluation_validation
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from research_agent.evaluation.recommendation_evaluator import (
    build_recommendation_traceability,
    evaluate_recommendations,
    score_recommendation_dimensions_from_memo,
    score_recommendations_from_memo,
    score_single_recommendation,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _rec(**kwargs) -> dict:
    defaults = {
        "id": "R1",
        "title": "Deploy liquid cooling systems for high-density AI racks",
        "summary": (
            "AI data centers should invest in direct liquid cooling infrastructure. "
            "However, the capital cost is significant and requires facility re-design. "
            "While the ROI is positive above 30 kW per rack, transition costs must be "
            "planned over 2–3 years. Liquid cooling reduces PUE by 0.2–0.4 points."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1", "H2"],
        "supporting_evidence": ["E001", "E002", "E003"],
        "key_risks": [
            "Capital cost may exceed budget for greenfield deployments",
            "Requires specialized maintenance expertise not widely available",
            "Vendor lock-in risk with proprietary cooling solutions",
        ],
        "trigger_conditions": ["rack density exceeds 30 kW", "PUE target below 1.3"],
        "confidence": "high",
        "confidence_rationale": (
            "Multiple independent sources confirm the 30 kW threshold for liquid cooling."
        ),
    }
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# score_single_recommendation
# ---------------------------------------------------------------------------

def test_full_recommendation_scores_high():
    result = score_single_recommendation(_rec())
    assert result["evidence_support_score"] >= 0.85
    assert result["reasoning_score"] >= 0.75
    assert result["tradeoff_score"] >= 0.55
    assert result["risk_score"] >= 0.90
    assert result["actionability_score"] >= 0.50
    assert result["recommendation_score"] >= 0.70


def test_empty_recommendation_scores_low():
    result = score_single_recommendation({})
    assert result["evidence_support_score"] == 0.0
    assert result["reasoning_score"] == 0.0
    assert result["risk_score"] == 0.0
    assert result["recommendation_score"] < 0.30


def test_evidence_support_zero_no_evidence():
    result = score_single_recommendation(_rec(supporting_evidence=[]))
    assert result["evidence_support_score"] == 0.0


def test_evidence_support_one_id():
    result = score_single_recommendation(_rec(supporting_evidence=["E001"]))
    assert result["evidence_support_score"] == 0.50


def test_evidence_support_match_bonus():
    result = score_single_recommendation(
        _rec(supporting_evidence=["E001", "E002"]),
        evidence_ids={"E001", "E002", "E003"},
    )
    assert result["evidence_support_score"] > 0.70


def test_reasoning_no_hypotheses():
    result = score_single_recommendation(_rec(supported_by_hypotheses=[]))
    assert result["reasoning_score"] < 0.60  # can still get rationale/summary credit


def test_reasoning_full_credit():
    result = score_single_recommendation(
        _rec(),
        hypothesis_ids={"H1", "H2"},
    )
    assert result["reasoning_score"] >= 0.90


def test_tradeoff_no_keywords():
    result = score_single_recommendation(_rec(summary="Invest in liquid cooling.", trigger_conditions=[]))
    assert result["tradeoff_score"] < 0.30


def test_tradeoff_with_keywords_and_triggers():
    result = score_single_recommendation(_rec())
    assert result["tradeoff_score"] >= 0.55


def test_risk_zero_risks():
    result = score_single_recommendation(_rec(key_risks=[]))
    assert result["risk_score"] == 0.0


def test_risk_one_short_risk():
    result = score_single_recommendation(_rec(key_risks=["costly"]))
    # one risk but < 20 chars — no specificity bonus
    assert result["risk_score"] == 0.50


def test_risk_three_specific_risks():
    result = score_single_recommendation(_rec())
    assert result["risk_score"] >= 0.90


def test_actionability_hedge_dominated():
    result = score_single_recommendation(_rec(
        title="Monitor cooling trends",
        summary="Evaluate and consider exploring potential cooling technologies.",
        time_horizon=None,
        priority=None,
    ))
    # dominated by hedge verbs, no action verbs
    assert result["actionability_score"] < 0.40


def test_actionability_strong_action_verb():
    result = score_single_recommendation(_rec(
        title="Deploy liquid cooling",
        time_horizon="near_term",
        priority="high",
    ))
    assert result["actionability_score"] >= 0.70


def test_traceability_in_result():
    result = score_single_recommendation(_rec())
    t = result["traceability"]
    assert t["recommendation_id"] == "R1"
    assert "E001" in t["evidence_ids"]
    assert "H1" in t["hypothesis_ids"]


# ---------------------------------------------------------------------------
# evaluate_recommendations
# ---------------------------------------------------------------------------

def test_evaluate_empty_returns_zeroes():
    result = evaluate_recommendations([])
    agg = result["aggregate"]
    assert agg["recommendation_count"] == 0
    assert agg["recommendation_score"] == 0.0
    assert result["recommendation_scores"] == []
    assert result["traceability"] == []


def test_evaluate_single_recommendation():
    result = evaluate_recommendations([_rec()])
    agg = result["aggregate"]
    assert agg["recommendation_count"] == 1
    assert agg["recommendation_score"] > 0.60
    assert len(result["recommendation_scores"]) == 1
    assert len(result["traceability"]) == 1


def test_evaluate_multiple_recommendations():
    recs = [_rec(), _rec(id="R2", key_risks=[], supporting_evidence=[])]
    result = evaluate_recommendations(recs)
    agg = result["aggregate"]
    assert agg["recommendation_count"] == 2
    # mean should be between best and worst
    full = evaluate_recommendations([_rec()])["aggregate"]["recommendation_score"]
    empty = evaluate_recommendations([_rec(id="R2", key_risks=[], supporting_evidence=[])])["aggregate"]["recommendation_score"]
    assert empty < agg["recommendation_score"] < full or agg["recommendation_score"] == full


def test_evaluate_dimension_means_present():
    result = evaluate_recommendations([_rec()])
    agg = result["aggregate"]
    for key in ("mean_evidence_support", "mean_reasoning", "mean_tradeoff", "mean_risk", "mean_actionability"):
        assert key in agg, f"Missing key: {key}"


def test_evaluate_traceability_structure():
    result = evaluate_recommendations([_rec()], hypothesis_ids={"H1"})
    t = result["traceability"][0]
    assert "recommendation_id" in t
    assert "evidence_ids" in t
    assert "hypothesis_ids" in t
    assert "challenge_ids" in t


# ---------------------------------------------------------------------------
# build_recommendation_traceability
# ---------------------------------------------------------------------------

def test_traceability_empty_rec():
    t = build_recommendation_traceability({})
    assert t["recommendation_id"] == ""
    assert t["evidence_ids"] == []
    assert t["hypothesis_ids"] == []
    assert t["challenge_ids"] == []


def test_traceability_full_rec():
    t = build_recommendation_traceability(_rec())
    assert t["recommendation_id"] == "R1"
    assert set(t["evidence_ids"]) == {"E001", "E002", "E003"}
    assert set(t["hypothesis_ids"]) == {"H1", "H2"}


def test_traceability_challenge_ids():
    rec = _rec()
    rec["supported_by_challenges"] = ["C1", "C2"]
    t = build_recommendation_traceability(rec)
    assert set(t["challenge_ids"]) == {"C1", "C2"}


# ---------------------------------------------------------------------------
# score_recommendations_from_memo (benchmark proxy)
# ---------------------------------------------------------------------------

def test_memo_proxy_no_inferences():
    assert score_recommendations_from_memo([]) == 0.1


def test_memo_proxy_few_short_inferences():
    score = score_recommendations_from_memo(["Consider cooling.", "Review power."])
    # < 60 chars each → no actionability credit, count credit = 2/4 = 0.5
    assert score < 0.60


def test_memo_proxy_many_actionable_inferences():
    infs = [
        "AI data centers should invest in direct liquid cooling for racks above 30 kW per rack.",
        "Operators must deploy immersion cooling to support GPU clusters exceeding 100 kW per rack.",
        "Facilities should transition away from air cooling for all AI-class deployments by 2027.",
        "Data center operators need to build modular power infrastructure to support 1 MW racks.",
    ]
    score = score_recommendations_from_memo(infs)
    assert score >= 0.70  # proxy scorer; exact value depends on heuristic weights


def test_memo_proxy_capped_at_1():
    infs = [
        f"Deploy new infrastructure for project {i} by investing heavily in cooling systems." for i in range(20)
    ]
    score = score_recommendations_from_memo(infs)
    assert score <= 1.0


# ---------------------------------------------------------------------------
# AgentScores integration
# ---------------------------------------------------------------------------

def test_agent_scorer_includes_recommendation_score():
    from research_agent.evaluation.agent_scorer import AgentScores, score_agents
    from unittest.mock import MagicMock

    memo = MagicMock()
    memo.coverage_matrix = []
    memo.source_notes = []
    memo.evidence = []
    memo.research_gaps = []
    memo.contradictions = []
    memo.confirmed_facts = []
    memo.inferences = [
        "AI data centers should adopt liquid cooling to support high-density GPU racks above 30 kW.",
        "Operators must invest in modular power distribution to enable 1 MW rack deployments.",
    ]

    qa_score = MagicMock()
    qa_score.citation_count = 5
    qa_score.citation_score = 0.7

    result = score_agents("Q1", "ai_data_centers", memo, qa_score)
    assert hasattr(result, "recommendation_score")
    assert result.recommendation_score > 0.0


def test_aggregate_includes_recommendation_score():
    from research_agent.evaluation.agent_scorer import AgentScores, aggregate_agent_scores

    s1 = AgentScores(question_id="Q1", domain="ai_dc", recommendation_score=0.8)
    s2 = AgentScores(question_id="Q2", domain="ai_dc", recommendation_score=0.6)
    agg = aggregate_agent_scores([s1, s2])
    assert "recommendation_score" in agg
    assert abs(agg["recommendation_score"] - 0.7) < 0.01


def test_aggregate_empty_has_recommendation_score():
    from research_agent.evaluation.agent_scorer import aggregate_agent_scores
    agg = aggregate_agent_scores([])
    assert agg["recommendation_score"] == 0.0


# ---------------------------------------------------------------------------
# QA agent wires recommendation_evaluation
# ---------------------------------------------------------------------------

def test_qa_agent_stores_recommendation_evaluation():
    from functional_agents.qa_agent import QAAgent
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.plan = {"subquestions": ["What is the power density?"]}
    ctx.evidence_notes = [{
        "coverage_by_subquestion": {"What is the power density?": {"coverage": "FULL"}},
        "evidence_by_subquestion": {"What is the power density?": [{}]},
        "evidence_summary": {"total_evidence_items": 5},
    }]
    ctx.research_object = {"evidence": []}
    ctx.profiles = []
    ctx.hypotheses = [{"hypothesis_id": "H1", "statement": "AI racks exceed 100 kW."}]
    ctx.hypothesis_challenges = []
    ctx.surviving_hypotheses = []
    ctx.validated_contradictions = []
    ctx.recommendations = [_rec()]
    ctx.recommendation_portfolio = {"near_term": [_rec()]}
    ctx.contradiction_metrics = {}

    agent = QAAgent()
    result_ctx = agent._execute(ctx)

    assert "recommendation_evaluation" in result_ctx.qa
    assert "recommendation_evaluation" in result_ctx.research_object

    ev = result_ctx.qa["recommendation_evaluation"]
    assert ev["aggregate"]["recommendation_count"] == 1
    assert ev["aggregate"]["recommendation_score"] > 0.0


# ===========================================================================
# J6.6a — Observability tests
# ===========================================================================

# ---------------------------------------------------------------------------
# missing_evidence_links and primary_penalty
# ---------------------------------------------------------------------------

def test_missing_evidence_links_true_when_no_evidence():
    result = score_single_recommendation(_rec(supporting_evidence=[]))
    assert result["missing_evidence_links"] is True


def test_missing_evidence_links_false_when_evidence_present():
    result = score_single_recommendation(_rec())
    assert result["missing_evidence_links"] is False


def test_primary_penalty_identified_no_evidence():
    result = score_single_recommendation(_rec(supporting_evidence=[]))
    assert result["primary_penalty"] == "missing_evidence_links"


def test_primary_penalty_identified_no_risks():
    result = score_single_recommendation(_rec(key_risks=[]))
    assert result["primary_penalty"] == "no_risk_identification"


def test_primary_penalty_none_when_all_healthy():
    result = score_single_recommendation(_rec())
    assert result["primary_penalty"] is None


def test_aggregate_score_alias_matches_recommendation_score():
    result = score_single_recommendation(_rec())
    assert result["aggregate_score"] == result["recommendation_score"]


# ---------------------------------------------------------------------------
# recommendation_warnings
# ---------------------------------------------------------------------------

def test_evaluate_warnings_emitted_for_weak_recommendation():
    weak = _rec(id="R_WEAK", supporting_evidence=[])
    result = evaluate_recommendations([_rec(), weak])
    warnings = result["recommendation_warnings"]
    assert any(w["recommendation_id"] == "R_WEAK" for w in warnings)


def test_evaluate_warnings_empty_when_all_healthy():
    result = evaluate_recommendations([_rec()])
    assert result["recommendation_warnings"] == []


def test_evaluate_warnings_contain_aggregate_score():
    weak = _rec(id="R_WEAK", supporting_evidence=[], key_risks=[])
    result = evaluate_recommendations([weak])
    w = result["recommendation_warnings"][0]
    assert "aggregate_score" in w
    assert w["aggregate_score"] >= 0.0


# ---------------------------------------------------------------------------
# recommendation_summary
# ---------------------------------------------------------------------------

def test_evaluate_recommendation_summary_present():
    result = evaluate_recommendations([_rec()])
    assert "recommendation_summary" in result


def test_recommendation_summary_fields():
    result = evaluate_recommendations([_rec()])
    s = result["recommendation_summary"]
    assert s["recommendation_count"] == 1
    assert "average_score" in s
    assert "lowest_score" in s
    assert "highest_score" in s


def test_recommendation_summary_lowest_highest_consistent():
    r1 = _rec(id="R1")
    r2 = _rec(id="R2", supporting_evidence=[], key_risks=[])
    result = evaluate_recommendations([r1, r2])
    s = result["recommendation_summary"]
    assert s["lowest_score"] <= s["average_score"] <= s["highest_score"]


def test_recommendation_summary_empty():
    result = evaluate_recommendations([])
    s = result["recommendation_summary"]
    assert s["recommendation_count"] == 0
    assert s["average_score"] == 0.0


# ---------------------------------------------------------------------------
# traceability includes missing_evidence_links
# ---------------------------------------------------------------------------

def test_traceability_includes_missing_evidence_links_flag():
    result = score_single_recommendation(_rec(supporting_evidence=[]))
    t = result["traceability"]
    assert "missing_evidence_links" in t
    assert t["missing_evidence_links"] is True


def test_traceability_missing_evidence_links_false_when_present():
    result = score_single_recommendation(_rec())
    t = result["traceability"]
    assert t["missing_evidence_links"] is False


# ---------------------------------------------------------------------------
# score_recommendation_dimensions_from_memo
# ---------------------------------------------------------------------------

def test_dimensions_from_memo_empty():
    dims = score_recommendation_dimensions_from_memo([])
    for key in ("evidence_support", "reasoning", "tradeoff", "risk", "actionability"):
        assert key in dims
        assert dims[key] == 0.1


def test_dimensions_from_memo_keys_present():
    dims = score_recommendation_dimensions_from_memo(["invest in liquid cooling because air cooling fails above 30 kW."])
    assert set(dims.keys()) == {"evidence_support", "reasoning", "tradeoff", "risk", "actionability"}


def test_dimensions_from_memo_actionability_increases_with_verbs():
    low = score_recommendation_dimensions_from_memo(["consider monitoring potential trends."])
    high = score_recommendation_dimensions_from_memo([
        "Deploy liquid cooling infrastructure across all AI racks exceeding 30 kW because air cooling fails."
    ])
    assert high["actionability"] >= low["actionability"]


def test_dimensions_from_memo_all_between_0_and_1():
    infs = [
        "Invest in immersion cooling because GPU rack density exceeds 100 kW; however, capital costs are high.",
        "Deploy modular power units to reduce risk of stranded assets.",
    ]
    dims = score_recommendation_dimensions_from_memo(infs)
    for k, v in dims.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"


# ---------------------------------------------------------------------------
# AgentScores dimension fields
# ---------------------------------------------------------------------------

def test_agent_scorer_stores_dimension_fields():
    from research_agent.evaluation.agent_scorer import score_agents
    from unittest.mock import MagicMock

    memo = MagicMock()
    memo.coverage_matrix = []
    memo.source_notes = []
    memo.evidence = []
    memo.research_gaps = []
    memo.contradictions = []
    memo.confirmed_facts = []
    memo.inferences = [
        "Deploy liquid cooling because rack power exceeds 30 kW — however, capital costs are high.",
    ]

    qa_score = MagicMock()
    qa_score.citation_count = 5
    qa_score.citation_score = 0.7

    result = score_agents("Q1", "ai_dc", memo, qa_score)
    for field in ("rec_evidence_support", "rec_reasoning", "rec_tradeoff", "rec_risk", "rec_actionability"):
        assert hasattr(result, field), f"Missing field: {field}"
        val = getattr(result, field)
        assert 0.0 <= val <= 1.0, f"{field}={val} out of range"


def test_aggregate_includes_recommendation_dimension_summary():
    from research_agent.evaluation.agent_scorer import AgentScores, aggregate_agent_scores

    s = AgentScores(
        question_id="Q1", domain="ai_dc",
        recommendation_score=0.8,
        rec_evidence_support=0.9, rec_reasoning=0.8,
        rec_tradeoff=0.6, rec_risk=1.0, rec_actionability=0.9,
    )
    agg = aggregate_agent_scores([s])
    assert "recommendation_dimension_summary" in agg
    ds = agg["recommendation_dimension_summary"]
    assert set(ds.keys()) == {"evidence_support", "reasoning", "tradeoff", "risk", "actionability"}
    assert abs(ds["evidence_support"] - 0.9) < 0.001


# ---------------------------------------------------------------------------
# EvaluationRun propagation
# ---------------------------------------------------------------------------

def test_evaluation_run_has_recommendation_dimension_summary():
    from research_agent.evaluation.runner import EvaluationRun
    run = EvaluationRun()
    run.recommendation_dimension_summary = {"evidence_support": 0.8, "reasoning": 0.7}
    assert run.recommendation_dimension_summary["evidence_support"] == 0.8


# ---------------------------------------------------------------------------
# Benchmark JSON report — recommendation_dimension_summary in summary block
# ---------------------------------------------------------------------------

def test_benchmark_json_report_includes_dimension_summary():
    from research_agent.evaluation.runner import EvaluationRun
    from research_agent.evaluation.report import build_json_report

    run = EvaluationRun()
    run.recommendation_score = 0.79
    run.recommendation_dimension_summary = {
        "evidence_support": 0.81,
        "reasoning": 0.88,
        "tradeoff": 0.76,
        "risk": 0.83,
        "actionability": 0.91,
    }
    report = build_json_report(run)
    assert "recommendation_score" in report["summary"]
    assert report["summary"]["recommendation_score"] == 0.79
    assert "recommendation_dimension_summary" in report["summary"]
    ds = report["summary"]["recommendation_dimension_summary"]
    assert ds["evidence_support"] == 0.81
    assert ds["actionability"] == 0.91


def test_agent_evaluation_dict_has_dimension_summary():
    from research_agent.evaluation.runner import EvaluationRun
    from research_agent.evaluation.report import build_json_report

    run = EvaluationRun()
    run.recommendation_dimension_summary = {"evidence_support": 0.7, "reasoning": 0.8,
                                            "tradeoff": 0.6, "risk": 0.9, "actionability": 0.85}
    report = build_json_report(run)
    ae = report["agent_evaluation"]
    assert "recommendation_dimension_summary" in ae["aggregate"]


# ---------------------------------------------------------------------------
# Benchmark MD report — Recommendation Evaluation section
# ---------------------------------------------------------------------------

def test_benchmark_md_report_includes_recommendation_section():
    from research_agent.evaluation.runner import EvaluationRun
    from research_agent.evaluation.report import build_md_report

    run = EvaluationRun()
    run.recommendation_score = 0.82
    run.recommendation_dimension_summary = {
        "evidence_support": 0.80,
        "reasoning": 0.88,
        "tradeoff": 0.75,
        "risk": 0.90,
        "actionability": 0.85,
    }
    md = build_md_report(run)
    assert "## Recommendation Evaluation" in md
    assert "Evidence Support" in md
    assert "Actionability" in md


# ---------------------------------------------------------------------------
# QA validation — recommendation_evaluation_validation block
# ---------------------------------------------------------------------------

def test_qa_agent_stores_recommendation_evaluation_validation():
    from functional_agents.qa_agent import QAAgent
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.plan = {"subquestions": ["What is the power density?"]}
    ctx.evidence_notes = [{
        "coverage_by_subquestion": {"What is the power density?": {"coverage": "FULL"}},
        "evidence_by_subquestion": {"What is the power density?": [{}]},
        "evidence_summary": {"total_evidence_items": 5},
    }]
    ctx.research_object = {"evidence": []}
    ctx.profiles = []
    ctx.hypotheses = [{"hypothesis_id": "H1", "statement": "AI racks exceed 100 kW."}]
    ctx.hypothesis_challenges = []
    ctx.surviving_hypotheses = []
    ctx.validated_contradictions = []
    ctx.recommendations = [_rec()]
    ctx.recommendation_portfolio = {"near_term": [_rec()]}
    ctx.contradiction_metrics = {}

    agent = QAAgent()
    result_ctx = agent._execute(ctx)

    assert "recommendation_evaluation_validation" in result_ctx.qa
    v = result_ctx.qa["recommendation_evaluation_validation"]
    assert "scores_present" in v
    assert "traceability_present" in v
    assert "dimension_scores_present" in v
    assert "warnings_present" in v
    assert "summary_present" in v
    assert v["scores_present"] is True
    assert v["dimension_scores_present"] is True
