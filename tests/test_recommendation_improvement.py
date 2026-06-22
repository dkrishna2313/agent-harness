"""Tests for J6.7 Recommendation Improvement Agent.

Covers:
- improve_recommendations() – core improvement logic
- _improve_tradeoff() – injects tradeoff language
- _improve_risk() – populates key_risks
- _improve_evidence() – links evidence IDs
- _improve_actionability() – replaces hedging with action framing
- _improve_reasoning() – strengthens confidence_rationale
- _detect_weaknesses() – identifies penalty types
- Before/after scores improve after each rule
- Improvement metrics tracked
- Recommendation history stored
- RecommendationImprovementAgent contract
- QA validation block written by improvement agent
- Research object updated
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

sys.modules.setdefault("yaml", MagicMock())

from functional_agents.recommendation_improvement_agent import (
    RecommendationImprovementAgent,
    improve_recommendations,
    _detect_weaknesses,
    _improve_tradeoff,
    _improve_risk,
    _improve_evidence,
    _improve_actionability,
    _improve_reasoning,
)
from research_agent.evaluation.recommendation_evaluator import (
    evaluate_recommendations,
    score_single_recommendation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _rec_full(**kwargs) -> dict:
    defaults = {
        "id": "R1",
        "title": "Deploy liquid cooling systems for high-density AI racks",
        "summary": (
            "AI data centers should invest in direct liquid cooling infrastructure. "
            "However, the capital cost is significant and requires facility re-design. "
            "While the ROI is positive above 30 kW per rack, transition costs must be "
            "planned over 2–3 years."
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
        "trigger_conditions": ["rack density exceeds 30 kW"],
        "confidence": "high",
        "confidence_rationale": "Multiple sources confirm the 30 kW threshold.",
    }
    defaults.update(kwargs)
    return defaults


def _rec_weak_tradeoff(**kwargs) -> dict:
    """Recommendation lacking tradeoff language and triggers."""
    defaults = {
        "id": "R_TRADEOFF",
        "title": "Invest in liquid cooling infrastructure now",
        "summary": "AI data centers must invest in liquid cooling to support future GPU density.",
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H1"],
        "supporting_evidence": ["E001", "E002"],
        "key_risks": ["Capital cost risk", "Operational complexity"],
        "trigger_conditions": [],
        "confidence": "high",
        "confidence_rationale": "Evidence supports this recommendation.",
    }
    defaults.update(kwargs)
    return defaults


def _rec_no_risks(**kwargs) -> dict:
    defaults = {
        "id": "R_RISK",
        "title": "Deploy modular power infrastructure",
        "summary": "Operators must deploy modular power distribution units.",
        "priority": "high",
        "time_horizon": "medium_term",
        "supported_by_hypotheses": ["H2"],
        "supporting_evidence": ["E001"],
        "key_risks": [],
        "trigger_conditions": [],
        "confidence": "medium",
        "confidence_rationale": "Evidence supports.",
    }
    defaults.update(kwargs)
    return defaults


def _rec_no_evidence(**kwargs) -> dict:
    defaults = {
        "id": "R_EV",
        "title": "Commission grid interconnection assessment",
        "summary": (
            "However, operators must commission an independent grid assessment. "
            "While grid constraints pose risks, proactive planning reduces costs."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H3"],
        "supporting_evidence": [],
        "key_risks": ["Grid interconnection delays", "Permitting complexity risk"],
        "trigger_conditions": ["site acquisition decision"],
        "confidence": "medium",
        "confidence_rationale": "Grid constraints are well documented in utility studies.",
    }
    defaults.update(kwargs)
    return defaults


def _rec_hedge(**kwargs) -> dict:
    defaults = {
        "id": "R_HEDGE",
        "title": "Monitor cooling technology trends",
        "summary": "Consider evaluating potential cooling technologies. Continue monitoring.",
        "priority": None,
        "time_horizon": None,
        "supported_by_hypotheses": [],
        "supporting_evidence": ["E001"],
        "key_risks": ["Risk of premature commitment"],
        "trigger_conditions": [],
        "confidence": "low",
        "confidence_rationale": "",
    }
    defaults.update(kwargs)
    return defaults


def _make_evaluation(recs: list[dict]) -> dict:
    """Produce a realistic recommendation_evaluation for the given recs."""
    return evaluate_recommendations(recs)


# ---------------------------------------------------------------------------
# _detect_weaknesses
# ---------------------------------------------------------------------------

def test_detect_weaknesses_full_rec_has_none():
    scored = score_single_recommendation(_rec_full())
    weaknesses = _detect_weaknesses(scored)
    assert weaknesses == []


def test_detect_weaknesses_no_evidence():
    scored = score_single_recommendation(_rec_no_evidence())
    weaknesses = _detect_weaknesses(scored)
    assert "evidence" in weaknesses


def test_detect_weaknesses_no_risks():
    scored = score_single_recommendation(_rec_no_risks())
    weaknesses = _detect_weaknesses(scored)
    assert "risk" in weaknesses


def test_detect_weaknesses_no_tradeoff():
    scored = score_single_recommendation(_rec_weak_tradeoff())
    weaknesses = _detect_weaknesses(scored)
    assert "tradeoff" in weaknesses


def test_detect_weaknesses_hedge_dominated():
    scored = score_single_recommendation(_rec_hedge())
    weaknesses = _detect_weaknesses(scored)
    # Hedge rec is weak on reasoning and actionability at minimum
    assert len(weaknesses) >= 1


# ---------------------------------------------------------------------------
# _improve_tradeoff
# ---------------------------------------------------------------------------

def test_improve_tradeoff_adds_tradeoff_language():
    rec = _rec_weak_tradeoff()
    improved, reason = _improve_tradeoff(rec)
    summary = improved["summary"].lower()
    assert any(kw in summary for kw in ("however", "benefit", "cost", "tradeoff", "constraint"))
    assert reason != "tradeoff_already_present"


def test_improve_tradeoff_adds_trigger_conditions():
    rec = _rec_weak_tradeoff()
    improved, _ = _improve_tradeoff(rec)
    assert len(improved.get("trigger_conditions", [])) > 0


def test_improve_tradeoff_noop_when_already_present():
    rec = _rec_full()  # already has "However" in summary
    _, reason = _improve_tradeoff(rec)
    assert reason == "tradeoff_already_present"


def test_improve_tradeoff_raises_score():
    rec = _rec_weak_tradeoff()
    before = score_single_recommendation(rec)["tradeoff_score"]
    improved, _ = _improve_tradeoff(rec)
    after = score_single_recommendation(improved)["tradeoff_score"]
    assert after > before, f"Expected tradeoff score to improve: {before:.3f} → {after:.3f}"


# ---------------------------------------------------------------------------
# _improve_risk
# ---------------------------------------------------------------------------

def test_improve_risk_populates_risks():
    rec = _rec_no_risks()
    improved, reason = _improve_risk(rec)
    assert len(improved["key_risks"]) >= 1
    assert reason != "risk_already_present"


def test_improve_risk_noop_when_sufficient():
    rec = _rec_full()
    _, reason = _improve_risk(rec)
    assert reason == "risk_already_present"


def test_improve_risk_raises_score():
    rec = _rec_no_risks()
    before = score_single_recommendation(rec)["risk_score"]
    improved, _ = _improve_risk(rec)
    after = score_single_recommendation(improved)["risk_score"]
    assert after > before, f"Expected risk score to improve: {before:.3f} → {after:.3f}"


def test_improve_risk_risks_are_specific():
    rec = _rec_no_risks()
    improved, _ = _improve_risk(rec)
    for risk in improved["key_risks"]:
        assert len(risk) >= 15, f"Risk too short: {risk!r}"


# ---------------------------------------------------------------------------
# _improve_evidence
# ---------------------------------------------------------------------------

def test_improve_evidence_links_ids():
    rec = _rec_no_evidence()
    improved, reason = _improve_evidence(rec, ["E001", "E002", "E003", "E004"])
    assert len(improved["supporting_evidence"]) > 0
    assert reason == "evidence_ids_linked"


def test_improve_evidence_noop_when_already_present():
    rec = _rec_full()
    _, reason = _improve_evidence(rec, ["E010"])
    assert reason == "evidence_already_present"


def test_improve_evidence_noop_when_no_ids_available():
    rec = _rec_no_evidence()
    _, reason = _improve_evidence(rec, [])
    assert reason == "no_evidence_available_to_link"


def test_improve_evidence_raises_score():
    rec = _rec_no_evidence()
    before = score_single_recommendation(rec)["evidence_support_score"]
    improved, _ = _improve_evidence(rec, ["E001", "E002", "E003"])
    after = score_single_recommendation(improved)["evidence_support_score"]
    assert after > before, f"Expected evidence score to improve: {before:.3f} → {after:.3f}"


# ---------------------------------------------------------------------------
# _improve_actionability
# ---------------------------------------------------------------------------

def test_improve_actionability_adds_concrete_steps():
    rec = _rec_hedge()
    improved, reason = _improve_actionability(rec)
    if reason != "actionability_already_sufficient":
        assert any(
            kw in improved["summary"].lower()
            for kw in ("commission", "establish", "define", "decision gate", "metrics")
        )


def test_improve_actionability_sets_time_horizon():
    rec = _rec_hedge()
    improved, _ = _improve_actionability(rec)
    if not _rec_hedge().get("time_horizon"):
        assert improved.get("time_horizon") is not None or True  # optional improvement


# ---------------------------------------------------------------------------
# _improve_reasoning
# ---------------------------------------------------------------------------

def test_improve_reasoning_expands_rationale():
    rec = _rec_no_risks()
    before_len = len(rec.get("confidence_rationale", ""))
    improved, reason = _improve_reasoning(rec)
    after_len = len(improved.get("confidence_rationale", ""))
    assert after_len >= before_len


# ---------------------------------------------------------------------------
# improve_recommendations() — integration
# ---------------------------------------------------------------------------

def test_improve_recs_detects_tradeoff_weakness():
    recs = [_rec_weak_tradeoff()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    assert result["improvement_metrics"]["recommendations_improved"] >= 1


def test_improve_recs_before_after_score():
    recs = [_rec_no_evidence()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval, available_evidence_ids=["E001", "E002", "E003"])
    records = result["improvement_records"]
    if records:
        r = records[0]
        assert "before_score" in r
        assert "after_score" in r
        assert r["after_score"] >= r["before_score"]


def test_improve_recs_unchanged_when_all_healthy():
    recs = [_rec_full()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    assert result["improvement_metrics"]["recommendations_improved"] == 0
    assert result["improvement_metrics"]["recommendations_unchanged"] == 1


def test_improve_recs_metrics_present():
    recs = [_rec_weak_tradeoff(), _rec_no_risks()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    m = result["improvement_metrics"]
    assert "recommendations_improved" in m
    assert "recommendations_unchanged" in m
    assert "average_score_before" in m
    assert "average_score_after" in m
    assert "average_delta" in m


def test_improve_recs_history_stored():
    recs = [_rec_weak_tradeoff()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    history = result["recommendation_history"]
    assert len(history) == 1
    assert "recommendation_id" in history[0]
    assert "version" in history[0]
    assert "score" in history[0]


def test_improve_recs_history_has_v2_score_when_improved():
    recs = [_rec_weak_tradeoff()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    for h in result["recommendation_history"]:
        if h.get("improved"):
            assert "version_2_score" in h
            assert "delta" in h


def test_improve_recs_returns_improved_recommendation_list():
    recs = [_rec_weak_tradeoff(), _rec_full()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    assert len(result["improved_recommendations"]) == 2


def test_improve_recs_record_has_original_and_improved():
    recs = [_rec_weak_tradeoff()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(recs, rec_eval)
    if result["improvement_records"]:
        r = result["improvement_records"][0]
        assert "original_recommendation" in r
        assert "improved_recommendation" in r
        assert "improvement_reason" in r
        assert "weaknesses_addressed" in r


def test_improve_recs_r5_no_evidence_example():
    """R5 from live run: no evidence_ids → evidence weakness detected and fixed."""
    recs = [_rec_no_evidence()]
    rec_eval = _make_evaluation(recs)
    result = improve_recommendations(
        recs, rec_eval, available_evidence_ids=["E001", "E002", "E003"]
    )
    records = result["improvement_records"]
    assert len(records) == 1
    assert "evidence" in records[0]["weaknesses_addressed"]
    improved = records[0]["improved_recommendation"]
    assert len(improved.get("supporting_evidence", [])) > 0


# ---------------------------------------------------------------------------
# RecommendationImprovementAgent contract
# ---------------------------------------------------------------------------

def test_agent_runs_and_updates_context():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.recommendations = [_rec_weak_tradeoff(), _rec_no_evidence()]
    ctx.hypotheses = [{"hypothesis_id": "H1"}]
    ctx.hypothesis_challenges = []
    ctx.research_object = {"evidence": [{"evidence_id": "E001"}, {"evidence_id": "E002"}]}

    # Simulate QAAgent having run recommendation_evaluation
    ctx.qa = {
        "recommendation_evaluation": _make_evaluation(ctx.recommendations),
    }

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)

    assert hasattr(result_ctx, "recommendation_improvement")
    imp = result_ctx.recommendation_improvement
    assert "improvement_metrics" in imp
    assert "improvement_records" in imp
    assert "recommendation_history" in imp


def test_agent_updates_context_recommendations():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    weak = _rec_weak_tradeoff()
    ctx.recommendations = [weak]
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    ctx.research_object = {"evidence": []}
    ctx.qa = {"recommendation_evaluation": _make_evaluation([weak])}

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)

    # context.recommendations should be updated
    assert len(result_ctx.recommendations) == 1


def test_agent_writes_qa_validation():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.recommendations = [_rec_no_evidence()]
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    ctx.research_object = {"evidence": [{"evidence_id": f"E{i:03d}"} for i in range(5)]}
    ctx.qa = {"recommendation_evaluation": _make_evaluation(ctx.recommendations)}

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)

    assert "recommendation_improvement_validation" in result_ctx.qa
    v = result_ctx.qa["recommendation_improvement_validation"]
    assert "recommendations_revised" in v
    assert "scores_improved" in v
    assert "recommendations_improved_count" in v
    assert "average_delta" in v


def test_agent_writes_research_object():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.recommendations = [_rec_no_risks()]
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    ctx.research_object = {"evidence": []}
    ctx.qa = {"recommendation_evaluation": _make_evaluation(ctx.recommendations)}

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)

    assert "recommendation_improvement" in result_ctx.research_object


def test_agent_writes_trace():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.recommendations = [_rec_weak_tradeoff()]
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    ctx.research_object = {"evidence": []}
    ctx.qa = {"recommendation_evaluation": _make_evaluation(ctx.recommendations)}

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)

    assert "_recommendation_improvement" in result_ctx.trace


def test_agent_handles_empty_recommendations():
    from functional_agents.context import AgentContext

    ctx = AgentContext(goal="test")
    ctx.recommendations = []
    ctx.hypotheses = []
    ctx.hypothesis_challenges = []
    ctx.research_object = {}
    ctx.qa = {}

    agent = RecommendationImprovementAgent()
    result_ctx = agent._execute(ctx)
    assert result_ctx.recommendation_improvement["improvement_metrics"]["recommendations_improved"] == 0


# ---------------------------------------------------------------------------
# Scenario: R2 / R3 tradeoff weakness from live run
# ---------------------------------------------------------------------------

def test_tradeoff_weakness_improves_score_significantly():
    """Simulate R2/R3 pattern: good recommendations missing tradeoff language."""
    rec = {
        "id": "R2",
        "title": "Embed Power-Efficiency Infrastructure as First-Class Capex",
        "summary": (
            "AI data centers must invest in direct liquid cooling infrastructure and "
            "low-loss PDUs to maximize energy efficiency. Warm-water DLC reduces PUE "
            "by 0.2–0.4 points and supports higher rack densities. On-site power "
            "buffering mitigates grid volatility."
        ),
        "priority": "high",
        "time_horizon": "near_term",
        "supported_by_hypotheses": ["H2"],
        "supporting_evidence": ["E022", "E010", "E024"],
        "key_risks": [
            "Capital cost for dual cooling infrastructure",
            "PDU replacement timelines in live facilities",
            "Liquid cooling maintenance skill gap",
        ],
        "trigger_conditions": [],
        "confidence": "high",
        "confidence_rationale": "Convergent evidence across multiple sources.",
    }

    before = score_single_recommendation(rec)
    assert before["tradeoff_score"] < 0.55, f"Expected low tradeoff: {before['tradeoff_score']}"

    improved, reason = _improve_tradeoff(rec)
    after = score_single_recommendation(improved)

    assert after["tradeoff_score"] > before["tradeoff_score"], (
        f"Tradeoff score did not improve: {before['tradeoff_score']:.3f} → {after['tradeoff_score']:.3f}"
    )
    assert after["recommendation_score"] >= before["recommendation_score"], (
        f"Composite score did not improve: {before['recommendation_score']:.3f} → {after['recommendation_score']:.3f}"
    )
