"""Recommendation quality evaluator (J6.6).

Scores each recommendation on five independent dimensions and aggregates
them into a single 0–1 composite score.

Dimensions
----------
evidence_support  – does the recommendation trace back to evidence IDs?
reasoning         – does it follow logically from hypotheses + rationale?
tradeoff          – does it acknowledge benefits, costs, and constraints?
risk              – does it identify specific failure modes?
actionability     – does it name concrete actions (not just "monitor")?

Public API
----------
score_single_recommendation(rec, evidence_ids, hypothesis_ids, challenge_ids)
evaluate_recommendations(recommendations, context_dict)
build_recommendation_traceability(rec)
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_ACTION_VERBS: frozenset[str] = frozenset({
    "invest", "deploy", "build", "develop", "prioritize", "implement",
    "adopt", "establish", "expand", "reduce", "increase", "commission",
    "procure", "integrate", "partner", "launch", "accelerate", "begin",
    "start", "transition", "shift", "require", "mandate", "fund",
    "negotiate", "secure", "contract", "design", "upgrade", "replace",
})
_HEDGE_ONLY_VERBS: frozenset[str] = frozenset({
    "monitor", "evaluate", "consider", "explore", "assess", "review",
    "investigate", "study", "examine", "analyze", "watch", "track",
    "observe", "await", "continue",
})
_TRADEOFF_KEYWORDS: frozenset[str] = frozenset({
    "however", "but", "although", "while", "trade-off", "tradeoff",
    "trade off", "cost", "benefit", "challenge", "constraint", "drawback",
    "limitation", "whereas", "despite", "downside", "upside", "offset",
    "compromise", "balance",
})

# Aggregate weight per dimension — must sum to 1.0
_DIMENSION_WEIGHTS: dict[str, float] = {
    "evidence_support": 0.25,
    "reasoning":        0.25,
    "tradeoff":         0.15,
    "risk":             0.20,
    "actionability":    0.15,
}


# ---------------------------------------------------------------------------
# Per-dimension scorers
# ---------------------------------------------------------------------------

def _score_evidence_support(rec: dict, available_evidence_ids: set[str]) -> float:
    """0–1: recommendation is grounded in specific evidence IDs."""
    ev_ids: list[str] = rec.get("supporting_evidence", [])
    n = len(ev_ids)
    if n == 0:
        return 0.0
    base = {1: 0.50, 2: 0.70, 3: 0.85}.get(n, 1.0)
    # Bonus if the IDs actually appear in the known evidence set
    if available_evidence_ids:
        matched = sum(1 for eid in ev_ids if eid in available_evidence_ids)
        match_ratio = matched / n
        base = min(1.0, base + match_ratio * 0.1)
    return round(base, 3)


def _score_reasoning(rec: dict, available_hypothesis_ids: set[str]) -> float:
    """0–1: recommendation logically follows from hypotheses and has rationale."""
    score = 0.0
    hyp_ids: list[str] = rec.get("supported_by_hypotheses", [])
    if hyp_ids:
        score += 0.40
        # Bonus if IDs actually reference known hypotheses
        if available_hypothesis_ids:
            matched = sum(1 for hid in hyp_ids if hid in available_hypothesis_ids)
            if matched > 0:
                score += 0.10
    rationale = rec.get("confidence_rationale", "")
    if len(rationale) >= 20:
        score += 0.25
    summary = rec.get("summary", "")
    if len(summary) >= 100:
        score += 0.25
    return min(1.0, round(score, 3))


def _score_tradeoff(rec: dict) -> float:
    """0–1: recommendation acknowledges benefits, costs, and constraints."""
    summary = rec.get("summary", "").lower()
    has_tradeoff_kw = any(kw in summary for kw in _TRADEOFF_KEYWORDS)
    has_triggers = len(rec.get("trigger_conditions", [])) > 0
    # Longer summaries are more likely to contain nuance
    has_length = len(summary) >= 150

    score = 0.0
    if has_tradeoff_kw:
        score += 0.55
    if has_triggers:
        score += 0.30
    if has_length:
        score += 0.15
    return min(1.0, round(score, 3))


def _score_risk(rec: dict) -> float:
    """0–1: recommendation identifies specific failure modes."""
    risks: list[str] = rec.get("key_risks", [])
    n = len(risks)
    if n == 0:
        return 0.0
    base = {1: 0.50, 2: 0.75}.get(n, 1.0)
    # Specificity bonus: risks that are ≥ 20 chars are likely non-trivial
    specific = sum(1 for r in risks if len(r) >= 20)
    specificity_bonus = min(0.15, (specific / n) * 0.15)
    return min(1.0, round(base + specificity_bonus, 3))


def _score_actionability(rec: dict) -> float:
    """0–1: recommendation names concrete actions over vague monitoring."""
    title = rec.get("title", "").lower()
    summary = rec.get("summary", "").lower()
    combined = title + " " + summary

    has_action = any(v in combined for v in _ACTION_VERBS)
    hedge_count = sum(1 for v in _HEDGE_ONLY_VERBS if v in combined)
    hedge_dominated = hedge_count >= 2 and not has_action

    score = 0.0
    if has_action:
        score += 0.50
    if not hedge_dominated:
        score += 0.20
    if rec.get("time_horizon") in ("near_term", "medium_term", "long_term"):
        score += 0.15
    if rec.get("priority") in ("high", "medium"):
        score += 0.15
    return min(1.0, round(score, 3))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_single_recommendation(
    rec: dict,
    evidence_ids: set[str] | None = None,
    hypothesis_ids: set[str] | None = None,
    challenge_ids: set[str] | None = None,
) -> dict:
    """Return a dimension-score dict for one recommendation.

    Parameters
    ----------
    rec:
        A recommendation dict (keys: id, title, summary, supporting_evidence,
        supported_by_hypotheses, key_risks, trigger_conditions, time_horizon,
        priority, confidence_rationale).
    evidence_ids, hypothesis_ids, challenge_ids:
        Sets of known IDs used to verify linkage.
    """
    ev_ids = evidence_ids or set()
    hyp_ids = hypothesis_ids or set()

    ev_sup = _score_evidence_support(rec, ev_ids)
    reasoning = _score_reasoning(rec, hyp_ids)
    tradeoff = _score_tradeoff(rec)
    risk = _score_risk(rec)
    actionability = _score_actionability(rec)

    aggregate = round(
        ev_sup        * _DIMENSION_WEIGHTS["evidence_support"]
        + reasoning   * _DIMENSION_WEIGHTS["reasoning"]
        + tradeoff    * _DIMENSION_WEIGHTS["tradeoff"]
        + risk        * _DIMENSION_WEIGHTS["risk"]
        + actionability * _DIMENSION_WEIGHTS["actionability"],
        3,
    )

    return {
        "recommendation_id": rec.get("id", ""),
        "title": rec.get("title", ""),
        "evidence_support_score": ev_sup,
        "reasoning_score": reasoning,
        "tradeoff_score": tradeoff,
        "risk_score": risk,
        "actionability_score": actionability,
        "recommendation_score": aggregate,
        "traceability": build_recommendation_traceability(rec),
    }


def evaluate_recommendations(
    recommendations: list[dict],
    *,
    evidence_ids: set[str] | None = None,
    hypothesis_ids: set[str] | None = None,
    challenge_ids: set[str] | None = None,
) -> dict:
    """Score all recommendations and return an aggregate evaluation dict.

    Returns
    -------
    {
        "recommendation_scores": [...],   # per-recommendation score dicts
        "aggregate": {                    # portfolio-level aggregates
            "recommendation_count": N,
            "mean_evidence_support": ...,
            "mean_reasoning": ...,
            "mean_tradeoff": ...,
            "mean_risk": ...,
            "mean_actionability": ...,
            "recommendation_score": ...,
        },
        "traceability": [...]             # traceability records
    }
    """
    if not recommendations:
        return {
            "recommendation_scores": [],
            "aggregate": {
                "recommendation_count": 0,
                "mean_evidence_support": 0.0,
                "mean_reasoning": 0.0,
                "mean_tradeoff": 0.0,
                "mean_risk": 0.0,
                "mean_actionability": 0.0,
                "recommendation_score": 0.0,
            },
            "traceability": [],
        }

    ev_ids = evidence_ids or set()
    hyp_ids = hypothesis_ids or set()
    ch_ids = challenge_ids or set()

    scored = [
        score_single_recommendation(rec, ev_ids, hyp_ids, ch_ids)
        for rec in recommendations
    ]
    n = len(scored)

    def _mean(key: str) -> float:
        return round(sum(s[key] for s in scored) / n, 3)

    traceability = [s["traceability"] for s in scored]

    return {
        "recommendation_scores": scored,
        "aggregate": {
            "recommendation_count": n,
            "mean_evidence_support": _mean("evidence_support_score"),
            "mean_reasoning": _mean("reasoning_score"),
            "mean_tradeoff": _mean("tradeoff_score"),
            "mean_risk": _mean("risk_score"),
            "mean_actionability": _mean("actionability_score"),
            "recommendation_score": _mean("recommendation_score"),
        },
        "traceability": traceability,
    }


def build_recommendation_traceability(rec: dict) -> dict:
    """Extract structured traceability from a recommendation dict.

    Returns
    -------
    {
        "recommendation_id": "R1",
        "evidence_ids": [...],
        "hypothesis_ids": [...],
        "challenge_ids": []
    }
    """
    return {
        "recommendation_id": rec.get("id", ""),
        "evidence_ids": list(rec.get("supporting_evidence", [])),
        "hypothesis_ids": list(rec.get("supported_by_hypotheses", [])),
        "challenge_ids": list(rec.get("supported_by_challenges", [])),
    }


# ---------------------------------------------------------------------------
# Benchmark proxy scorer (used by score_agents when ResearchMemo has no
# functional-pipeline recommendations — inferences serve as a proxy)
# ---------------------------------------------------------------------------

def score_recommendations_from_memo(memo_inferences: list[str]) -> float:
    """Return a 0–1 recommendation proxy score from ResearchMemo.inferences.

    Used in the benchmark evaluation path where functional-pipeline
    RecommendationAgent output is not available.
    """
    n = len(memo_inferences)
    if n == 0:
        return 0.1  # minimal credit
    # Count inferences that look actionable (≥ 60 chars with action verb)
    actionable = sum(
        1 for inf in memo_inferences
        if len(inf) >= 60 and any(v in inf.lower() for v in _ACTION_VERBS)
    )
    count_score = min(1.0, n / 4.0)        # ≥4 inferences → full credit
    quality_score = min(1.0, actionable / max(1, n))
    return round(count_score * 0.5 + quality_score * 0.5, 3)
