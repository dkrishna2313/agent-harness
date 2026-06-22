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
# Diagnostics helpers
# ---------------------------------------------------------------------------

_PENALTY_THRESHOLD = 0.50  # dimension score below this is a "significant penalty"


def _identify_primary_penalty(
    ev_sup: float,
    reasoning: float,
    tradeoff: float,
    risk: float,
    actionability: float,
) -> str | None:
    """Return the name of the lowest-scoring dimension if it falls below the penalty
    threshold, weighted by its contribution to the aggregate score.  Returns None
    when all dimensions are healthy (≥ threshold)."""
    weighted: list[tuple[float, str]] = [
        (ev_sup   * _DIMENSION_WEIGHTS["evidence_support"], "missing_evidence_links"),
        (reasoning * _DIMENSION_WEIGHTS["reasoning"],       "weak_reasoning"),
        (tradeoff  * _DIMENSION_WEIGHTS["tradeoff"],        "no_tradeoff_awareness"),
        (risk      * _DIMENSION_WEIGHTS["risk"],            "no_risk_identification"),
        (actionability * _DIMENSION_WEIGHTS["actionability"], "low_actionability"),
    ]
    raw: list[tuple[float, str]] = [
        (ev_sup,   "missing_evidence_links"),
        (reasoning, "weak_reasoning"),
        (tradeoff,  "no_tradeoff_awareness"),
        (risk,      "no_risk_identification"),
        (actionability, "low_actionability"),
    ]
    worst_raw = min(raw, key=lambda t: t[0])
    if worst_raw[0] < _PENALTY_THRESHOLD:
        return worst_raw[1]
    return None


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

    # Diagnostics
    missing_evidence_links = len(rec.get("supporting_evidence", [])) == 0
    primary_penalty = _identify_primary_penalty(ev_sup, reasoning, tradeoff, risk, actionability)

    traceability = build_recommendation_traceability(rec)
    traceability["missing_evidence_links"] = missing_evidence_links

    return {
        "recommendation_id": rec.get("id", ""),
        "title": rec.get("title", ""),
        "evidence_support_score": ev_sup,
        "reasoning_score": reasoning,
        "tradeoff_score": tradeoff,
        "risk_score": risk,
        "actionability_score": actionability,
        "recommendation_score": aggregate,
        "aggregate_score": aggregate,  # alias for spec compatibility
        "missing_evidence_links": missing_evidence_links,
        "primary_penalty": primary_penalty,
        "traceability": traceability,
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
    _EMPTY: dict = {
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
        "recommendation_summary": {
            "recommendation_count": 0,
            "average_score": 0.0,
            "lowest_score": 0.0,
            "highest_score": 0.0,
        },
        "recommendation_warnings": [],
        "traceability": [],
    }
    if not recommendations:
        return _EMPTY

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

    all_scores = [s["recommendation_score"] for s in scored]
    traceability = [s["traceability"] for s in scored]

    # Collect warnings for any recommendation with a primary penalty
    warnings: list[dict] = []
    for s in scored:
        if s.get("primary_penalty"):
            warnings.append({
                "recommendation_id": s["recommendation_id"],
                "issue": s["primary_penalty"],
                "aggregate_score": s["recommendation_score"],
            })
        elif s.get("missing_evidence_links"):
            warnings.append({
                "recommendation_id": s["recommendation_id"],
                "issue": "missing_evidence_links",
                "aggregate_score": s["recommendation_score"],
            })

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
        "recommendation_summary": {
            "recommendation_count": n,
            "average_score": round(sum(all_scores) / n, 3),
            "lowest_score": round(min(all_scores), 3),
            "highest_score": round(max(all_scores), 3),
        },
        "recommendation_warnings": warnings,
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

_RE_LOGICAL_CONN = re.compile(
    r"\bbecause\b|\bsince\b|\btherefore\b|\bthus\b|\bconsequently\b|\bhence\b",
    re.IGNORECASE,
)
_RE_RISK_WORD = re.compile(
    r"\brisk\b|\bchallenge\b|\bbarrier\b|\bconstraint\b|\bfail\b|\bhazard\b",
    re.IGNORECASE,
)


def score_recommendation_dimensions_from_memo(memo_inferences: list[str]) -> dict[str, float]:
    """Return per-dimension proxy scores from ResearchMemo.inferences.

    Used in the benchmark evaluation path where functional-pipeline
    RecommendationAgent output is not available.  Scores are estimates only —
    each dimension uses lightweight text heuristics on the inference strings.
    """
    n = len(memo_inferences)
    if n == 0:
        return {
            "evidence_support": 0.1,
            "reasoning": 0.1,
            "tradeoff": 0.1,
            "risk": 0.1,
            "actionability": 0.1,
        }

    # evidence_support: inferences with citation-style patterns ("according to", "data shows")
    re_evidence = re.compile(
        r"\baccording\s+to\b|\bdata\s+shows?\b|\bevidence\s+suggests?\b"
        r"|\bsource\b|\bcited\b|\breport\b|\bstudy\b|\banalysis\b",
        re.IGNORECASE,
    )
    ev_fraction = sum(1 for inf in memo_inferences if re_evidence.search(inf)) / n

    # reasoning: logical connectors present
    logic_fraction = sum(1 for inf in memo_inferences if _RE_LOGICAL_CONN.search(inf)) / n

    # tradeoff: tradeoff keywords
    trade_fraction = sum(
        1 for inf in memo_inferences
        if any(kw in inf.lower() for kw in _TRADEOFF_KEYWORDS)
    ) / n

    # risk: risk words
    risk_fraction = sum(1 for inf in memo_inferences if _RE_RISK_WORD.search(inf)) / n

    # actionability: action verbs + length
    actionable_count = sum(
        1 for inf in memo_inferences
        if len(inf) >= 60 and any(v in inf.lower() for v in _ACTION_VERBS)
    )
    action_fraction = actionable_count / n

    count_bonus = min(0.30, n / 4.0 * 0.30)  # up to +0.30 for having ≥4 inferences

    def _score(fraction: float) -> float:
        return round(min(1.0, 0.40 + fraction * 0.60 + count_bonus), 3)

    return {
        "evidence_support": _score(ev_fraction),
        "reasoning": _score(logic_fraction),
        "tradeoff": _score(trade_fraction),
        "risk": _score(risk_fraction),
        "actionability": _score(action_fraction),
    }


def score_recommendations_from_memo(memo_inferences: list[str]) -> float:
    """Return a 0–1 composite recommendation proxy score from ResearchMemo.inferences.

    Used in the benchmark evaluation path where functional-pipeline
    RecommendationAgent output is not available.
    """
    dims = score_recommendation_dimensions_from_memo(memo_inferences)
    return round(
        dims["evidence_support"] * _DIMENSION_WEIGHTS["evidence_support"]
        + dims["reasoning"]      * _DIMENSION_WEIGHTS["reasoning"]
        + dims["tradeoff"]       * _DIMENSION_WEIGHTS["tradeoff"]
        + dims["risk"]           * _DIMENSION_WEIGHTS["risk"]
        + dims["actionability"]  * _DIMENSION_WEIGHTS["actionability"],
        3,
    )
