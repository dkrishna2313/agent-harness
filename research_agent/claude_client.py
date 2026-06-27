"""Anthropic Claude integration for the research workflow."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Protocol

LOGGER = logging.getLogger(__name__)

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from .evidence_filter import sanitize_evidence_items
from .prompts import SYSTEM_PROMPT
from .schemas import (
    Chunk,
    ClaudeCallTrace,
    EvidenceItem,
    ResearchMemo,
    ResearchPlan,
    SourceDocument,
    assign_evidence_ids,
)

DEFAULT_MODEL = "claude-sonnet-4-6"
# Haiku is available as an opt-in via ANTHROPIC_EXTRACTION_MODEL env var, but
# defaults to Sonnet — Haiku extracts significantly fewer items per question
# which degrades citation coverage in the synthesized memo.
DEFAULT_EXTRACTION_MODEL = DEFAULT_MODEL


class EvidenceExtractionPayload(BaseModel):
    """Strict payload — used only to generate the JSON schema for the tool definition."""

    evidence_items: list[EvidenceItem] = Field(default_factory=list)


class _RawEvidencePayload(BaseModel):
    """Lenient payload — used to validate Claude's response.

    Items are kept as raw dicts so that ``extract_evidence`` can validate them
    one-by-one and discard individual failures without losing the entire batch.
    """

    evidence_items: list = Field(default_factory=list)


class MemoSynthesisPayload(BaseModel):
    executive_summary: str = ""
    confirmed_facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    power_implications: list[str] = Field(default_factory=list)
    cooling_implications: list[str] = Field(default_factory=list)
    networking_implications: list[str] = Field(default_factory=list)
    rack_architecture_implications: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class ResearchPlanningPayload(BaseModel):
    """Structured output for PlannerAgent (J5.1)."""

    research_type: str = Field(
        description="Question classification: FACT_LOOKUP, COMPARISON, EXPLANATION, or RESEARCH"
    )
    subquestions: list[str] = Field(
        default_factory=list,
        description="3-7 focused subquestions that decompose the main question",
    )
    investigation_areas: list[str] = Field(
        default_factory=list,
        description="4-8 topic areas to investigate (e.g. Power, Cooling, Economics)",
    )
    profiles_used: list[str] = Field(
        default_factory=list,
        description="Profile names whose domain knowledge informed this plan",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of the classification and planning choices",
    )


class DecisionModelPayload(BaseModel):
    """Structured output for ProblemFramingAgent (J6.1).

    Transforms a business goal into a structured Decision Model that guides
    the rest of the research pipeline.
    """

    objective: str = Field(
        description="The core decision objective, restated as a precise research goal"
    )
    decision_areas: list[str] = Field(
        default_factory=list,
        description="3-6 key decision areas or dimensions the research must address",
    )
    critical_uncertainties: list[str] = Field(
        default_factory=list,
        description="2-5 critical unknowns that most affect the decision outcome",
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="3-6 specific research questions derived from the goal",
    )
    evidence_requirements: list[str] = Field(
        default_factory=list,
        description="Types of evidence needed (e.g. market data, technical specs, case studies)",
    )


class ResearchStrategyPayload(BaseModel):
    """Structured output for ResearchStrategyAgent (J6.2).

    Translates the Decision Model into an executable research plan that guides
    profile selection, evidence gathering, and coverage targeting.
    """

    profile_priorities: dict[str, int] = Field(
        default_factory=dict,
        description="Profile name → integer priority rank (1 = highest). Lists all profiles in order of relevance to the decision model.",
    )
    research_question_priorities: list[dict] = Field(
        default_factory=list,
        description='Ordered list of {question: str, priority: int} dicts ranked by decision impact.',
    )
    required_evidence: list[str] = Field(
        default_factory=list,
        description="Specific evidence items needed (e.g. 'AI power demand forecasts', 'SMR deployment schedules')",
    )
    source_priorities: list[str] = Field(
        default_factory=list,
        description="Source types in priority order (e.g. 'regulatory filings', 'grid operator reports')",
    )
    coverage_targets: dict[str, str] = Field(
        default_factory=dict,
        description="Topic/area → required coverage level: 'strong', 'moderate', or 'light'",
    )
    strategy_rationale: str = Field(
        default="",
        description="2-3 sentence explanation of the strategy choices",
    )


class HypothesisItem(BaseModel):
    """A single competing hypothesis (J6.3)."""

    id: str = Field(description="Short identifier, e.g. 'H1'")
    title: str = Field(description="One-line hypothesis title")
    summary: str = Field(description="2-4 sentence explanation of the hypothesis")
    type: str = Field(
        default="general",
        description="Hypothesis category, e.g. 'constraint_dominant', 'technology_option', 'portfolio_strategy'",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs (Exxx) that support this hypothesis",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs that contradict or weaken this hypothesis",
    )
    evidence_gaps: list[str] = Field(
        default_factory=list,
        description="Types of evidence that are absent but needed to test this hypothesis",
    )
    confidence: str = Field(
        default="medium",
        description="Confidence level: 'high', 'medium', or 'low'",
    )
    confidence_rationale: str = Field(
        default="",
        description="1-2 sentence explanation of why this confidence level was assigned",
    )
    decision_implications: list[str] = Field(
        default_factory=list,
        description="Concrete strategic implications for the original decision",
    )
    disconfirming_evidence_needed: list[str] = Field(
        default_factory=list,
        description="Specific evidence that would weaken or invalidate this hypothesis",
    )


class HypothesisPayload(BaseModel):
    """Structured output for HypothesisAgent (J6.3)."""

    hypotheses: list[HypothesisItem] = Field(
        default_factory=list,
        description="3-5 competing hypotheses generated from evidence and decision model",
    )
    synthesis_note: str = Field(
        default="",
        description="1-2 sentence overview of the hypothesis landscape",
    )


class ChallengeItem(BaseModel):
    """Challenge analysis for one hypothesis (J6.4)."""

    hypothesis_id: str = Field(description="The hypothesis ID this challenge addresses, e.g. 'H1'")
    challenge_summary: str = Field(description="1-3 sentence summary of the main challenge")
    hidden_assumptions: list[str] = Field(
        default_factory=list,
        description="Implicit assumptions the hypothesis relies on that are unverified",
    )
    weak_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence quality issues: vendor projections, thin data, unsupported claims",
    )
    contradicting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs (Exxx) that contradict or weaken the hypothesis",
    )
    missing_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence that is absent but needed to validate the hypothesis",
    )
    falsification_tests: list[str] = Field(
        default_factory=list,
        description="Specific observable conditions that would invalidate the hypothesis",
    )
    robustness: str = Field(
        default="medium",
        description="Overall robustness of the hypothesis: 'low', 'medium', or 'high'",
    )


class SurvivingHypothesis(BaseModel):
    """Post-challenge survival status for one hypothesis (J6.4)."""

    hypothesis_id: str = Field(description="The hypothesis ID")
    survival_status: str = Field(
        description="'strong', 'moderate', or 'weak' — how well the hypothesis survived challenges"
    )
    reason: str = Field(description="1-2 sentence rationale for the survival status")


class ChallengePayload(BaseModel):
    """Structured output for ChallengeAgent (J6.4)."""

    hypothesis_challenges: list[ChallengeItem] = Field(
        default_factory=list,
        description="One ChallengeItem per hypothesis",
    )
    surviving_hypotheses: list[SurvivingHypothesis] = Field(
        default_factory=list,
        description="Survival status for each hypothesis after challenge analysis",
    )
    challenge_synthesis: str = Field(
        default="",
        description="1-2 sentence overview of which hypotheses survived best and why",
    )


class RecommendationItem(BaseModel):
    """A single actionable recommendation derived from challenged hypotheses (J6.5)."""

    id: str = Field(description="Short identifier, e.g. 'R1'")
    # J7.2 – stable recommendation_id (REC-001 format) and assumption back-links
    recommendation_id: str = Field(default="", description="Stable ID e.g. 'REC-001'; auto-derived from id when empty")
    supported_assumption_ids: list[str] = Field(
        default_factory=list,
        description="assumption_ids from the Decision Model that this recommendation depends on",
    )
    title: str = Field(description="One-line recommendation title")
    summary: str = Field(description="2-4 sentence explanation of what to do and why")
    priority: str = Field(default="medium", description="'high', 'medium', or 'low'")
    time_horizon: str = Field(
        default="near_term",
        description="'near_term' (2026-2030), 'medium_term' (2030-2035), or 'long_term' (2035+)",
    )
    supported_by_hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypothesis IDs (H1, H2, …) that justify this recommendation",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="Evidence IDs (Exxx) that directly support this recommendation",
    )
    key_risks: list[str] = Field(
        default_factory=list,
        description="Specific risks that could undermine this recommendation",
    )
    trigger_conditions: list[str] = Field(
        default_factory=list,
        description="Future events that would change or activate this recommendation",
    )
    confidence: str = Field(default="medium", description="'high', 'medium', or 'low'")
    confidence_rationale: str = Field(
        default="",
        description="1-2 sentences explaining the confidence level",
    )


class RecommendationPortfolio(BaseModel):
    """Time-horizon grouping of recommendation IDs (J6.5)."""

    near_term: list[str] = Field(default_factory=list, description="Recommendation IDs for 2026-2030")
    medium_term: list[str] = Field(default_factory=list, description="Recommendation IDs for 2030-2035")
    long_term: list[str] = Field(default_factory=list, description="Recommendation IDs for 2035+")


class RecommendationPayload(BaseModel):
    """Structured output for RecommendationAgent (J6.5)."""

    recommendations: list[RecommendationItem] = Field(
        default_factory=list,
        description="3-5 actionable recommendations derived from surviving hypotheses",
    )
    recommendation_portfolio: RecommendationPortfolio = Field(
        default_factory=RecommendationPortfolio,
        description="Recommendations grouped by time horizon",
    )
    synthesis_note: str = Field(
        default="",
        description="1-2 sentence overview of the recommendation set",
    )


class AssumptionItem(BaseModel):
    """A single strategic assumption (J7.1)."""

    assumption_id: str = Field(description="Short unique ID e.g. 'A-001'")
    statement: str = Field(description="What must be true for the recommendation to hold")
    category: str = Field(
        description=(
            "One of: Technology, Market, Economics, Regulation, Policy, Supply Chain, "
            "Competition, Customer, Execution, Geopolitics, Environment, Infrastructure, Finance, Other"
        )
    )
    importance: str = Field(description="Critical | Important | Supporting")
    evidence_support: str = Field(description="Strong | Moderate | Weak | None")
    confidence: str = Field(description="High | Medium | Low")
    rationale: str = Field(description="Why this assumption matters strategically")
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items that support this assumption",
    )
    conflicts_with: list[str] = Field(
        default_factory=list,
        description="assumption_ids that contradict this assumption",
    )
    status: str = Field(default="Active", description="Active | Validated | Invalidated")


class AssumptionPayload(BaseModel):
    """Structured output for AssumptionAgent (J7.1)."""

    assumptions: list[AssumptionItem] = Field(
        default_factory=list,
        description="5-10 strategic assumptions that must hold for the recommendations to be valid",
    )
    conflict_pairs: list[list[str]] = Field(
        default_factory=list,
        description="Pairs of assumption_ids that conflict with each other",
    )


class RiskItem(BaseModel):
    """A single strategic risk (J7.3)."""

    risk_id: str = Field(description="Short unique ID e.g. 'RSK-001'")
    statement: str = Field(description="What could go wrong / what could cause an assumption to fail")
    category: str = Field(
        description=(
            "One of: Technology, Market, Economics, Regulation, Policy, Supply Chain, "
            "Competition, Customer, Execution, Geopolitics, Environment, Infrastructure, Finance, Other"
        )
    )
    severity: str = Field(description="High | Medium | Low — impact if the risk materialises")
    likelihood: str = Field(description="High | Medium | Low — probability of materialising")
    evidence_support: str = Field(description="Strong | Moderate | Weak | None")
    confidence: str = Field(description="High | Medium | Low — confidence in this risk assessment")
    rationale: str = Field(description="Why this risk matters strategically")
    related_assumption_ids: list[str] = Field(
        default_factory=list,
        description="assumption_ids whose validity this risk threatens",
    )
    affected_recommendation_ids: list[str] = Field(
        default_factory=list,
        description="REC-NNN ids of recommendations that would be affected if this risk materialises",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items that support or inform this risk",
    )
    mitigation_notes: str = Field(default="", description="High-level mitigation actions or hedges")
    status: str = Field(default="Active", description="Active | Mitigated | Retired")


class RiskPayload(BaseModel):
    """Structured output for RiskAgent (J7.3)."""

    risks: list[RiskItem] = Field(
        default_factory=list,
        description="5-10 strategic risks that could cause assumptions to fail",
    )


_SCHEMA_ADAPTERS = {
    "research_plan": TypeAdapter(ResearchPlan),
    "research_planning": TypeAdapter(ResearchPlanningPayload),
    "problem_framing": TypeAdapter(DecisionModelPayload),
    "research_strategy": TypeAdapter(ResearchStrategyPayload),
    "hypothesis_generation": TypeAdapter(HypothesisPayload),
    "challenge_generation": TypeAdapter(ChallengePayload),
    "recommendation_generation": TypeAdapter(RecommendationPayload),
    "assumption_generation": TypeAdapter(AssumptionPayload),     # J7.1
    "risk_generation": TypeAdapter(RiskPayload),                 # J7.3
    # Used for the tool-definition schema sent to Claude (strict EvidenceItem types).
    "evidence_extraction": TypeAdapter(EvidenceExtractionPayload),
    # Used for response validation (lenient — items validated per-item in extract_evidence).
    "evidence_extraction_raw": TypeAdapter(_RawEvidencePayload),
    "memo_synthesis": TypeAdapter(MemoSynthesisPayload),
}


class LLMClient(Protocol):
    """Minimal interface required by the agent."""

    is_mock: bool
    model: str
    call_traces: list[ClaudeCallTrace]


class MockClaudeClient:
    """Deterministic client used when Claude is unavailable."""

    is_mock = True
    model = "mock-claude"

    def __init__(self) -> None:
        self.call_traces: list[ClaudeCallTrace] = []

    def plan_research_question(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None = None,
        research_strategy: dict | None = None,
    ) -> ResearchPlanningPayload:
        q = question.lower()
        if any(w in q for w in ("compare", "vs", "versus", "difference between")):
            research_type = "COMPARISON"
        elif any(w in q for w in ("why", "how does", "explain", "what causes")):
            research_type = "EXPLANATION"
        elif any(w in q for w in ("what is", "what are", "how many", "how much", "list")):
            research_type = "FACT_LOOKUP"
        else:
            research_type = "RESEARCH"

        profiles_used = [p.get("name", "") for p in profiles_context if p.get("name")]
        # Seed from decision model when available (goal-driven runs)
        subquestions = (
            list(decision_model.get("research_questions", []))
            if decision_model else []
        ) or [
            f"What are the key facts about: {question}?",
            "What evidence exists in the available sources?",
            "What are the main constraints or limitations?",
            "What are the practical implications?",
            "What gaps remain in the available evidence?",
        ]
        investigation_areas = (
            list(decision_model.get("decision_areas", []))
            if decision_model else []
        ) or ["Overview", "Key Facts", "Evidence Quality", "Implications", "Open Questions"]
        return ResearchPlanningPayload(
            research_type=research_type,
            subquestions=subquestions,
            investigation_areas=investigation_areas,
            profiles_used=profiles_used,
            reasoning="Mock plan seeded from decision model." if decision_model else "Mock deterministic plan.",
        )

    def frame_problem(
        self,
        goal: str,
        profiles_context: list[dict],
    ) -> "DecisionModelPayload":
        """Return a deterministic decision model for the given business goal."""
        return DecisionModelPayload(
            objective=f"Research and analyse: {goal}",
            decision_areas=["Market Landscape", "Technical Feasibility", "Risk Assessment", "Investment Criteria"],
            critical_uncertainties=["Market timing", "Competitive dynamics", "Regulatory environment"],
            research_questions=[
                f"What is the current state of: {goal}?",
                "What are the key technical and market constraints?",
                "What evidence exists on investment returns and risk factors?",
                "What are the strategic options and their trade-offs?",
            ],
            evidence_requirements=["Market data", "Technical specifications", "Case studies", "Analyst reports"],
        )

    def generate_research_strategy(
        self,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> "ResearchStrategyPayload":
        """Return a deterministic research strategy from a decision model."""
        profiles = [p.get("name", "") for p in profiles_context if p.get("name")]
        rqs = decision_model.get("research_questions", [])
        areas = decision_model.get("decision_areas", [])
        uncertainties = decision_model.get("critical_uncertainties", [])
        evidence_reqs = decision_model.get("evidence_requirements", [])
        return ResearchStrategyPayload(
            profile_priorities={p: i + 1 for i, p in enumerate(profiles)},
            research_question_priorities=[
                {"question": q, "priority": i + 1} for i, q in enumerate(rqs)
            ],
            required_evidence=evidence_reqs or [
                "Primary data sources",
                "Expert assessments",
                "Quantitative benchmarks",
            ],
            source_priorities=["primary research", "industry reports", "expert analysis", "case studies"],
            coverage_targets={
                **{area: "strong" for area in areas[:2]},
                **{area: "moderate" for area in areas[2:]},
                **{u: "strong" for u in uncertainties[:1]},
            },
            strategy_rationale="Mock strategy: profiles ranked by order, questions ranked by position, coverage targets set to strong/moderate.",
        )


    def generate_hypotheses(
        self,
        decision_model: dict,
        research_strategy: dict,
        evidence_items: list[dict],
        profile_coverage: dict,
        contradictions: list[dict],
    ) -> "HypothesisPayload":
        """Return deterministic competing hypotheses from evidence and context."""
        decision_areas = decision_model.get("decision_areas", ["Area A", "Area B", "Area C"])
        objective = decision_model.get("objective", "Evaluate the strategic opportunity")
        # Sample evidence IDs from evidence_items
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        sup1 = ev_ids[:2] if len(ev_ids) >= 2 else ev_ids
        sup2 = ev_ids[2:4] if len(ev_ids) >= 4 else ev_ids[:1]
        con1 = ev_ids[4:5] if len(ev_ids) >= 5 else []

        h1 = HypothesisItem(
            id="H1",
            title=f"{decision_areas[0] if decision_areas else 'Structural constraints'} dominate the strategic outlook",
            summary=(
                f"The primary factor shaping {objective} is {decision_areas[0] if decision_areas else 'structural constraints'}. "
                "Evidence points to foundational limitations that constrain near-term options. "
                "Actors who address these constraints first will gain a durable advantage."
            ),
            type="constraint_dominant",
            supporting_evidence=sup1,
            contradicting_evidence=con1,
            evidence_gaps=["Region-specific constraint data", "Time-series projections"],
            confidence="medium",
            confidence_rationale="Supported by available evidence but limited by sparse longitudinal data.",
            decision_implications=[
                f"Prioritise addressing {decision_areas[0] if decision_areas else 'constraints'} first",
                "Build contingency plans for constraint-limited scenarios",
                "Invest in monitoring of leading constraint indicators",
            ],
            disconfirming_evidence_needed=[
                f"Evidence that {decision_areas[0] if decision_areas else 'constraints'} are being resolved faster than projected",
                "Data showing alternative pathways circumvent the constraint",
            ],
        )

        h2 = HypothesisItem(
            id="H2",
            title=f"{decision_areas[1] if len(decision_areas) > 1 else 'Technology options'} unlock mid-term opportunities",
            summary=(
                f"Emerging developments in {decision_areas[1] if len(decision_areas) > 1 else 'technology'} "
                "create viable pathways that are not yet reflected in current market positioning. "
                "Decision-makers who take early positions on these options will outperform later entrants. "
                "The risk is in the timing — too early or too late both carry penalties."
            ),
            type="technology_option",
            supporting_evidence=sup2,
            contradicting_evidence=[],
            evidence_gaps=["Forward-looking cost trajectory data", "Deployment schedule evidence"],
            confidence="low",
            confidence_rationale="Hypothesis is plausible but evidence is sparse; requires more forward-looking data.",
            decision_implications=[
                "Maintain optionality by avoiding irreversible commitments near-term",
                "Monitor leading indicators of technology readiness",
                f"Consider staged entry into {decision_areas[1] if len(decision_areas) > 1 else 'technology'} plays",
            ],
            disconfirming_evidence_needed=[
                "Evidence that deployment costs are not declining",
                "Evidence of regulatory barriers blocking near-term adoption",
            ],
        )

        h3 = HypothesisItem(
            id="H3",
            title="Hybrid portfolio strategy outperforms single-path approaches",
            summary=(
                "No single strategic path dominates across all scenarios. "
                "A diversified approach across multiple decision areas reduces variance and preserves optionality. "
                "The evidence base, while incomplete, suggests that the uncertainty profile favours a portfolio over a concentrated bet."
            ),
            type="portfolio_strategy",
            supporting_evidence=ev_ids[-1:] if ev_ids else [],
            contradicting_evidence=[],
            evidence_gaps=["Portfolio-level outcome studies", "Cross-strategy comparative data"],
            confidence="medium",
            confidence_rationale="Consistent with general decision theory; the specific evidence base is thin.",
            decision_implications=[
                "Structure decisions as a portfolio with defined allocation thresholds",
                "Avoid over-commitment to any single technology or market pathway",
                "Build governance mechanisms to rebalance as uncertainty resolves",
            ],
            disconfirming_evidence_needed=[
                "Evidence that one pathway clearly dominates on cost-risk terms",
                "Evidence that portfolio management costs outweigh diversification benefits",
            ],
        )

        return HypothesisPayload(
            hypotheses=[h1, h2, h3],
            synthesis_note=(
                f"Three competing hypotheses span the strategic landscape for: {objective}. "
                "H1 reflects a constraint-driven view, H2 an opportunity-driven view, "
                "and H3 a portfolio view. Evidence gaps are significant across all three."
            ),
        )

    def generate_challenges(
        self,
        hypotheses: list[dict],
        evidence_items: list[dict],
        contradictions: list[dict],
        research_gaps: list[dict],
        profile_coverage: dict,
    ) -> "ChallengePayload":
        """Return deterministic challenges for each hypothesis."""
        challenges = []
        surviving = []
        # Map hypothesis IDs to robustness so we can vary them deterministically
        robustness_cycle = ["medium", "low", "high"]
        status_cycle = ["moderate", "weak", "strong"]

        # Collect evidence IDs for cross-referencing
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        contra_ids = [c.get("evidence_id_1", "") or c.get("item_a_id", "") for c in contradictions if c]

        for i, h in enumerate(hypotheses):
            hid = h.get("id", f"H{i+1}")
            title = h.get("title", "")
            sup_ev = h.get("supporting_evidence", [])
            con_ev = h.get("contradicting_evidence", [])
            gaps = h.get("evidence_gaps", [])
            robustness = robustness_cycle[i % 3]
            status = status_cycle[i % 3]

            challenges.append(ChallengeItem(
                hypothesis_id=hid,
                challenge_summary=(
                    f"Hypothesis '{title[:60]}' relies on unverified assumptions "
                    "and is constrained by evidence gaps that limit confidence."
                ),
                hidden_assumptions=[
                    f"Assumes {title[:40]} conditions remain stable over the decision horizon",
                    "Assumes current regulatory and market frameworks persist",
                    "Assumes decision-maker has operational capacity to execute the implied strategy",
                ],
                weak_evidence=[
                    "Relies on vendor projections rather than independently verified operating data",
                    "Evidence base is skewed toward early-stage deployments, not at-scale operations",
                ] + ([f"Supporting evidence ({sup_ev[0]}) from a single source type"] if sup_ev else []),
                contradicting_evidence=con_ev[:2] or contra_ids[:1],
                missing_evidence=gaps[:2] or [
                    "Independent third-party cost and performance benchmarks",
                    "Long-run operational data from comparable deployments",
                ],
                falsification_tests=[
                    f"If the primary assumption underlying '{hid}' is refuted by new data, downgrade immediately",
                    "If regulatory timeline slips by more than 18 months, reassess viability",
                    "If independent cost data diverges >30% from vendor projections, reject supporting evidence",
                ],
                robustness=robustness,
            ))
            surviving.append(SurvivingHypothesis(
                hypothesis_id=hid,
                survival_status=status,
                reason=(
                    f"{hid} survives as '{status}': the core mechanism is plausible "
                    "but evidence gaps and hidden assumptions limit immediate confidence."
                ),
            ))

        return ChallengePayload(
            hypothesis_challenges=challenges,
            surviving_hypotheses=surviving,
            challenge_synthesis=(
                f"Challenge analysis reviewed {len(hypotheses)} hypothesis/hypotheses. "
                "No single hypothesis is strongly confirmed; all carry material assumptions "
                "that require further evidence before strategic commitment."
            ),
        )

    def generate_recommendations(
        self,
        hypotheses: list[dict],
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
        validated_contradictions: list[dict] | None = None,
    ) -> "RecommendationPayload":
        """Return deterministic recommendations derived from surviving hypotheses."""
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        horizon_cycle = ["near_term", "medium_term", "near_term", "long_term"]
        priority_cycle = ["high", "medium", "high", "low"]
        confidence_cycle = ["medium", "low", "medium", "high"]
        status_by_id = {s.get("hypothesis_id", ""): s.get("survival_status", "moderate") for s in surviving_hypotheses}

        recs = []
        for i, h in enumerate(hypotheses):
            hid = h.get("id", f"H{i+1}")
            title = h.get("title", "")
            horizon = horizon_cycle[i % 4]
            priority = priority_cycle[i % 4]
            confidence = confidence_cycle[i % 4]
            survival = status_by_id.get(hid, "moderate")
            sup_ev = ev_ids[i*2 : i*2+3] if len(ev_ids) > i*2 else ev_ids[:2]
            challenge = hypothesis_challenges[i] if i < len(hypothesis_challenges) else {}
            _raw_risks = challenge.get("key_risks") or challenge.get("weak_evidence") or []
            risks = _raw_risks[:2] if _raw_risks else [
                f"Evidence supporting {hid} may not generalise at scale",
                "Execution capacity may be constrained in target timeframe",
            ]

            recs.append(RecommendationItem(
                id=f"R{i+1}",
                title=f"Act on {title[:50]}",
                summary=(
                    f"Based on {hid} (survival: {survival}), this recommendation addresses "
                    f"the core strategic implication. "
                    f"Confidence is {confidence} given the challenge findings."
                ),
                priority=priority,
                time_horizon=horizon,
                supported_by_hypotheses=[hid],
                supporting_evidence=sup_ev,
                key_risks=risks[:2] if isinstance(risks, list) else [str(risks)],
                trigger_conditions=[
                    f"When leading indicators for {hid} confirm trajectory",
                    "When regulatory or market conditions change materially",
                ],
                confidence=confidence,
                confidence_rationale=f"Inherits from {hid} survival status '{survival}'; challenge findings constrain confidence.",
            ))

        near = [r.id for r in recs if r.time_horizon == "near_term"]
        mid = [r.id for r in recs if r.time_horizon == "medium_term"]
        lng = [r.id for r in recs if r.time_horizon == "long_term"]

        return RecommendationPayload(
            recommendations=recs,
            recommendation_portfolio=RecommendationPortfolio(
                near_term=near, medium_term=mid, long_term=lng,
            ),
            synthesis_note=(
                f"{len(recs)} recommendations generated from {len(hypotheses)} challenged hypotheses. "
                "Near-term actions focus on highest-survival hypotheses."
            ),
        )


    def generate_assumptions(
        self,
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
    ) -> "AssumptionPayload":
        """Return deterministic mock strategic assumptions (J7.1)."""
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

        categories = ["Technology", "Market", "Economics", "Regulation", "Supply Chain",
                      "Execution", "Infrastructure", "Policy", "Competition", "Finance"]
        importances = ["Critical", "Critical", "Important", "Important", "Supporting",
                       "Critical", "Important", "Supporting", "Important", "Critical"]
        supports = ["Strong", "Moderate", "Weak", "Strong", "Moderate",
                    "Moderate", "Strong", "Weak", "Moderate", "Strong"]
        confidences = ["High", "Medium", "Low", "High", "Medium",
                       "Medium", "High", "Low", "Medium", "High"]

        templates = [
            "The underlying technology is sufficiently mature for commercial deployment at scale",
            "Market demand will remain at projected levels over the investment horizon",
            "Capital costs will not materially exceed current estimates",
            "The regulatory environment will remain stable and permissive",
            "Supply chain constraints will be resolved within the planning timeframe",
            "The organisation has sufficient execution capacity to deliver the programme",
            "Infrastructure dependencies (power, cooling, connectivity) will be available",
            "Policy support will continue throughout the deployment period",
            "Competitive dynamics will not materially shift the economic case",
            "Financing conditions will remain favourable at the required scale",
        ]

        assumptions = []
        for i, hyp in enumerate(surviving_hypotheses[:7]):
            idx = i % len(templates)
            sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
            assumptions.append(AssumptionItem(
                assumption_id=f"A-{i+1:03d}",
                statement=templates[idx],
                category=categories[idx],
                importance=importances[idx],
                evidence_support=supports[idx],
                confidence=confidences[idx],
                rationale=(
                    f"Derived from hypothesis '{hyp.get('title', hyp.get('id', f'H{i+1}'))}'. "
                    f"This must hold for the strategic recommendation to remain valid."
                ),
                evidence_ids=sup_ev,
                conflicts_with=[],
                status="Active",
            ))

        # Add remaining mock assumptions if fewer hypotheses than templates
        for j in range(len(assumptions), min(5, len(templates))):
            sup_ev = ev_ids[j*2 : j*2+2] if len(ev_ids) > j*2 else ev_ids[:1]
            assumptions.append(AssumptionItem(
                assumption_id=f"A-{j+1:03d}",
                statement=templates[j],
                category=categories[j],
                importance=importances[j],
                evidence_support=supports[j],
                confidence=confidences[j],
                rationale=f"Strategic assumption relevant to: {question[:80]}",
                evidence_ids=sup_ev,
                conflicts_with=[],
                status="Active",
            ))

        # Simple mock conflict: assumption 2 (Market demand stable) vs assumption 9 (Competition shifts)
        conflict_pairs: list[list[str]] = []
        if len(assumptions) >= 9:
            assumptions[1].conflicts_with.append(assumptions[8].assumption_id)
            assumptions[8].conflicts_with.append(assumptions[1].assumption_id)
            conflict_pairs.append([assumptions[1].assumption_id, assumptions[8].assumption_id])

        return AssumptionPayload(assumptions=assumptions, conflict_pairs=conflict_pairs)

    def generate_risks(
        self,
        assumptions: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> "RiskPayload":
        """Generate strategic risks from assumptions (J7.3) — mock version."""
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

        risk_templates = [
            ("Technology maturity proves insufficient, causing deployment delays or failures", "Technology", "High", "Medium"),
            ("Market demand shifts materially below projections, undermining the economic case", "Market", "High", "Medium"),
            ("Capital costs escalate beyond estimates, impairing project viability", "Economics", "High", "Low"),
            ("Regulatory changes impose unforeseen restrictions or obligations", "Regulation", "Medium", "Medium"),
            ("Supply chain disruptions delay or prevent timely execution", "Supply Chain", "Medium", "Low"),
        ]

        risks = []
        for i, (stmt, cat, sev, lik) in enumerate(risk_templates):
            # Link to the assumption at the same index (if it exists)
            related_a = [assumptions[i]["assumption_id"]] if i < len(assumptions) else []
            # Derive affected recommendations from the linked assumption
            affected_r: list[str] = []
            if related_a and i < len(assumptions):
                affected_r = list(assumptions[i].get("supported_recommendation_ids", []))
            sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
            risks.append(RiskItem(
                risk_id=f"RSK-{i+1:03d}",
                statement=stmt,
                category=cat,
                severity=sev,
                likelihood=lik,
                evidence_support="Moderate",
                confidence="Medium",
                rationale=f"Strategic risk relevant to: {question[:80]}",
                related_assumption_ids=related_a,
                affected_recommendation_ids=affected_r,
                evidence_ids=sup_ev,
                mitigation_notes="",
                status="Active",
            ))

        return RiskPayload(risks=risks)


class ClaudeClient:
    """Thin Anthropic SDK wrapper for structured research calls."""

    is_mock = False

    def __init__(
        self,
        *,
        model: str | None = None,
        extraction_model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4000,
        anthropic_client: Any | None = None,
        use_extraction_cache: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL") or DEFAULT_MODEL
        # Extraction uses a fast cheap model by default; override with env var or arg.
        self.extraction_model = (
            extraction_model
            or os.getenv("ANTHROPIC_EXTRACTION_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )
        self.max_tokens = max_tokens
        self.call_traces: list[ClaudeCallTrace] = []

        from .extraction_cache import ExtractionCache
        self._extraction_cache: ExtractionCache | None = (
            ExtractionCache() if use_extraction_cache else None
        )
        LOGGER.debug(
            "ClaudeClient: synthesis_model=%s  extraction_model=%s  cache=%s",
            self.model,
            self.extraction_model,
            "enabled" if self._extraction_cache else "disabled",
        )

        if anthropic_client is not None:
            self._client = anthropic_client
            return

        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for Claude runs.")

        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("Install anthropic to use Claude.") from exc

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def frame_problem(
        self,
        goal: str,
        profiles_context: list[dict],
    ) -> DecisionModelPayload:
        """Transform a business goal into a structured Decision Model (J6.1)."""
        payload = self._call_json(
            operation="problem_framing",
            schema_name="problem_framing",
            prompt=_problem_framing_prompt(goal, profiles_context),
            max_tokens=2000,
        )
        return DecisionModelPayload.model_validate(payload)

    def generate_research_strategy(
        self,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> ResearchStrategyPayload:
        """Transform a Decision Model into an executable research strategy (J6.2)."""
        payload = self._call_json(
            operation="generate_research_strategy",
            schema_name="research_strategy",
            prompt=_strategy_prompt(decision_model, profiles_context),
            max_tokens=2000,
        )
        return ResearchStrategyPayload.model_validate(payload)

    def generate_hypotheses(
        self,
        decision_model: dict,
        research_strategy: dict,
        evidence_items: list[dict],
        profile_coverage: dict,
        contradictions: list[dict],
    ) -> HypothesisPayload:
        """Generate competing hypotheses from evidence and context (J6.3)."""
        payload = self._call_json(
            operation="generate_hypotheses",
            schema_name="hypothesis_generation",
            prompt=_hypothesis_prompt(
                decision_model, research_strategy,
                evidence_items, profile_coverage, contradictions,
            ),
            max_tokens=8000,
        )
        return HypothesisPayload.model_validate(payload)

    def generate_challenges(
        self,
        hypotheses: list[dict],
        evidence_items: list[dict],
        contradictions: list[dict],
        research_gaps: list[dict],
        profile_coverage: dict,
    ) -> ChallengePayload:
        """Challenge each hypothesis to surface weaknesses and surviving strength (J6.4)."""
        payload = self._call_json(
            operation="generate_challenges",
            schema_name="challenge_generation",
            prompt=_challenge_prompt(
                hypotheses, evidence_items, contradictions, research_gaps, profile_coverage,
            ),
            max_tokens=10000,
        )
        return ChallengePayload.model_validate(payload)

    def generate_recommendations(
        self,
        hypotheses: list[dict],
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
        validated_contradictions: list[dict] | None = None,
    ) -> RecommendationPayload:
        """Generate actionable recommendations from challenged hypotheses (J6.5)."""
        payload = self._call_json(
            operation="generate_recommendations",
            schema_name="recommendation_generation",
            prompt=_recommendation_prompt(
                hypotheses, surviving_hypotheses, hypothesis_challenges,
                evidence_items, decision_model, research_strategy,
                validated_contradictions=validated_contradictions or [],
            ),
            max_tokens=10000,
        )
        return RecommendationPayload.model_validate(payload)

    def generate_assumptions(
        self,
        surviving_hypotheses: list[dict],
        hypothesis_challenges: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
        research_strategy: dict,
    ) -> AssumptionPayload:
        """Identify strategic assumptions that must hold for recommendations to remain valid (J7.1)."""
        payload = self._call_json(
            operation="generate_assumptions",
            schema_name="assumption_generation",
            prompt=_assumption_prompt(
                surviving_hypotheses, hypothesis_challenges,
                evidence_items, decision_model, research_strategy,
            ),
            max_tokens=8000,
        )
        return AssumptionPayload.model_validate(payload)

    def generate_risks(
        self,
        assumptions: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> RiskPayload:
        """Identify strategic risks that could cause assumptions to fail (J7.3)."""
        payload = self._call_json(
            operation="generate_risks",
            schema_name="risk_generation",
            prompt=_risk_prompt(assumptions, recommendations, evidence_items, decision_model),
            max_tokens=8000,
        )
        return RiskPayload.model_validate(payload)

    def plan_research_question(
        self,
        question: str,
        profiles_context: list[dict],
        decision_model: dict | None = None,
        research_strategy: dict | None = None,
    ) -> ResearchPlanningPayload:
        """Classify the question and generate a structured research plan (J5.1 / J6.1a)."""
        payload = self._call_json(
            operation="plan_research_question",
            schema_name="research_planning",
            prompt=_planning_prompt(
                question, profiles_context,
                decision_model=decision_model,
                research_strategy=research_strategy,
            ),
            max_tokens=2000,
        )
        return ResearchPlanningPayload.model_validate(payload)

    def create_research_plan(
        self,
        question: str,
        source_texts: Sequence[SourceDocument],
    ) -> ResearchPlan:
        payload = self._call_json(
            operation="create_research_plan",
            schema_name="research_plan",
            prompt=_research_plan_prompt(question, source_texts),
        )
        return ResearchPlan.model_validate(payload)

    def extract_evidence(
        self,
        question: str,
        source_texts: Sequence[SourceDocument],
    ) -> list[EvidenceItem]:
        # Each evidence item serialises to roughly 150 tokens; use a generous
        # ceiling so the response is never truncated mid-JSON.
        # response_schema_name uses the lenient schema so per-item validation
        # can discard bad items without failing the whole batch.
        payload = self._call_json(
            operation="extract_evidence",
            schema_name="evidence_extraction",
            prompt=_evidence_prompt(question, source_texts),
            max_tokens=max(self.max_tokens, 16_000),
            response_schema_name="evidence_extraction_raw",
        )
        raw_items = payload.get("evidence_items", [])
        LOGGER.debug("extract_evidence: raw item count from payload=%d", len(raw_items))

        validated: list[EvidenceItem] = []
        discarded = 0
        for item in raw_items:
            try:
                validated.append(EvidenceItem.model_validate(item))
            except Exception as exc:
                LOGGER.debug(
                    "extract_evidence: discarding item due to validation error: %s", exc
                )
                discarded += 1

        if discarded:
            LOGGER.warning(
                "extract_evidence: discarded %d of %d items due to validation errors",
                discarded,
                len(raw_items),
            )

        clean = sanitize_evidence_items(validated, stage="claude_extract_evidence")
        result = assign_evidence_ids(clean)
        LOGGER.debug("extract_evidence: final EvidenceItem count=%d", len(result))
        return result

    def extract_evidence_from_chunks(
        self,
        question: str,
        chunks: Sequence[Chunk],
        *,
        prompt_override: str | None = None,
    ) -> list[EvidenceItem]:
        chunk_list = list(chunks)

        # Cache read — skip when a custom prompt is supplied (diagnostic / non-production use).
        if prompt_override is None and self._extraction_cache is not None:
            cached = self._extraction_cache.get(question, chunk_list)
            if cached is not None:
                from research_agent.log import PROGRESS
                LOGGER.log(PROGRESS, "[extraction_cache] hit  chunks=%d  items=%d", len(chunk_list), len(cached))
                return cached

        prompt = prompt_override if prompt_override is not None else _evidence_chunk_prompt(question, chunk_list)
        payload = self._call_json(
            operation="extract_evidence",
            schema_name="evidence_extraction",
            prompt=prompt,
            max_tokens=max(self.max_tokens, 16_000),
            response_schema_name="evidence_extraction_raw",
            model_override=self.extraction_model,
        )
        raw_items = payload.get("evidence_items", [])
        LOGGER.debug("extract_evidence_from_chunks: raw item count from payload=%d", len(raw_items))

        validated: list[EvidenceItem] = []
        discarded = 0
        for item in raw_items:
            try:
                validated.append(EvidenceItem.model_validate(item))
            except Exception as exc:
                LOGGER.debug(
                    "extract_evidence_from_chunks: discarding item due to validation error: %s", exc
                )
                discarded += 1

        if discarded:
            LOGGER.warning(
                "extract_evidence_from_chunks: discarded %d of %d items due to validation errors",
                discarded,
                len(raw_items),
            )

        clean = sanitize_evidence_items(validated, stage="claude_extract_from_chunks")
        result = assign_evidence_ids(clean)
        LOGGER.debug("extract_evidence_from_chunks: final EvidenceItem count=%d", len(result))

        # Cache write — skip when a custom prompt was used.
        if prompt_override is None and self._extraction_cache is not None:
            self._extraction_cache.put(question, chunk_list, result)

        return result

    def synthesize_memo(
        self,
        question: str,
        evidence_items: Sequence[EvidenceItem],
    ) -> ResearchMemo:
        payload = self._call_json(
            operation="synthesize_memo",
            schema_name="memo_synthesis",
            prompt=_memo_prompt(question, evidence_items),
            max_tokens=max(self.max_tokens, 12_000),
        )
        return ResearchMemo(
            title=f"Research Memo: {question}",
            question=question,
            executive_summary=_string_value(payload.get("executive_summary")),
            confirmed_facts=_string_list(payload.get("confirmed_facts")),
            inferences=_string_list(payload.get("inferences")),
            power_implications=_string_list(payload.get("power_implications")),
            cooling_implications=_string_list(payload.get("cooling_implications")),
            networking_implications=_string_list(payload.get("networking_implications")),
            rack_architecture_implications=_string_list(
                payload.get("rack_architecture_implications")
            ),
            open_questions=_string_list(payload.get("open_questions")),
            source_notes=list(evidence_items),
            evidence=list(evidence_items),
        )

    def _call_json(
        self,
        *,
        operation: str,
        schema_name: str,
        prompt: str,
        max_tokens: int | None = None,
        response_schema_name: str | None = None,
        model_override: str | None = None,
    ) -> dict[str, Any] | list[Any]:
        # response_schema_name lets callers use one schema for the tool definition
        # (what Claude sees) and a different, more lenient schema for parsing the
        # response (e.g. evidence_extraction_raw for per-item validation).
        _response_schema = response_schema_name or schema_name
        _model = model_override or self.model
        request_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = self._client.messages.create(
                model=_model,
                max_tokens=max_tokens or self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[_tool_definition(operation, schema_name)],
                tool_choice={"type": "tool", "name": operation},
            )
            stop_reason = getattr(response, "stop_reason", None)
            output_tokens = getattr(getattr(response, "usage", None), "output_tokens", None)
            LOGGER.debug(
                "%s: stop_reason=%s output_tokens=%s max_tokens=%s",
                operation,
                stop_reason,
                output_tokens,
                max_tokens or self.max_tokens,
            )
            if stop_reason == "max_tokens":
                raise RuntimeError(
                    f"{operation}: response truncated (stop_reason=max_tokens, "
                    f"limit={max_tokens or self.max_tokens}). "
                    "The tool input is incomplete and would silently validate as empty. "
                    "Increase max_tokens for this operation."
                )
            tool_input = _response_tool_input(response)
            LOGGER.debug(
                "%s: tool_input present=%s raw_length=%s",
                operation,
                tool_input is not None,
                len(str(tool_input)) if tool_input is not None else 0,
            )
            if tool_input is not None:
                tool_input = _normalize_tool_input(tool_input)
                payload = _validate_payload(tool_input, _response_schema)
            else:
                text = _response_text(response)
                payload = parse_or_repair_json(
                    text,
                    _response_schema,
                    {
                        "operation": operation,
                        "expected_shape": _schema_description(_response_schema),
                        "repair": lambda repair_prompt: self._repair_json_text(
                            operation=operation,
                            prompt=repair_prompt,
                            max_tokens=max_tokens or self.max_tokens,
                        ),
                    },
                )
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=operation,
                    model_name=_model,
                    request_timestamp=request_timestamp,
                    success=True,
                    token_usage=_token_usage(response),
                )
            )
            return payload
        except Exception as exc:
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=operation,
                    model_name=_model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                )
            )
            raise

    def _repair_json_text(self, *, operation: str, prompt: str, max_tokens: int) -> str:
        request_timestamp = datetime.now(timezone.utc).isoformat()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=True,
                    token_usage=_token_usage(response),
                )
            )
            return _response_text(response)
        except Exception as exc:
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                )
            )
            raise


def create_research_plan(
    question: str,
    source_texts: Sequence[SourceDocument],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ResearchPlan:
    return ClaudeClient(model=model, api_key=api_key).create_research_plan(question, source_texts)


def extract_evidence(
    question: str,
    source_texts: Sequence[SourceDocument],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[EvidenceItem]:
    return ClaudeClient(model=model, api_key=api_key).extract_evidence(question, source_texts)


def synthesize_memo(
    question: str,
    evidence_items: Sequence[EvidenceItem],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> ResearchMemo:
    return ClaudeClient(model=model, api_key=api_key).synthesize_memo(question, evidence_items)


def extract_evidence_from_chunks(
    question: str,
    chunks: Sequence[Chunk],
    *,
    model: str | None = None,
    api_key: str | None = None,
) -> list[EvidenceItem]:
    return ClaudeClient(model=model, api_key=api_key).extract_evidence_from_chunks(question, chunks)


def aggregate_call_traces(call_traces: Sequence[ClaudeCallTrace]) -> dict[str, Any]:
    if not call_traces:
        return {
            "model_name": None,
            "request_timestamp": None,
            "response_success": None,
            "token_usage": None,
            "errors": [],
        }

    token_usage: dict[str, int] = {}
    errors: list[str] = []
    for trace in call_traces:
        for key, value in trace.token_usage.items():
            token_usage[key] = token_usage.get(key, 0) + value
        if trace.error:
            errors.append(trace.error)

    return {
        "model_name": call_traces[-1].model_name,
        "request_timestamp": call_traces[0].request_timestamp,
        "response_success": all(trace.success for trace in call_traces),
        "token_usage": token_usage or None,
        "errors": errors,
    }


def _planning_prompt(
    question: str,
    profiles_context: list[dict],
    decision_model: dict | None = None,
    research_strategy: dict | None = None,
) -> str:
    """Build the PlannerAgent prompt for question classification and decomposition (J5.1 / J6.1a / J6.2)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    # Decision Model context — injected when available (goal-driven runs)
    dm_section = ""
    if decision_model:
        dm_section = f"""
Decision Model (pre-derived from business goal — use this to ground your plan):
  Objective: {decision_model.get('objective', '')}
  Decision areas: {', '.join(decision_model.get('decision_areas', []))}
  Critical uncertainties: {', '.join(decision_model.get('critical_uncertainties', []))}
  Research questions: {'; '.join(decision_model.get('research_questions', []))}
  Evidence requirements: {', '.join(decision_model.get('evidence_requirements', []))}

Your subquestions and investigation areas should be aligned with the Decision Model above.
"""

    # Research Strategy context — injected when available (J6.2)
    rs_section = ""
    if research_strategy:
        rq_prios = research_strategy.get("research_question_priorities", [])
        rq_ordered = "; ".join(
            rqp.get("question", "") for rqp in sorted(rq_prios, key=lambda x: x.get("priority", 99))
        )
        coverage = ", ".join(
            f"{k}={v}" for k, v in list(research_strategy.get("coverage_targets", {}).items())[:5]
        )
        rs_section = f"""
Research Strategy (use this to prioritise subquestions and structure your plan):
  Priority questions (most important first): {rq_ordered}
  Required evidence: {', '.join(research_strategy.get('required_evidence', [])[:4])}
  Source priorities: {', '.join(research_strategy.get('source_priorities', [])[:4])}
  Coverage targets: {coverage}

Align your subquestions with the priority question order above.
"""

    return f"""You are a research planning agent. Analyze the question below and produce a structured research plan.

Question:
{question}

Domain profiles loaded:{profile_lines if profile_lines else " (none)"}
{dm_section}{rs_section}
Instructions:
1. Classify the research_type as exactly one of:
   - FACT_LOOKUP: asking for a specific fact, number, or definition
   - COMPARISON: comparing two or more entities, technologies, or options
   - EXPLANATION: asking why or how something works
   - RESEARCH: broad investigation requiring synthesis across multiple topics

2. Generate 3-7 focused subquestions that decompose the main question into
   answerable parts. Draw on the domain profiles to make subquestions specific.
   If a Decision Model is provided, map subquestions to the research_questions above.

3. Generate 4-8 investigation areas (short topic labels like "Power Requirements",
   "Deployment Timeline", "Economics") that structure the research.
   If a Decision Model is provided, align areas with the decision_areas above.

4. List which profile names informed this plan in profiles_used.

5. Write a brief reasoning (2-3 sentences) explaining your classification.

Return structured JSON only.
"""


def _problem_framing_prompt(goal: str, profiles_context: list[dict]) -> str:
    """Build the ProblemFramingAgent prompt for decision model generation (J6.1)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    return f"""You are a strategic research planning agent. Transform the business goal below into a structured Decision Model that will guide a research pipeline.

Business Goal:
{goal}

Domain profiles available:{profile_lines if profile_lines else " (none)"}

Instructions:
1. Restate the goal as a precise research objective (1-2 sentences).

2. Identify 3-6 key decision areas — the dimensions that must be understood to act on this goal (e.g. "Market readiness", "Technical feasibility", "Regulatory landscape").

3. Identify 2-5 critical uncertainties — the unknowns that most affect the decision outcome.

4. Generate 3-6 specific, answerable research questions derived directly from the goal and decision areas. Draw on the domain profiles to make questions specific and actionable.

5. List 2-5 evidence requirements — the types of evidence needed to answer the research questions (e.g. "Benchmark performance data", "Vendor cost sheets", "Industry analyst reports").

Return structured JSON only.
"""


def _hypothesis_prompt(
    decision_model: dict,
    research_strategy: dict,
    evidence_items: list[dict],
    profile_coverage: dict,
    contradictions: list[dict],
) -> str:
    """Build the HypothesisAgent prompt (J6.3)."""
    objective = decision_model.get("objective", "")
    areas = ", ".join(decision_model.get("decision_areas", []))
    uncertainties = "\n".join(f"  - {u}" for u in decision_model.get("critical_uncertainties", []))

    # Summarise evidence (max 20 items for prompt size)
    ev_lines = ""
    for e in evidence_items[:20]:
        eid = e.get("evidence_id", "?")
        claim = e.get("claim", e.get("text", ""))[:120]
        source = e.get("source_document", "")
        ev_lines += f"\n  [{eid}] {claim} (source: {source})"

    # Contradictions
    contra_lines = ""
    for c in contradictions[:5]:
        contra_lines += f"\n  - {c.get('topic', '?')}: {c.get('summary', '')[:100]}"

    # Coverage summary
    coverage_lines = ""
    for profile, level in list(profile_coverage.items())[:6]:
        coverage_lines += f"\n  - {profile}: {level}"

    rq_list = "\n".join(
        f"  {i+1}. {rqp.get('question', '')}"
        for i, rqp in enumerate(
            sorted(research_strategy.get("research_question_priorities", []), key=lambda x: x.get("priority", 99))
        )
    )

    return f"""You are a strategic analysis agent. Generate 3-5 competing hypotheses that explain the evidence gathered for this decision.

Decision Objective:
{objective}

Decision Areas: {areas}

Critical Uncertainties:
{uncertainties if uncertainties else "  (none listed)"}

Research Questions (priority order):
{rq_list if rq_list else "  (none)"}

Evidence Collected (up to 20 items):
{ev_lines if ev_lines else "  (no evidence available)"}

Profile Coverage:{coverage_lines if coverage_lines else " (none)"}

Contradictions Detected:{contra_lines if contra_lines else " (none)"}

Instructions:
1. Generate 3-5 COMPETING hypotheses. Each should represent a meaningfully different strategic interpretation of the evidence.

2. For each hypothesis, include:
   - id: "H1", "H2", etc.
   - title: one-line hypothesis
   - summary: 2-4 sentences explaining the hypothesis
   - type: one of constraint_dominant, technology_option, portfolio_strategy, market_timing, risk_concentration, or similar
   - supporting_evidence: list of evidence IDs (from the list above) that support it
   - contradicting_evidence: list of evidence IDs that weaken or contradict it
   - evidence_gaps: list of evidence types that are absent but needed to test it
   - confidence: "high", "medium", or "low"
   - confidence_rationale: 1-2 sentences explaining the confidence level
   - decision_implications: 2-4 concrete strategic implications
   - disconfirming_evidence_needed: 2-3 specific evidence items that would invalidate this hypothesis

3. Hypotheses should be mutually distinguishable — avoid restating the same claim.

4. Write a 1-2 sentence synthesis_note summarising the hypothesis landscape.

Return structured JSON only.
"""


def _challenge_prompt(
    hypotheses: list[dict],
    evidence_items: list[dict],
    contradictions: list[dict],
    research_gaps: list[dict],
    profile_coverage: dict,
) -> str:
    """Build the ChallengeAgent prompt (J6.4)."""
    hyp_lines = ""
    for h in hypotheses:
        hid = h.get("id", "?")
        title = h.get("title", "")
        summary = h.get("summary", "")
        sup = ", ".join(h.get("supporting_evidence", [])[:5]) or "none"
        con = ", ".join(h.get("contradicting_evidence", [])[:5]) or "none"
        gaps = "; ".join(h.get("evidence_gaps", [])[:3]) or "none stated"
        conf = h.get("confidence", "medium")
        hyp_lines += (
            f"\n{hid}: {title}\n"
            f"  Summary: {summary}\n"
            f"  Supporting evidence: {sup}\n"
            f"  Contradicting evidence: {con}\n"
            f"  Evidence gaps: {gaps}\n"
            f"  Confidence: {conf}\n"
        )

    ev_lines = ""
    for e in evidence_items[:20]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:120]
        src = e.get("source_document", "")
        ev_lines += f"  {eid}: {claim} (source: {src})\n"

    contra_lines = ""
    for c in contradictions[:5]:
        cid = c.get("contradiction_id", "?")
        topic = c.get("topic", "")
        sev = c.get("severity", "")
        contra_lines += f"  {cid} [{sev}]: {topic}\n"

    gap_lines = ""
    for g in research_gaps[:5]:
        gap_lines += f"  - {g.get('gap', g) if isinstance(g, dict) else g}\n"

    cov_lines = "\n".join(
        f"  {pname}: {level}" for pname, level in profile_coverage.items()
    ) or "  (not available)"

    return f"""\
You are a rigorous intellectual adversary tasked with stress-testing strategic hypotheses.

## Hypotheses to Challenge
{hyp_lines}

## Evidence Available (up to 20 items)
{ev_lines or "  (none)"}

## Known Contradictions
{contra_lines or "  (none)"}

## Research Gaps
{gap_lines or "  (none)"}

## Profile Coverage
{cov_lines}

---

## Your Task

For EACH hypothesis above, produce a ChallengeItem with:

1. challenge_summary — 1-3 sentences on the main weakness
2. hidden_assumptions — 2-4 implicit assumptions the hypothesis relies on (things it takes for granted without evidence)
3. weak_evidence — 2-3 specific evidence quality problems (e.g. vendor projections, single-source claims, outdated data)
4. contradicting_evidence — list evidence IDs (Exxx) from the evidence list above that weaken this hypothesis; explain briefly why each matters
5. missing_evidence — 2-3 types of evidence that are absent but would be needed to validate the hypothesis
6. falsification_tests — 2-3 specific, observable conditions that would definitively invalidate the hypothesis
7. robustness — "high" (withstands most challenges), "medium" (withstands some), or "low" (undermined by major gaps or contradictions)

Then produce a SurvivingHypothesis for each:
- survival_status: "strong" (core logic intact), "moderate" (survives with caveats), "weak" (significant doubts)
- reason: 1-2 sentences explaining the survival status

Finally, write a 1-2 sentence challenge_synthesis summarising which hypotheses survived best and why.

Be adversarial. Your job is to find problems, not validate. Do not restate the hypothesis — critique it.

Return structured JSON only.
"""


def _recommendation_prompt(
    hypotheses: list[dict],
    surviving_hypotheses: list[dict],
    hypothesis_challenges: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
    research_strategy: dict,
    validated_contradictions: list[dict] | None = None,
) -> str:
    """Build the RecommendationAgent prompt (J6.5)."""
    objective = decision_model.get("objective", "")
    decision_areas = decision_model.get("decision_areas", [])

    # Survival lookup
    survival_by_id = {s.get("hypothesis_id", ""): s for s in surviving_hypotheses}
    challenge_by_id = {c.get("hypothesis_id", ""): c for c in hypothesis_challenges}

    hyp_lines = ""
    for h in hypotheses:
        hid = h.get("id", "?")
        title = h.get("title", "")
        summary = h.get("summary", "")
        sup = ", ".join(h.get("supporting_evidence", [])[:4]) or "none"
        sv = survival_by_id.get(hid, {})
        status = sv.get("survival_status", "unknown")
        reason = sv.get("reason", "")
        ch = challenge_by_id.get(hid, {})
        robustness = ch.get("robustness", "unknown")
        key_challenge = ch.get("challenge_summary", "")[:100]
        hyp_lines += (
            f"\n{hid}: {title}  [survival={status}, robustness={robustness}]\n"
            f"  Summary: {summary}\n"
            f"  Supporting evidence: {sup}\n"
            f"  Survival reason: {reason}\n"
            f"  Key challenge: {key_challenge}\n"
        )

    ev_lines = ""
    for e in evidence_items[:20]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:120]
        src = e.get("source_document", "")
        ev_lines += f"  {eid}: {claim} (source: {src})\n"

    areas_text = "\n".join(f"  - {a}" for a in decision_areas[:6]) or "  (not specified)"

    # J6.5a – validated contradictions block
    contra_lines = ""
    for c in (validated_contradictions or [])[:10]:
        cid = c.get("contradiction_id", "?")
        topic = c.get("topic", "")
        sev = c.get("severity", "")
        claim_a = c.get("evidence_a_claim", "")[:80]
        claim_b = c.get("evidence_b_claim", "")[:80]
        contra_lines += f"  {cid} [{sev}] {topic}: \"{claim_a}\" vs \"{claim_b}\"\n"

    return f"""\
You are a strategic advisor translating challenged hypotheses into actionable recommendations.

## Decision Context
Objective: {objective}
Decision Areas:
{areas_text}

## Hypotheses with Challenge Results
{hyp_lines}

## Evidence Available (up to 20 items)
{ev_lines or "  (none)"}

## Validated Contradictions (post-suppression, use to flag risks)
{contra_lines or "  (none detected)"}

---

## Your Task

Generate 3-5 strategic recommendations. Each recommendation must:

1. Be DERIVED from one or more surviving hypotheses — do not invent new strategic logic
2. Be grounded in specific evidence IDs from the evidence list above
3. Include key risks that could undermine it
4. Include trigger conditions — future events that change or activate the recommendation
5. Be classified by time horizon: "near_term" (2026-2030), "medium_term" (2030-2035), or "long_term" (2035+)
6. Have a confidence level ("high"/"medium"/"low") that reflects the underlying hypothesis robustness and challenge findings

Rules:
- Recommendations from "weak" survival hypotheses should have LOW priority and LOW confidence
- Recommendations from "strong" or "moderate" survival hypotheses may have MEDIUM or HIGH priority
- Do not repeat hypothesis summaries as recommendations — translate them into specific actions
- Include a recommendation_portfolio that groups recommendation IDs by time horizon
- Write a 1-2 sentence synthesis_note summarising the recommendation set

Return structured JSON only.
"""


def _assumption_prompt(
    surviving_hypotheses: list[dict],
    hypothesis_challenges: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
    research_strategy: dict,
) -> str:
    """Build the AssumptionAgent prompt (J7.1)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))
    decision_areas = decision_model.get("decision_areas", decision_model.get("investigation_areas", []))

    survival_by_id = {s.get("hypothesis_id", ""): s for s in surviving_hypotheses}
    challenge_by_id = {c.get("hypothesis_id", ""): c for c in hypothesis_challenges}

    hyp_lines = ""
    for h in surviving_hypotheses:
        hid = h.get("hypothesis_id", h.get("id", "?"))
        title = h.get("title", "")
        status = h.get("survival_status", "")
        reason = h.get("reason", "")
        hyp_lines += f"\n  {hid}: {title}  [status={status}]\n  Reason: {reason}\n"

    ev_lines = ""
    for e in evidence_items[:25]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:120]
        src = e.get("source_document", e.get("source", ""))[:40]
        ev_lines += f"\n  {eid}: {claim}  [source: {src}]"

    return f"""You are a senior strategy consultant producing a Strategic Assumption analysis.

STRATEGIC QUESTION: {question}

DECISION AREAS: {', '.join(str(a) for a in decision_areas[:8])}

SURVIVING HYPOTHESES:
{hyp_lines or "  (none provided)"}

EVIDENCE AVAILABLE:
{ev_lines or "  (none provided)"}

TASK:
Identify 5–10 strategic assumptions that MUST hold for the research findings and emerging
recommendations to remain valid. These are NOT observations or findings — they are
conditions that are currently assumed to be true but could prove false.

For each assumption:
1. State precisely what must be true
2. Assign the most appropriate category
3. Rate importance: Critical (recommendation fails if false) | Important | Supporting
4. Rate evidence support: how well do the evidence items back this assumption?
5. Rate confidence: your confidence that this assumption currently holds
6. Write a rationale: why does this assumption matter strategically?
7. List evidence_ids from the available evidence that support this assumption (use exact IDs)
8. Identify any conflicts with other assumptions in your list

CONFLICT DETECTION:
If two assumptions are mutually contradictory or in tension, flag them in conflict_pairs.
Also set conflicts_with on each assumption involved.

CRITICAL RULES:
- Avoid trivial or obvious assumptions ("data exists", "team will work hard")
- Avoid duplicates — each assumption must be distinct
- Focus on strategic conditions, not operational details
- Assumptions should be falsifiable — a reasonable analyst could challenge them
- Use the exact evidence_id strings from the list above (e.g. "EV-001")

Return structured JSON matching the assumption_generation schema.
"""


def _risk_prompt(
    assumptions: list[dict],
    recommendations: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
) -> str:
    """Build the RiskAgent prompt (J7.3)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))

    a_lines = ""
    for a in assumptions:
        a_id = a.get("assumption_id", "?")
        stmt = a.get("statement", "")[:120]
        imp = a.get("importance", "")
        rec_ids = ", ".join(a.get("supported_recommendation_ids", [])) or "none"
        a_lines += f"\n  {a_id} [{imp}]: {stmt}\n    → supports recommendations: {rec_ids}\n"

    r_lines = ""
    for r in recommendations:
        r_id = r.get("recommendation_id", r.get("id", "?"))
        title = r.get("title", r.get("recommendation", ""))[:100]
        a_ids = ", ".join(r.get("supported_assumption_ids", [])) or "none"
        r_lines += f"\n  {r_id}: {title}\n    → rests on assumptions: {a_ids}\n"

    ev_lines = ""
    for e in evidence_items[:20]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:100]
        src = e.get("source_document", e.get("source", ""))[:40]
        ev_lines += f"\n  {eid}: {claim}  [source: {src}]"

    return f"""You are a senior strategy consultant producing a Strategic Risk analysis.

STRATEGIC QUESTION: {question}

STRATEGIC ASSUMPTIONS (what must be true for recommendations to hold):
{a_lines or "  (none provided)"}

RECOMMENDATIONS (what the research recommends):
{r_lines or "  (none provided)"}

EVIDENCE AVAILABLE:
{ev_lines or "  (none provided)"}

TASK:
Identify 5–10 strategic risks — conditions or events that could cause one or more assumptions
above to fail, thereby threatening the validity of one or more recommendations.

For each risk:
1. State precisely what could go wrong (the risk event or condition)
2. Assign the most appropriate category
3. Rate severity: High (recommendation collapses), Medium, Low
4. Rate likelihood: High, Medium, Low
5. Rate evidence support: how strongly do the available evidence items signal this risk?
6. Rate confidence: your confidence in this risk assessment
7. Write a rationale: why does this risk matter strategically?
8. List related_assumption_ids from the assumptions above (use exact IDs)
9. List affected_recommendation_ids: which recommendations would be undermined?
   (Traverse from the related assumptions to their supported_recommendation_ids)
10. List evidence_ids from the available evidence that inform this risk
11. Optionally provide brief mitigation_notes

CRITICAL RULES:
- Each risk must threaten at least one named assumption — no floating risks
- Derive affected_recommendation_ids by following the assumption→recommendation links
- Do not duplicate risks — each must be distinct
- Avoid trivial risks ("team may not cooperate") — focus on strategic conditions
- Use exact assumption IDs (e.g. "A-001") and recommendation IDs (e.g. "REC-001")
- Risks are forward-looking threats, not observations or findings

Return structured JSON matching the risk_generation schema.
"""


def _strategy_prompt(decision_model: dict, profiles_context: list[dict]) -> str:
    """Build the ResearchStrategyAgent prompt (J6.2)."""
    profile_lines = ""
    for p in profiles_context:
        name = p.get("name", "unknown")
        desc = p.get("description", "")
        topics = ", ".join(p.get("key_topics", []))
        profile_lines += f"\n- {name}: {desc}"
        if topics:
            profile_lines += f" (key topics: {topics})"

    dm_areas = "\n".join(f"  - {a}" for a in decision_model.get("decision_areas", []))
    dm_questions = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(decision_model.get("research_questions", [])))
    dm_uncertainties = "\n".join(f"  - {u}" for u in decision_model.get("critical_uncertainties", []))
    dm_evidence = "\n".join(f"  - {e}" for e in decision_model.get("evidence_requirements", []))

    return f"""You are a research strategy agent. Given a Decision Model and available domain profiles, produce an executable research strategy.

Decision Model:
  Objective: {decision_model.get("objective", "")}
  Decision Areas:
{dm_areas}
  Research Questions:
{dm_questions}
  Critical Uncertainties:
{dm_uncertainties}
  Evidence Requirements:
{dm_evidence}

Domain Profiles:{profile_lines if profile_lines else " (none)"}

Instructions:
1. Rank each profile by its relevance to this decision model (1 = most relevant). Include all available profiles.

2. Order the research questions by decision impact — most important first. Return a list of {{question, priority}} objects.

3. List the specific evidence items needed to satisfy the decision model's evidence requirements. Be concrete (e.g. "AI power demand forecasts 2024–2030" not just "forecasts").

4. List source types in priority order (e.g. "grid operator reports", "peer-reviewed studies", "vendor datasheets").

5. For each decision area and critical uncertainty, assign a coverage target: "strong", "moderate", or "light".

6. Write 2-3 sentences explaining the strategic choices.

Return structured JSON only.
"""


def _research_plan_prompt(question: str, source_texts: Sequence[SourceDocument]) -> str:
    return f"""Create a concise research plan for this local-source research question.

Question:
{question}

Available sources:
{_source_inventory(source_texts)}

Return JSON only with this shape:
{{
  "research_questions": ["..."],
  "key_topics": ["..."],
  "source_priorities": ["..."]
}}
"""


def _evidence_prompt(question: str, source_texts: Sequence[SourceDocument]) -> str:
    return f"""Extract source-grounded evidence for this question.

Question:
{question}

Rules — MAXIMIZE RECALL:
- Use only the source text below.
- Each evidence_snippet must be copied or tightly paraphrased from one source.
- Use categories only from: architecture, power, cooling, networking, rack architecture, operations, other.
- Do not invent evidence IDs; evidence_id is assigned by the harness after extraction.
- Extract EVERY distinct atomic factual claim present in the source text.
- One claim = one fact. Decompose compound statements: "X requires Y, enabling Z" → three separate items.
- Aim for 10-30 items per source for evidence-dense content; fewer only for sparse content. No upper cap.
- Include numeric claims, specifications, constraints, timelines, and policy statements.
- Return JSON only.

CRITICAL — claim field rules (violations cause the item to be discarded):
- Each claim must summarise ONLY what its own source document states.
- Do NOT compare against other sources or reference claims from other documents.
- Do NOT use the words: contradicts, contradicting, inconsistent, conflicting, in contrast to.
- Do NOT write phrases like "Unlike other sources…", "This contradicts…", "This is inconsistent with…".
- A valid claim: "HALEU fuel is not commercially available from OECD member suppliers."
- An INVALID claim: "This contradicts claims of global HALEU availability." ← will be discarded.

JSON shape:
{{
  "evidence_items": [
    {{
      "claim": "...",
      "source_document": "filename.ext",
      "evidence_snippet": "...",
      "category": "architecture",
      "relevance": "...",
      "confidence": "high"
    }}
  ]
}}

Sources:
{_source_blocks(source_texts)}
"""


_SYNTHESIS_FIELDS = ("evidence_id", "claim", "source_document", "evidence_snippet", "category")


def _slim_evidence(item: EvidenceItem) -> dict[str, Any]:
    """Return only the fields Claude needs for synthesis (drops scoring noise)."""
    d = item.model_dump()
    return {k: d[k] for k in _SYNTHESIS_FIELDS}


def _memo_prompt(question: str, evidence_items: Sequence[EvidenceItem]) -> str:
    evidence_json = json.dumps([_slim_evidence(i) for i in evidence_items], indent=2)
    return f"""Synthesize a Markdown memo payload from the source-grounded evidence.

Question:
{question}

Rules:
- Use only the provided evidence items.
- Evidence IDs and source document names are assigned by the harness.
- Every entry in confirmed_facts, power_implications, cooling_implications, networking_implications, and rack_architecture_implications must end with exactly one citation in this format: [Source: filename.pdf, Evidence: E001].
- Use only source_document and evidence_id values present in the provided evidence.
- Do not invent source names or evidence IDs.
- Distinguish confirmed facts from inferences.
- Keep entries concise.
- Return JSON only, not Markdown.

JSON shape:
{{
  "executive_summary": "...",
  "confirmed_facts": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "inferences": ["..."],
  "power_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "cooling_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "networking_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "rack_architecture_implications": ["Claim text. [Source: filename.pdf, Evidence: E001]"],
  "open_questions": ["..."]
}}

Evidence:
{evidence_json}
"""


def _source_inventory(source_texts: Sequence[SourceDocument]) -> str:
    if not source_texts:
        return "No sources loaded."
    return "\n".join(
        f"- {source.path.name}: {source.char_count} extracted characters" for source in source_texts
    )


def _source_blocks(source_texts: Sequence[SourceDocument], *, max_chars_per_source: int = 12_000) -> str:
    if not source_texts:
        return "No sources loaded."
    blocks: list[str] = []
    for source in source_texts:
        blocks.append(
            "\n".join(
                [
                    f"Source document: {source.path.name}",
                    f"Path: {source.path}",
                    "Text:",
                    source.text[:max_chars_per_source],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _evidence_chunk_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    return f"""Extract source-grounded evidence for this question.

Question:
{question}

Rules — MAXIMIZE RECALL:
- Use only the source text below.
- Each evidence_snippet must be copied or tightly paraphrased from one chunk.
- Use categories only from: architecture, power, cooling, networking, rack architecture, operations, other.
- Do not invent evidence IDs; evidence_id is assigned by the harness after extraction.
- Set source_chunk_id to the Chunk ID shown in the header for the chunk you drew evidence from.
- Extract EVERY distinct atomic factual claim present in the source text.
- One claim = one fact. Decompose compound statements: "X requires Y, enabling Z" → three separate items.
- Aim for 10-30 items per chunk for evidence-dense content; fewer only for sparse content. No upper cap.
- Include numeric claims, specifications, constraints, timelines, and policy statements.
- Return JSON only.

CRITICAL — claim field rules (violations cause the item to be discarded):
- Each claim must summarise ONLY what its own source chunk states.
- Do NOT compare against other chunks or reference claims from other documents.
- Do NOT use the words: contradicts, contradicting, inconsistent, conflicting, in contrast to.
- Do NOT write phrases like "Unlike other sources…", "This contradicts…", "This is inconsistent with…".
- A valid claim: "The BWRX-300 is designed for construction in 24–36 months using modular techniques."
- An INVALID claim: "This contradicts estimates of longer construction timelines." ← will be discarded.

JSON shape:
{{
  "evidence_items": [
    {{
      "claim": "...",
      "source_document": "filename.ext",
      "source_chunk_id": "filename_ext_C001",
      "evidence_snippet": "...",
      "category": "architecture",
      "relevance": "...",
      "confidence": "high"
    }}
  ]
}}

Chunks:
{_chunk_blocks(chunks)}
"""


def _chunk_blocks(chunks: Sequence[Chunk]) -> str:
    """Format pre-selected chunks for the evidence extraction prompt.

    Selection and budget enforcement are handled upstream by
    ``select_relevant_chunks``; this function formats whatever it receives.
    """
    if not chunks:
        return "No chunks available."
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            "\n".join(
                [
                    f"Chunk ID: {chunk.chunk_id}",
                    f"Document: {chunk.document_name}",
                    f"Chunk: {chunk.chunk_number}",
                    "Text:",
                    chunk.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _response_text(response: Any) -> str:
    parts: list[str] = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _response_tool_input(response: Any) -> Any | None:
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "tool_use" and hasattr(block, "input"):
            return block.input
        if hasattr(block, "input") and getattr(block, "name", None):
            return block.input
    return None


def parse_or_repair_json(
    raw_response: str,
    schema_name: str,
    repair_prompt_context: dict[str, Any],
) -> dict[str, Any]:
    """Parse Claude JSON, repair once if needed, and validate with Pydantic."""

    try:
        return _validate_payload(_parse_json_text(raw_response), schema_name)
    except Exception as first_error:
        repair = repair_prompt_context.get("repair")
        if repair is None:
            raise ValueError(f"{schema_name} JSON parse failed: {first_error}") from first_error

        repair_prompt = _repair_prompt(raw_response, schema_name, repair_prompt_context)
        repaired_response = repair(repair_prompt)
        try:
            return _validate_payload(_parse_json_text(repaired_response), schema_name)
        except Exception as second_error:
            raise ValueError(
                f"{schema_name} JSON parse failed after repair: {second_error}"
            ) from second_error


def _parse_json_text(text: str) -> Any:
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(text.strip())


def _json_candidates(text: str) -> list[str]:
    stripped = text.strip()
    candidates: list[str] = []

    fence_matches = re_findall_json_fences(stripped)
    candidates.extend(fence_matches)

    if stripped:
        candidates.append(stripped)

    for opener, closer in (("{", "}"), ("[", "]")):
        start = stripped.find(opener)
        end = stripped.rfind(closer)
        if start >= 0 and end >= start:
            candidates.append(stripped[start : end + 1])

    deduped: list[str] = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def re_findall_json_fences(text: str) -> list[str]:
    import re

    return [
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    ]


def _normalize_tool_input(tool_input: Any) -> Any:
    """Coerce string-encoded list fields to actual lists before schema validation.

    Some models return ``{"evidence_items": "[{...}]"}`` — a JSON-encoded string
    where a list is expected.  Pydantic v2 does not coerce strings to lists, so
    we pre-process the dict and decode any string-valued list fields.
    """
    if not isinstance(tool_input, dict):
        return tool_input
    result = dict(tool_input)
    for key, value in result.items():
        if isinstance(value, str) and value.strip().startswith("["):
            try:
                decoded = json.loads(value)
                if isinstance(decoded, list):
                    result[key] = decoded
            except (json.JSONDecodeError, ValueError):
                pass
    return result


def _validate_payload(payload: Any, schema_name: str) -> dict[str, Any]:
    if schema_name not in _SCHEMA_ADAPTERS:
        raise ValueError(f"Unknown schema: {schema_name}")
    validated = _SCHEMA_ADAPTERS[schema_name].validate_python(payload)
    if isinstance(validated, BaseModel):
        return validated.model_dump()
    return validated


def _repair_prompt(
    raw_response: str,
    schema_name: str,
    repair_prompt_context: dict[str, Any],
) -> str:
    expected_shape = repair_prompt_context.get("expected_shape", _schema_description(schema_name))
    operation = repair_prompt_context.get("operation", schema_name)
    return f"""The previous Claude response for {operation} was not valid JSON for schema {schema_name}.

Return valid JSON only. Do not include markdown fences, comments, or prose.

Expected schema:
{expected_shape}

Invalid response:
{raw_response}
"""


def _tool_definition(operation: str, schema_name: str) -> dict[str, Any]:
    return {
        "name": operation,
        "description": f"Return structured JSON for {operation}.",
        "input_schema": _SCHEMA_ADAPTERS[schema_name].json_schema(),
    }


def _schema_description(schema_name: str) -> str:
    if schema_name not in _SCHEMA_ADAPTERS:
        return "{}"
    return json.dumps(_SCHEMA_ADAPTERS[schema_name].json_schema(), indent=2)


def _token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    values: dict[str, int] = {}
    for attr in ("input_tokens", "output_tokens"):
        value = getattr(usage, attr, None)
        if isinstance(value, int):
            values[attr] = value
    return values


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def validation_error_message(exc: ValidationError) -> str:
    return "; ".join(error["msg"] for error in exc.errors())
