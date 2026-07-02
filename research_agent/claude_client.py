"""Anthropic Claude integration for the research workflow."""

from __future__ import annotations

import json
import logging
import os
import time
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
        description='At most 6 {question: str, priority: int} dicts ranked by decision impact. "question" is a SHORT label (≤12 words), not the full restated question.',
    )
    required_evidence: list[str] = Field(
        default_factory=list,
        description="At most 6 concrete evidence items, each ≤12 words (e.g. 'AI power demand forecasts 2024-2030').",
    )
    source_priorities: list[str] = Field(
        default_factory=list,
        description="At most 5 source types, each ≤6 words (e.g. 'grid operator reports').",
    )
    coverage_targets: dict[str, str] = Field(
        default_factory=dict,
        description="At most 8 entries: topic/area → coverage level, exactly 'strong', 'moderate', or 'light'.",
    )
    strategy_rationale: str = Field(
        default="",
        description="At most 2 sentences explaining the strategy choices.",
    )


class ExecutiveDecisionStreamPayload(BaseModel):
    """One executive workstream in the Decision Architecture (J9.3)."""

    title: str = Field(description="Short workstream title, e.g. 'Power Procurement'.")
    executive_objective: str = Field(
        default="", description="What this stream must decide, ≤25 words."
    )
    related_strategic_themes: list[str] = Field(
        default_factory=list, description="At most 3 strategic themes this stream advances."
    )
    research_questions: list[str] = Field(
        default_factory=list,
        description="At most 3 supporting research questions (children of this stream), each ≤20 words.",
    )
    expected_outputs: list[str] = Field(
        default_factory=list, description="At most 2 concrete deliverables, each ≤12 words."
    )


class DecisionArchitecturePayload(BaseModel):
    """Structured output for Executive Framing (J9.3).

    Reframes the engagement as an executive decision. Bounded to stay compact
    (mirrors the J9.1b anti-truncation discipline).
    """

    executive_decision_statement: str = Field(
        default="", description="The decision being made, ≤2 sentences, executive voice."
    )
    executive_context: str = Field(
        default="", description="Why this decision matters now, ≤3 sentences."
    )
    strategic_themes: list[str] = Field(
        default_factory=list, description="At most 8 high-level consulting workstream themes."
    )
    decision_streams: list[ExecutiveDecisionStreamPayload] = Field(
        default_factory=list, description="4-6 executive decision streams. Research questions live under these."
    )
    executive_unknowns: list[str] = Field(
        default_factory=list,
        description="At most 6 unknowns most likely to change the recommendation (not ordinary research gaps).",
    )
    board_decisions_required: list[str] = Field(
        default_factory=list, description="At most 6 executive approvals required before implementation."
    )
    success_definition: list[str] = Field(
        default_factory=list, description="At most 6 measurable decision outcomes."
    )
    in_scope: list[str] = Field(
        default_factory=list, description="At most 8 areas explicitly in scope."
    )
    out_of_scope_items: list[str] = Field(
        default_factory=list, description="At most 6 areas explicitly excluded (only if clearly implied)."
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
        description="3-7 highest-leverage strategic assumptions that must hold for the recommendations to be valid",
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


class OpportunityItem(BaseModel):
    """A single strategic opportunity (J7.4)."""

    opportunity_id: str = Field(description="Short unique ID e.g. 'OPP-001'")
    statement: str = Field(description="What additional value becomes possible when the assumption exceeds expectations")
    category: str = Field(
        description=(
            "One of: Technology, Market, Economics, Regulation, Policy, Supply Chain, "
            "Competition, Customer, Execution, Geopolitics, Environment, Infrastructure, Finance, Other"
        )
    )
    impact: str = Field(description="High | Medium | Low — magnitude of upside if the opportunity is captured")
    likelihood: str = Field(description="High | Medium | Low — probability of the favourable condition materialising")
    evidence_support: str = Field(description="Strong | Moderate | Weak | None")
    confidence: str = Field(description="High | Medium | Low — confidence in this opportunity assessment")
    rationale: str = Field(description="Why this opportunity matters strategically and what makes it achievable")
    related_assumption_ids: list[str] = Field(
        default_factory=list,
        description="assumption_ids whose upside scenario this opportunity describes",
    )
    enabled_recommendation_ids: list[str] = Field(
        default_factory=list,
        description="REC-NNN ids of recommendations that would be amplified if this opportunity is captured",
    )
    evidence_ids: list[str] = Field(
        default_factory=list,
        description="IDs of evidence items that support or inform this opportunity",
    )
    exploitation_notes: str = Field(default="", description="High-level actions to capture this opportunity")
    status: str = Field(default="Active", description="Active | Realized | Expired")


class OpportunityPayload(BaseModel):
    """Structured output for OpportunityAgent (J7.4)."""

    opportunities: list[OpportunityItem] = Field(
        default_factory=list,
        description="5-10 strategic opportunities that become available when assumptions exceed expectations",
    )


class StrategicOptionItem(BaseModel):
    """A single strategic option synthesising the full J7 graph (J7.5)."""

    option_id: str = Field(description="Short unique ID e.g. 'OPT-A'")
    title: str = Field(description="Short descriptive name for this option (5-10 words)")
    description: str = Field(description="What this option entails — 2-4 sentences")
    strategic_objective: str = Field(description="The primary objective this option is trying to achieve")
    expected_outcomes: list[str] = Field(
        default_factory=list,
        description="2-5 specific outcomes expected from pursuing this option",
    )
    supporting_assumption_ids: list[str] = Field(
        default_factory=list,
        description="assumption_ids this option depends on (use exact IDs e.g. 'A-001')",
    )
    associated_risk_ids: list[str] = Field(
        default_factory=list,
        description="RSK-NNN ids of risks this option must manage or mitigate",
    )
    associated_opportunity_ids: list[str] = Field(
        default_factory=list,
        description="OPP-NNN ids of opportunities this option is positioned to capture",
    )
    supporting_recommendation_ids: list[str] = Field(
        default_factory=list,
        description="REC-NNN ids of recommendations this option implements",
    )
    advantages: list[str] = Field(
        default_factory=list,
        description="2-4 key advantages of this option",
    )
    disadvantages: list[str] = Field(
        default_factory=list,
        description="2-4 key disadvantages or trade-offs of this option",
    )
    implementation_complexity: str = Field(
        default="Medium",
        description="Low | Medium | High — overall difficulty of implementation",
    )
    estimated_time_horizon: str = Field(
        default="Medium-term",
        description="Near-term | Medium-term | Long-term",
    )
    capital_intensity: str = Field(
        default="Medium",
        description="Low | Medium | High — relative capital requirement",
    )
    confidence: str = Field(
        default="Medium",
        description="High | Medium | Low — confidence that this option will succeed",
    )
    recommended: bool = Field(
        default=False,
        description="True for exactly ONE option — the preferred course of action",
    )
    rationale: str = Field(
        description="Why this option is (or is not) the preferred choice — compare against alternatives",
    )


class StrategicOptionPayload(BaseModel):
    """Structured output for StrategicOptionAgent (J7.5)."""

    options: list[StrategicOptionItem] = Field(
        default_factory=list,
        description="Between 2 and 5 genuinely different strategic options; exactly one has recommended=True",
    )


class DecisionMatrixEntryItem(BaseModel):
    """Per-option row in the decision matrix (J7.6)."""

    option_id: str
    strategic_fit: str = Field(description="Very High | High | Medium | Low | Very Low")
    implementation_risk: str = Field(description="Very High | High | Medium | Low | Very Low")
    execution_complexity: str = Field(description="Very High | High | Medium | Low | Very Low")
    capital_requirement: str = Field(description="Very High | High | Medium | Low | Very Low")
    expected_return: str = Field(description="Very High | High | Medium | Low | Very Low")
    time_to_value: str = Field(description="Very High | High | Medium | Low | Very Low")
    dependency_strength: str = Field(description="Very High | High | Medium | Low | Very Low")
    assumption_strength: str = Field(description="Very High | High | Medium | Low | Very Low")
    risk_exposure: str = Field(description="Very High | High | Medium | Low | Very Low")
    opportunity_capture: str = Field(description="Very High | High | Medium | Low | Very Low")
    overall_score: str = Field(description="Very High | High | Medium | Low | Very Low")
    strengths: list[str] = Field(default_factory=list, description="2-4 key strengths of this option")
    weaknesses: list[str] = Field(default_factory=list, description="2-4 key weaknesses of this option")


class DecisionAnalysisItem(BaseModel):
    """Structured output for DecisionAnalysisAgent (J7.6)."""

    analysis_id: str = Field(description="Unique ID e.g. 'DA-001'")
    recommended_option_id: str = Field(description="option_id of the preferred Strategic Option")
    executive_summary: str = Field(description="2-4 sentence plain-English explanation of why this option wins")
    comparison_dimensions: list[str] = Field(
        default_factory=list,
        description="List of dimension names used in the decision matrix",
    )
    option_rankings: list[str] = Field(
        default_factory=list,
        description="option_ids ordered from most to least preferred",
    )
    decision_matrix: list[DecisionMatrixEntryItem] = Field(
        default_factory=list,
        description="One row per Strategic Option rating each across all comparison dimensions",
    )
    key_tradeoffs: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit tradeoff statements. Format: 'Higher X → Lower Y'. "
            "Derive ONLY from the existing graph — do not invent new tradeoffs."
        ),
    )
    key_uncertainties: list[str] = Field(
        default_factory=list,
        description="Uncertainties or assumption failures that could shift the preferred option",
    )
    sensitivity_analysis: str = Field(
        description=(
            "Explain which specific assumption IDs, if they fail, would change the preferred option. "
            "Reference assumption_ids by name. Do NOT invent new scenarios."
        ),
    )
    confidence_summary: str = Field(description="Overall confidence level and key limiting factors")
    rationale: str = Field(description="Full justification for why the recommended option wins over each alternative")
    confidence: str = Field(default="Medium", description="High | Medium | Low")


class DecisionAnalysisPayload(BaseModel):
    """Structured output wrapper for DecisionAnalysisAgent (J7.6)."""

    analysis: DecisionAnalysisItem
    # PH1a — LLM output normalization diagnostics (None when not applicable).
    normalization: dict | None = None


class ExecutiveConfidenceItem(BaseModel):
    """Structured output for ExecutiveConfidenceAgent (J7.7)."""

    confidence_id: str = Field(description="Unique ID e.g. 'EC-001'")
    overall_confidence: str = Field(
        default="Medium",
        description="High | Medium | Low — synthesised over the full decision graph",
    )
    decision_readiness: str = Field(
        default="Needs Additional Validation",
        description="Ready for Decision | Needs Additional Validation | Not Ready",
    )
    board_recommendation: str = Field(
        default="Proceed with Conditions",
        description="Proceed | Proceed with Conditions | Delay Pending Evidence | Reject",
    )
    confidence_rationale: str = Field(
        description=(
            "2-4 sentence plain-English rationale for the overall confidence level. "
            "Must derive from the existing graph — no new reasoning."
        ),
    )
    confidence_drivers: list[str] = Field(
        default_factory=list,
        description="Factors from the existing graph that raise confidence (e.g. 'Strong evidence base')",
    )
    confidence_limiters: list[str] = Field(
        default_factory=list,
        description="Factors from the existing graph that lower confidence (e.g. 'Vendor-only sources')",
    )
    critical_unknowns: list[str] = Field(
        default_factory=list,
        description="Unknowns that must resolve before the decision can be made confidently",
    )
    validation_priorities: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered due-diligence checklist of the most important items to validate before approving. "
            "Derive from critical_unknowns and assumption gaps. 3-7 items."
        ),
    )
    confidence_if_assumptions_hold: str = Field(
        description="Confidence level (High/Medium/Low) and rationale if all Critical assumptions hold",
    )
    confidence_if_assumptions_fail: str = Field(
        description="Confidence level and rationale if key Critical assumptions fail",
    )
    decision_horizon: str = Field(
        default="",
        description="When a decision should be made (e.g. 'Q3 2026', 'Before Series B close')",
    )
    last_updated: str = Field(default="", description="ISO timestamp; populated by the system")


class ExecutiveConfidencePayload(BaseModel):
    """Structured output wrapper for ExecutiveConfidenceAgent (J7.7)."""

    confidence: ExecutiveConfidenceItem


class StrategicSynthesisPayload(BaseModel):
    """Cross-domain strategic synthesis (J10.7).

    Integrates independent per-Decision-Domain reasoning into one executive
    perspective. Executive reasoning ONLY — no recommendations or implementation
    plans. Bounded to stay compact (J9.1b discipline).
    """

    executive_summary: str = Field(
        default="", description="Cross-domain executive perspective, <=4 sentences."
    )
    cross_domain_findings: list[str] = Field(
        default_factory=list, description="At most 8 findings spanning multiple Decision Domains."
    )
    cross_domain_dependencies: list[str] = Field(
        default_factory=list,
        description="At most 8 dependencies, form 'A requires/depends-on B' across domains.",
    )
    cross_domain_conflicts: list[str] = Field(
        default_factory=list, description="At most 6 tensions/conflicts between domains."
    )
    strategic_levers: list[str] = Field(
        default_factory=list, description="At most 6 leverage points that move multiple domains."
    )
    dominant_constraints: list[str] = Field(
        default_factory=list, description="At most 6 constraints that bind the overall decision."
    )
    emerging_themes: list[str] = Field(
        default_factory=list, description="At most 8 themes emerging across domains."
    )


_SCHEMA_ADAPTERS = {
    "research_plan": TypeAdapter(ResearchPlan),
    "research_planning": TypeAdapter(ResearchPlanningPayload),
    "problem_framing": TypeAdapter(DecisionModelPayload),
    "executive_framing": TypeAdapter(DecisionArchitecturePayload),  # J9.3
    "research_strategy": TypeAdapter(ResearchStrategyPayload),
    "hypothesis_generation": TypeAdapter(HypothesisPayload),
    "challenge_generation": TypeAdapter(ChallengePayload),
    "recommendation_generation": TypeAdapter(RecommendationPayload),
    "assumption_generation": TypeAdapter(AssumptionPayload),     # J7.1
    "risk_generation": TypeAdapter(RiskPayload),                 # J7.3
    "opportunity_generation": TypeAdapter(OpportunityPayload),   # J7.4
    "strategic_option_generation": TypeAdapter(StrategicOptionPayload),  # J7.5
    "decision_analysis_generation": TypeAdapter(DecisionAnalysisPayload),  # J7.6
    "strategic_synthesis": TypeAdapter(StrategicSynthesisPayload),  # J10.7
    "executive_confidence_generation": TypeAdapter(ExecutiveConfidencePayload),  # J7.7
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
        strategic_synthesis: dict | None = None,
    ) -> "RecommendationPayload":
        """Return deterministic recommendations derived from surviving hypotheses.

        J10.8 — accepts strategic_synthesis for signature parity; the deterministic
        mock derives recommendations from hypotheses/evidence and does not vary on it.
        """
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

    def generate_opportunities(
        self,
        assumptions: list[dict],
        recommendations: list[dict],
        risks: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> "OpportunityPayload":
        """Generate strategic opportunities from assumptions (J7.4) — mock version."""
        ev_ids = [e.get("evidence_id", "") for e in evidence_items if e.get("evidence_id")]
        question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

        opp_templates = [
            ("Technology matures faster than expected, enabling earlier deployment and first-mover advantage", "Technology", "High", "Medium"),
            ("Market demand accelerates above projections, creating a larger addressable opportunity", "Market", "High", "Medium"),
            ("Capital costs decline faster than modelled, improving project economics materially", "Economics", "High", "Low"),
            ("Regulatory environment becomes more favourable, reducing compliance burden and opening new markets", "Regulation", "Medium", "Medium"),
            ("Supply chain innovations reduce lead times, enabling faster scale-up than planned", "Supply Chain", "Medium", "Low"),
        ]

        opportunities = []
        for i, (stmt, cat, imp, lik) in enumerate(opp_templates):
            related_a = [assumptions[i]["assumption_id"]] if i < len(assumptions) else []
            enabled_r: list[str] = []
            if related_a and i < len(assumptions):
                enabled_r = list(assumptions[i].get("supported_recommendation_ids", []))
            sup_ev = ev_ids[i*2 : i*2+2] if len(ev_ids) > i*2 else ev_ids[:1]
            opportunities.append(OpportunityItem(
                opportunity_id=f"OPP-{i+1:03d}",
                statement=stmt,
                category=cat,
                impact=imp,
                likelihood=lik,
                evidence_support="Moderate",
                confidence="Medium",
                rationale=f"Strategic opportunity relevant to: {question[:80]}",
                related_assumption_ids=related_a,
                enabled_recommendation_ids=enabled_r,
                evidence_ids=sup_ev,
                exploitation_notes="",
                status="Active",
            ))

        return OpportunityPayload(opportunities=opportunities)

    def generate_strategic_options(
        self,
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> "StrategicOptionPayload":
        """Generate strategic options synthesising the J7 graph (J7.5) — mock version."""
        question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

        # Collect IDs for linkage
        a_ids = [a.get("assumption_id", "") for a in assumptions if a.get("assumption_id")]
        r_ids = [r.get("risk_id", "") for r in risks if r.get("risk_id")]
        o_ids = [o.get("opportunity_id", "") for o in opportunities if o.get("opportunity_id")]
        rec_ids = [r.get("recommendation_id", r.get("id", "")) for r in recommendations if r.get("recommendation_id") or r.get("id")]

        def _slice(lst, start, n): return lst[start:start+n] if lst else []

        options = [
            StrategicOptionItem(
                option_id="OPT-A",
                title="Aggressive first-mover investment",
                description=(
                    f"Commit maximum capital immediately to capture the earliest advantage on: {question[:60]}. "
                    "Accept higher execution risk in exchange for market leadership."
                ),
                strategic_objective="Establish first-mover advantage and market leadership",
                expected_outcomes=[
                    "Early capacity secured before competitor entry",
                    "Premium positioning in the market",
                    "Accelerated learning curve",
                ],
                supporting_assumption_ids=_slice(a_ids, 0, 2),
                associated_risk_ids=_slice(r_ids, 0, 2),
                associated_opportunity_ids=_slice(o_ids, 0, 2),
                supporting_recommendation_ids=_slice(rec_ids, 0, 1),
                advantages=["Speed to market", "Captures first-mover upside"],
                disadvantages=["High capital at risk", "Limited optionality"],
                implementation_complexity="High",
                estimated_time_horizon="Near-term",
                capital_intensity="High",
                confidence="Medium",
                recommended=False,
                rationale="High reward but high execution risk; preferred only if market timing is critical.",
            ),
            StrategicOptionItem(
                option_id="OPT-B",
                title="Phased deployment preserving optionality",
                description=(
                    f"Deploy in staged tranches, validating assumptions at each gate before committing further capital. "
                    "Balances speed with risk management."
                ),
                strategic_objective="Optimise risk-adjusted returns through staged commitment",
                expected_outcomes=[
                    "Capital deployed only when assumptions validated",
                    "Ability to course-correct at each gate",
                    "Sustainable long-term position",
                ],
                supporting_assumption_ids=_slice(a_ids, 1, 3),
                associated_risk_ids=_slice(r_ids, 1, 3),
                associated_opportunity_ids=_slice(o_ids, 1, 2),
                supporting_recommendation_ids=_slice(rec_ids, 0, 2),
                advantages=["Lower downside risk", "Preserves optionality", "Validates assumptions iteratively"],
                disadvantages=["Slower to full scale", "May cede first-mover advantage"],
                implementation_complexity="Medium",
                estimated_time_horizon="Medium-term",
                capital_intensity="Medium",
                confidence="High",
                recommended=True,
                rationale=(
                    "Preferred over OPT-A because it manages the key risks identified while still "
                    "capturing the primary opportunities. Preferred over OPT-C because it commits "
                    "meaningfully rather than hedging across all vectors."
                ),
            ),
            StrategicOptionItem(
                option_id="OPT-C",
                title="Conservative multi-partner ecosystem strategy",
                description=(
                    "Build through partnerships and ecosystem relationships rather than direct ownership, "
                    "minimising upfront capital while establishing strategic positioning."
                ),
                strategic_objective="Minimise capital risk while establishing ecosystem optionality",
                expected_outcomes=[
                    "Lower capital commitment",
                    "Diversified risk across partners",
                    "Ecosystem relationships established",
                ],
                supporting_assumption_ids=_slice(a_ids, 2, 2),
                associated_risk_ids=_slice(r_ids, 2, 2),
                associated_opportunity_ids=_slice(o_ids, 2, 2),
                supporting_recommendation_ids=_slice(rec_ids, 1, 2),
                advantages=["Lowest capital at risk", "Flexible exit options"],
                disadvantages=["Diluted upside", "Dependency on partner alignment", "Slower decision-making"],
                implementation_complexity="Low",
                estimated_time_horizon="Long-term",
                capital_intensity="Low",
                confidence="Medium",
                recommended=False,
                rationale="Lower risk but also lower return; appropriate only if capital preservation is the primary constraint.",
            ),
        ]

        return StrategicOptionPayload(options=options)

    def generate_decision_analysis(
        self,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        decision_model: dict,
    ) -> "DecisionAnalysisPayload":
        """Generate decision analysis comparing strategic options (J7.6) — mock version."""
        question = decision_model.get("strategic_question", decision_model.get("objective", "the decision"))

        # Find the recommended option
        rec_opt = next((o for o in strategic_options if o.get("recommended")), None)
        rec_id = rec_opt["option_id"] if rec_opt else (strategic_options[0]["option_id"] if strategic_options else "OPT-A")
        opt_ids = [o.get("option_id", f"OPT-{i}") for i, o in enumerate(strategic_options)]

        # Build a matrix row per option
        _score_map = {"OPT-A": "High", "OPT-B": "Very High", "OPT-C": "Medium"}

        def _row(opt: dict) -> DecisionMatrixEntryItem:
            oid = opt.get("option_id", "OPT-X")
            score = _score_map.get(oid, "Medium")
            low = "Low" if oid == "OPT-A" else ("Very High" if oid == "OPT-C" else "Medium")
            return DecisionMatrixEntryItem(
                option_id=oid,
                strategic_fit=score,
                implementation_risk="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "Medium"),
                execution_complexity="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "Medium"),
                capital_requirement="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "Medium"),
                expected_return="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "High"),
                time_to_value="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "Medium"),
                dependency_strength="High",
                assumption_strength=score,
                risk_exposure="High" if oid == "OPT-A" else ("Low" if oid == "OPT-C" else "Medium"),
                opportunity_capture="High" if oid in ("OPT-A", "OPT-B") else "Medium",
                overall_score=score,
                strengths=["Speed to market", "Captures full upside"] if oid == "OPT-A" else (
                    ["Balanced risk-return", "Preserves optionality"] if oid == "OPT-B" else
                    ["Lowest capital at risk", "Flexible exit options"]
                ),
                weaknesses=["High capital at risk", "Limited optionality"] if oid == "OPT-A" else (
                    ["Slower to full scale", "May cede first-mover advantage"] if oid == "OPT-B" else
                    ["Diluted upside", "Partner dependency"]
                ),
            )

        matrix = [_row(o) for o in strategic_options]
        rankings = [rec_id] + [oid for oid in opt_ids if oid != rec_id]

        # Build assumption-sensitivity references
        a_ids = [a.get("assumption_id", "") for a in assumptions if a.get("assumption_id")]
        sens_ref = f"If {a_ids[0]} fails" if a_ids else "If the primary assumption fails"
        sens_ref2 = f"If {a_ids[1]} proves optimistic" if len(a_ids) > 1 else "If market timing shifts"

        analysis = DecisionAnalysisItem(
            analysis_id="DA-001",
            recommended_option_id=rec_id,
            executive_summary=(
                f"The preferred option for '{question[:60]}' balances risk-adjusted returns with "
                "capital discipline. It outperforms alternatives on strategic fit and opportunity "
                "capture while managing the key risks identified in the assumption graph. "
                "This analysis is derived entirely from the existing reasoning graph."
            ),
            comparison_dimensions=[
                "Strategic Fit", "Implementation Risk", "Execution Complexity",
                "Capital Requirement", "Expected Return", "Time to Value",
                "Assumption Strength", "Risk Exposure", "Opportunity Capture",
            ],
            option_rankings=rankings,
            decision_matrix=matrix,
            key_tradeoffs=[
                "Higher capital commitment → lower execution risk and faster time-to-value",
                "Longer implementation horizon → greater strategic flexibility and optionality",
                "Higher assumption robustness → lower sensitivity to market timing",
                "Ecosystem partnership approach → reduced upside but diversified downside",
            ],
            key_uncertainties=[
                f"{sens_ref} the risk-return balance shifts materially toward the conservative option",
                f"{sens_ref2} the aggressive option becomes relatively more attractive",
                "Regulatory environment changes could invalidate cross-option assumptions",
            ],
            sensitivity_analysis=(
                f"{sens_ref}, the recommended option would no longer dominate — the conservative "
                f"option would become preferred. {sens_ref2}, the aggressive option's time-to-value "
                "advantage would amplify, making it competitive with the recommended option. "
                "The recommended option remains preferred under the majority of plausible scenarios."
            ),
            confidence_summary=(
                "Medium confidence. The analysis relies on the assumption set which has moderate "
                "evidence support. Key limiting factors: capital cost estimates and regulatory "
                "timeline assumptions carry the most uncertainty."
            ),
            rationale=(
                f"The recommended option ({rec_id}) wins over alternatives because it achieves the "
                "highest overall score on the decision matrix, particularly on strategic fit and "
                "opportunity capture, while maintaining manageable risk exposure. Unlike the aggressive "
                "option, it does not require all assumptions to hold simultaneously. Unlike the "
                "conservative option, it commits meaningfully rather than hedging across all vectors, "
                "which would dilute expected returns below the required threshold."
            ),
            confidence="Medium",
        )

        return DecisionAnalysisPayload(analysis=analysis)

    def generate_strategic_synthesis(
        self,
        domain_plans: list[dict],
        domain_evidence: list[dict],
        domain_hypotheses: list[dict],
        decision_architecture: dict,
    ) -> "StrategicSynthesisPayload":
        """Cross-domain strategic synthesis (J10.7) — deterministic mock version."""
        titles = [
            (d.get("decision_domain_title") or d.get("title") or f"Domain {i + 1}")
            for i, d in enumerate(domain_hypotheses or domain_plans or [])
        ]
        themes = list(decision_architecture.get("strategic_themes", [])) or titles
        statement = decision_architecture.get("decision_statement", "the decision")

        findings = [
            f"{t}: {len((domain_hypotheses[i] or {}).get('hypotheses', []))} hypotheses generated"
            for i, t in enumerate(titles)
        ][:8]
        dependencies = [
            f"{titles[i]} depends on {titles[i + 1]}" for i in range(len(titles) - 1)
        ][:8]
        conflicts = (
            [f"Tension between {titles[0]} and {titles[-1]}"] if len(titles) >= 2 else []
        )
        return StrategicSynthesisPayload(
            executive_summary=(
                f"Cross-domain synthesis for {statement}: {len(titles)} decision domains integrated."
            ),
            cross_domain_findings=findings,
            cross_domain_dependencies=dependencies,
            cross_domain_conflicts=conflicts,
            strategic_levers=themes[:6],
            dominant_constraints=list(decision_architecture.get("executive_unknowns", []))[:6],
            emerging_themes=themes[:8],
        )

    def generate_executive_confidence(
        self,
        decision_analysis: dict,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        scenarios: list[dict],
        decision_model: dict,
    ) -> "ExecutiveConfidencePayload":
        """Generate executive confidence assessment over the J7 graph (J7.7) — mock version."""
        # Derive confidence from assumption importance distribution
        critical = sum(1 for a in assumptions if a.get("importance") == "Critical")
        weak_ev = sum(1 for a in assumptions if a.get("evidence_support") == "Weak")
        high_risks = sum(1 for r in risks if r.get("severity") == "High")
        da_conf = decision_analysis.get("confidence", "Medium")

        # Map to OverallConfidence
        if da_conf == "High" and critical <= 2 and weak_ev == 0:
            overall = "High"
            readiness = "Ready for Decision"
            board_rec = "Proceed"
        elif da_conf == "Low" or high_risks >= 3 or weak_ev >= 2:
            overall = "Low"
            readiness = "Not Ready"
            board_rec = "Delay Pending Evidence"
        else:
            overall = "Medium"
            readiness = "Needs Additional Validation"
            board_rec = "Proceed with Conditions"

        a_ids = [a.get("assumption_id", "") for a in assumptions if a.get("assumption_id")]
        crit_a = [a for a in assumptions if a.get("importance") == "Critical"]
        crit_ids = [a.get("assumption_id", "") for a in crit_a]

        drivers = [
            f"Decision analysis confidence: {da_conf}",
            f"{len(strategic_options)} strategic options explicitly compared",
        ]
        if len(opportunities) > 0:
            drivers.append(f"{len(opportunities)} strategic opportunities identified")
        if high_risks == 0:
            drivers.append("No High-severity risks in the risk register")

        limiters = []
        if weak_ev > 0:
            limiters.append(f"{weak_ev} assumption(s) have Weak evidence support")
        if high_risks > 0:
            limiters.append(f"{high_risks} High-severity risk(s) require mitigation")
        if critical > 0:
            limiters.append(f"{critical} Critical assumption(s) must hold for the strategy to succeed")
        if not limiters:
            limiters.append("Evidence base relies primarily on vendor-sourced material")

        unknowns = [f"Resolution of {aid}" for aid in crit_ids[:3]] if crit_ids else [
            "Primary assumption validation",
            "Independent evidence verification",
        ]

        priorities = []
        for a in crit_a[:4]:
            stmt = a.get("statement", "")[:60]
            priorities.append(f"Validate: {stmt}")
        if not priorities:
            priorities = ["Validate critical assumptions with independent evidence"]
        if high_risks > 0:
            priorities.append(f"Mitigate {high_risks} High-severity risk(s) before commitment")

        if_hold = (
            f"High confidence — if all {len(crit_a)} Critical assumption(s) hold, "
            "the recommended option achieves its strategic objectives with manageable risk."
        ) if crit_a else "High confidence — core assumptions are well-supported."

        if_fail = (
            f"Low confidence — if Critical assumption(s) ({', '.join(crit_ids[:2])}) fail, "
            "the strategy's risk-return profile shifts materially and the preferred option may no longer dominate."
        ) if crit_ids else "Medium confidence — strategy remains viable under partial assumption failure."

        item = ExecutiveConfidenceItem(
            confidence_id="EC-001",
            overall_confidence=overall,
            decision_readiness=readiness,
            board_recommendation=board_rec,
            confidence_rationale=(
                f"Overall confidence is {overall}. The decision analysis identified {len(strategic_options)} "
                f"strategic options with {critical} Critical assumptions underpinning the recommended path. "
                f"{len(risks)} risks have been identified, of which {high_risks} are High-severity. "
                "This assessment synthesises the full J7 decision graph."
            ),
            confidence_drivers=drivers,
            confidence_limiters=limiters,
            critical_unknowns=unknowns,
            validation_priorities=priorities,
            confidence_if_assumptions_hold=if_hold,
            confidence_if_assumptions_fail=if_fail,
            decision_horizon="Q3 2026",
        )
        return ExecutiveConfidencePayload(confidence=item)


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

    def frame_executive_decision(
        self,
        engagement: dict | None,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> DecisionArchitecturePayload:
        """Executive Framing (J9.3): produce a Decision Architecture by reasoning.

        Reframes the engagement as an executive decision — statement, context,
        strategic themes, decision streams (with research questions as children),
        executive unknowns, board decisions, success criteria, and scope.
        """
        # J9.3 — 4000 covers the bounded architecture (~1500 tokens) with headroom
        # for tool-call JSON of nested streams. Schema + prompt cap counts so this
        # is not driven higher; the agent falls back to deterministic derivation if
        # the response still truncates.
        payload = self._call_json(
            operation="executive_framing",
            schema_name="executive_framing",
            prompt=_executive_framing_prompt(engagement, decision_model, profiles_context),
            max_tokens=4000,
        )
        return DecisionArchitecturePayload.model_validate(payload)

    def generate_research_strategy(
        self,
        decision_model: dict,
        profiles_context: list[dict],
    ) -> ResearchStrategyPayload:
        """Transform a Decision Model into an executable research strategy (J6.2)."""
        # J9.1b — 2000 is ample for the bounded strategy object (~600-800 tokens
        # incl. tool-call JSON). The prompt and ResearchStrategyPayload schema now
        # cap counts and forbid restating the brief, so this is not raised; the
        # agent falls back to a deterministic bounded strategy if truncation recurs.
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
            max_tokens=5000,
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
            max_tokens=6000,
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
        strategic_synthesis: dict | None = None,
    ) -> RecommendationPayload:
        """Generate actionable recommendations from challenged hypotheses (J6.5).

        J10.8 — an optional Strategic Synthesis block shapes reasoning and
        prioritisation (evidence citations still come from evidence items).
        """
        payload = self._call_json(
            operation="generate_recommendations",
            schema_name="recommendation_generation",
            prompt=_recommendation_prompt(
                hypotheses, surviving_hypotheses, hypothesis_challenges,
                evidence_items, decision_model, research_strategy,
                validated_contradictions=validated_contradictions or [],
                strategic_synthesis=strategic_synthesis,
            ),
            max_tokens=6000,
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
            max_tokens=5000,
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
            max_tokens=5000,
        )
        return RiskPayload.model_validate(payload)

    def generate_opportunities(
        self,
        assumptions: list[dict],
        recommendations: list[dict],
        risks: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> OpportunityPayload:
        """Identify strategic opportunities from upside assumption scenarios (J7.4)."""
        payload = self._call_json(
            operation="generate_opportunities",
            schema_name="opportunity_generation",
            prompt=_opportunity_prompt(assumptions, recommendations, risks, evidence_items, decision_model),
            max_tokens=5000,
        )
        return OpportunityPayload.model_validate(payload)

    def generate_strategic_options(
        self,
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        evidence_items: list[dict],
        decision_model: dict,
    ) -> StrategicOptionPayload:
        """Generate strategic options synthesising the J7 reasoning graph (J7.5)."""
        payload = self._call_json(
            operation="generate_strategic_options",
            schema_name="strategic_option_generation",
            prompt=_strategic_options_prompt(
                assumptions, risks, opportunities, recommendations, evidence_items, decision_model
            ),
            max_tokens=6000,
        )
        return StrategicOptionPayload.model_validate(payload)

    def generate_decision_analysis(
        self,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        decision_model: dict,
    ) -> DecisionAnalysisPayload:
        """Generate decision analysis comparing strategic options (J7.6)."""
        payload = self._call_json(
            operation="generate_decision_analysis",
            schema_name="decision_analysis_generation",
            prompt=_decision_analysis_prompt(
                strategic_options, assumptions, risks, opportunities, recommendations, decision_model
            ),
            max_tokens=6000,
        )
        # PH1a — normalize the 'analysis' object at the boundary BEFORE typed
        # validation. The model intermittently returns 'analysis' as a stringified
        # JSON object (or a plain string); normalization deserializes the former
        # and drops the latter so a malformed payload degrades to the caller's
        # deterministic fallback instead of raising a raw pydantic error.
        from .llm_normalize import normalize_llm_object
        raw_analysis = payload.get("analysis") if isinstance(payload, dict) else payload
        norm_obj, diag = normalize_llm_object(
            raw_analysis,
            required_fields=("recommended_option_id",),
            component="decision_analysis",
        )
        # When normalization fails, pass the original through so model_validate
        # raises (the agent catches it and falls back); flag the fallback.
        analysis_value = norm_obj if norm_obj is not None else raw_analysis
        if norm_obj is None:
            diag["fallback_used"] = True
        result = DecisionAnalysisPayload.model_validate({"analysis": analysis_value})
        result.normalization = diag
        return result

    def generate_strategic_synthesis(
        self,
        domain_plans: list[dict],
        domain_evidence: list[dict],
        domain_hypotheses: list[dict],
        decision_architecture: dict,
    ) -> StrategicSynthesisPayload:
        """Cross-domain strategic synthesis (J10.7) — one integration call."""
        payload = self._call_json(
            operation="generate_strategic_synthesis",
            schema_name="strategic_synthesis",
            prompt=_strategic_synthesis_prompt(
                domain_plans, domain_evidence, domain_hypotheses, decision_architecture,
            ),
            max_tokens=4000,
        )
        return StrategicSynthesisPayload.model_validate(payload)

    def generate_executive_confidence(
        self,
        decision_analysis: dict,
        strategic_options: list[dict],
        assumptions: list[dict],
        risks: list[dict],
        opportunities: list[dict],
        recommendations: list[dict],
        scenarios: list[dict],
        decision_model: dict,
    ) -> ExecutiveConfidencePayload:
        """Generate executive confidence assessment over the J7 graph (J7.7)."""
        payload = self._call_json(
            operation="generate_executive_confidence",
            schema_name="executive_confidence_generation",
            prompt=_executive_confidence_prompt(
                decision_analysis, strategic_options, assumptions,
                risks, opportunities, recommendations, scenarios, decision_model,
            ),
            max_tokens=6000,
        )
        return ExecutiveConfidencePayload.model_validate(payload)

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
            max_tokens=max(self.max_tokens, 6_500),
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
            max_tokens=max(self.max_tokens, 4_500),
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
        _t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=_model,
                max_tokens=max_tokens or self.max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                tools=[_tool_definition(operation, schema_name)],
                tool_choice={"type": "tool", "name": operation},
            )
            _llm_ms = (time.monotonic() - _t0) * 1000
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
                    duration_ms=round(_llm_ms, 1),
                )
            )
            return payload
        except Exception as exc:
            _llm_ms = (time.monotonic() - _t0) * 1000
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=operation,
                    model_name=_model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                    duration_ms=round(_llm_ms, 1),
                )
            )
            raise

    def _repair_json_text(self, *, operation: str, prompt: str, max_tokens: int) -> str:
        request_timestamp = datetime.now(timezone.utc).isoformat()
        _t0 = time.monotonic()
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            _llm_ms = (time.monotonic() - _t0) * 1000
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=True,
                    token_usage=_token_usage(response),
                    duration_ms=round(_llm_ms, 1),
                )
            )
            return _response_text(response)
        except Exception as exc:
            _llm_ms = (time.monotonic() - _t0) * 1000
            self.call_traces.append(
                ClaudeCallTrace(
                    operation=f"{operation}_repair",
                    model_name=self.model,
                    request_timestamp=request_timestamp,
                    success=False,
                    error=str(exc),
                    duration_ms=round(_llm_ms, 1),
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


def _executive_framing_prompt(
    engagement: dict | None,
    decision_model: dict,
    profiles_context: list[dict],
) -> str:
    """Build the Executive Framing prompt (J9.3).

    Feeds the structured engagement plus the problem-framing outputs so the model
    reasons about how a strategy consultant would structure the engagement as an
    executive decision. Kept tightly bounded to avoid output truncation.
    """
    eng = engagement or {}

    def _lines(label: str, items: list) -> str:
        items = [str(i).strip() for i in (items or []) if str(i).strip()]
        return f"\n{label}:\n" + "\n".join(f"  - {i}" for i in items) if items else ""

    profiles = ", ".join(p.get("name", "") for p in profiles_context if p.get("name")) or "(none)"

    engagement_block = ""
    if eng:
        engagement_block = "Engagement:\n"
        for key in ("title", "client", "industry", "current_situation", "decision_horizon"):
            val = (eng.get(key) or "").strip() if isinstance(eng.get(key), str) else eng.get(key)
            if val:
                engagement_block += f"  {key}: {val}\n"
        engagement_block += _lines("  Objectives", eng.get("objectives"))
        engagement_block += _lines("  Priorities", eng.get("priorities"))
        engagement_block += _lines("  Constraints", eng.get("constraints"))
        engagement_block += _lines("  Success criteria", eng.get("success_criteria"))
        engagement_block += _lines("  Known unknowns", eng.get("known_unknowns"))

    dm_block = (
        f"Objective: {decision_model.get('objective', '')}"
        + _lines("Decision areas", decision_model.get("decision_areas"))
        + _lines("Research questions", decision_model.get("research_questions"))
        + _lines("Critical uncertainties", decision_model.get("critical_uncertainties"))
    )

    return f"""You are a senior strategy consultant framing an executive engagement. Reframe the material below as an EXECUTIVE DECISION — not a research plan. Research is a supporting workstream, not the primary product.

{engagement_block}
Problem framing so far:
{dm_block}

Available domain profiles: {profiles}

Produce a Decision Architecture the way an experienced consultant would structure it for a board:
1. executive_decision_statement: the decision being made, ≤2 sentences, executive voice (not a research question).
2. executive_context: why this decision matters now, ≤3 sentences.
3. strategic_themes: at most 8 consulting workstream themes.
4. decision_streams: 4-6 streams. Each has a title, an executive_objective (≤25 words), related_strategic_themes (≤3), research_questions (AT MOST 3, ≤20 words each — these are the supporting analyses that live UNDER the stream), and expected_outputs (≤2). Every research question must sit under a stream.
5. executive_unknowns: at most 6 unknowns most likely to CHANGE the recommendation (not ordinary research gaps).
6. board_decisions_required: at most 6 concrete executive approvals needed before implementation (e.g. "Approve capital allocation").
7. success_definition: at most 6 measurable decision outcomes.
8. in_scope / out_of_scope_items: what is explicitly in and out of scope (only list exclusions that are clearly implied; do not invent).

Keep every field tight. Return structured JSON only — no prose outside the JSON fields.
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

    # Summarise evidence (max 12 items for prompt size — J8.8b)
    ev_lines = ""
    for e in evidence_items[:12]:
        eid = e.get("evidence_id", "?")
        claim = e.get("claim", e.get("text", ""))[:100]
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

    return f"""You are a strategic analysis agent. Generate 3-4 competing hypotheses that explain the evidence gathered for this decision.

Decision Objective:
{objective}

Decision Areas: {areas}

Critical Uncertainties:
{uncertainties if uncertainties else "  (none listed)"}

Research Questions (priority order):
{rq_list if rq_list else "  (none)"}

Evidence Collected (up to 12 items):
{ev_lines if ev_lines else "  (no evidence available)"}

Profile Coverage:{coverage_lines if coverage_lines else " (none)"}

Contradictions Detected:{contra_lines if contra_lines else " (none)"}

Instructions:
1. Generate 3-4 COMPETING hypotheses. Each should represent a meaningfully different strategic interpretation of the evidence.

2. For each hypothesis, include:
   - id: "H1", "H2", etc.
   - title: one-line hypothesis
   - summary: 1-2 sentences explaining the hypothesis
   - type: one of constraint_dominant, technology_option, portfolio_strategy, market_timing, risk_concentration, or similar
   - supporting_evidence: list of evidence IDs (from the list above) that support it
   - contradicting_evidence: list of evidence IDs that weaken or contradict it
   - evidence_gaps: 2-3 evidence types that are absent but needed to test it
   - confidence: "high", "medium", or "low"
   - confidence_rationale: 1 sentence explaining the confidence level
   - decision_implications: 2-3 concrete strategic implications
   - disconfirming_evidence_needed: 2 specific evidence items that would invalidate this hypothesis

3. Hypotheses should be mutually distinguishable — avoid restating the same claim.

4. Write a 1 sentence synthesis_note summarising the hypothesis landscape.

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
        summary = h.get("summary", "")[:150]
        sup = ", ".join(h.get("supporting_evidence", [])[:4]) or "none"
        con = ", ".join(h.get("contradicting_evidence", [])[:3]) or "none"
        gaps = "; ".join(h.get("evidence_gaps", [])[:2]) or "none stated"
        conf = h.get("confidence", "medium")
        hyp_lines += (
            f"\n{hid}: {title}\n"
            f"  Summary: {summary}\n"
            f"  Supporting evidence: {sup}  Contradicting: {con}\n"
            f"  Evidence gaps: {gaps}  Confidence: {conf}\n"
        )

    ev_lines = ""
    for e in evidence_items[:10]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:100]
        src = e.get("source_document", "")
        ev_lines += f"  {eid}: {claim} (source: {src})\n"

    contra_lines = ""
    for c in contradictions[:5]:
        cid = c.get("contradiction_id", "?")
        topic = c.get("topic", "")
        sev = c.get("severity", "")
        contra_lines += f"  {cid} [{sev}]: {topic}\n"

    gap_lines = ""
    for g in research_gaps[:3]:
        gap_lines += f"  - {g.get('gap', g) if isinstance(g, dict) else g}\n"

    cov_lines = "\n".join(
        f"  {pname}: {level}" for pname, level in profile_coverage.items()
    ) or "  (not available)"

    return f"""\
You are a rigorous intellectual adversary tasked with stress-testing strategic hypotheses.

## Hypotheses to Challenge
{hyp_lines}

## Evidence Available (up to 10 items)
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

1. challenge_summary — 1-2 sentences on the main weakness
2. hidden_assumptions — 2-3 implicit assumptions the hypothesis takes for granted without evidence
3. weak_evidence — 2 specific evidence quality problems (e.g. vendor projections, single-source claims)
4. contradicting_evidence — list evidence IDs (Exxx) from the list above that weaken this hypothesis
5. missing_evidence — 2 types of evidence absent but needed to validate the hypothesis
6. falsification_tests — 2 specific, observable conditions that would definitively invalidate the hypothesis
7. robustness — "high" (withstands most challenges), "medium" (withstands some), or "low" (major gaps)

Then produce a SurvivingHypothesis for each:
- survival_status: "strong" (core logic intact), "moderate" (survives with caveats), "weak" (significant doubts)
- reason: 1 sentence explaining the survival status

Finally, write a 1 sentence challenge_synthesis summarising which hypotheses survived best and why.

Be adversarial. Your job is to find problems, not validate. Do not restate the hypothesis — critique it.

Return structured JSON only.
"""


# J10.8 — bounded caps for the Strategic Synthesis section of the recommendation
# prompt (mirrors J9.1b discipline). Kept in sync with recommendation_agent's
# diagnostics counters.
_SYNTH_SUMMARY_MAX_CHARS = 600
_SYNTH_LIST_CAP = 5


def _strategic_synthesis_section(strategic_synthesis: dict | None) -> str:
    """Render a BOUNDED Strategic Synthesis block for the recommendation prompt (J10.8)."""
    if not strategic_synthesis:
        return ""

    def _lst(key: str) -> str:
        items = [str(x).strip() for x in (strategic_synthesis.get(key) or []) if str(x).strip()]
        items = items[:_SYNTH_LIST_CAP]
        return "\n".join(f"  - {x}" for x in items) if items else "  (none)"

    summary = (strategic_synthesis.get("executive_summary") or "").strip()[:_SYNTH_SUMMARY_MAX_CHARS]
    return f"""
## Strategic Synthesis (executive cross-domain perspective — shape reasoning & prioritisation, not evidence citations)
Executive summary: {summary or "(none)"}
Cross-domain findings:
{_lst("cross_domain_findings")}
Cross-domain dependencies:
{_lst("cross_domain_dependencies")}
Cross-domain conflicts:
{_lst("cross_domain_conflicts")}
Strategic levers:
{_lst("strategic_levers")}
Dominant constraints:
{_lst("dominant_constraints")}
Emerging themes:
{_lst("emerging_themes")}
"""


def _recommendation_prompt(
    hypotheses: list[dict],
    surviving_hypotheses: list[dict],
    hypothesis_challenges: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
    research_strategy: dict,
    validated_contradictions: list[dict] | None = None,
    strategic_synthesis: dict | None = None,
) -> str:
    """Build the RecommendationAgent prompt (J6.5; J10.8 adds synthesis context)."""
    objective = decision_model.get("objective", "")
    decision_areas = decision_model.get("decision_areas", [])

    # Survival lookup
    survival_by_id = {s.get("hypothesis_id", ""): s for s in surviving_hypotheses}
    challenge_by_id = {c.get("hypothesis_id", ""): c for c in hypothesis_challenges}

    hyp_lines = ""
    for h in hypotheses:
        hid = h.get("id", "?")
        title = h.get("title", "")
        summary = h.get("summary", "")[:150]
        sup = ", ".join(h.get("supporting_evidence", [])[:3]) or "none"
        sv = survival_by_id.get(hid, {})
        status = sv.get("survival_status", "unknown")
        reason = sv.get("reason", "")[:100]
        ch = challenge_by_id.get(hid, {})
        robustness = ch.get("robustness", "unknown")
        key_challenge = ch.get("challenge_summary", "")[:100]
        hyp_lines += (
            f"\n{hid}: {title}  [survival={status}, robustness={robustness}]\n"
            f"  Summary: {summary}\n"
            f"  Evidence: {sup}  Challenge: {key_challenge}\n"
        )

    ev_lines = ""
    for e in evidence_items[:10]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:100]
        src = e.get("source_document", "")
        ev_lines += f"  {eid}: {claim} (source: {src})\n"

    areas_text = "\n".join(f"  - {a}" for a in decision_areas[:6]) or "  (not specified)"

    # J6.5a – validated contradictions block
    contra_lines = ""
    for c in (validated_contradictions or [])[:5]:
        cid = c.get("contradiction_id", "?")
        topic = c.get("topic", "")
        sev = c.get("severity", "")
        contra_lines += f"  {cid} [{sev}] {topic}\n"

    return f"""\
You are a strategic advisor translating challenged hypotheses into actionable recommendations.

## Decision Context
Objective: {objective}
Decision Areas:
{areas_text}
{_strategic_synthesis_section(strategic_synthesis)}
## Hypotheses with Challenge Results
{hyp_lines}

## Evidence Available (up to 10 items)
{ev_lines or "  (none)"}

## Validated Contradictions
{contra_lines or "  (none detected)"}

---

## Your Task

Generate 3-4 strategic recommendations. Each recommendation must:

1. Be DERIVED from one or more surviving hypotheses — do not invent new strategic logic
2. Be grounded in specific evidence IDs from the evidence list above
3. Include 1-2 key risks that could undermine it
4. Include 1-2 trigger conditions — future events that change or activate the recommendation
5. Be classified by time horizon: "near_term" (2026-2030), "medium_term" (2030-2035), or "long_term" (2035+)
6. Have a confidence level ("high"/"medium"/"low") reflecting hypothesis robustness and challenge findings
7. Include a 1-2 sentence summary (not a restatement of the hypothesis — translate to specific action)

Rules:
- Recommendations from "weak" survival hypotheses should have LOW priority and LOW confidence
- Recommendations from "strong" or "moderate" survival hypotheses may have MEDIUM or HIGH priority
- Include a recommendation_portfolio grouping IDs by time horizon
- Write a 1 sentence synthesis_note summarising the recommendation set

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
    for e in evidence_items[:12]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:100]
        src = e.get("source_document", e.get("source", ""))[:35]
        ev_lines += f"\n  {eid}: {claim}  [source: {src}]"

    return f"""You are a senior strategy consultant producing a Strategic Assumption analysis.

STRATEGIC QUESTION: {question}

DECISION AREAS: {', '.join(str(a) for a in decision_areas[:8])}

SURVIVING HYPOTHESES:
{hyp_lines or "  (none provided)"}

EVIDENCE AVAILABLE:
{ev_lines or "  (none provided)"}

TASK:
Identify the 3–5 HIGHEST-LEVERAGE strategic assumptions that must hold for the research
findings and emerging recommendations to remain valid.

The objective is NOT completeness. The objective is strategic leverage.

SELECTION CRITERIA — include an assumption only if it passes at least one of:
- Would the board ask whether this assumption is valid before approving the strategy?
- If this assumption proved false, would the strategic recommendation materially change?
- Does this assumption affect capital allocation, strategic direction, timing, or risk profile?

PRIORITISATION — keep only the top 3–5:
- Prefer assumptions that change WHAT is decided, not HOW it is implemented
- Prefer independent, genuinely uncertain assumptions
- Avoid implementation details, operational assumptions, and administrative prerequisites

For each retained assumption:
1. State precisely what must be true
2. Assign the most appropriate category
3. Rate importance: Critical (recommendation fails if false) | Important | Supporting
4. Rate evidence support: how well do the evidence items back this assumption?
5. Rate confidence: your confidence that this assumption currently holds
6. Write a 1-2 sentence rationale: why is this assumption load-bearing for the strategy?
7. List evidence_ids from the available evidence that support this assumption (use exact IDs)
8. Identify any conflicts with other assumptions in your list

CONFLICT DETECTION:
If two assumptions are mutually contradictory, flag them in conflict_pairs and set conflicts_with.

QUALITY RULES:
- Produce the smallest set that captures all materially different strategic risks
- Avoid duplicates — each assumption must be independently falsifiable
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
    for e in evidence_items[:8]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:80]
        src = e.get("source_document", e.get("source", ""))[:35]
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
Identify 4–7 strategic risks — conditions or events that could cause one or more assumptions
above to fail, thereby threatening the validity of one or more recommendations.

For each risk:
1. State precisely what could go wrong (the risk event or condition)
2. Assign the most appropriate category
3. Rate severity: High (recommendation collapses), Medium, Low
4. Rate likelihood: High, Medium, Low
5. Rate evidence support: how strongly do the available evidence items signal this risk?
6. Rate confidence: your confidence in this risk assessment
7. Write a 1-2 sentence rationale: why does this risk matter strategically?
8. List related_assumption_ids from the assumptions above (use exact IDs)
9. List affected_recommendation_ids derived from the assumption→recommendation links
10. List evidence_ids from the available evidence that inform this risk
11. Optionally provide brief mitigation_notes (1 sentence max)

CRITICAL RULES:
- Each risk must threaten at least one named assumption — no floating risks
- Do not duplicate risks — each must be distinct
- Avoid trivial risks — focus on strategic conditions
- Use exact assumption IDs (e.g. "A-001") and recommendation IDs (e.g. "REC-001")

Return structured JSON matching the risk_generation schema.
"""


def _opportunity_prompt(
    assumptions: list[dict],
    recommendations: list[dict],
    risks: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
) -> str:
    """Build the OpportunityAgent prompt (J7.4)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))

    a_lines = ""
    for a in assumptions:
        a_id = a.get("assumption_id", "?")
        stmt = a.get("statement", "")[:120]
        imp = a.get("importance", "")
        rec_ids = ", ".join(a.get("supported_recommendation_ids", [])) or "none"
        a_lines += f"\n  {a_id} [{imp}]: {stmt}\n    → enables recommendations: {rec_ids}\n"

    r_lines = ""
    for r in recommendations:
        r_id = r.get("recommendation_id", r.get("id", "?"))
        title = r.get("title", r.get("recommendation", ""))[:100]
        r_lines += f"\n  {r_id}: {title}\n"

    risk_lines = ""
    for rk in risks[:8]:
        rk_id = rk.get("risk_id", "?")
        stmt = rk.get("statement", "")[:100]
        a_ids = ", ".join(rk.get("related_assumption_ids", [])) or "none"
        risk_lines += f"\n  {rk_id}: {stmt}\n    → threatens assumptions: {a_ids}\n"

    ev_lines = ""
    for e in evidence_items[:8]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:80]
        src = e.get("source_document", e.get("source", ""))[:35]
        ev_lines += f"\n  {eid}: {claim}  [source: {src}]"

    return f"""You are a senior strategy consultant producing a Strategic Opportunity analysis.

STRATEGIC QUESTION: {question}

STRATEGIC ASSUMPTIONS (what must be true for recommendations to hold):
{a_lines or "  (none provided)"}

RECOMMENDATIONS (what the research recommends):
{r_lines or "  (none provided)"}

STRATEGIC RISKS (downside scenarios already identified):
{risk_lines or "  (none provided)"}

EVIDENCE AVAILABLE:
{ev_lines or "  (none provided)"}

TASK:
Identify 4–7 strategic opportunities — upside scenarios that become available when one
or more assumptions prove MORE FAVOURABLE than expected, not merely satisfied.

This is not about positive risks. It is about what additional value, advantage, or
acceleration becomes possible when an assumption is exceeded.

For each opportunity:
1. State precisely what additional value becomes possible (the upside scenario)
2. Assign the most appropriate category
3. Rate impact: High (transformative advantage), Medium, Low
4. Rate likelihood: High, Medium, Low
5. Rate evidence support: how strongly do evidence items signal this upside is plausible?
6. Rate confidence: your confidence in this opportunity assessment
7. Write a 1-2 sentence rationale: why does this opportunity matter strategically?
8. List related_assumption_ids from the assumptions above (use exact IDs)
9. List enabled_recommendation_ids derived from the assumption→recommendation links
10. List evidence_ids from the available evidence that support this opportunity
11. Optionally provide brief exploitation_notes (1 sentence max)

OPPORTUNITY QUALITY CRITERIA:
- Each opportunity must be grounded in a named assumption — no floating opportunities
- Be specific about the upside mechanism — avoid generic "things could go better"
- Opportunities should be distinct from each other and from the risks already identified
- Use exact assumption IDs (e.g. "A-001") and recommendation IDs (e.g. "REC-001")

Return structured JSON matching the opportunity_generation schema.
"""


def _strategic_options_prompt(
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    evidence_items: list[dict],
    decision_model: dict,
) -> str:
    """Build the StrategicOptionAgent prompt (J7.5)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))

    a_lines = ""
    for a in assumptions:
        a_id = a.get("assumption_id", "?")
        stmt = a.get("statement", "")[:100]
        imp = a.get("importance", "")
        a_lines += f"\n  {a_id} [{imp}]: {stmt}"

    risk_lines = ""
    for r in risks:
        r_id = r.get("risk_id", "?")
        stmt = r.get("statement", "")[:100]
        sev = r.get("severity", "")
        a_ids = ", ".join(r.get("related_assumption_ids", [])) or "none"
        risk_lines += f"\n  {r_id} [{sev}]: {stmt}  (threatens: {a_ids})"

    opp_lines = ""
    for o in opportunities:
        o_id = o.get("opportunity_id", "?")
        stmt = o.get("statement", "")[:100]
        imp = o.get("impact", "")
        a_ids = ", ".join(o.get("related_assumption_ids", [])) or "none"
        opp_lines += f"\n  {o_id} [{imp}]: {stmt}  (via: {a_ids})"

    rec_lines = ""
    for r in recommendations:
        r_id = r.get("recommendation_id", r.get("id", "?"))
        title = r.get("title", r.get("recommendation", ""))[:100]
        a_ids = ", ".join(r.get("supported_assumption_ids", [])) or "none"
        rec_lines += f"\n  {r_id}: {title}  (assumptions: {a_ids})"

    ev_lines = ""
    for e in evidence_items[:6]:
        eid = e.get("evidence_id", "")
        claim = e.get("claim", "")[:80]
        ev_lines += f"\n  {eid}: {claim}"

    return f"""You are a senior strategy consultant producing a Strategic Options analysis.

STRATEGIC QUESTION: {question}

STRATEGIC ASSUMPTIONS (what must be true):
{a_lines or "  (none provided)"}

STRATEGIC RISKS (downside scenarios):
{risk_lines or "  (none provided)"}

STRATEGIC OPPORTUNITIES (upside scenarios):
{opp_lines or "  (none provided)"}

RECOMMENDATIONS (research-derived actions):
{rec_lines or "  (none provided)"}

EVIDENCE (selected):
{ev_lines or "  (none provided)"}

TASK:
Generate 2-4 genuinely different strategic options. Each option is a coherent, internally
consistent course of action — a strategic posture, not a list of tasks.

Options must be meaningfully different in: risk appetite, capital commitment, time horizon,
which opportunities they pursue, and which risks they accept vs mitigate.

For each option:
1. Give it a short descriptive title (5-10 words)
2. Describe what it entails (1-3 sentences)
3. State the strategic objective it pursues (1 sentence)
4. List 2-3 expected outcomes
5. List supporting_assumption_ids it depends on (exact IDs e.g. "A-001")
6. List associated_risk_ids it must manage (exact IDs e.g. "RSK-001")
7. List associated_opportunity_ids it captures (exact IDs e.g. "OPP-001")
8. List supporting_recommendation_ids it implements (exact IDs e.g. "REC-001")
9. List 2-3 advantages
10. List 2-3 disadvantages
11. Rate implementation_complexity: Low | Medium | High
12. Rate estimated_time_horizon: Near-term | Medium-term | Long-term
13. Rate capital_intensity: Low | Medium | High
14. Rate confidence: High | Medium | Low
15. Set recommended: true for EXACTLY ONE option
16. Write a rationale (2-3 sentences) comparing this option against the others

SELECTION RULES:
- Exactly one option must have recommended=True
- The recommended option rationale must explicitly compare it against the alternatives
- Non-recommended options must still be complete and analytically sound
- Use exact IDs when referencing assumptions, risks, opportunities, and recommendations

Return structured JSON matching the strategic_option_generation schema.
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

    return f"""You are a research strategy agent. Given a Decision Model and available domain profiles, produce a CONCISE, executable research strategy.

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

Instructions — keep the output tight; this is a routing plan, not a report:
1. Rank each profile by relevance (1 = most relevant). Include all available profiles.

2. Order the research questions by decision impact. Return {{question, priority}} objects. For "question", write a SHORT label of ≤12 words — do NOT restate the full question text and do NOT repeat the objective or engagement brief. Maximum 6 questions.

3. required_evidence: at most 6 concrete evidence items, each ≤12 words (e.g. "AI power demand forecasts 2024-2030").

4. source_priorities: at most 5 source types, each ≤6 words.

5. coverage_targets: one entry per decision area / critical uncertainty (at most 8 total), value is exactly "strong", "moderate", or "light".

6. strategy_rationale: at most 2 sentences.

Hard limits: no field may exceed the counts above. Return structured JSON only — no prose, preamble, or explanation outside the JSON fields.
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
    return f"""Synthesize a research memo from the source-grounded evidence below.

Question:
{question}

Rules:
- Use only the provided evidence items.
- Every entry in confirmed_facts, power_implications, cooling_implications, networking_implications, and rack_architecture_implications must end with exactly one citation: [Source: filename.pdf, Evidence: E001].
- Use only source_document and evidence_id values present in the provided evidence.
- Do not invent source names or evidence IDs.
- Keep each entry concise (1 sentence).
- Return JSON only.

Output limits: confirmed_facts ≤6, inferences ≤3, each implication section ≤4, open_questions ≤2.

JSON shape:
{{
  "executive_summary": "...",
  "confirmed_facts": ["Claim. [Source: filename.pdf, Evidence: E001]"],
  "inferences": ["..."],
  "power_implications": ["Claim. [Source: filename.pdf, Evidence: E001]"],
  "cooling_implications": ["Claim. [Source: filename.pdf, Evidence: E001]"],
  "networking_implications": ["Claim. [Source: filename.pdf, Evidence: E001]"],
  "rack_architecture_implications": ["Claim. [Source: filename.pdf, Evidence: E001]"],
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

Rules:
- Use only the source text below.
- Each evidence_snippet must be copied or tightly paraphrased from one chunk.
- Use categories only from: architecture, power, cooling, networking, rack architecture, operations, other.
- Do not invent evidence IDs; evidence_id is assigned by the harness after extraction.
- Set source_chunk_id to the Chunk ID shown in the header for the chunk you drew evidence from.
- Extract 3-6 items per chunk — prioritise facts that directly address the question.
- Include numeric claims, specifications, and policy statements relevant to the question.
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


def _decision_analysis_prompt(
    strategic_options: list[dict],
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    decision_model: dict,
) -> str:
    """Build the DecisionAnalysisAgent prompt (J7.6)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))

    def _fmt(items: list[dict], id_key: str, label_key: str = "statement") -> str:
        return "\n".join(
            f"  [{item.get(id_key, '?')}] {item.get(label_key, item.get('title', ''))}"
            for item in items
        ) or "  (none)"

    opts_detail = ""
    for opt in strategic_options:
        oid = opt.get("option_id", "?")
        rec = " [RECOMMENDED]" if opt.get("recommended") else ""
        opts_detail += (
            f"\n  {oid}{rec}: {opt.get('title', '')}\n"
            f"    Description: {opt.get('description', '')[:200]}\n"
            f"    Assumptions: {opt.get('supporting_assumption_ids', [])}  "
            f"Risks: {opt.get('associated_risk_ids', [])}  "
            f"Opps: {opt.get('associated_opportunity_ids', [])}\n"
            f"    Complexity: {opt.get('implementation_complexity', '')}  "
            f"Capital: {opt.get('capital_intensity', '')}  "
            f"Horizon: {opt.get('estimated_time_horizon', '')}\n"
            f"    Rationale: {opt.get('rationale', '')[:200]}\n"
        )

    return f"""You are a strategic decision analyst. Produce an explicit, rigorous comparison of the Strategic Options below, explaining WHY one option is preferred.

DECISION QUESTION
{question}

STRATEGIC OPTIONS
{opts_detail}
ASSUMPTIONS
{_fmt(assumptions, 'assumption_id')}

RISKS
{_fmt(risks, 'risk_id')}

OPPORTUNITIES
{_fmt(opportunities, 'opportunity_id')}

RECOMMENDATIONS
{_fmt(recommendations, 'recommendation_id', 'title')}

TASK
Produce a DecisionAnalysis object that:

1. RATES each option across all comparison dimensions in the decision matrix.
   Dimensions: Strategic Fit, Implementation Risk, Execution Complexity, Capital Requirement,
   Expected Return, Time to Value, Dependency Strength, Assumption Strength, Risk Exposure,
   Opportunity Capture. Use: Very High | High | Medium | Low | Very Low.

2. RANKS all options from most to least preferred.

3. IDENTIFIES 3-4 explicit tradeoffs of the form "Higher X → Lower Y".
   Derive ONLY from the existing graph. Do NOT invent new tradeoffs.

4. EXPLAINS sensitivity: which specific assumption_ids, if they fail, would change the preferred option.
   Reference assumption_ids by name. Do NOT generate new scenarios.

5. JUSTIFIES the preferred option against each alternative in 2-3 sentences total.

CONSTRAINTS
- Do NOT generate new options, evidence, or scenarios.
- Everything must derive from the existing graph above.
- Exactly one recommended_option_id (must match an existing option_id).
- Be specific: name assumption IDs, risk IDs, opportunity IDs where relevant.

Return structured JSON matching the decision_analysis_generation schema.
"""


def _strategic_synthesis_prompt(
    domain_plans: list[dict],
    domain_evidence: list[dict],
    domain_hypotheses: list[dict],
    decision_architecture: dict,
) -> str:
    """Build the cross-domain Strategic Synthesis prompt (J10.7)."""

    def _domain_block(i: int) -> str:
        hyps = (domain_hypotheses[i] if i < len(domain_hypotheses) else {}) or {}
        title = hyps.get("decision_domain_title") or f"Domain {i + 1}"
        h_titles = [h.get("title", "") for h in (hyps.get("hypotheses") or [])][:3]
        return f"- {title}: " + ("; ".join(t for t in h_titles if t) or "(no hypotheses)")

    n = max(len(domain_hypotheses), len(domain_plans))
    domains = "\n".join(_domain_block(i) for i in range(n)) or "(no domains)"
    themes = ", ".join(decision_architecture.get("strategic_themes", [])) or "(none)"
    statement = decision_architecture.get("decision_statement", "")

    return f"""You are a senior strategy consultant integrating independent Decision Domain analyses into ONE executive perspective. This is executive reasoning — NOT recommendations and NOT implementation plans.

Decision: {statement}
Strategic themes: {themes}

Decision Domains (title: top hypotheses):
{domains}

Make the IMPLICIT relationships across domains EXPLICIT. Produce:
1. executive_summary: the integrated cross-domain perspective, <=4 sentences.
2. cross_domain_findings: <=8 findings that span multiple domains.
3. cross_domain_dependencies: <=8, each of the form "A requires/depends-on B" across domains.
4. cross_domain_conflicts: <=6 tensions between domains.
5. strategic_levers: <=6 leverage points that move multiple domains at once.
6. dominant_constraints: <=6 constraints that bind the whole decision.
7. emerging_themes: <=8 themes emerging across domains.

Do NOT produce recommendations or actions. Return structured JSON only.
"""


def _executive_confidence_prompt(
    decision_analysis: dict,
    strategic_options: list[dict],
    assumptions: list[dict],
    risks: list[dict],
    opportunities: list[dict],
    recommendations: list[dict],
    scenarios: list[dict],
    decision_model: dict,
) -> str:
    """Build the ExecutiveConfidenceAgent prompt (J7.7)."""
    question = decision_model.get("strategic_question", decision_model.get("objective", ""))
    da_conf = decision_analysis.get("confidence", "Medium")
    da_summary = decision_analysis.get("executive_summary", "")
    da_rationale = decision_analysis.get("rationale", "")
    rec_id = decision_analysis.get("recommended_option_id", "")

    def _fmt(items: list[dict], id_key: str, label_key: str = "statement") -> str:
        return "\n".join(
            f"  [{item.get(id_key, '?')}] importance={item.get('importance', item.get('severity', ''))} "
            f"evidence={item.get('evidence_support', '')} conf={item.get('confidence', '')} "
            f"— {item.get(label_key, item.get('title', ''))[:100]}"
            for item in items
        ) or "  (none)"

    sens = decision_analysis.get("sensitivity_analysis", "")
    tradeoffs = "\n".join(f"  - {t}" for t in decision_analysis.get("key_tradeoffs", [])) or "  (none)"
    uncertainties = "\n".join(f"  - {u}" for u in decision_analysis.get("key_uncertainties", [])) or "  (none)"

    scenario_summary = ""
    if scenarios:
        scenario_summary = "\n".join(
            f"  [{s.get('scenario_id', '?')}] {s.get('name', '')} — outcome: {s.get('strategic_outcome', '')[:60]}"
            for s in scenarios[:4]
        )
    else:
        scenario_summary = "  (no scenarios available)"

    return f"""You are an executive decision advisor. Synthesise the completed J7 decision graph into a single executive confidence assessment: "Should an executive approve this recommendation today?"

DECISION QUESTION
{question}

DECISION ANALYSIS SUMMARY
Recommended option: {rec_id}
Analysis confidence: {da_conf}
Summary: {da_summary[:250]}
Rationale: {da_rationale[:250]}

KEY TRADEOFFS
{tradeoffs}

KEY UNCERTAINTIES
{uncertainties}

SENSITIVITY ANALYSIS
{sens[:300] if isinstance(sens, str) else str(sens)[:300]}

STRATEGIC ASSUMPTIONS
{_fmt(assumptions, 'assumption_id')}

STRATEGIC RISKS
{_fmt(risks, 'risk_id')}

STRATEGIC OPPORTUNITIES
{_fmt(opportunities, 'opportunity_id')}

SCENARIO ANALYSIS
{scenario_summary}

TASK
Produce an ExecutiveConfidence object that:

1. RATES overall_confidence (High/Medium/Low) based on: evidence strength, Critical assumption count,
   High-severity risk concentration, and option sensitivity to assumption failures.

2. DETERMINES decision_readiness:
   - "Ready for Decision" — strong evidence, assumptions validated, risks mitigated
   - "Needs Additional Validation" — material unknowns remain but not blocking
   - "Not Ready" — critical evidence gaps or unmitigated blocking risks

3. ISSUES a board_recommendation: "Proceed" | "Proceed with Conditions" | "Delay Pending Evidence" | "Reject"

4. IDENTIFIES 3-4 critical_unknowns — specific items that must resolve before deciding.

5. PRODUCES validation_priorities — 3-5 concrete due-diligence actions. Reference assumption_ids
   and risk_ids by name. Be specific (e.g. "Validate A-001 cost assumption via independent estimate").

6. PROVIDES conditional analysis:
   - confidence_if_assumptions_hold: confidence if Critical assumptions are confirmed
   - confidence_if_assumptions_fail: confidence if Critical assumptions fail

CONSTRAINTS
- Do NOT generate new strategic reasoning, options, or evidence.
- Everything must derive from the decision graph provided above.
- Reference assumption IDs and risk IDs specifically.

Return structured JSON matching the executive_confidence_generation schema.
"""
