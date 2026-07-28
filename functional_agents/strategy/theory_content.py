"""PH12.2/PH12.2b — TheoryContent: models for theory-specific content, coverage, confidence.

TheoryContent carries a theory's assigned assumptions, risks, opportunities,
evidence, success conditions, and recommendations — all with canonical source IDs
and assignment lineage.

PH12.2b adds discrimination fields: relationship classification, discrimination scores,
distinctive/shared splits per category, and multi-state homogenization tracking.

Backward-compatible defaults throughout (empty lists/dicts, None for optional fields).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Lineage entries
# ---------------------------------------------------------------------------

class ContentLineageEntry(BaseModel):
    """One item in a theory's content lineage."""

    source_id: str = ""
    assignment_type: str = ""   # option_link | recommendation_link | risk_link |
    #                             opportunity_link | posture_match | semantic_inference |
    #                             sensitivity | symmetric_fallback
    via_ids: list[str] = Field(default_factory=list)  # intermediate objects in the chain
    relevance_score: float = 0.0
    rationale: str = ""

    # PH12.2b — discrimination fields (set by discrimination_calculator)
    relationship_classification: str = ""  # explicit_discriminating | explicit_shared |
    #                                         posture_discriminating | semantic_discriminating |
    #                                         sensitivity | fallback
    discrimination_score: float = 0.0     # 1.0 - (shared_count / total_theories)
    relationship_scope: str = ""           # theory_unique | theory_subset | global_shared
    shared_across_theory_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class EvidenceLineageEntry(BaseModel):
    """One evidence item with potentially multiple lineage paths."""

    evidence_id: str = ""
    assignment_type: str = ""
    relevance_score: float = 0.0
    rationale: str = ""
    lineage_paths: list[dict[str, str]] = Field(default_factory=list)
    # Each path: {"source_type": "assumption", "source_id": "A-001"}

    # PH12.2b — discrimination fields (set by discrimination_calculator)
    relationship_classification: str = ""
    discrimination_score: float = 0.0
    relationship_scope: str = ""
    shared_across_theory_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "allow"}


class SuccessConditionEntry(BaseModel):
    """One theory-specific success condition."""

    text: str = ""
    source_type: str = ""      # option | opportunity | recommendation | choice | engagement
    source_ids: list[str] = Field(default_factory=list)
    assignment_type: str = ""
    relevance_score: float = 0.0

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Content coverage and confidence
# ---------------------------------------------------------------------------

class ContentCoverage(BaseModel):
    """Theory-specific content coverage across object categories."""

    assumptions: float = 0.0      # fraction of total that are explicitly linked
    risks: float = 0.0
    opportunities: float = 0.0
    recommendations: float = 0.0
    evidence: float = 0.0
    success_conditions: float = 0.0
    overall: float = 0.0
    status: str = "insufficient"  # sufficient | partial | fallback_heavy | insufficient
    fallback_count: int = 0
    explicit_count: int = 0

    # PH12.2b — discrimination-aware coverage fractions (set post-hoc by discrimination_calculator)
    canonical: float = 0.0            # fraction of total assigned items that are canonical (explicit/posture)
    distinctive: float = 0.0          # fraction of assigned that are theory-distinctive (discrimination_score > 0)
    shared_context: float = 0.0       # fraction of assigned that are globally shared (discrimination_score == 0)
    evidence_distinctive: float = 0.0 # fraction of evidence that is theory-distinctive

    model_config = {"extra": "allow"}

    @field_validator("status")
    @classmethod
    def _valid_status(cls, v: str) -> str:
        valid = {"sufficient", "partial", "fallback_heavy", "insufficient"}
        if v not in valid:
            raise ValueError(f"ContentCoverage.status must be one of {sorted(valid)}, got {v!r}")
        return v

    @classmethod
    def compute(
        cls,
        total_assumptions: int,
        total_risks: int,
        total_opportunities: int,
        total_recommendations: int,
        total_evidence: int,
        assigned_assumptions: int,
        assigned_risks: int,
        assigned_opportunities: int,
        assigned_recommendations: int,
        assigned_evidence: int,
        assigned_success_conditions: int,
        explicit_count: int,
        fallback_count: int,
    ) -> "ContentCoverage":
        def _frac(a: int, t: int) -> float:
            return round(min(float(a) / float(t), 1.0), 4) if t > 0 else (1.0 if a == 0 else 0.0)

        a_cov = _frac(assigned_assumptions, max(total_assumptions, 1))
        r_cov = _frac(assigned_risks, max(total_risks, 1))
        o_cov = _frac(assigned_opportunities, max(total_opportunities, 1))
        rec_cov = _frac(assigned_recommendations, max(total_recommendations, 1))
        ev_cov = _frac(assigned_evidence, max(total_evidence, 1))
        sc_cov = 1.0 if assigned_success_conditions > 0 else 0.0

        overall = round((a_cov + r_cov + o_cov + rec_cov + ev_cov + sc_cov) / 6.0, 4)
        total_assigned = explicit_count + fallback_count
        fallback_share = fallback_count / total_assigned if total_assigned > 0 else 0.0

        if overall >= 0.70 and fallback_share <= 0.30:
            status = "sufficient"
        elif overall >= 0.40 and fallback_share <= 0.60:
            status = "partial"
        elif fallback_share > 0.60:
            status = "fallback_heavy"
        else:
            status = "insufficient"

        return cls(
            assumptions=a_cov,
            risks=r_cov,
            opportunities=o_cov,
            recommendations=rec_cov,
            evidence=ev_cov,
            success_conditions=sc_cov,
            overall=overall,
            status=status,
            fallback_count=fallback_count,
            explicit_count=explicit_count,
        )


class ContentConfidence(BaseModel):
    """Content confidence separate from theory evaluation confidence."""

    level: str = "Low"        # High | Medium | Low
    explicit_share: float = 0.0     # fraction of items from explicit relationships
    mapping_confidence: str = ""    # inherits from option mapping
    evidence_coverage: float = 0.0
    fallback_share: float = 0.0
    posture_match_share: float = 0.0
    contradiction_share: float = 0.0
    rationale: str = ""

    # PH12.2b — discrimination-aware shares (set by discrimination_calculator)
    explicit_discriminating_share: float = 0.0  # explicit items that are theory-distinctive
    explicit_shared_share: float = 0.0           # explicit items that are globally shared
    posture_discriminating_share: float = 0.0   # posture items that are theory-distinctive
    distinctive_evidence_share: float = 0.0     # evidence items that are theory-distinctive

    model_config = {"extra": "allow"}

    @classmethod
    def compute(
        cls,
        explicit_count: int,
        fallback_count: int,
        posture_match_count: int,
        contradiction_count: int,
        mapping_confidence: str,
        evidence_coverage: float,
        # PH12.2b optional discrimination params
        explicit_discriminating_count: int = 0,
        explicit_shared_count: int = 0,
        posture_discriminating_count: int = 0,
        distinctive_evidence_count: int = 0,
        total_evidence_count: int = 0,
    ) -> "ContentConfidence":
        total = explicit_count + fallback_count + posture_match_count
        if total == 0:
            return cls(level="Low", rationale="No content assigned.")

        explicit_share = explicit_count / total
        fallback_share = fallback_count / total
        posture_share  = posture_match_count / total
        contra_share   = contradiction_count / total if total > 0 else 0.0

        # PH12.2b discrimination-aware shares
        explicit_disc_share = explicit_discriminating_count / total if total > 0 else 0.0
        explicit_sh_share   = explicit_shared_count / total if total > 0 else 0.0
        posture_disc_share  = posture_discriminating_count / total if total > 0 else 0.0
        dist_ev_share       = (distinctive_evidence_count / total_evidence_count
                               if total_evidence_count > 0 else 0.0)

        # Confidence matrix — PH12.2b: High requires discriminating explicit share >= 0.40
        # and mapping must not be None
        discriminating_share = explicit_disc_share + posture_disc_share
        has_discrimination_data = explicit_discriminating_count + explicit_shared_count > 0

        if (explicit_share >= 0.60 and mapping_confidence in ("High", "Medium")
                and evidence_coverage >= 0.50 and fallback_share <= 0.20
                and (not has_discrimination_data or discriminating_share >= 0.40)):
            level = "High"
        elif (explicit_share + posture_share >= 0.40
              and fallback_share <= 0.50
              and evidence_coverage >= 0.20):
            level = "Medium"
        else:
            level = "Low"

        mapping_str = f"mapping={mapping_confidence}" if mapping_confidence else ""
        rationale = (
            f"explicit={explicit_count}, posture={posture_match_count}, "
            f"fallback={fallback_count}; "
            f"ev_coverage={evidence_coverage:.2f}; {mapping_str}"
        ).strip("; ")

        return cls(
            level=level,
            explicit_share=round(explicit_share, 4),
            mapping_confidence=mapping_confidence,
            evidence_coverage=round(evidence_coverage, 4),
            fallback_share=round(fallback_share, 4),
            posture_match_share=round(posture_share, 4),
            contradiction_share=round(contra_share, 4),
            rationale=rationale,
            explicit_discriminating_share=round(explicit_disc_share, 4),
            explicit_shared_share=round(explicit_sh_share, 4),
            posture_discriminating_share=round(posture_disc_share, 4),
            distinctive_evidence_share=round(dist_ev_share, 4),
        )


# ---------------------------------------------------------------------------
# TheoryContent — main model
# ---------------------------------------------------------------------------

class TheoryContent(BaseModel):
    """Theory-specific content with canonical IDs and assignment lineage.

    Carries the resolved content for one TheoryOfWinning. All new fields have
    backward-compatible defaults (empty lists / None / default models).
    """

    theory_id: str = ""
    mapped_option_id: str | None = None
    mapping_confidence: str = ""

    # Assigned canonical IDs (sorted, deduplicated)
    recommendation_ids: list[str] = Field(default_factory=list)
    assumption_ids: list[str] = Field(default_factory=list)
    risk_ids: list[str] = Field(default_factory=list)
    opportunity_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)

    # PH12.2b — distinctive/shared splits (set by discrimination_calculator)
    distinctive_assumption_ids: list[str] = Field(default_factory=list)
    shared_assumption_ids: list[str] = Field(default_factory=list)
    distinctive_risk_ids: list[str] = Field(default_factory=list)
    shared_risk_ids: list[str] = Field(default_factory=list)
    distinctive_opportunity_ids: list[str] = Field(default_factory=list)
    shared_opportunity_ids: list[str] = Field(default_factory=list)
    distinctive_recommendation_ids: list[str] = Field(default_factory=list)
    shared_recommendation_ids: list[str] = Field(default_factory=list)
    distinctive_evidence_ids: list[str] = Field(default_factory=list)
    shared_evidence_ids: list[str] = Field(default_factory=list)

    # PH12.2b — homogenization state for this theory (set by content_differentiation)
    homogenization_state: str = "none"  # none | partial | substantial | full

    # Structured success conditions
    success_conditions: list[SuccessConditionEntry] = Field(default_factory=list)

    # Lineage by category
    content_lineage: dict[str, list[ContentLineageEntry]] = Field(
        default_factory=lambda: {
            "assumptions": [],
            "risks": [],
            "opportunities": [],
            "recommendations": [],
            "evidence": [],
            "success_conditions": [],
        }
    )

    # Evidence with multi-path lineage
    evidence_lineage: list[EvidenceLineageEntry] = Field(default_factory=list)

    # Coverage and confidence
    coverage: ContentCoverage = Field(default_factory=ContentCoverage)
    confidence: ContentConfidence = Field(default_factory=ContentConfidence)

    # Fallback metadata
    content_fallbacks: list[dict[str, Any]] = Field(default_factory=list)

    # Diagnostics
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"extra": "allow"}
