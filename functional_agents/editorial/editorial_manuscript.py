"""EditorialManuscript — canonical editorial artifact for the Editorial Platform (PH6.3).

The manuscript is the destination that future writer agents populate.
It sits between the EditorialBrief (structured inputs) and the deliverable
renderers (Markdown, DOCX, PPTX).

For PH6.3 all content fields (paragraphs, bullet_groups, tables, figures) are
empty — the scaffold is established here; writers populate it in PH6.4+.

Contains no markdown, no DOCX/PPTX formatting, and no rendering logic.

Serialisation: editorial_manuscript.to_dict() → JSON-safe dict.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass
class ManuscriptProvenance:
    """Full traceability chain for a manuscript section.

    Links the section back through EditorialBrief → ResearchObject → DecisionModel.
    """

    brief_id: str = ""                  # EditorialBrief.metadata.brief_id
    brief_section_key: str = ""         # e.g. "executive_summary"
    decision_model_id: str = ""
    research_object_id: str = ""
    analysis_id: str = ""
    confidence_id: str = ""
    risk_ids: list[str] = field(default_factory=list)
    opportunity_ids: list[str] = field(default_factory=list)
    recommendation_ids: list[str] = field(default_factory=list)
    assumption_ids: list[str] = field(default_factory=list)
    option_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ManuscriptSection — shared structure for all section types
# ---------------------------------------------------------------------------

@dataclass
class ManuscriptSection:
    """Shared editorial structure for every manuscript section.

    PH6.3: content fields are empty scaffolds — writers populate in PH6.4+.
    """

    title: str
    subtitle: str = ""
    paragraphs: list[str] = field(default_factory=list)
    bullet_groups: list[list[str]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    figures: list[dict[str, Any]] = field(default_factory=list)
    provenance: ManuscriptProvenance = field(default_factory=ManuscriptProvenance)
    source_section_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Concrete section types
# ---------------------------------------------------------------------------

@dataclass
class ExecutiveSummaryManuscriptSection(ManuscriptSection):
    """Manuscript section for the board-level executive opener."""


@dataclass
class DecisionAnalysisManuscriptSection(ManuscriptSection):
    """Manuscript section for the strategic option comparison narrative."""


@dataclass
class RecommendationManuscriptSection(ManuscriptSection):
    """Manuscript section for the action-oriented recommendations."""


@dataclass
class RiskManuscriptSection(ManuscriptSection):
    """Manuscript section for key risks and mitigations."""


@dataclass
class OpportunityManuscriptSection(ManuscriptSection):
    """Manuscript section for strategic opportunities."""


@dataclass
class ConfidenceManuscriptSection(ManuscriptSection):
    """Manuscript section for executive confidence and decision readiness."""


@dataclass
class AppendixManuscriptSection(ManuscriptSection):
    """Manuscript section for evidence provenance and citations."""


# ---------------------------------------------------------------------------
# Manuscript metadata
# ---------------------------------------------------------------------------

@dataclass
class ManuscriptMetadata:
    """Identity and provenance of this EditorialManuscript."""

    manuscript_id: str
    created_at: str
    brief_id: str              # EditorialBrief.metadata.brief_id
    pipeline_run_id: str
    decision_model_id: str
    research_object_id: str
    question: str
    profiles: list[str] = field(default_factory=list)
    execution_profile: str = ""


# ---------------------------------------------------------------------------
# EditorialManuscript — canonical editorial destination
# ---------------------------------------------------------------------------

@dataclass
class EditorialManuscript:
    """The canonical editorial artifact that writer agents will populate (PH6.3+).

    Built from an EditorialBrief by the EditorialCoordinator.
    Contains no prose yet — content fields are populated by writers in PH6.4+.
    Contains no markdown, DOCX, or PPTX formatting.
    Read-only from the perspective of renderers.
    """

    metadata: ManuscriptMetadata
    executive_summary: ExecutiveSummaryManuscriptSection
    decision_analysis: DecisionAnalysisManuscriptSection
    recommendations: RecommendationManuscriptSection
    strategic_risks: RiskManuscriptSection
    strategic_opportunities: OpportunityManuscriptSection
    executive_confidence: ConfidenceManuscriptSection
    appendix: AppendixManuscriptSection

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
