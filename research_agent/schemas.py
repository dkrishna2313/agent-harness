"""Pydantic schemas used across the research workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

SourceType = Literal[
    "nvidia_technical",
    "nvidia_marketing",
    "vendor_whitepaper",
    "vendor_brief",
    "press_release",
    "independent_technical",
    "blog",
    "synthetic",
    "unknown",
    # Profile-driven types (J1.3)
    "authoritative_primary",   # score 5: government/intergovernmental bodies (DOE, IAEA, NRC)
    "industry_vendor",         # score 4: industry vendors and associations (NuScale, WNA)
]

EvidenceCategory = Literal[
    # AI data-center categories
    "architecture",
    "power",
    "cooling",
    "networking",
    "rack architecture",
    "operations",
    "gpu",
    "facility",
    "resiliency",
    # SMR / nuclear categories
    "economics",
    "construction",
    "licensing",
    "reactor design",
    "grid integration",
    "fuel cycle",
    "deployment timeline",
    "safety",
    "waste management",
    "bwrx",
    "nuscale",
    # generic fallback
    "other",
]


class SourceQuality(BaseModel):
    """Source quality classification for a document."""

    source_document: str
    source_type: SourceType
    source_quality_score: int = Field(ge=1, le=5)
    rationale: str


class SourceDocument(BaseModel):
    """Extracted text and metadata for a local source file."""

    model_config = ConfigDict(frozen=True)

    path: Path
    title: str
    extension: str
    text: str = Field(repr=False)

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)


class SourceLoadError(BaseModel):
    """Non-fatal source loading error."""

    path: Path
    message: str
    exception_type: str | None = None


class Chunk(BaseModel):
    """A fixed-size text chunk derived from a source document."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_name: str
    chunk_number: int
    text: str = Field(repr=False)
    start_offset: int
    end_offset: int
    # K1.0 – web retrieval provenance.  Defaults keep backward compatibility.
    source_type: str = "local"  # "local" or "web"
    source_url: str = ""        # non-empty for web chunks

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.text)


class SourceCollection(BaseModel):
    """Documents loaded from a source directory."""

    root: Path
    documents: list[SourceDocument] = Field(default_factory=list)
    errors: list[SourceLoadError] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A structured source-backed evidence note."""

    evidence_id: str = ""
    claim: str
    source_document: str
    source_chunk_id: str = ""
    evidence_snippet: str
    category: EvidenceCategory
    relevance: str
    confidence: Literal["high", "medium", "low"]
    relevance_score: int = Field(default=3, ge=1, le=5)
    source_quality_score: int = Field(default=3, ge=1, le=5)
    source_quality_class: str = ""  # e.g. "nvidia_technical", "synthetic"
    specificity_score: int = Field(default=3, ge=1, le=5)
    overall_score: float = Field(default=3.0, ge=1, le=5)
    # J1.6 – entity and scope fields; populated by contradiction detection.
    # Empty string means not yet extracted (backward-compatible default).
    entity: str = ""        # named entity in the claim, e.g. "GB200 NVL72", "Reactor Alpha"
    entity_type: str = ""   # entity category, e.g. "rack", "reactor_unit"
    scope: str = ""         # measurement scope, e.g. "rack", "component", "fleet"
    # J3.1 – evidence classification; populated by evidence_enricher.
    evidence_type: str = ""  # metric | fact | comparison | causal | forecast | risk | constraint | timeline
    topics: list[str] = Field(default_factory=list)  # profile topics this evidence belongs to
    # J3.2 – retrieval diversity; populated by evidence_enricher.
    perspective: str = ""  # research dimension, e.g. "economics" | "fuel" | "cooling"


class ResearchPlan(BaseModel):
    """Claude-generated plan for a research run."""

    research_questions: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    source_priorities: list[str] = Field(default_factory=list)


class ClaudeCallTrace(BaseModel):
    """Observability metadata for one Claude request."""

    operation: str
    model_name: str
    request_timestamp: str
    success: bool
    token_usage: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class Contradiction(BaseModel):
    """A detected contradiction between two evidence items."""

    contradiction_id: str
    topic: str
    evidence_a_id: str
    evidence_b_id: str
    evidence_a_claim: str
    evidence_b_claim: str
    evidence_a_source: str
    evidence_b_source: str
    severity: Literal["low", "medium", "high"]
    explanation: str
    source_quality_a: int = Field(default=3, ge=1, le=5)
    source_quality_b: int = Field(default=3, ge=1, le=5)
    confidence: Literal["high", "medium", "low"] = "medium"
    # Metric type for each side — populated by numeric/categorical checks (J1.4).
    # Empty string means the field is not applicable or could not be inferred.
    metric_type_a: str = ""
    metric_type_b: str = ""
    # Contradiction topic provenance (J1.5 profile-aware topics).
    # "profile:<name>" when classified by a loaded profile;
    # ""              when no profile was active (hard-coded fallback).
    topic_source: str = ""
    # Entity / scope tracing — populated by duration conflict checks (J1.6).
    # entity_a / entity_b: significant tokens identifying the subject of each
    # claim (e.g. "alpha reactor").  comparison_reason: one-line explanation
    # of why the engine decided to compare the two items.
    entity_a: str = ""
    entity_b: str = ""
    comparison_reason: str = ""
    # J1.6 – physical measurement scope for each side of the contradiction.
    scope_a: str = ""
    scope_b: str = ""


class SuppressedComparison(BaseModel):
    """A pair of evidence items considered for contradiction but suppressed.

    Suppression reasons (J1.6):
    - ``scope_mismatch``        : items measure different physical scopes
                                  (e.g. rack-level vs component-level power)
    - ``milestone_progression`` : year values represent different lifecycle
                                  stages (e.g. approval year vs operation year)
    """

    evidence_a_id: str
    evidence_b_id: str
    evidence_a_claim: str
    evidence_b_claim: str
    reason: str            # "scope_mismatch" | "metric_scope_mismatch" | "milestone_progression" | "entity_mismatch"
    scope_a: str = ""
    scope_b: str = ""
    metric_a: str = ""     # populated for metric_scope_mismatch
    metric_b: str = ""
    detail: str = ""       # human-readable explanation of why it was suppressed


class ResearchGap(BaseModel):
    """An identified gap in the evidence corpus relative to the research question."""

    gap_id: str
    topic: str
    priority: Literal["high", "medium", "low"]
    description: str
    rationale: str


class CoverageArea(BaseModel):
    """Coverage assessment for one question topic."""

    topic: str
    evidence_count: int
    source_count: int
    coverage_level: Literal["strong", "moderate", "weak", "none"]
    rationale: str


class EvaluationWarning(BaseModel):
    """Warning-mode quality check result."""

    code: str
    message: str
    severity: str = "warning"


class ResearchMemo(BaseModel):
    """Structured memo that can be rendered to Markdown."""

    title: str
    question: str
    executive_summary: str
    confirmed_facts: list[str] = Field(default_factory=list)
    inferences: list[str] = Field(default_factory=list)
    power_implications: list[str] = Field(default_factory=list)
    cooling_implications: list[str] = Field(default_factory=list)
    networking_implications: list[str] = Field(default_factory=list)
    rack_architecture_implications: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    source_notes: list[EvidenceItem] = Field(default_factory=list)
    evaluation_warnings: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    research_plan: ResearchPlan | None = None
    claude_model_name: str | None = None
    claude_request_timestamp: str | None = None
    claude_response_success: bool | None = None
    claude_token_usage: dict[str, int] | None = None
    claude_call_traces: list[ClaudeCallTrace] = Field(default_factory=list)
    claude_errors: list[str] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    research_gaps: list[ResearchGap] = Field(default_factory=list)
    coverage_matrix: list[CoverageArea] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalScore(BaseModel):
    """Retrieval scoring result for one chunk."""

    chunk_id: str
    document_name: str
    keyword_score: float
    topic_match_score: float
    document_priority_score: float  # normalized source quality (quality_score / 5.0)
    source_quality_score: int = Field(default=3, ge=1, le=5)  # raw 1-5 quality score
    overall_retrieval_score: float


class ChunkDiagnostic(BaseModel):
    """Per-chunk diagnostic record written to the trace."""

    chunk_id: str
    document_name: str
    chunk_size: int
    relevance_score: float
    evidence_candidate_count: int
    sent_to_claude: bool
    evidence_items_created: int
    extraction_decision: Literal["accepted", "rejected", "not_sent"]
    rejection_reason: str | None = None


def assign_evidence_ids(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Assign stable sequential evidence IDs to evidence items."""

    return [
        item.model_copy(update={"evidence_id": f"E{index:03d}"})
        for index, item in enumerate(items, start=1)
    ]
