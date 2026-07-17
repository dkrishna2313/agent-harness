"""Canonical knowledge models — J8.0 ontology, v2 schema (PH5.5a).

Evidence v2 adds passage-level provenance, content signals, grounding, and
relationship fields to the Evidence record (PH5.5a schema foundation).
All new fields are optional or defaulted — v1 payloads deserialize unchanged.

RetrievalProvenance is a new first-class type for runtime scoring traceability.
"""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field

# ---------------------------------------------------------------------------
# Literals / enums
# ---------------------------------------------------------------------------

EvidenceState = Literal[
    "ACTIVE",
    "SUPERSEDED",
    "LOW_CONFIDENCE",
    "RETRACTED",
    "ARCHIVED",
]

CredibilityLevel = Literal["HIGH", "MEDIUM", "LOW"]

ReviewStatus = Literal["UNREVIEWED", "AUTO_REVIEWED", "HUMAN_REVIEWED"]

ContradictionType = Literal["DIRECT", "SCOPE", "TEMPORAL", "METHODOLOGICAL"]

DetectionMethod = Literal["AUTOMATED", "HUMAN"]

ContradictionSeverity = Literal["HIGH", "MEDIUM", "LOW"]

ResolutionStatus = Literal["OPEN", "RESOLVED", "ACCEPTED_AMBIGUITY"]

RunStatus = Literal["RUNNING", "COMPLETED", "FAILED"]

EvidenceType = Literal[
    "STRATEGIC",      # Investment, deployment, policy, market claims — Planner primary retrieval
    "TECHNICAL",      # Specifications, parameters, engineering claims — Planner secondary retrieval
    "PROVENANCE",     # Authorship, publication info — stored but excluded from normal retrieval
    "ADMINISTRATIVE", # Document IDs, revisions, trademarks, boilerplate — excluded from retrieval
]

# ---------------------------------------------------------------------------
# v2 Evidence literals (PH5.5a)
# ---------------------------------------------------------------------------

EvidenceConfidence = Literal["HIGH", "MEDIUM", "LOW"]

GroundingStrength = Literal["STRONG", "MODERATE", "WEAK"]

RetrievalMode = Literal["lexical", "semantic", "hybrid"]

RerankerType = Literal["passthrough", "llm", "none"]

SourceType = Literal[
    "PDF",
    "HTML",
    "TXT",
    "DOCX",
    "API",
]

# ---------------------------------------------------------------------------
# Source — immutable, content-addressed
# ---------------------------------------------------------------------------


class Source(BaseModel):
    """Represents an original document exactly as obtained from the world.

    A Source record is written once at ingestion and never modified.
    source_id is derived from the SHA-256 fingerprint of canonical_text.
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    uri: str
    title: str
    author: str | None = None
    publisher: str | None = None
    publication_date: date | None = None
    retrieved_date: date
    fingerprint: str
    language: str = "en"
    document_type: SourceType
    domain: str
    subtitle: str | None = None
    organization: str | None = None
    copyright: str | None = None
    document_version: str | None = None
    document_number: str | None = None
    canonical_text: str = Field(repr=False)
    page_count: int | None = None

    @computed_field
    @property
    def char_count(self) -> int:
        return len(self.canonical_text)

    @classmethod
    def compute_fingerprint(cls, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @classmethod
    def compute_source_id(cls, fingerprint: str) -> str:
        """source_id is the first 32 chars of the fingerprint."""
        return fingerprint[:32]


# ---------------------------------------------------------------------------
# Evidence — append-only
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """Represents one atomic, source-backed claim.

    Evidence records are append-only: never modified after creation.
    If a claim is superseded, a new Evidence record is created referencing
    the old one via superseded_by / supersedes.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    evidence_type: EvidenceType = "STRATEGIC"
    supporting_source_ids: list[str] = Field(default_factory=list)
    profile_ids: list[str] = Field(default_factory=list)
    extraction_run_id: str
    entity: str = ""
    entity_type: str = ""
    scope: str = ""
    category: str = ""
    supersedes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
    contradiction_ids: list[str] = Field(default_factory=list)

    # --- v2: Schema identity (PH5.5a) ---
    schema_version: str = "1.0"
    corpus_version: str | None = None

    # --- v2: Passage-level provenance (PH5.5a; populated by PH5.5b extraction) ---
    excerpt: str | None = None
    page_number: int | None = None
    section_heading: str | None = None
    chunk_id: str | None = None
    char_offset_start: int | None = None
    char_offset_end: int | None = None

    # --- v2: Content signals (PH5.5a; populated by PH5.5b extraction) ---
    topics: list[str] = Field(default_factory=list)
    temporal_reference: str | None = None
    is_quantitative: bool = False

    # --- v2: Extraction quality (PH5.5a; populated by PH5.5b extraction) ---
    evidence_confidence: EvidenceConfidence | None = None

    # --- v2: Grounding (PH5.5a; populated by PH5.5e assembly) ---
    subquestion_assignments: list[str] = Field(default_factory=list)
    investigation_area_assignments: list[str] = Field(default_factory=list)
    grounding_strength: GroundingStrength | None = None
    coverage_contribution: GroundingStrength | None = None

    # --- v2: Relationships (PH5.5a; populated by downstream agents) ---
    corroborates: list[str] = Field(default_factory=list)
    informed_hypotheses: list[str] = Field(default_factory=list)
    informed_recommendations: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def statement_fingerprint(self) -> str:
        """SHA-256[:16] of normalised statement — short deduplication key (v1 compat)."""
        normalised = " ".join(self.statement.lower().split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]

    @computed_field
    @property
    def content_fingerprint(self) -> str:
        """Full SHA-256 of normalised statement — canonical v2 content identity."""
        normalised = " ".join(self.statement.lower().split())
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# KnowledgeMetadata — mutable quality and lifecycle state
# ---------------------------------------------------------------------------


class KnowledgeMetadata(BaseModel):
    """Mutable quality and lifecycle metadata for one Evidence record.

    Separated from Evidence to preserve Evidence immutability.
    Quality assessments evolve; Evidence records do not.
    """

    evidence_id: str
    state: EvidenceState = "ACTIVE"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    credibility: CredibilityLevel = "MEDIUM"
    relevance_score: float = Field(default=3.0, ge=1.0, le=5.0)
    source_quality_score: float = Field(default=3.0, ge=1.0, le=5.0)
    specificity_score: float = Field(default=3.0, ge=1.0, le=5.0)
    overall_score: float = Field(default=3.0, ge=1.0, le=5.0)
    review_status: ReviewStatus = "UNREVIEWED"
    version: int = 1
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    quality_flags: list[str] = Field(default_factory=list)
    retrieval_enabled: bool = True
    retrieval_priority: int = Field(default=3, ge=1, le=5)
    strategic_value: float = Field(default=0.5, ge=0.0, le=1.0)

    def compute_overall_score(self) -> float:
        return round(
            (self.relevance_score + self.source_quality_score + self.specificity_score) / 3.0,
            2,
        )


# ---------------------------------------------------------------------------
# ExtractionRun — narrow provenance record for evidence construction
# ---------------------------------------------------------------------------


class ExtractionRun(BaseModel):
    """Records the parameters and outcome of a KnowledgeBuilder execution.

    ExtractionRun is narrow by design: it exists solely to provide provenance
    and context for Evidence construction and supersession. It is not a
    general-purpose audit log (that is a future J9 capability).
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    source_ids: list[str] = Field(default_factory=list)
    model_version: str
    prompt_version: str
    evidence_ids_produced: list[str] = Field(default_factory=list)
    evidence_ids_superseded: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    status: RunStatus = "RUNNING"
    sources_scanned: int = 0
    sources_skipped: int = 0
    sources_rebuilt: int = 0
    duplicates_merged: int = 0
    embeddings_generated: int = 0
    duration_seconds: float | None = None


# ---------------------------------------------------------------------------
# Contradiction — first-class KB object (structure only for J8.1)
# ---------------------------------------------------------------------------


class Contradiction(BaseModel):
    """A known conflict between two Evidence records.

    Contradictions are computed offline and stored permanently.
    Not regenerated on each research run.
    Note: contradiction detection is implemented in J8.4.
    This model is defined here so the store can hold them.
    """

    contradiction_id: str = Field(default_factory=lambda: str(uuid4()))
    evidence_id_a: str
    evidence_id_b: str
    contradiction_type: ContradictionType
    detection_method: DetectionMethod = "AUTOMATED"
    severity: ContradictionSeverity = "MEDIUM"
    resolution_status: ResolutionStatus = "OPEN"
    resolution_notes: str | None = None
    detected_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# EvidenceProfile — join table
# ---------------------------------------------------------------------------


class EvidenceProfile(BaseModel):
    evidence_id: str
    profile_id: str
    relevance_score: float = Field(default=3.0, ge=1.0, le=5.0)
    tagged_by: Literal["AUTOMATED", "HUMAN"] = "AUTOMATED"


# ---------------------------------------------------------------------------
# RetrievalProvenance — runtime scoring record (v2, PH5.5a)
# ---------------------------------------------------------------------------


class RetrievalProvenance(BaseModel):
    """Runtime retrieval provenance for one evidence item returned by a query.

    Not persisted to the Evidence JSONL — stored per query alongside the
    assembled evidence set. Enables full scoring traceability for grounding
    and reproducibility auditing without polluting the Evidence corpus with
    query-specific state.

    PH5.5c adds: retrieval_timestamp, retrieved_candidate_count, reranked,
    reranker_model — all describing the retrieval process, never influencing it.
    """

    evidence_id: str
    retrieval_query: str
    retrieval_mode: RetrievalMode
    retrieval_model_version: str | None = None
    retrieval_rank: int
    hybrid_score: float
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    metadata_factor: float | None = None
    reranker: RerankerType = "passthrough"
    rerank_score: float | None = None
    rerank_rationale: str | None = None
    # PH5.5c — process provenance fields
    retrieval_timestamp: str | None = None          # ISO 8601, when provenance was captured
    retrieved_candidate_count: int = 0              # matched_candidates from the retrieval pass
    reranked: bool = False                          # True when LLM reranker produced the final order
    reranker_model: str | None = None               # e.g. "claude-haiku-4-5-20251001"


# ---------------------------------------------------------------------------
# SourceManifestEntry — tracks what has been indexed per source
# ---------------------------------------------------------------------------


class SourceManifestEntry(BaseModel):
    """Tracks the indexing state of one Source.

    Used by the KnowledgeBuilder to implement incremental builds.
    If a source's fingerprint matches the manifest entry, it is skipped.
    """

    source_id: str
    fingerprint: str
    domain: str
    uri: str
    evidence_ids: list[str] = Field(default_factory=list)
    metadata_ids: list[str] = Field(default_factory=list)
    last_built: datetime = Field(default_factory=datetime.utcnow)
    extraction_run_id: str = ""
