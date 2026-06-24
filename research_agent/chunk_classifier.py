"""Chunk classification for evidence yield optimisation (JH1).

Classifies chunks as evidence_dense | context | boilerplate | reference | unknown,
assigns an extraction priority (high | medium | low | skip), and computes
candidate signals (numeric_claim_count, named_entity_count, …).

The classifier is deterministic — no ML, no LLM call, no external dependencies.
It uses regex patterns and heuristics tuned to technical PDF extracts.

Public API
----------
classify_chunk(chunk)         -> ChunkClassification
compute_candidate_signals(text) -> CandidateSignals
is_boilerplate(text)          -> bool
is_high_priority_section(text) -> bool
PRIORITY_BOOST                  dict[ExtractionPriority, float]   (for scoring)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ChunkType = Literal["evidence_dense", "context", "boilerplate", "reference", "unknown"]
ExtractionPriority = Literal["high", "medium", "low", "skip"]

# Priority numeric boost values used by chunk selector
PRIORITY_BOOST: dict[str, float] = {
    "high":   1.0,
    "medium": 0.5,
    "low":    0.0,
    "skip":  -999.0,   # sentinel — excluded from selection
}


@dataclass
class CandidateSignals:
    """Heuristic signals that predict evidence density for a chunk."""

    numeric_claim_count: int = 0      # "42 MW", "3.7%", "$1.2 billion"
    named_entity_count: int = 0       # Capitalized multi-word phrases
    date_count: int = 0               # Years and date patterns
    unit_count: int = 0               # Technical units (kW, MW, GW, %, USD…)
    comparative_claim_count: int = 0  # "more than", "higher than", "vs", "compared"
    policy_or_standard_terms: int = 0 # FERC, IEEE, ASHRAE, PJM, MISO, ISO, DOE…

    @property
    def total(self) -> int:
        return (
            self.numeric_claim_count
            + self.named_entity_count
            + self.date_count
            + self.unit_count
            + self.comparative_claim_count
            + self.policy_or_standard_terms
        )

    @property
    def signal_score(self) -> float:
        """Normalised 0–1 score capped at total ≥ 10 = 1.0."""
        return min(self.total / 10.0, 1.0)

    def to_dict(self) -> dict[str, int]:
        return {
            "numeric_claim_count":       self.numeric_claim_count,
            "named_entity_count":        self.named_entity_count,
            "date_count":                self.date_count,
            "unit_count":                self.unit_count,
            "comparative_claim_count":   self.comparative_claim_count,
            "policy_or_standard_terms":  self.policy_or_standard_terms,
        }


@dataclass
class ChunkClassification:
    """Result of classify_chunk() for one chunk."""

    chunk_id: str
    chunk_type: ChunkType
    extraction_priority: ExtractionPriority
    candidate_signals: CandidateSignals = field(default_factory=CandidateSignals)
    classification_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "chunk_id":             self.chunk_id,
            "chunk_type":           self.chunk_type,
            "extraction_priority":  self.extraction_priority,
            "candidate_signals":    self.candidate_signals.to_dict(),
            "classification_reason": self.classification_reason,
        }


# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

_TOC_PATTERNS: list[re.Pattern] = [
    re.compile(r"table\s+of\s+contents", re.I),
    re.compile(r"^\s*contents\s*$", re.I | re.M),
    re.compile(r"(?:\d+\s+){3,}\d+"),          # runs of numbers (page numbers)
    re.compile(r"\.{4,}\s*\d+"),               # "Section name ....... 12"
    re.compile(r"(?:^|\n)\s*\d{1,3}\s+[A-Z][^\n]{0,60}\n\s*\d{1,3}\s+[A-Z]"),  # numbered TOC entries
]

_LEGAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"all rights reserved", re.I),
    re.compile(r"©\s*\d{4}", re.I),
    re.compile(r"copyright\s+\d{4}", re.I),
    re.compile(r"confidential\s+and\s+proprietary", re.I),
    re.compile(r"no part of this\s+(?:publication|document|report)", re.I),
    re.compile(r"without\s+(?:the\s+)?(?:prior\s+)?written\s+permission", re.I),
    re.compile(r"terms\s+and\s+conditions", re.I),
    re.compile(r"liability\s+disclaimer", re.I),
    re.compile(r"this\s+document\s+(?:is\s+)?(?:provided|intended)\s+for\s+informational", re.I),
]

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"page\s+\d+\s+of\s+\d+", re.I),          # "Page 3 of 22"
    re.compile(r"click\s+here\s+to", re.I),
    re.compile(r"(?:forward|back|next|previous)\s+page", re.I),
    re.compile(r"^\s*(?:figure|table|exhibit)\s+\d+\.?\s*$", re.I | re.M),
    re.compile(r"^\s*(?:notes?|source|sources?):\s*$", re.I | re.M),
    re.compile(r"(?:nav(?:igation)?|menu|header|footer)\b", re.I),
    re.compile(r"^\s*\d{1,3}\s*$", re.M),                 # lone page numbers
]

_REFERENCE_PATTERNS: list[re.Pattern] = [
    re.compile(r"references?\s*\n", re.I),
    re.compile(r"bibliography\s*\n", re.I),
    re.compile(r"works\s+cited\s*\n", re.I),
    re.compile(r"^\s*\[\d+\]\s+", re.M),                  # [1] citation format
    re.compile(r"^\s*\d+\.\s+[A-Z][^.]{10,}\.\s*(?:doi|http|www|isbn)", re.I | re.M),
    re.compile(r"doi:\s*10\.\d{4}/", re.I),
]

# ---------------------------------------------------------------------------
# High-value section patterns
# ---------------------------------------------------------------------------

_HIGH_PRIORITY_SECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"executive\s+summary", re.I),
    re.compile(r"key\s+(?:findings|results|takeaways|insights|conclusions)", re.I),
    re.compile(r"\bconclusions?\b", re.I),
    re.compile(r"\brecommendations?\b", re.I),
    re.compile(r"market\s+outlook", re.I),
    re.compile(r"scenario\s+analysis", re.I),
    re.compile(r"technology\s+roadmap", re.I),
    re.compile(r"cost\s+(?:estimates?|analysis|projections?|breakdown)", re.I),
    re.compile(r"policy\s+implications?", re.I),
    re.compile(r"investment\s+(?:outlook|analysis|opportunities?)", re.I),
    re.compile(r"technical\s+specifications?", re.I),
    re.compile(r"performance\s+(?:metrics|data|benchmarks?)", re.I),
    re.compile(r"financial\s+(?:projections?|analysis|results?)", re.I),
]

# ---------------------------------------------------------------------------
# Evidence-density signals
# ---------------------------------------------------------------------------

_NUMERIC_CLAIM_RE = re.compile(
    r"""
    (?:
        \d[\d,]*\.?\d*          # number (with optional comma thousands sep)
        \s*                     # optional space
        (?:
            %                  |  # percentage
            MW|GW|kW|TWh|GWh|MWh |  # power
            °C|°F              |  # temperature
            (?:USD|GBP|EUR|\$|€|£)\s*(?:billion|million|thousand|bn|mn|k)?  |
            (?:billion|million|thousand)\s*(?:dollars|pounds|euros)?         |
            (?:TB|GB|PB)/s     |  # bandwidth
            (?:PUE|CUE|WUE)    |  # efficiency ratios
            years?             |
            months?            |
            MW(?:th|e)?        |  # thermal/electric MW
            tons?|tonnes?
        )
    )
    """,
    re.VERBOSE | re.I,
)

_UNIT_RE = re.compile(
    r"""\b(?:
        MW|GW|kW|TWh|GWh|MWh|kWh |
        kV|MV|GV                  |
        MW(?:th|e)                |
        GHz|MHz|THz               |
        GB/s|TB/s|PB/s            |
        (?:bn|mn)\b               |
        billion|million|trillion  |
        (?:USD|GBP|EUR)\b         |
        PUE|CUE|WUE               |
        GPUs?|TPUs?|CPUs?|ASICs?  |
        racks?|servers?
    )\b""",
    re.VERBOSE | re.I,
)

_DATE_RE = re.compile(
    r"\b(?:20\d{2}|19\d{2})\b"                         # years 19xx-20xx
    r"|Q[1-4]\s*20\d{2}"                                # Q1 2024
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+20\d{2}\b",
    re.I,
)

_COMPARATIVE_RE = re.compile(
    r"\b(?:more|less|higher|lower|greater|fewer|faster|slower|larger|smaller)\s+than\b"
    r"|\bcompared\s+(?:to|with)\b"
    r"|\bvs\.?\s+\w"
    r"|\bversus\b"
    r"|\bunlike\b"
    r"|\bwhile\b.*\bwhereas\b"
    r"|\brelative\s+to\b",
    re.I,
)

_POLICY_RE = re.compile(
    r"""\b(?:
        FERC|NERC|ERCOT|PJM|MISO|CAISO|SPP|NYISO|ISO-NE |
        IEEE|ASHRAE|NFPA|NEC|IEC                         |
        DOE|EPA|CPUC|PUCT|NARUC                          |
        Order\s+\d{3,4}                                  |  # FERC Order 2023
        interconnection\s+(?:queue|process|agreement)    |
        transmission\s+(?:planning|operator|owner)       |
        renewable\s+portfolio\s+standard                 |
        IRA|Inflation\s+Reduction\s+Act                  |
        ISO\s+\d{2,5}                                    |
        IEC\s+\d{2,5}                                    |
        ANSI
    )\b""",
    re.VERBOSE | re.I,
)

_NAMED_ENTITY_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"              # Capitalized multi-word phrase
    r"|\b[A-Z]{2,}\b",                                   # Acronym
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_candidate_signals(text: str) -> CandidateSignals:
    """Compute evidence-density signals for a text block."""
    return CandidateSignals(
        numeric_claim_count     = len(_NUMERIC_CLAIM_RE.findall(text)),
        named_entity_count      = len(_NAMED_ENTITY_RE.findall(text)),
        date_count              = len(_DATE_RE.findall(text)),
        unit_count              = len(_UNIT_RE.findall(text)),
        comparative_claim_count = len(_COMPARATIVE_RE.findall(text)),
        policy_or_standard_terms = len(_POLICY_RE.findall(text)),
    )


def is_boilerplate(text: str) -> bool:
    """Return True if the text looks like non-evidence boilerplate."""
    # Very short chunks are often covers or navigation
    stripped = text.strip()
    if len(stripped) < 150:
        return True

    toc_score = sum(1 for p in _TOC_PATTERNS if p.search(text))
    if toc_score >= 2:
        return True

    legal_score = sum(1 for p in _LEGAL_PATTERNS if p.search(text))
    if legal_score >= 2:
        return True

    boiler_score = sum(1 for p in _BOILERPLATE_PATTERNS if p.search(text))
    if boiler_score >= 3:
        return True

    # If more than 60% of "sentences" are ≤ 8 words (navigation / list items)
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if s.strip()]
    if sentences:
        short = sum(1 for s in sentences if len(s.split()) <= 8)
        if short / len(sentences) > 0.6 and len(sentences) >= 5:
            return True

    return False


def is_reference_section(text: str) -> bool:
    """Return True if this chunk is primarily a references / bibliography section."""
    ref_score = sum(1 for p in _REFERENCE_PATTERNS if p.search(text))
    return ref_score >= 2


def is_high_priority_section(text: str) -> bool:
    """Return True if the text starts or is headed by a high-value section."""
    # Check first 400 chars (headers tend to appear early)
    header_zone = text[:400]
    return any(p.search(header_zone) for p in _HIGH_PRIORITY_SECTION_PATTERNS)


def classify_chunk(chunk_id: str, text: str) -> ChunkClassification:
    """Classify a chunk and return its type, priority, and candidate signals.

    Parameters
    ----------
    chunk_id : str
        The chunk's stable ID (used in the returned object only).
    text : str
        The raw chunk text.
    """
    signals = compute_candidate_signals(text)

    # ── boilerplate / reference short-circuits ──────────────────────────────
    if is_boilerplate(text):
        return ChunkClassification(
            chunk_id=chunk_id,
            chunk_type="boilerplate",
            extraction_priority="skip",
            candidate_signals=signals,
            classification_reason="boilerplate: short text, navigation, legal, or TOC pattern",
        )

    if is_reference_section(text):
        return ChunkClassification(
            chunk_id=chunk_id,
            chunk_type="reference",
            extraction_priority="skip",
            candidate_signals=signals,
            classification_reason="reference section: bibliography / citation list",
        )

    # ── evidence-dense detection ─────────────────────────────────────────────
    # High priority: starts with a known high-value section header
    if is_high_priority_section(text):
        return ChunkClassification(
            chunk_id=chunk_id,
            chunk_type="evidence_dense",
            extraction_priority="high",
            candidate_signals=signals,
            classification_reason="high-priority section header (exec summary / findings / recommendations)",
        )

    # Evidence dense: strong signal score
    if signals.signal_score >= 0.6 or (signals.numeric_claim_count >= 3 and signals.unit_count >= 2):
        return ChunkClassification(
            chunk_id=chunk_id,
            chunk_type="evidence_dense",
            extraction_priority="high",
            candidate_signals=signals,
            classification_reason=(
                f"evidence_dense: signals={signals.total} "
                f"(numeric={signals.numeric_claim_count}, units={signals.unit_count}, "
                f"policy={signals.policy_or_standard_terms})"
            ),
        )

    if signals.signal_score >= 0.3 or signals.numeric_claim_count >= 1 or signals.policy_or_standard_terms >= 1:
        return ChunkClassification(
            chunk_id=chunk_id,
            chunk_type="evidence_dense",
            extraction_priority="medium",
            candidate_signals=signals,
            classification_reason=(
                f"evidence_candidate: signals={signals.total} "
                f"(numeric={signals.numeric_claim_count}, policy={signals.policy_or_standard_terms})"
            ),
        )

    # Context text: prose without strong evidence signals
    return ChunkClassification(
        chunk_id=chunk_id,
        chunk_type="context",
        extraction_priority="low",
        candidate_signals=signals,
        classification_reason=f"context: low signal score ({signals.signal_score:.2f}), no priority headers",
    )
