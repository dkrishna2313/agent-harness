"""Coverage matrix computation.

For every question topic detected in the research question, a CoverageArea
records how thoroughly the extracted evidence corpus covers that topic.

Coverage levels — determined solely by evidence quantity and source diversity
-------------------------------------------------------------------------------
strong   ≥ STRONG_MIN_EVIDENCE items  AND  ≥ STRONG_MIN_SOURCES distinct sources
moderate ≥ 2 evidence items  (any source count)
weak     exactly 1 evidence item
none     0 evidence items

Source quality is intentionally excluded from the level gate.  It enriches the
rationale text (e.g. "high-quality sources") but never reduces the level that
evidence count and source diversity already earned.

Research gaps are also intentionally excluded from the level calculation.
They describe *what is missing*, not *how much was found*.  Gap information
is appended to the rationale so it remains visible without penalising a
well-evidenced topic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .evaluator import classify_question_topics
from .schemas import CoverageArea, EvidenceItem, ResearchGap, SourceQuality

if TYPE_CHECKING:
    from .profile import DomainProfile

# ---------------------------------------------------------------------------
# Thresholds (evidence-based only; quality does not gate any level)
# ---------------------------------------------------------------------------
STRONG_MIN_EVIDENCE = 5
STRONG_MIN_SOURCES = 2

# ---------------------------------------------------------------------------
# Topic → evidence categories that count toward coverage of that topic
# ---------------------------------------------------------------------------
_TOPIC_CATEGORIES: dict[str, frozenset[str]] = {
    "power":             frozenset({"power"}),
    "cooling":           frozenset({"cooling"}),
    "networking":        frozenset({"networking"}),
    "rack architecture": frozenset({"rack architecture", "architecture"}),
    "operations":        frozenset({"operations"}),
    "backup/resiliency": frozenset({"operations", "power"}),
}

# ---------------------------------------------------------------------------
# Topic → keywords matched against gap.topic for rationale notes
# ---------------------------------------------------------------------------
_TOPIC_GAP_KEYWORDS: dict[str, frozenset[str]] = {
    "power":             frozenset({"rack power", "power delivery", "ups", "backup", "generator", "power quality"}),
    "cooling":           frozenset({"cooling technology", "cdu", "heat rejection", "facility integration", "water temp"}),
    "networking":        frozenset({"bandwidth", "topology", "optics", "switch architecture"}),
    "rack architecture": frozenset({"rack architecture", "rack", "architecture"}),
    "operations":        frozenset({"commissioning", "monitoring", "maintenance", "resiliency"}),
    "backup/resiliency": frozenset({"ups", "backup", "resiliency", "generator"}),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_coverage_matrix(
    question: str,
    evidence: Sequence[EvidenceItem],
    *,
    research_gaps: Sequence[ResearchGap] | None = None,
    source_quality_map: dict[str, SourceQuality] | None = None,
    profile: "DomainProfile | None" = None,
) -> list[CoverageArea]:
    """Return a CoverageArea for every relevant topic.

    When *profile* is supplied the topics come from ``profile.coverage_topics``
    (fixed, domain-defined set) and evidence is counted either via the
    profile's category mapping or via keyword matching against evidence text.

    When *profile* is ``None`` the legacy behaviour is used: topics are
    detected from *question* and evidence is counted by ``EvidenceItem.category``.

    Results are sorted alphabetically by topic so the output is deterministic.
    """
    if profile is not None:
        topics = list(profile.coverage_topics)
    else:
        topics = sorted(classify_question_topics(question))

    if not topics:
        return []

    gaps = list(research_gaps or [])
    areas: list[CoverageArea] = []

    for topic in sorted(topics):
        if profile is not None:
            topic_evidence = _evidence_for_topic_profile(evidence, topic, profile)
            if profile.coverage_gap_keywords and topic in profile.coverage_gap_keywords:
                gap_keywords = frozenset(kw.lower() for kw in profile.coverage_gap_keywords[topic])
            else:
                gap_keywords = frozenset(
                    kw.lower() for kw in profile.topic_keywords.get(topic, [topic])
                )
        else:
            categories = _TOPIC_CATEGORIES.get(topic, frozenset({topic}))
            topic_evidence = [item for item in evidence if item.category in categories]
            gap_keywords = _TOPIC_GAP_KEYWORDS.get(topic, frozenset({topic}))

        evidence_count = len(topic_evidence)
        sources: set[str] = {item.source_document for item in topic_evidence}
        source_count = len(sources)
        avg_quality = _avg_quality(sources, source_quality_map)

        # Gaps for this topic — used in rationale only, not in level
        topic_gaps = [
            g for g in gaps
            if any(kw in g.topic.lower() for kw in gap_keywords)
        ]
        high_gaps = [g for g in topic_gaps if g.priority == "high"]

        level = _coverage_level(
            evidence_count=evidence_count,
            source_count=source_count,
        )
        rationale = _build_rationale(
            topic=topic,
            evidence_count=evidence_count,
            source_count=source_count,
            avg_quality=avg_quality,
            level=level,
            high_gaps=high_gaps,
        )

        areas.append(CoverageArea(
            topic=topic,
            evidence_count=evidence_count,
            source_count=source_count,
            coverage_level=level,
            rationale=rationale,
        ))

    return areas


def _evidence_for_topic_profile(
    evidence: Sequence[EvidenceItem],
    topic: str,
    profile: "DomainProfile",
) -> list[EvidenceItem]:
    """Return evidence items relevant to *topic* using the profile's configuration.

    Strategy (in order of preference):
    1. If the profile defines a ``topic_categories`` mapping for this topic,
       filter by ``item.category`` (exact match, fast, legacy-compatible).
    2. Otherwise fall back to keyword matching against the concatenation of
       ``item.claim`` and ``item.evidence_snippet``.
    """
    # Check if the profile has an explicit topic→category mapping
    if profile.topic_categories and topic in profile.topic_categories:
        explicit_cats = frozenset(profile.topic_categories[topic])
        return [item for item in evidence if item.category in explicit_cats]

    # No explicit mapping: keyword-based fallback
    keywords = frozenset(kw.lower() for kw in profile.topic_keywords.get(topic, [topic]))
    if not keywords:
        return []
    return [
        item for item in evidence
        if _item_contains_any(item, keywords)
    ]


def _item_contains_any(item: EvidenceItem, keywords: frozenset[str]) -> bool:
    """Return True if any keyword appears in the item's text fields."""
    haystack = f"{item.claim} {item.evidence_snippet}".lower()
    return any(kw in haystack for kw in keywords)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _coverage_level(
    *,
    evidence_count: int,
    source_count: int,
) -> str:
    """Return the coverage level based on evidence count and source diversity only.

    Source quality and research gaps are intentionally absent: quality enriches
    the rationale; gaps are a separate instrument that measures *missing*
    information, not *found* evidence.
    """
    if evidence_count == 0:
        return "none"
    if evidence_count == 1:
        return "weak"
    if evidence_count >= STRONG_MIN_EVIDENCE and source_count >= STRONG_MIN_SOURCES:
        return "strong"
    return "moderate"


def _build_rationale(
    *,
    topic: str,
    evidence_count: int,
    source_count: int,
    avg_quality: float,
    level: str,
    high_gaps: list[ResearchGap],
) -> str:
    """Compose a human-readable rationale string.

    Structure:
      1. Lead: evidence-coverage strength statement
      2. Quality note (if notable)
      3. Gap note (if high-priority gaps exist) — informational only
    """
    if level == "none":
        return f"No evidence items found for '{topic}'."

    source_word = "source" if source_count == 1 else "sources"

    if level == "weak":
        lead = f"Weak evidence coverage. Only 1 evidence item found for '{topic}'."
    elif level == "moderate":
        lead = (
            f"Moderate evidence coverage. "
            f"{evidence_count} evidence items across {source_count} {source_word}."
        )
    else:  # strong
        lead = (
            f"Strong evidence coverage. "
            f"{evidence_count} evidence items across {source_count} {source_word}."
        )

    quality_note = _quality_note(avg_quality)
    parts = [lead]
    if quality_note:
        parts.append(quality_note.capitalize() + ".")

    if high_gaps:
        gap_labels = _gap_label_list(high_gaps)
        parts.append(f"Some research gaps remain regarding {gap_labels}.")

    return " ".join(parts)


def _gap_label_list(gaps: list[ResearchGap]) -> str:
    """Return a short comma-joined list of gap topics (max 3, then '…')."""
    labels = [g.topic for g in gaps[:3]]
    suffix = "…" if len(gaps) > 3 else ""
    return ", ".join(labels) + suffix


def _avg_quality(sources: set[str], sq_map: dict[str, SourceQuality] | None) -> float:
    if not sq_map or not sources:
        return 3.0
    scores = [sq_map[s].source_quality_score for s in sources if s in sq_map]
    return sum(scores) / len(scores) if scores else 3.0


def _quality_note(avg_quality: float) -> str:
    """Return a short quality descriptor phrase, or empty string if unremarkable."""
    if avg_quality >= 4.5:
        return "high-quality sources"
    if avg_quality >= 4.0:
        return "good-quality sources"
    if avg_quality <= 1.5:
        return "low-quality sources only"
    return ""
