"""Evidence quality enrichment (J3.1).

Adds evidence_type and topics to EvidenceItem objects after initial extraction.
This is a post-extraction pass — it never modifies claim/snippet text, only
annotates structured fields.

Evidence type taxonomy
----------------------
metric      – contains a numeric measurement with a unit (kW, MW, %, years, GPUs…)
comparison  – explicitly compares two things (more than, higher than, vs, unlike…)
causal      – expresses a cause-effect relationship (because, due to, leads to…)
forecast    – forward-looking or projected information (will, expected, by 2030…)
risk        – identifies a risk, challenge, or barrier
constraint  – identifies a limiting condition (depends on, requires, limited by…)
timeline    – time duration or schedule without a plain numeric measurement
fact        – default; verifiable statement not in the above categories
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from .perspectives import classify_perspective
from .schemas import EvidenceItem

if TYPE_CHECKING:
    from .profile import DomainProfile


# ---------------------------------------------------------------------------
# Evidence type detection
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific; first match wins.
# Each entry: (type_name, compiled_pattern)
_TYPE_RULES: list[tuple[str, re.Pattern]] = [
    (
        "comparison",
        re.compile(
            r"\b(more than|less than|higher than|lower than|greater than|"
            r"compared to|unlike|whereas|by contrast|in contrast|"
            r"outperforms|exceeds|surpasses|vs\.?|versus|relative to|"
            r"advantage over|disadvantage compared)\b",
            re.I,
        ),
    ),
    (
        "causal",
        re.compile(
            r"\b(because|due to|as a result|therefore|thus|hence|"
            r"leads? to|results? in|causes?|enables?|prevents?|delays?|"
            r"limits?|driven by|triggered by|attributable to|owing to|"
            r"shortage[s]? .{0,30}delay|supply .{0,20}constrain)\b",
            re.I,
        ),
    ),
    (
        "risk",
        re.compile(
            r"\b(risk|challenge|barrier|uncertain|unlikely|may fail|"
            r"concern|threat|vulnerability|exposure|hazard|danger|"
            r"problematic|difficult|if not|could jeopardize|at risk)\b",
            re.I,
        ),
    ),
    (
        "forecast",
        re.compile(
            r"\b(will|expect(?:ed|s)?|project(?:ed|s)?|anticipate[sd]?|"
            r"forecast[s]?|by 20\d\d|target[s]?|plan(?:ned|s)?|goal|aim[s]?|"
            r"predict(?:ed|s)?|trajectory|outlook|future)\b",
            re.I,
        ),
    ),
    (
        "constraint",
        re.compile(
            r"\b(limited by|depends on|dependent on|requires?|must|cannot|"
            r"constrained|bottleneck|cap(?:acity)? constraint|"
            r"insufficient|inadequate|lacking|shortage)\b",
            re.I,
        ),
    ),
    (
        "timeline",
        re.compile(
            r"\b(\d{1,2}[-–]\d{1,2}\s*years?|\d+\s*months?|\d+\s*years?|"
            r"quarter[s]?|schedule|duration|timeline|by Q[1-4]|"
            r"first concrete|commissioning date|completion date|"
            r"construction period|build time)\b",
            re.I,
        ),
    ),
    (
        "metric",
        re.compile(
            r"\b\d[\d,.]*\s*"
            r"(?:kw|mw|gw|kwh|mwh|gwh|kwt|mwt|gwt|kwe|mwe|gwe|"
            r"tb/s|gb/s|mb/s|"
            r"gpu[s]?|cpu[s]?|gb|tb|pb|"
            r"percent|%|usd|\$|€|billion|million|trillion|"
            r"years?|months?|weeks?|days?|"
            r"degrees?|°[cf]|celsius|fahrenheit|"
            r"bar|psi|atm|pascal|"
            r"liter[s]?|gallon[s]?|"
            r"watt[s]?|ampere[s]?|volt[s]?|"
            r"tonne[s]?|ton[s]?|kg|pound[s]?|"
            r"meter[s]?|km|mile[s]?|"
            r"rpm|hz|khz|mhz|ghz)(?:\b|(?=\W|$))",
            re.I,
        ),
    ),
]


def classify_evidence_type(text: str) -> str:
    """Return the most specific evidence type for *text*.

    Evaluates rules in priority order; returns 'fact' if nothing matches.
    """
    for type_name, pattern in _TYPE_RULES:
        if pattern.search(text):
            return type_name
    return "fact"


# ---------------------------------------------------------------------------
# Topic tagging
# ---------------------------------------------------------------------------

def tag_evidence_topics(
    text: str,
    profile: "DomainProfile | None",
) -> list[str]:
    """Return the profile topics whose keywords appear in *text*.

    Falls back to an empty list when no profile is provided (topics are
    then populated by the coverage-oriented pass in agent.py).
    """
    if profile is None:
        return []
    text_lower = text.lower()
    matched: list[str] = []
    for topic, keywords in profile.topic_keywords.items():
        if any(kw.lower() in text_lower for kw in keywords):
            matched.append(topic)
    return matched


# ---------------------------------------------------------------------------
# Main enrichment pass
# ---------------------------------------------------------------------------

def enrich_evidence_with_metadata(
    items: list[EvidenceItem],
    profile: "DomainProfile | None" = None,
) -> list[EvidenceItem]:
    """Annotate evidence_type and topics on every item in-place-style.

    Returns a new list of EvidenceItems with the fields populated.
    Existing non-empty values are preserved (idempotent).
    """
    enriched: list[EvidenceItem] = []
    for item in items:
        updates: dict = {}
        # evidence_type: detect if not yet set
        if not item.evidence_type:
            updates["evidence_type"] = classify_evidence_type(
                item.claim + " " + item.evidence_snippet
            )
        # topics: tag if not yet set
        if not item.topics:
            updates["topics"] = tag_evidence_topics(
                item.claim + " " + item.evidence_snippet, profile
            )
        # J3.2 – perspective: classify if not yet set
        if not item.perspective:
            updates["perspective"] = classify_perspective(
                item.claim + " " + item.evidence_snippet,
                item.source_document,
            )
        if updates:
            enriched.append(item.model_copy(update=updates))
        else:
            enriched.append(item)
    return enriched


# ---------------------------------------------------------------------------
# Density metrics and distribution aggregation
# ---------------------------------------------------------------------------

def build_evidence_density_stats(
    items: list[EvidenceItem],
    chunks_processed: int,
) -> dict:
    """Return J3.1.6 density metrics and J3.1.8 distribution stats."""
    n = len(items)
    type_dist: Counter[str] = Counter(
        it.evidence_type or "unknown" for it in items
    )
    topic_dist: Counter[str] = Counter(
        topic for it in items for topic in it.topics
    )
    return {
        "chunks_processed": chunks_processed,
        "evidence_items": n,
        "evidence_per_chunk": round(n / chunks_processed, 2) if chunks_processed else 0.0,
        "evidence_type_distribution": dict(type_dist),
        "topic_distribution": dict(topic_dist),
    }
