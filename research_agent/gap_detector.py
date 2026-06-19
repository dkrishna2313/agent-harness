"""Research gap detection: identify insufficiently supported topics in the evidence corpus."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from .evaluator import classify_question_topics
from .schemas import EvidenceItem, ResearchGap

if TYPE_CHECKING:
    from .profile import DomainProfile

# ---------------------------------------------------------------------------
# Keyword sets for each sub-topic we look for in evidence text
# Each entry: (topic_label, search_terms, priority, description, rationale)
# ---------------------------------------------------------------------------

_POWER_SUBTOPICS: list[tuple[str, set[str], str, str, str]] = [
    (
        "Rack Power Consumption",
        {"rack power", "kw", "mw", "power draw", "power consumption", "watt"},
        "high",
        "No explicit rack-level power consumption figure found.",
        "Power planning requires a concrete rack-level power target (kW per rack).",
    ),
    (
        "Power Delivery Infrastructure",
        {"pdu", "busway", "power distribution", "power delivery", "breaker", "circuit"},
        "high",
        "No power delivery or PDU infrastructure guidance found.",
        "Facility teams need PDU ratings and feed topology to plan electrical infrastructure.",
    ),
    (
        "UPS and Backup Power",
        {"ups", "uninterruptible", "battery backup", "bbu", "backup power"},
        "medium",
        "No UPS or battery backup requirements found.",
        "Runtime and capacity targets for backup power are needed for resiliency planning.",
    ),
    (
        "Generator and Utility Interconnect",
        {"generator", "utility", "interconnect", "genset", "transfer switch", "grid connection"},
        "medium",
        "No generator or utility interconnect specifications found.",
        "Generator sizing and interconnect requirements affect site planning.",
    ),
    (
        "Power Quality",
        {"power quality", "harmonic", "voltage regulation", "power factor", "transient"},
        "low",
        "No power quality specifications found.",
        "Power quality requirements (harmonics, voltage tolerance) are needed for facility design.",
    ),
]

_COOLING_SUBTOPICS: list[tuple[str, set[str], str, str, str]] = [
    (
        "Cooling Technology",
        {"liquid cool", "direct liquid", "dlc", "air cool", "immersion", "rear door", "cooling technology"},
        "high",
        "No cooling technology specification found.",
        "Operators must know whether liquid or air cooling is required before facility planning.",
    ),
    (
        "CDU Requirements",
        {"cdu", "coolant distribution", "cooling distribution unit", "manifold"},
        "high",
        "No CDU sizing or specification found.",
        "CDU capacity and connection specifications are required for liquid cooling deployment.",
    ),
    (
        "Water Temperature and Flow Rate",
        {"supply temperature", "water temperature", "flow rate", "coolant temperature", "°c", "degrees"},
        "medium",
        "No coolant supply temperature or flow rate specifications found.",
        "Facility chiller and pipe sizing depend on required supply temperature and flow rate.",
    ),
    (
        "Heat Rejection",
        {"heat rejection", "heat load", "thermal output", "btuh", "heat dissipation"},
        "medium",
        "No total heat rejection load specification found.",
        "HVAC and cooling plant sizing requires the total heat rejection per rack.",
    ),
    (
        "Facility Cooling Integration",
        {"facility integration", "raised floor", "in-row", "overhead", "cooling integration", "chiller"},
        "low",
        "No facility-side cooling integration guidance found.",
        "Integration guidance is needed to connect rack cooling to facility infrastructure.",
    ),
]

_NETWORKING_SUBTOPICS: list[tuple[str, set[str], str, str, str]] = [
    (
        "Network Bandwidth",
        {"bandwidth", "throughput", "gb/s", "tb/s", "network speed"},
        "high",
        "No network bandwidth specification found.",
        "Bandwidth requirements drive switch selection and cabling plant design.",
    ),
    (
        "Network Topology",
        {"topology", "spine", "leaf", "fat tree", "rail", "network topology"},
        "medium",
        "No network topology guidance found.",
        "Topology affects switch count, cabling, and latency for scale-out deployments.",
    ),
    (
        "Optical Transceivers",
        {"optic", "transceiver", "pluggable", "qsfp", "dac", "aoc", "fiber"},
        "medium",
        "No optical transceiver or cabling specifications found.",
        "Transceiver type and cable reach requirements drive infrastructure costs.",
    ),
    (
        "Switch Architecture",
        {"switch", "asic", "switching fabric", "switch architecture"},
        "low",
        "No switch architecture or vendor recommendations found.",
        "Switch selection impacts latency, power, and operational complexity.",
    ),
]

_OPERATIONS_SUBTOPICS: list[tuple[str, set[str], str, str, str]] = [
    (
        "Commissioning",
        {"commission", "installation", "bring-up", "initial setup", "rack installation"},
        "medium",
        "No commissioning or installation guidance found.",
        "Step-by-step commissioning procedures reduce deployment risk.",
    ),
    (
        "Monitoring and Telemetry",
        {"monitor", "telemetry", "alert", "dashboard", "observability", "health check"},
        "medium",
        "No monitoring or telemetry requirements found.",
        "Monitoring requirements drive management network and tooling selection.",
    ),
    (
        "Maintenance Procedures",
        {"maintenance", "servicing", "hot-swap", "replacement", "field replace"},
        "low",
        "No maintenance or field-servicing guidance found.",
        "Maintenance windows and procedures affect operational planning.",
    ),
    (
        "Resiliency and Redundancy",
        {"redundan", "resilien", "failover", "ha ", "high availability", "n+1"},
        "low",
        "No resiliency or redundancy specifications found.",
        "Redundancy levels for power and cooling feed site availability calculations.",
    ),
]

# Map question topic → subtopics to check
_TOPIC_SUBTOPIC_MAP: dict[str, list[tuple[str, set[str], str, str, str]]] = {
    "power": _POWER_SUBTOPICS,
    "cooling": _COOLING_SUBTOPICS,
    "networking": _NETWORKING_SUBTOPICS,
    "operations": _OPERATIONS_SUBTOPICS,
    "backup/resiliency": _OPERATIONS_SUBTOPICS,
    "rack architecture": [],  # covered implicitly by power/cooling checks
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_gaps(
    question: str,
    evidence: Sequence[EvidenceItem],
    profile: "DomainProfile | None" = None,
) -> list[ResearchGap]:
    """Identify under-evidenced subtopics relative to the detected question topics.

    When *profile* is supplied the topic set and gap-check definitions come from
    the profile's ``research_gap_checks`` mapping.  When *profile* is ``None``
    the legacy hard-coded subtopic lists are used.
    """
    if profile is not None:
        question_topics = profile.classify_question_topics(question)
        subtopic_map = _profile_to_subtopic_map(profile)
    else:
        question_topics = classify_question_topics(question)
        subtopic_map = _TOPIC_SUBTOPIC_MAP

    if not question_topics:
        return []

    # Build a single lowercased evidence corpus for fast substring search
    corpus = " ".join(
        (item.claim + " " + item.evidence_snippet).lower() for item in evidence
    )

    gaps: list[ResearchGap] = []
    seen_topics: set[str] = set()  # deduplicate by topic label

    for qt in sorted(question_topics):  # sorted for determinism
        subtopics = subtopic_map.get(qt, [])
        for topic_label, keywords, priority, description, rationale in subtopics:
            if topic_label in seen_topics:
                continue
            if not _corpus_has_coverage(corpus, keywords):
                seen_topics.add(topic_label)
                gaps.append(ResearchGap(
                    gap_id="",        # assigned below
                    topic=topic_label,
                    priority=priority,  # type: ignore[arg-type]
                    description=description,
                    rationale=rationale,
                ))

    # Sort: high → medium → low, then alphabetically by topic within each tier
    _PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
    gaps.sort(key=lambda g: (_PRIORITY_ORDER[g.priority], g.topic))

    return _assign_gap_ids(gaps)


def _profile_to_subtopic_map(
    profile: "DomainProfile",
) -> dict[str, list[tuple[str, set[str], str, str, str]]]:
    """Convert a profile's ``research_gap_checks`` into the internal tuple format."""
    result: dict[str, list[tuple[str, set[str], str, str, str]]] = {}
    for topic, checks in profile.research_gap_checks.items():
        result[topic] = [
            (
                check.topic,
                set(check.keywords),
                check.priority,
                check.description,
                check.rationale,
            )
            for check in checks
        ]
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _corpus_has_coverage(corpus: str, keywords: set[str]) -> bool:
    """Return True if ANY keyword appears in the corpus."""
    return any(kw in corpus for kw in keywords)


def _assign_gap_ids(gaps: list[ResearchGap]) -> list[ResearchGap]:
    return [
        g.model_copy(update={"gap_id": f"G{i:03d}"})
        for i, g in enumerate(gaps, start=1)
    ]
