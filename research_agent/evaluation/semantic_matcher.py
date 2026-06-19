"""Semantic matching layer for benchmark scoring (J3.1b).

Matching pipeline
-----------------
1. Exact substring        → similarity 1.0, confidence HIGH  → accept
2. Extra synonyms         → similarity 0.95, confidence HIGH → accept
3. Registry synonym       → similarity 0.95, confidence HIGH → accept*
4. Token Jaccard overlap  → similarity computed:
     HIGH   ≥ 0.92        → accept*
     MEDIUM ≥ 0.85        → accept if not blocked by anti-synonym
     LOW    < 0.85        → reject

*Anti-synonym check applied to all non-exact matches: if the matched phrase
 contains any anti-synonym for the expected term → reject ("anti_synonym").

must_not_include terms always use exact matching (J3.1a.6 — never relaxed).

Public API
----------
score_term_coverage(terms, answer_text, *, alternatives, threshold) → list[SemanticMatch]
semantic_match(expected, answer_text, *, threshold, extra_synonyms)  → SemanticMatch
compute_match_stats(matches)                                          → dict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Registry (J3.1b.2 / J3.1b.3)
# ---------------------------------------------------------------------------
# Each entry defines a canonical term, its explicit synonyms, and a list of
# anti-synonyms that MUST NOT satisfy the canonical term even if they produce
# a similarity score above the acceptance threshold.
#
# YAML-equivalent representation:
#
#   economy of scale:
#     synonyms: [economies of scale, economics-of-scale, scale advantage, ...]
#     anti_synonyms: [learning rate, interest rate, foak, discount rate]
#
#   load following:
#     synonyms: [load-following, grid flexibility, operational flexibility, ...]
#     anti_synonyms: [baseload, base load, baseload power, continuous operation]

@dataclass(frozen=True)
class RegistryEntry:
    """One entry in the domain synonym registry."""

    canonical: str
    synonyms: frozenset[str]        # HIGH-confidence matches (sim = 0.95)
    anti_synonyms: frozenset[str]   # blocked even at HIGH similarity


_REGISTRY: list[RegistryEntry] = [

    # ------------------------------------------------------------------ SMR

    RegistryEntry(
        canonical="economy of scale",
        synonyms=frozenset({
            "economies of scale", "economics-of-scale", "economics of scale",
            "scale advantage", "scale economics", "scale economy",
            "diseconomy of scale", "diseconomies of scale",
            "economy-of-scale", "scale disadvantage", "scale effect",
        }),
        anti_synonyms=frozenset({
            # Learning rate / FOAK / NOAK are related but NOT synonyms for
            # economy-of-scale — they belong to the nth-of-a-kind cost curve
            "learning rate", "foak", "noak",
            "first-of-a-kind", "nth-of-a-kind",
            "interest rate", "discount rate",
        }),
    ),

    RegistryEntry(
        canonical="load following",
        synonyms=frozenset({
            "load-following", "load follow", "load-follow",
            "grid flexibility", "grid-flexibility",
            "operational flexibility", "reactor flexibility",
            "power-following", "flexible operation", "flexible operations",
            "flexible dispatch", "dispatchable",
            "variable output", "ramping", "ramp capability", "ramp rate",
            "load-following capability",
        }),
        anti_synonyms=frozenset({
            # Baseload is the OPPOSITE of load-following
            "baseload", "base load", "baseload power", "base-load",
            "continuous operation", "constant output", "flat output",
        }),
    ),

    RegistryEntry(
        canonical="first-of-a-kind",
        synonyms=frozenset({
            "foak", "first of a kind", "first of kind", "first-of-kind",
        }),
        anti_synonyms=frozenset({
            "noak", "nth-of-a-kind", "nth of a kind",
        }),
    ),

    RegistryEntry(
        canonical="nth-of-a-kind",
        synonyms=frozenset({
            "noak", "nth of a kind", "nth-of-kind",
            "learning rate", "nth unit", "serial production learning",
        }),
        anti_synonyms=frozenset({
            "foak", "first-of-a-kind", "first of a kind",
        }),
    ),

    RegistryEntry(
        canonical="levelized cost",
        synonyms=frozenset({
            "levelized cost of electricity", "lcoe",
            "cost of electricity", "cost of energy", "lcos",
        }),
        anti_synonyms=frozenset({
            "overnight cost", "capital cost", "capex",
        }),
    ),

    RegistryEntry(
        canonical="overnight capital cost",
        synonyms=frozenset({
            "overnight cost", "capital cost", "construction capital",
            "capex", "capital expenditure", "overnight construction cost",
        }),
        anti_synonyms=frozenset({
            "levelized cost", "lcoe", "operating cost", "opex",
        }),
    ),

    RegistryEntry(
        canonical="commercial operation",
        synonyms=frozenset({
            "commercial operations", "commercial service",
            "grid connection", "operational startup", "first power",
            "commercial startup", "enter service", "enter commercial operation",
        }),
        anti_synonyms=frozenset({
            "construction start", "first concrete", "groundbreaking",
        }),
    ),

    RegistryEntry(
        canonical="factory fabrication",
        synonyms=frozenset({
            "factory production", "modular construction",
            "factory manufacturing", "serial production", "modular fabrication",
            "factory built", "factory-built", "off-site fabrication",
            "shop fabrication", "modular build", "factory assembly",
            "factory-fabricated",
        }),
        anti_synonyms=frozenset({
            "on-site construction", "field construction", "bespoke construction",
        }),
    ),

    RegistryEntry(
        canonical="construction duration",
        synonyms=frozenset({
            "construction period", "build time", "construction time",
            "construction schedule", "build schedule", "construction timeline",
        }),
        anti_synonyms=frozenset(),
    ),

    RegistryEntry(
        canonical="passive safety",
        synonyms=frozenset({
            "passive cooling", "passive shutdown",
            "passive safety system", "decay heat removal",
            "gravity-fed cooling", "natural circulation",
        }),
        anti_synonyms=frozenset({
            "active safety", "active cooling", "active shutdown",
        }),
    ),

    RegistryEntry(
        canonical="haleu",
        synonyms=frozenset({
            "high-assay low-enriched uranium",
            "high assay low enriched uranium", "haleu fuel",
            "enriched uranium supply", "fuel enrichment",
        }),
        anti_synonyms=frozenset({
            "natural uranium", "low-enriched uranium", "leu",
            "depleted uranium",
        }),
    ),

    # --------------------------------------------------------- AI Data-center

    RegistryEntry(
        canonical="liquid cooling",
        synonyms=frozenset({
            "direct liquid cooling", "dlc",
            "water cooling", "liquid-cooled", "water-cooled",
            "coolant distribution", "cdu", "liquid heat removal",
        }),
        anti_synonyms=frozenset({
            "air cooling", "air-cooled", "free cooling", "adiabatic cooling",
        }),
    ),

    RegistryEntry(
        canonical="rack power",
        synonyms=frozenset({
            "rack-level power", "total rack power",
            "rack power consumption", "per-rack power", "rack power draw",
        }),
        anti_synonyms=frozenset({
            "chip power", "gpu power", "component power",
        }),
    ),

    RegistryEntry(
        canonical="nvlink",
        synonyms=frozenset({
            "nv-link", "nvlink switch", "nvlink fabric",
            "nvlink bandwidth", "nvlink interconnect",
        }),
        anti_synonyms=frozenset({
            "infiniband", "ethernet", "pcie", "nvme",
        }),
    ),

    RegistryEntry(
        canonical="hbm",
        synonyms=frozenset({
            "high bandwidth memory", "hbm2", "hbm3",
            "hbm3e",
        }),
        anti_synonyms=frozenset({
            "gddr", "gddr6", "gddr6x", "lpddr", "ddr5",
        }),
    ),

    RegistryEntry(
        canonical="power distribution",
        synonyms=frozenset({
            "power delivery", "pdu",
            "power distribution unit", "busway", "busbar",
        }),
        anti_synonyms=frozenset(),
    ),
]

# ---------------------------------------------------------------------------
# Index structures for O(1) lookup
# ---------------------------------------------------------------------------

# canonical → entry
_CANONICAL_LOOKUP: dict[str, RegistryEntry] = {
    e.canonical.lower(): e for e in _REGISTRY
}

# any synonym (including canonical) → canonical
_TERM_TO_CANONICAL: dict[str, str] = {}
for _entry in _REGISTRY:
    _TERM_TO_CANONICAL[_entry.canonical.lower()] = _entry.canonical.lower()
    for _syn in _entry.synonyms:
        _TERM_TO_CANONICAL[_syn.lower()] = _entry.canonical.lower()


def _get_entry(term: str) -> RegistryEntry | None:
    """Return the RegistryEntry for *term*, or None if not registered."""
    canonical = _TERM_TO_CANONICAL.get(term.lower())
    return _CANONICAL_LOOKUP.get(canonical) if canonical else None


# ---------------------------------------------------------------------------
# Confidence bands (J3.1b.1 / J3.1b.5)
# ---------------------------------------------------------------------------

_SIM_HIGH: float = 0.92    # HIGH  → accept (anti-synonym check still applied)
_SIM_MEDIUM: float = 0.85  # MEDIUM → accept unless anti-synonym blocks it
#                            LOW    → sim < 0.85 → always reject


def _confidence_band(similarity: float) -> str:
    if similarity >= _SIM_HIGH:
        return "HIGH"
    if similarity >= _SIM_MEDIUM:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "of", "to", "for", "is", "are",
    "was", "were", "be", "been", "by", "on", "at", "from", "into", "that",
    "this", "with", "as", "its", "their", "our",
})


def _tokens(text: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def _token_jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _anti_synonym_blocked(
    expected_lower: str,
    matched_phrase: str,
    extra_synonyms: Sequence[str],
) -> bool:
    """Return True if *matched_phrase* is blocked by the anti-synonym registry.

    Checks both the registry entry for *expected_lower* and any extra synonyms
    that happen to be registered terms.
    """
    phrase_lower = matched_phrase.lower()
    entry = _get_entry(expected_lower)
    if entry:
        if any(anti.lower() in phrase_lower for anti in entry.anti_synonyms):
            return True
    # Also check from the perspective of extras that resolve to a canonical
    for alt in extra_synonyms:
        alt_entry = _get_entry(alt)
        if alt_entry and any(anti.lower() in phrase_lower for anti in alt_entry.anti_synonyms):
            return True
    return False


# ---------------------------------------------------------------------------
# SemanticMatch (J3.1b.1 / J3.1b.4 / J3.1b.6)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SemanticMatch:
    """Result of matching one expected term against answer text.

    Fields
    ------
    expected      : original must_include term
    matched_phrase: phrase found in the answer (empty if no match)
    similarity    : 0.0–1.0 (raw score before confidence/anti-synonym gate)
    matched       : final acceptance decision (False if LOW or anti_synonym blocked)
    match_type    : "exact" | "synonym" | "token_overlap" | "none"
    confidence    : "HIGH" | "MEDIUM" | "LOW" | "NONE"
    reason        : explanation — why accepted or why rejected
    """

    expected: str
    matched_phrase: str
    similarity: float
    matched: bool
    match_type: str
    confidence: str = "NONE"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "expected": self.expected,
            "matched_phrase": self.matched_phrase,
            "similarity": round(self.similarity, 3),
            "matched": self.matched,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Core match function (J3.1b)
# ---------------------------------------------------------------------------

def semantic_match(
    expected: str,
    answer_text: str,
    *,
    threshold: float = 0.85,
    extra_synonyms: Sequence[str] = (),
) -> SemanticMatch:
    """Return a :class:`SemanticMatch` for *expected* term against *answer_text*.

    Matching pipeline
    -----------------
    1. Exact substring → confidence HIGH, auto-accept
    2. extra_synonyms exact substrings → confidence HIGH, anti-synonym check
    3. Registry synonym exact substring → confidence HIGH, anti-synonym check
    4. Token Jaccard window → confidence from similarity band, anti-synonym check

    Parameters
    ----------
    threshold:
        Minimum similarity floor.  Defaults to 0.85 (MEDIUM band).
        Values below MEDIUM band are always rejected regardless of threshold.
    extra_synonyms:
        Additional phrases to try (e.g. benchmark acceptable_alternatives).
    """
    answer_lower = answer_text.lower()
    exp_lower = expected.lower()

    # 1. Exact substring — always HIGH, no anti-synonym needed
    if exp_lower in answer_lower:
        return SemanticMatch(
            expected, expected, 1.0, True, "exact",
            "HIGH", "exact_substring_match",
        )

    # 2. Extra synonyms (acceptable_alternatives) — exact substring
    for alt in extra_synonyms:
        alt_lower = alt.lower()
        if alt_lower in answer_lower:
            blocked = _anti_synonym_blocked(exp_lower, alt, extra_synonyms)
            if blocked:
                return SemanticMatch(
                    expected, alt, 0.95, False, "synonym",
                    "HIGH", "anti_synonym",
                )
            return SemanticMatch(
                expected, alt, 0.95, 0.95 >= threshold, "synonym",
                "HIGH", "extra_synonym_match",
            )

    # 3. Registry synonym — exact substring
    entry = _get_entry(exp_lower)
    if entry:
        candidates = sorted(
            (s for s in entry.synonyms if s.lower() != exp_lower),
            key=len,
            reverse=True,
        )
        for syn in candidates:
            if syn.lower() in answer_lower:
                blocked = _anti_synonym_blocked(exp_lower, syn, extra_synonyms)
                if blocked:
                    return SemanticMatch(
                        expected, syn, 0.95, False, "synonym",
                        "HIGH", "anti_synonym",
                    )
                return SemanticMatch(
                    expected, syn, 0.95, 0.95 >= threshold, "synonym",
                    "HIGH", "synonym_registry_match",
                )

    # 4. Token Jaccard across answer windows (sentence-level)
    windows = [w.strip() for w in re.split(r"[.!?;]|\s{3,}", answer_text) if w.strip()]
    best_sim = 0.0
    best_window = ""
    for window in windows:
        sim = _token_jaccard(expected, window)
        if sim > best_sim:
            best_sim = sim
            best_window = window[:120]

    if best_sim < threshold:
        conf = _confidence_band(best_sim)
        return SemanticMatch(
            expected, best_window or "", best_sim, False, "none",
            conf, "low_confidence",
        )

    conf = _confidence_band(best_sim)

    # LOW band always rejects
    if conf == "LOW":
        return SemanticMatch(
            expected, best_window, best_sim, False, "token_overlap",
            "LOW", "low_confidence",
        )

    # MEDIUM / HIGH — check anti-synonym
    blocked = _anti_synonym_blocked(exp_lower, best_window, extra_synonyms)
    if blocked:
        return SemanticMatch(
            expected, best_window, best_sim, False, "token_overlap",
            conf, "anti_synonym",
        )

    reason = "token_overlap_high" if conf == "HIGH" else "token_overlap_medium"
    return SemanticMatch(
        expected, best_window, best_sim, True, "token_overlap",
        conf, reason,
    )


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_term_coverage(
    terms: list[str],
    answer_text: str,
    *,
    alternatives: list[str] | None = None,
    threshold: float = 0.85,
) -> list[SemanticMatch]:
    """Return a :class:`SemanticMatch` for every term in *terms*.

    Parameters
    ----------
    terms:
        ``must_include`` term list from the benchmark question.
    answer_text:
        Flattened answer string to search through.
    alternatives:
        ``acceptable_alternatives`` — tried as extra synonyms for every term.
    threshold:
        Minimum similarity floor.  Defaults to 0.85 (MEDIUM band).
    """
    alts = alternatives or []
    return [
        semantic_match(term, answer_text, threshold=threshold, extra_synonyms=alts)
        for term in terms
    ]


# ---------------------------------------------------------------------------
# Match statistics (J3.1b.7)
# ---------------------------------------------------------------------------

def compute_match_stats(matches: list[SemanticMatch]) -> dict:
    """Return J3.1b.7 trace statistics for a set of SemanticMatches.

    Returns
    -------
    dict with keys:
        total_terms, exact_matches, synonym_matches, semantic_matches,
        rejected_semantic_matches, exact_matches_found (alias),
        semantic_matches_found (alias), unmatched, semantic_match_rate
    """
    total = len(matches)
    exact = sum(1 for m in matches if m.match_type == "exact" and m.matched)
    synonym = sum(1 for m in matches if m.match_type == "synonym" and m.matched)
    overlap = sum(1 for m in matches if m.match_type == "token_overlap" and m.matched)
    rejected = sum(
        1 for m in matches
        if not m.matched and m.match_type in ("synonym", "token_overlap")
    )
    hit = sum(1 for m in matches if m.matched)
    anti_blocked = sum(1 for m in matches if m.reason == "anti_synonym")

    # J3.1b.7 keys
    return {
        "total_terms": total,
        # J3.1b.7 statistics
        "exact_matches": exact,
        "synonym_matches": synonym,
        "semantic_matches": overlap,
        "rejected_semantic_matches": rejected,
        "anti_synonym_blocks": anti_blocked,
        "unmatched": total - hit,
        # backward-compat aliases used by scorer.py
        "exact_matches_found": exact,
        "semantic_matches_found": synonym + overlap,
        "semantic_match_rate": round((synonym + overlap) / total, 3) if total else 0.0,
    }
