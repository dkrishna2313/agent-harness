"""Contradiction detection between extracted evidence items.

Architecture (J1.6)
-------------------
Detection runs in two explicit phases:

Phase 1 – Evidence Enrichment (``enrich_evidence_items``)
    Each evidence item is annotated with:
        entity       – named product/reactor/component, e.g. "GB200 NVL72"
        entity_type  – category: "rack_system", "reactor", "power_subsystem", …
        scope        – measurement scale: "rack", "unit", "component", …

    Callers (agent.py) run this phase **before** storing evidence in the memo,
    so the enriched fields appear in every downstream output (trace, markdown).

Phase 2 – Pairwise Contradiction Detection (``detect_contradictions``)
    For every pair (A, B) of enriched evidence items, three gates must all pass
    before a Contradiction is emitted:

        Gate 1 – Entity compatibility
            If both items name a specific entity and those names differ → suppress.
            "Reactor Alpha" vs "Reactor Beta" is not a contradiction.
            Unknown entity on either side → gate passes (conservative).

        Gate 2 – Scope compatibility
            ``_INCOMPATIBLE_SCOPE_PAIRS`` defines which scale mismatches are
            physically impossible to contradict.
            component ↔ rack, unit ↔ fleet, etc. → suppress.
            "unknown" scope on either side → gate passes.

        Gate 3 – Metric / lifecycle compatibility
            Numeric: same unit + metric type, values differ > 20%.
            Categorical: opposing exclusive terms, same metric class.
            Lifecycle: different milestone classes → suppress
            (construction_approval ≠ commercial_operation).

    Suppressed pairs are appended to ``out_suppressed`` when the caller passes
    an empty list, and are surfaced in the trace.

Scope vocabulary (J1.6.2)
--------------------------
    Hardware (fine → coarse):
        component   – shelf, PSU, single card/tray
        subsystem   – NVSwitch fabric, cooling loop, power plane
        tray        – compute tray, cooling tray
        node        – single GPU node / server
        rack        – full rack (NVL72, MGX, …)
        cluster     – multiple racks / pod
        facility    – data centre building

    Nuclear (fine → coarse):
        unit        – single reactor unit / block
        reactor     – synonym for unit (either may appear in text)
        site        – plant site (may host multiple units)
        fleet       – all units of one operator / country
        province    – sub-national region
        country     – national aggregate

Lifecycle milestone classes (J1.6.5)
-------------------------------------
    construction_approval, construction_start, civil_works,
    equipment_installation, fuel_loading, commercial_operation,
    fleet_completion
    → Different classes represent progression milestones, never contradictions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .profile import DomainProfile
from .schemas import Contradiction, EvidenceItem, SourceQuality, SuppressedComparison

# ---------------------------------------------------------------------------
# Unit → topic mapping (for Contradiction.topic)
# ---------------------------------------------------------------------------

_UNIT_KEYWORDS: frozenset[str] = frozenset({
    "kw", "mw", "gw", "w", "gpu", "tb", "gb", "°c", "c", "rack", "node", "%", "x",
})

_UNIT_TO_TOPIC: dict[str, str] = {
    "kw": "rack power",
    "mw": "rack power",
    "gw": "rack power",
    "°c": "cooling temperature",
    "c": "cooling temperature",
    "gpu": "gpu count",
}

# ---------------------------------------------------------------------------
# Categorical exclusive pairs (term_set_A, term_set_B)
# A claim matching A and another matching B are in opposition.
# ---------------------------------------------------------------------------

# J6.5b – Known distinct product model pairs that cannot contradict each other.
# Each entry is (family_A_keywords, family_B_keywords).  When a claim contains
# keywords from family_A and the other claim contains keywords from family_B
# (or vice versa), the comparison is suppressed.
_DISTINCT_PRODUCT_PAIRS: list[tuple[frozenset[str], frozenset[str]]] = [
    # NVL form-factor size mismatch (same generation, different chassis sizes)
    (frozenset({"nvl8"}),  frozenset({"nvl36"})),
    (frozenset({"nvl8"}),  frozenset({"nvl72"})),
    (frozenset({"nvl36"}), frozenset({"nvl72"})),
    # System-class mismatch (DGX server vs HGX baseboard)
    (frozenset({"dgx"}), frozenset({"hgx"})),
    # Generation progression: Blackwell → Rubin
    (frozenset({"gb200", "blackwell", "b200", "b300", "b100"}),
     frozenset({"rubin", "vera rubin", "r100"})),
    # Generation progression: Hopper → Blackwell
    (frozenset({"hopper", "h100", "h200"}),
     frozenset({"gb200", "blackwell", "b200", "b300"})),
    # Generation progression: Hopper → Rubin
    (frozenset({"hopper", "h100", "h200"}),
     frozenset({"rubin", "vera rubin"})),
]

# Generation families — used to classify suppression reason as generation_progression
_GENERATION_FAMILIES: list[frozenset[str]] = [
    frozenset({"hopper", "h100", "h200"}),
    frozenset({"gb200", "blackwell", "b200", "b300", "b100"}),
    frozenset({"rubin", "vera rubin", "r100"}),
]


def _product_compatibility_check(claim_a: str, claim_b: str) -> tuple[bool, str]:
    """Return (is_compatible, suppression_reason) for two claims.

    suppression_reason is "" when compatible.
    suppression_reason is "generation_progression" for cross-generation comparisons.
    suppression_reason is "product_mismatch" for same-gen different-product comparisons.
    """
    t_a = claim_a.lower()
    t_b = claim_b.lower()

    for family_a, family_b in _DISTINCT_PRODUCT_PAIRS:
        has_a_in_a = bool(family_a & {kw for kw in family_a if kw in t_a})
        has_b_in_b = bool(family_b & {kw for kw in family_b if kw in t_b})
        has_b_in_a = bool(family_b & {kw for kw in family_b if kw in t_a})
        has_a_in_b = bool(family_a & {kw for kw in family_a if kw in t_b})

        if (has_a_in_a and has_b_in_b) or (has_b_in_a and has_a_in_b):
            combined = family_a | family_b
            is_gen = any(combined & g for g in _GENERATION_FAMILIES)
            reason = "generation_progression" if is_gen else "product_mismatch"
            return False, reason

    return True, ""


EXCLUSIVE_PAIRS: list[tuple[set[str], set[str]]] = [
    (
        {"air cool", "air-cool", "air cooled", "air-cooled"},
        {"liquid cool", "liquid-cool", "water cool", "dlc", "direct liquid"},
    ),
    ({"2025"}, {"2026"}),
    ({"2026"}, {"2027"}),
    ({"2027"}, {"2028"}),
    ({"2028"}, {"2029"}),
    ({"2029"}, {"2030"}),
    ({"2030"}, {"2031"}),
    ({"nvl72"}, {"nvl36"}),
    ({"72 gpu", "72-gpu"}, {"144 gpu", "144-gpu"}),
    ({"single-phase", "single phase"}, {"two-phase", "two phase"}),
]

# ---------------------------------------------------------------------------
# J1.6.1 Entity extraction
# ---------------------------------------------------------------------------
# Patterns are tried in order; first match wins.
# Each entry: (compiled_re, entity_name_template, entity_type)
# The template may use group references: {0} = whole match, {1} = group 1, …
# ---------------------------------------------------------------------------

_ENTITY_RULES: list[tuple[re.Pattern[str], str, str]] = [
    # NVIDIA GB200 NVL-series (e.g. "GB200 NVL72", "GB200 NVL36")
    (re.compile(r"\bgb200\s+nvl(\d+)\b", re.IGNORECASE), "GB200 NVL{1}", "rack_system"),
    # Generic NVL-series
    (re.compile(r"\bnvl(\d+)\b", re.IGNORECASE), "NVL{1}", "rack_system"),
    # Blackwell / Hopper rack references
    (re.compile(r"\bblackwell\s+(?:ultra\s+)?(?:nvl\d+\s+)?(?:rack|system)\b", re.IGNORECASE), "Blackwell Rack", "rack_system"),
    (re.compile(r"\bhopper\s+(?:nvl\d+\s+)?(?:rack|system)\b", re.IGNORECASE), "Hopper Rack", "rack_system"),
    # HGX / DGX references
    (re.compile(r"\b(hgx\s*[a-z]?\d{2,3})\b", re.IGNORECASE), "{1}", "rack_system"),
    (re.compile(r"\b(dgx\s*[a-z]?\d+)\b", re.IGNORECASE), "{1}", "rack_system"),
    # BWRX-300 and similar alphanumeric reactor model codes
    (re.compile(r"\b(bwrx-?\d+)\b", re.IGNORECASE), "{1_upper}", "reactor"),
    (re.compile(r"\b(xe-\d+[a-z]*)\b", re.IGNORECASE), "{1_upper}", "reactor"),
    (re.compile(r"\b(smr-\d+[a-z]*)\b", re.IGNORECASE), "{1_upper}", "reactor"),
    # NuScale SMR
    (re.compile(r"\bnuscale\s+(?:smr|power\s+module|voygr)\b", re.IGNORECASE), "NuScale SMR", "reactor"),
    # Named reactor pattern: "Reactor Alpha", "Reactor 1", "Reactor Unit 2"
    (re.compile(r"\breactor\s+(?:unit\s+)?([a-z][a-z0-9]*|\d+)\b", re.IGNORECASE), "Reactor {1_title}", "reactor"),
    # "Unit N" / "Block N" nuclear references
    (re.compile(r"\b(?:unit|block)\s+(\d+)\b", re.IGNORECASE), "Unit {1}", "reactor"),
    # Power shelf
    (re.compile(r"\bpower\s+shelf\b", re.IGNORECASE), "Power Shelf", "power_subsystem"),
    # PSU / power supply
    (re.compile(r"\b(?:psu|power\s+supply(?:\s+unit)?)\b", re.IGNORECASE), "PSU", "power_subsystem"),
    # Compute / cooling tray
    (re.compile(r"\bcompute\s+tray\b", re.IGNORECASE), "Compute Tray", "compute_subsystem"),
    (re.compile(r"\bcooling\s+tray\b", re.IGNORECASE), "Cooling Tray", "cooling_subsystem"),
    # Generic rack (weak — only if no stronger match above)
    (re.compile(r"\bthe\s+rack\b", re.IGNORECASE), "Rack", "rack_system"),
]


def _expand_template(template: str, m: re.Match[str]) -> str:
    """Expand a simple {N} / {N_upper} / {N_title} template against *m*."""
    result = template
    for i in range(1, 10):
        if m.lastindex is None or i > m.lastindex:
            break
        val = m.group(i) or ""
        result = result.replace(f"{{{i}}}", val)
        result = result.replace(f"{{{i}_upper}}", val.upper())
        result = result.replace(f"{{{i}_title}}", val.title())
    return result


def _extract_entity(text: str) -> tuple[str, str]:
    """Return (entity_name, entity_type) for the primary named entity in *text*.

    The original case of *text* is preserved in the returned name.
    Returns ``("", "")`` when no entity is recognised.
    """
    for pattern, template, entity_type in _ENTITY_RULES:
        m = pattern.search(text)
        if m:
            name = _expand_template(template, m)
            return name, entity_type
    return "", ""


# ---------------------------------------------------------------------------
# J1.6.2 Scope extraction
# ---------------------------------------------------------------------------

# Sub-rack hardware components (most specific — checked first)
_COMPONENT_TERMS: tuple[str, ...] = (
    "power shelf",
    " psu",
    "power supply unit",
    "power supply",
    "module bay",
    " drawer",
    "power module",
    "network module",
    # Generic shelf references — catch "per shelf", "each shelf", "1U shelf",
    # "shelf module" even when "power" doesn't appear before the noun.
    " shelf",
    "shelf module",
)

# Tray-level: compute/power/cooling trays sit above individual shelf components
_TRAY_TERMS: tuple[str, ...] = (
    "compute tray",
    "power tray",
    "cooling tray",
    " tray",
    "mgx tray",
    "mgx module",
    "nvswitch tray",
)

# Rack-level products / references
_RACK_TERMS: tuple[str, ...] = (
    " rack",
    "nvl72",
    "nvl36",
    "gb200 nvl",
    "mgx rack",
    "mgx system",
    "server rack",
    "compute rack",
)

# Single GPU node / server
_NODE_TERMS: tuple[str, ...] = (
    " node",
    "server node",
    "gpu node",
    "compute node",
    "per server",     # J6.5b: "per server" means node-level measurement
    "each server",    # J6.5b: "each server" is node-level
)

# Cluster / data-centre aggregate
_CLUSTER_TERMS: tuple[str, ...] = (
    "cluster",
    "data center",
    "datacenter",
    "data-center",
    " pod ",
    "row of rack",
    "row of server",
)

# Facility / building
_FACILITY_TERMS: tuple[str, ...] = (
    "facility",
    "campus",
    "data centre",
    "building",
)

# Single nuclear reactor unit
_UNIT_RE = re.compile(
    r"\breactor\s*(?:unit)?\b|\bunit\s+\d\b|\bblock\s+\d\b|\bsmr\s+unit\b",
    re.IGNORECASE,
)

# Nuclear plant site (may host multiple units)
_SITE_TERMS: tuple[str, ...] = (
    "reactor site",
    "plant site",
    "construction site",
    "nuclear site",
)

# Fleet / national aggregate
_FLEET_TERMS: tuple[str, ...] = (
    " fleet",
    "fleet-wide",
    "multiple reactor",
    "all reactor",
    "reactor fleet",
)

_COUNTRY_RE = re.compile(
    r"\bnational\b|\bnation-wide\b|\bcountry.?wide\b|\bprovince\b|\bregional\b",
    re.IGNORECASE,
)


def _extract_scope(text: str) -> str:
    """Return the physical measurement scope tag for *text* (lowercased input).

    Returns ``"unknown"`` when no scope keyword is found.  Checks finer-grained
    scopes first so "power shelf in the rack" → "component", not "rack".
    """
    t = text.lower()

    # ---- Hardware scopes (fine → coarse) ----
    # J6.5b: chip/die is a component-level scope (finer than shelf/PSU)
    if re.search(r"\bgpu\s+chip\b|\bchip\s+(?:power|tdp|die)\b|\bper\s+(?:gpu|chip|die)\b", t):
        return "component"
    if any(kw in t for kw in _COMPONENT_TERMS):
        return "component"
    if any(kw in t for kw in _TRAY_TERMS):
        return "tray"
    if re.search(r"\bgpu\b|\bcpu\b|\bsoc\b|\bnpu\b", t):
        return "node"
    if any(kw in t for kw in _RACK_TERMS):
        return "rack"
    if any(kw in t for kw in _NODE_TERMS):
        return "node"
    if any(kw in t for kw in _CLUSTER_TERMS):
        return "cluster"
    if any(kw in t for kw in _FACILITY_TERMS):
        return "facility"

    # ---- Nuclear scopes (fine → coarse) ----
    if _UNIT_RE.search(text):
        return "unit"
    if any(kw in t for kw in _SITE_TERMS):
        return "site"
    if any(kw in t for kw in _FLEET_TERMS):
        return "fleet"
    if _COUNTRY_RE.search(text):
        return "country"

    return "unknown"


def _entity_type_to_scope(entity_type: str) -> str:
    """Map an entity_type to its natural scope when text-based extraction fails."""
    return {
        "rack_system":        "rack",
        "reactor":            "unit",
        "power_subsystem":    "component",
        "compute_subsystem":  "component",
        "cooling_subsystem":  "component",
    }.get(entity_type, "unknown")


# ---------------------------------------------------------------------------
# J1.6.5 Scope compatibility matrix
# ---------------------------------------------------------------------------

_INCOMPATIBLE_SCOPE_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({"component", "rack"}),
    frozenset({"component", "tray"}),   # tray is a mid-level aggregate
    frozenset({"component", "node"}),
    frozenset({"component", "cluster"}),
    frozenset({"component", "facility"}),
    frozenset({"tray", "rack"}),
    frozenset({"tray", "cluster"}),
    frozenset({"tray", "facility"}),
    frozenset({"node", "rack"}),
    frozenset({"node", "cluster"}),
    frozenset({"node", "facility"}),
    # J6.5a – rack vs campus-scale scopes (e.g. 132 kW/rack vs 100 MW campus)
    frozenset({"rack", "cluster"}),
    frozenset({"rack", "facility"}),
    frozenset({"cluster", "facility"}),
    frozenset({"unit", "fleet"}),
    frozenset({"unit", "country"}),
    frozenset({"unit", "province"}),
    frozenset({"site", "fleet"}),       # plant site vs national fleet
    frozenset({"reactor", "fleet"}),    # "reactor" as synonym for "unit"
    frozenset({"reactor", "country"}),
})


def _scopes_compatible(scope_a: str, scope_b: str) -> bool:
    """Return True when the two scopes can be meaningfully compared.

    ``"unknown"`` on either side always passes (conservative: don't suppress
    when we cannot determine the scope).
    """
    if scope_a == "unknown" or scope_b == "unknown":
        return True
    if scope_a == scope_b:
        return True
    # Treat "reactor" and "unit" as synonyms
    norm_a = "unit" if scope_a == "reactor" else scope_a
    norm_b = "unit" if scope_b == "reactor" else scope_b
    if norm_a == norm_b:
        return True
    return frozenset({norm_a, norm_b}) not in _INCOMPATIBLE_SCOPE_PAIRS


# ---------------------------------------------------------------------------
# J1.6.1 Entity compatibility
# ---------------------------------------------------------------------------

def _entity_names_compatible(name_a: str, name_b: str) -> bool:
    """Return True when two entity names can represent the same subject.

    Rules:
    * Empty string on either side → compatible (unknown entity → don't suppress).
    * Exact match (case-insensitive) → compatible.
    * One is a prefix of the other (e.g. "NVL72" vs "GB200 NVL72") → compatible.
    * Otherwise → incompatible (claims are about different named things).
    """
    if not name_a or not name_b:
        return True
    a = name_a.lower().strip()
    b = name_b.lower().strip()
    if a == b:
        return True
    if a in b or b in a:
        return True
    return False


# ---------------------------------------------------------------------------
# Evidence enrichment (public API for agent.py)
# ---------------------------------------------------------------------------

def enrich_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    """Populate ``entity``, ``entity_type``, and ``scope`` on each item.

    Items that already have non-empty fields are left unchanged.
    Returns a new list — the originals are not mutated.
    """
    enriched = []
    for item in items:
        # Already populated → leave as-is
        if item.entity or item.entity_type or item.scope:
            enriched.append(item)
            continue

        entity_name, entity_type = _extract_entity(item.claim)
        scope = _extract_scope(item.claim)

        # Fall back: derive scope from entity type if text-based extraction failed
        if scope == "unknown" and entity_type:
            scope = _entity_type_to_scope(entity_type)

        enriched.append(item.model_copy(update={
            "entity": entity_name,
            "entity_type": entity_type,
            "scope": scope,
        }))
    return enriched


def build_extraction_stats(items: list[EvidenceItem]) -> dict:
    """Return J1.6.7 extraction statistics for the trace."""
    n = len(items)
    entities = sum(1 for it in items if it.entity)
    scopes   = sum(1 for it in items if it.scope and it.scope != "unknown")
    return {
        "evidence_items":    n,
        "entities_detected": entities,
        "scopes_detected":   scopes,
        "entity_coverage_pct": round(entities / n * 100, 1) if n else 0.0,
        "scope_coverage_pct":  round(scopes  / n * 100, 1) if n else 0.0,
    }


# ---------------------------------------------------------------------------
# Metric-type extraction (J1.4 + J1.6.3/J1.6.4)
# ---------------------------------------------------------------------------

_CONTEXT_WINDOW = 80  # characters examined around a matched value

# ---------------------------------------------------------------------------
# J6.5d – Numeric Semantic Classifier
# ---------------------------------------------------------------------------

# Keywords immediately before a NUMBER UNIT that indicate the number is a
# technology threshold / limit, not a measured rack power value.
_RE_NS_THRESHOLD_BEFORE = re.compile(
    r"\bpower\s+densit\w+\s+above\s*\Z"       # "rack power densities above [N] kW"
    r"|\bdensit\w+\s+above\s*\Z"               # "densities above [N] kW"
    r"|\binadequate\s+(?:for\s+\w+\s+)?above\s*\Z"  # "inadequate ... above [N]"
    r"|\badequate\s+(?:for\s+\w+\s+)?below\s*\Z"    # "adequate ... below [N]"
    r"|\bbecomes?\s+\w+\s+(?:above|below)\s*\Z"     # "becomes inadequate above [N]"
    r"|\beffective\s+(?:up\s+to|below)\s*\Z"
    r"|\bviable\s+(?:up\s+to|below)\s*\Z"
    r"|\bthreshold\s+of\s*\Z"
    r"|\blimit\s+of\s*\Z"
    r"|\bup\s+to\s*\Z",                        # "effective up to [N] kW"
    re.IGNORECASE,
)

# Keywords in the vicinity of a number that mark it as historical/reference context.
_RE_NS_HISTORICAL = re.compile(
    r"\bcloud\s+era\b|\btraditional\b|\blegacy\b|\bconventional\b"
    r"|\bprevious(?:ly)?\b|\bhistoric(?:al(?:ly)?)?\b|\bolder\s+generation\b"
    r"|\bpre.(?:ai|gpu|generative|llm)\b|\bbefore\s+(?:ai|llm|gen\s*ai)\b"
    r"|\btypically\s+operat\w+\b|\bstandard\s+(?:rack|server)\b"
    r"|\b20[01]\d\s+era\b",
    re.IGNORECASE,
)

# "compared to traditional/standard/legacy X" sub-clauses that introduce
# reference numbers that should not be compared against AI-era measurements.
_RE_NS_COMPARISON_CLAUSE = re.compile(
    r"\bcompared\s+to\s+(?:traditional|conventional|standard|legacy|older|previous|typical)\b"
    r"|\bversus\s+(?:traditional|conventional|standard)\b"
    r"|\bunlike\s+(?:traditional|conventional|standard)\b",
    re.IGNORECASE,
)


def _find_value_position(claim: str, value: float, unit: str) -> int | None:
    """Return the character offset of 'value unit' in *claim* (case-insensitive)."""
    val_int = int(value)
    val_str = str(val_int) if value == float(val_int) else str(value)
    for pat in (
        rf"\b{re.escape(val_str)}\s*{re.escape(unit)}\b",
        rf"\b{re.escape(unit)}\s*{re.escape(val_str)}\b",
    ):
        m = re.search(pat, claim, re.IGNORECASE)
        if m:
            return m.start()
    return None


def _classify_value_semantic(claim: str, value: float, unit: str) -> str:
    """Return the semantic role of *value* in *claim* (J6.5d).

    Returns one of:
        primary              – a direct measurement of the subject
        threshold            – a technology limit/boundary, not a rack power figure
        historical_comparison – a reference value for an older era or comparison class
    """
    pos = _find_value_position(claim, value, unit)
    if pos is None:
        return "primary"

    # 120 chars before the value and 80 chars after it
    window_before = claim[max(0, pos - 120):pos]
    window_after = claim[pos:min(len(claim), pos + 80)]

    # Threshold: "rack power densities above N kW", "adequate below N kW", etc.
    if _RE_NS_THRESHOLD_BEFORE.search(window_before[-50:]):
        return "threshold"

    # Historical: "cloud era", "traditional", etc. in the window around the value
    combined = window_before[-80:] + window_after
    if _RE_NS_HISTORICAL.search(combined):
        return "historical_comparison"

    # Comparison sub-clause: "compared to traditional server racks at N kW"
    if _RE_NS_COMPARISON_CLAUSE.search(window_before):
        return "historical_comparison"

    return "primary"


# J6.5b – range extraction for range-vs-average compatibility gate
def _extract_range_for_unit(text: str, unit: str) -> tuple[float, float] | None:
    """Extract [lo, hi] bounds from a range expression like '30–100 kW' in *text*."""
    pattern = re.compile(
        rf"\b(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\+?\s*{re.escape(unit)}\b",
        re.IGNORECASE,
    )
    m = pattern.search(text)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


_RE_RATE = re.compile(
    r"per\s+year|/\s*year|per\s+annum|annually|per\s+month|/\s*month"
    r"|throughput\b|licensing\s+rate|build\s+rate|install\s+rate",
    re.IGNORECASE,
)
_RE_TARGET = re.compile(
    r"\bby\s+20[3-9]\d\b|target\b|goal\b"
    r"|capacity\s+target|capacity\s+goal"
    r"|total\s+capacity|cumulative\s+capacity",
    re.IGNORECASE,
)
_RE_CURRENT = re.compile(
    r"\bcurrent\b|\btoday\b|\bexisting\b|\bpresent\b|\bnow\b|\boperating\b",
    re.IGNORECASE,
)

# J6.5a – temporal kind detector (used by Gate 6 in _check_numeric_conflict)
_RE_FUTURE_PROJECTION = re.compile(
    r"\bby\s+20[3-9]\d\b|\bproject(?:ed|ion)\b|\bforecast\b|\bexpect(?:ed)?\b"
    r"|\bpipeline\b|\bplanned\b|\bproposes?\b|\broadmap\b|\bfuture\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# J6.5c – Comparison context classification
# ---------------------------------------------------------------------------

_RE_CTX_EXAMPLE = re.compile(
    r"\bfor\s+example\b|\bfor\s+instance\b|\be\.g\b|\billustrat\w+\b"
    r"|\bhypothetical\b|\bconsider\s+a\s+(?:scenario|case|deployment)\b"
    r"|\bexample\s+(?:deployment|data.cent\w*|facility|use.?case|rack)\b"
    r"|\bsimple\s+example\b|\bsample\s+(?:deployment|rack)\b"
    r"|\bsuch\s+as\s+(?:a|an)\b",
    re.IGNORECASE,
)
_RE_CTX_REFERENCE = re.compile(
    r"\breference\s+(?:architecture|design|configuration|rack|deployment)\b"
    r"|\bstandard\s+(?:configuration|rack|design|deployment)\b"
    r"|\btypical\s+(?:rack|configuration|data.cent\w*|deployment|design)\b"
    r"|\bcanonical\s+(?:rack|design|deployment)\b"
    r"|\barchitecture\s+(?:calls?\s+for|requires?|specif\w+)\b",
    re.IGNORECASE,
)
_RE_CTX_INDUSTRY_AVG = re.compile(
    r"\bindustry\s+average\b|\baverage\s+(?:across|data.cent\w+|facility|rack)\b"
    r"|\bmost\s+(?:data.cent\w+|facilities|deployments|hyperscalers?)\b"
    r"|\btypically\s+(?:see|require|use|draw|consume|reach)\b"
    r"|\bacross\s+the\s+industry\b|\bgenerally\s+(?:require|draw|consume|see)\b"
    r"|\bindustry.wide\b|\bmarket\s+average\b|\bnorm\s+(?:is|for)\b",
    re.IGNORECASE,
)
_RE_CTX_VENDOR = re.compile(
    r"\bnvidia\s+(?:rates?|claims?|specif\w+|states?|announc\w+|sheet)\b"
    r"|\bvendor\s+(?:spec|claim|rating|data.sheet)\b"
    r"|\bdata\s+sheet\b|\bproduct\s+spec\w*\b|\bmanufacturer\s+(?:spec|claim|rating)\b"
    r"|\bspec(?:ified|ification)\s+(?:calls?\s+for|states?|shows?|gives?)\b",
    re.IGNORECASE,
)


def _classify_comparison_context(claim: str) -> str:
    """Classify the deployment/measurement context of *claim* (J6.5c).

    Returns one of:
        example_deployment    – illustrative / hypothetical figures
        reference_architecture – reference / standard / typical design
        industry_average       – average across deployments, not a single site
        vendor_claim           – product-specific vendor spec or data sheet
        future_projection      – projected, planned, roadmap target
        current_deployment     – operating today, existing installation
        unknown                – insufficient signal
    """
    if _RE_CTX_EXAMPLE.search(claim):
        return "example_deployment"
    if _RE_CTX_REFERENCE.search(claim):
        return "reference_architecture"
    if _RE_CTX_INDUSTRY_AVG.search(claim):
        return "industry_average"
    if _RE_CTX_VENDOR.search(claim):
        return "vendor_claim"
    # Fall back to temporal kind
    kind = _extract_numeric_kind_from_text(claim)
    if kind == "target":
        return "future_projection"
    if kind == "current":
        return "current_deployment"
    return "unknown"


# Pairs of contexts that cannot produce a meaningful contradiction.
# When both contexts are known and form one of these pairs, suppress.
_INCOMPATIBLE_CONTEXT_PAIRS: frozenset[frozenset[str]] = frozenset({
    # Illustrative examples cannot contradict concrete measurements
    frozenset({"example_deployment", "reference_architecture"}),
    frozenset({"example_deployment", "current_deployment"}),
    frozenset({"example_deployment", "future_projection"}),
    frozenset({"example_deployment", "vendor_claim"}),
    frozenset({"example_deployment", "industry_average"}),
    # Vendor spec for one product vs industry-wide average are different populations
    frozenset({"vendor_claim", "industry_average"}),
    # Backstop for current vs future (Gate 6 catches most cases; this catches the rest)
    frozenset({"current_deployment", "future_projection"}),
})


def _extract_numeric_kind_from_text(text: str) -> str:
    """Return the temporal kind of a numeric claim: 'current', 'target', 'rate', or 'unknown'."""
    if _RE_RATE.search(text):
        return "rate"
    if _RE_TARGET.search(text) or _RE_FUTURE_PROJECTION.search(text):
        return "target"
    if _RE_CURRENT.search(text):
        return "current"
    return "unknown"


# ---- Year context patterns -----------------------------------------------

_RE_YEAR_EVENT = re.compile(
    r"\bconference\b|\bsummit\b|\bgtc\b|\bexpo\b|\bevent\b"
    r"|\bwebinar\b|\bsymposium\b|\bannouncement\b|\bshow\b",
    re.IGNORECASE,
)
_RE_YEAR_DEPLOYMENT = re.compile(
    r"commercial\s+operat|\boperational\b|\bonline\b|\bgo\s+live\b"
    r"|\bdeployment\b|\bcommissioning\b|\bcommercial\s+service\b",
    re.IGNORECASE,
)
_RE_YEAR_PRODUCT = re.compile(
    r"\blaunch\b|\brelease\b|\bplatform\b|\barchitecture\b|\bgeneration\b"
    r"|\bavailable\b|\bavailability\b|\bship\b|\bshipping\b|\bproduct\b",
    re.IGNORECASE,
)
_RE_YEAR_CONSTRUCTION = re.compile(
    r"\bconstruction\b|\bground.?break|\bfirst\s+concrete\b|\bsite\s+prep"
    r"|\bapproval\b|\bapproved\b|\bpermit\b|\blicensing\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# J1.6.5 Lifecycle milestone classification
# ---------------------------------------------------------------------------

_MILESTONE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bconstruction\s+approv|\bpermit\b|\blicens\b|\bregulatory\s+approv", re.IGNORECASE),
     "construction_approval"),
    (re.compile(r"\bground.?break|\bfirst\s+concrete\b|\bconstruction\s+start", re.IGNORECASE),
     "construction_start"),
    (re.compile(r"\bcivil\s+works?\b|\bsite\s+prep|\bexcavat", re.IGNORECASE),
     "civil_works"),
    (re.compile(r"\bequipment\s+install|\bmodule\s+install|\bassembl", re.IGNORECASE),
     "equipment_installation"),
    (re.compile(r"\bfuel\s+load|\bnuclear\s+first\s+light\b|\bfirst\s+criticality\b", re.IGNORECASE),
     "fuel_loading"),
    (re.compile(r"commercial\s+operat|\bcommercial\s+service\b|\bcommission|\bgo.?live\b|\bpower\s+generat", re.IGNORECASE),
     "commercial_operation"),
    (re.compile(r"\bfleet\s+(?:complet|deploy|target)|\ball\s+units?\s+(?:online|operat)", re.IGNORECASE),
     "fleet_completion"),
]

# Milestone types that represent progression — comparing across these classes
# is a milestone difference, not a factual contradiction (J1.6.5).
_LIFECYCLE_MILESTONE_TYPES: frozenset[str] = frozenset({
    "construction_approval",
    "construction_start",
    "civil_works",
    "equipment_installation",
    "fuel_loading",
    "commercial_operation",
    "fleet_completion",
    # Legacy names kept for backward compatibility
    "year_construction",
    "year_deployment",
})


def _year_context_type(text: str, year_str: str) -> str:
    """Classify what lifecycle milestone a year reference represents in *text*."""
    idx = text.find(year_str)
    if idx == -1:
        return "year_generic"
    start  = max(0, idx - _CONTEXT_WINDOW)
    end    = min(len(text), idx + len(year_str) + _CONTEXT_WINDOW)
    window = text[start:end]

    # Check full milestone patterns first (more specific)
    for pattern, milestone_type in _MILESTONE_PATTERNS:
        if pattern.search(window):
            return milestone_type

    if _RE_YEAR_EVENT.search(window):
        return "year_event"
    if _RE_YEAR_DEPLOYMENT.search(window):
        return "commercial_operation"
    if _RE_YEAR_PRODUCT.search(window):
        return "year_product"
    if _RE_YEAR_CONSTRUCTION.search(window):
        return "construction_approval"
    return "year_generic"


_SCOPED_POWER_PREFIXES: tuple[str, ...] = (
    "rack_power_",
    "power_shelf_power_",
    "tray_power_",
    "node_power_",
    "gpu_power_",
    "cluster_power_",
    "facility_power_",
    "installed_capacity_",
    "fleet_capacity_",
)


def _is_power_scope_mismatch(type_a: str, type_b: str) -> bool:
    """Return True when both types are scope-prefixed power metrics but differ."""
    a_scoped = any(type_a.startswith(p) for p in _SCOPED_POWER_PREFIXES)
    b_scoped = any(type_b.startswith(p) for p in _SCOPED_POWER_PREFIXES)
    return a_scoped and b_scoped and type_a != type_b


def _metric_types_compatible(type_a: str, type_b: str) -> bool:
    """Return True when the two metric types measure the same thing.

    Blocking rules:
    * ``year_generic``   → skip (ambiguous).
    * ``year_event``     → never comparable.
    * Different lifecycle milestone classes (J1.6.5) → milestone progression,
      not a contradiction; incompatible.
    * Scope-prefixed power names that differ → incompatible.
    * rate vs non-rate → incompatible.
    """
    # Year / lifecycle rules — checked before identity so that year_event vs
    # year_event and year_generic vs year_generic are still blocked.
    is_year_a = type_a.startswith("year_") or type_a in _LIFECYCLE_MILESTONE_TYPES
    is_year_b = type_b.startswith("year_") or type_b in _LIFECYCLE_MILESTONE_TYPES

    if is_year_a or is_year_b:
        if type_a == "year_generic" or type_b == "year_generic":
            return False
        # year_event is never comparable (events are not factual claims about
        # the same timeline; two conference years are not contradictions).
        if type_a == "year_event" or type_b == "year_event":
            return False
        # Both are lifecycle milestones but different classes → progression,
        # not contradiction.
        if (type_a in _LIFECYCLE_MILESTONE_TYPES
                and type_b in _LIFECYCLE_MILESTONE_TYPES
                and type_a != type_b):
            return False
        return True

    if type_a == type_b:
        return True

    # Scope-prefixed power types (backstop; scope gate fires first)
    _POWER_SCOPES = (
        "rack_power_", "power_shelf_power_", "tray_power_", "node_power_",
        "gpu_power_", "cluster_power_", "facility_power_",
        "installed_capacity_", "fleet_capacity_",
    )
    a_scoped = any(type_a.startswith(p) for p in _POWER_SCOPES)
    b_scoped = any(type_b.startswith(p) for p in _POWER_SCOPES)
    if a_scoped and b_scoped and type_a != type_b:
        return False

    # Rate vs non-rate
    if type_a.endswith("_rate") != type_b.endswith("_rate"):
        return False

    return True


def _classify_numeric_metric_type(window: str, unit: str, scope: str = "unknown") -> str:
    """Return a semantic metric-type tag (J1.6.3) for a numeric *unit* value."""
    # Watts — GPU/chip TDP only (rack-scale uses kW/MW)
    if unit == "w":
        if scope in ("node", "chip"):
            return "gpu_power_w"
        return "w_value"

    if unit in ("gw", "mw", "kw", "tw"):
        if _RE_RATE.search(window):
            kind = "rate"
        elif _RE_TARGET.search(window):
            kind = "target"
        elif _RE_CURRENT.search(window):
            kind = "current"
        else:
            kind = "level"

        scope_prefix = {
            "rack":     "rack_power",
            "component":"power_shelf_power",
            "tray":     "tray_power",
            "node":     "node_power",
            "cluster":  "cluster_power",
            "facility": "facility_power",
            "unit":     "installed_capacity",
            "reactor":  "installed_capacity",
            "fleet":    "fleet_capacity",
        }.get(scope)
        if scope_prefix:
            return f"{scope_prefix}_{unit}"
        return f"{unit}_{kind}"

    return f"{unit}_value"


def _extract_value_and_metric(
    text: str, unit: str, scope: str = "unknown"
) -> tuple[float, str] | None:
    """Find the first *unit* occurrence in *text* and return (value, metric_type)."""
    for pattern in (
        rf"\b(\d+(?:\.\d+)?)\s*{re.escape(unit)}\b",
        rf"\b{re.escape(unit)}\s*(\d+(?:\.\d+)?)\b",
    ):
        m = re.search(pattern, text)
        if m:
            value = float(m.group(1))
            start = max(0, m.start() - _CONTEXT_WINDOW)
            end   = min(len(text), m.end() + _CONTEXT_WINDOW)
            metric_type = _classify_numeric_metric_type(text[start:end], unit, scope)
            return value, metric_type
    return None


# ---------------------------------------------------------------------------
# Duration conflict detection
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"\b(\d+)(?:\s*[-–]\s*(\d+))?\s*(months?|years?)",
    re.IGNORECASE,
)

_MONTH_MULT: dict[str, int] = {
    "month": 1, "months": 1,
    "year":  12, "years": 12,
}

_RE_DUR_CONSTRUCTION = re.compile(
    r"\bconstruction\b|\bbuild\b|\bsite\s+work\b|\binstall", re.IGNORECASE,
)
_RE_DUR_LICENSING = re.compile(
    r"\blicensing\b|\bregulatory\b|\breview\b|\bapproval\b|\bpermit", re.IGNORECASE,
)
_RE_DUR_LEAD_TIME = re.compile(
    r"\blead[\s-]time\b|\bprocurement\b|\bdelivery\b|\bmanufactur", re.IGNORECASE,
)

_DURATION_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "for", "in", "on", "at", "by", "from", "with", "and", "or",
    "not", "no", "its", "it", "that", "this", "these", "those",
    "can", "could", "will", "would", "may", "might", "must", "shall",
    "approximately", "about", "around", "roughly", "estimated", "expected",
    "projected", "anticipated", "typically", "usually", "often",
    "construction", "duration", "timeline", "schedule", "period", "time",
    "months", "years", "month", "year", "weeks", "days",
    "build", "commissioning", "licensing", "review", "approval",
    "completion", "deployment", "operation", "startup",
    "large", "small", "new", "first", "second", "next", "last", "initial",
    "planned", "proposed", "current", "historical", "traditional", "typical",
    "designed", "reactor", "nuclear", "plant", "facility",
})


def _extract_duration(text: str) -> tuple[float, float, str] | None:
    m = _DURATION_RE.search(text)
    if not m:
        return None
    lo_str, hi_str, unit = m.group(1), m.group(2), m.group(3).lower()
    mult = _MONTH_MULT.get(unit, 1)
    lo = float(lo_str) * mult
    hi = float(hi_str) * mult if hi_str else lo
    return lo, hi, unit


def _duration_metric_type(text: str) -> str:
    if _RE_DUR_CONSTRUCTION.search(text):
        return "duration_construction"
    if _RE_DUR_LICENSING.search(text):
        return "duration_licensing"
    if _RE_DUR_LEAD_TIME.search(text):
        return "duration_lead_time"
    return "duration_generic"


def _entity_tokens(text: str) -> frozenset[str]:
    words = re.findall(r"\b[a-z][a-z0-9_-]*\b", text)
    return frozenset(w for w in words if w not in _DURATION_STOP_WORDS and len(w) >= 3)


# ---------------------------------------------------------------------------
# Per-check functions
# ---------------------------------------------------------------------------

def _check_numeric_conflict(
    a: EvidenceItem,
    b: EvidenceItem,
    out_suppressed: list[SuppressedComparison] | None = None,
) -> Contradiction | None:
    """Return a Contradiction when A and B contain conflicting numeric values.

    Gates (in order):
    1. Both claims have at least one numeric token and share a unit keyword.
    2. Both have extractable values for that unit.
    3. Entity compatibility (J1.6.1) — different named entities → suppress.
    4. Scope compatibility (J1.6.2) — incompatible scales → suppress.
    5. Metric-type compatibility (J1.4) — different measurement classes → skip.
    6. Values differ by more than 20%.
    """
    claim_a = a.claim.lower()
    claim_b = b.claim.lower()

    if not re.search(r"\b\d+(?:\.\d+)?\b", claim_a):
        return None
    if not re.search(r"\b\d+(?:\.\d+)?\b", claim_b):
        return None

    # Use pre-populated scope/entity when available (set by enrich_evidence_items)
    scope_a = a.scope if a.scope else _extract_scope(claim_a)
    scope_b = b.scope if b.scope else _extract_scope(claim_b)

    for unit in _UNIT_KEYWORDS:
        if unit not in claim_a or unit not in claim_b:
            continue

        result_a = _extract_value_and_metric(claim_a, unit, scope_a)
        result_b = _extract_value_and_metric(claim_b, unit, scope_b)
        if result_a is None or result_b is None:
            continue

        val_a, metric_type_a = result_a
        val_b, metric_type_b = result_b

        # Gate 3: entity compatibility
        if not _entity_names_compatible(a.entity, b.entity):
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="entity_mismatch",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': entity '{a.entity}' vs '{b.entity}' — "
                        f"different named subjects cannot contradict."
                    ),
                ))
            continue

        # Gate 3.5 (J6.5b): product / generation mismatch
        _prod_compat, _prod_reason = _product_compatibility_check(a.claim, b.claim)
        if not _prod_compat:
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason=_prod_reason,
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': {_prod_reason} — claims reference "
                        f"different product models or GPU generations."
                    ),
                ))
            continue

        # Gate 4: scope compatibility
        if not _scopes_compatible(scope_a, scope_b):
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="scope_mismatch",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': {scope_a} scope ({val_a} {unit}) "
                        f"vs {scope_b} scope ({val_b} {unit}) — "
                        f"different measurement levels cannot contradict."
                    ),
                ))
            continue

        # Gate 5: metric-type compatibility
        if not _metric_types_compatible(metric_type_a, metric_type_b):
            if out_suppressed is not None and _is_power_scope_mismatch(metric_type_a, metric_type_b):
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="metric_scope_mismatch",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    metric_a=metric_type_a,
                    metric_b=metric_type_b,
                    detail=(
                        f"Metric '{metric_type_a}' (scope={scope_a}) is incompatible with "
                        f"'{metric_type_b}' (scope={scope_b}) — different power measurement levels."
                    ),
                ))
            continue

        # Gate 6 (J6.5a): temporal progression — current state vs future target
        kind_a = _extract_numeric_kind_from_text(claim_a)
        kind_b = _extract_numeric_kind_from_text(claim_b)
        _temporal_pair = frozenset({kind_a, kind_b})
        if _temporal_pair in (frozenset({"current", "target"}), frozenset({"current", "rate"})):
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="temporal_progression",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': temporal kinds differ — "
                        f"A is '{kind_a}' ({val_a} {unit}), B is '{kind_b}' ({val_b} {unit}). "
                        f"Current state vs future projection/rate are not contradictions."
                    ),
                ))
            continue

        # Gate 6.5 (J6.5c): comparison context incompatibility
        _ctx_a = _classify_comparison_context(a.claim)
        _ctx_b = _classify_comparison_context(b.claim)
        # example_deployment suppresses against anything — illustrative figures
        # are never a basis for contradiction with concrete measurements.
        _ctx_fire = (
            _ctx_a == "example_deployment" or _ctx_b == "example_deployment"
            or (
                _ctx_a != "unknown" and _ctx_b != "unknown"
                and frozenset({_ctx_a, _ctx_b}) in _INCOMPATIBLE_CONTEXT_PAIRS
            )
        )
        if _ctx_fire:
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="context_mismatch",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': context '{_ctx_a}' ({val_a} {unit}) "
                        f"vs '{_ctx_b}' ({val_b} {unit}) — "
                        f"incompatible measurement contexts cannot contradict."
                    ),
                ))
            continue

        # Gate 7 (J6.5b): range vs point-value compatibility
        _range_a = _extract_range_for_unit(claim_a, unit)
        _range_b = _extract_range_for_unit(claim_b, unit)
        if _range_a and not _range_b:
            _lo, _hi = _range_a
            if _lo * 0.85 <= val_b <= _hi * 1.15:
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="range_average_compatible",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Unit '{unit}': A states range [{_lo}, {_hi}] {unit}, "
                            f"B states {val_b} {unit} which falls within that range."
                        ),
                    ))
                continue
        elif _range_b and not _range_a:
            _lo, _hi = _range_b
            if _lo * 0.85 <= val_a <= _hi * 1.15:
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="range_average_compatible",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Unit '{unit}': B states range [{_lo}, {_hi}] {unit}, "
                            f"A states {val_a} {unit} which falls within that range."
                        ),
                    ))
                continue

        # Gate 8 (J6.5d): numeric semantic classification
        # Suppress when either extracted value is a threshold or historical reference.
        _sem_a = _classify_value_semantic(claim_a, val_a, unit)
        _sem_b = _classify_value_semantic(claim_b, val_b, unit)
        if _sem_a == "threshold" or _sem_b == "threshold":
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="threshold_vs_measurement",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': sem_a='{_sem_a}' ({val_a} {unit}), "
                        f"sem_b='{_sem_b}' ({val_b} {unit}) — "
                        f"threshold/limit values cannot contradict measurements."
                    ),
                ))
            continue
        if _sem_a == "historical_comparison" or _sem_b == "historical_comparison":
            if out_suppressed is not None:
                out_suppressed.append(SuppressedComparison(
                    evidence_a_id=a.evidence_id or "?",
                    evidence_b_id=b.evidence_id or "?",
                    evidence_a_claim=a.claim,
                    evidence_b_claim=b.claim,
                    reason="historical_progression",
                    scope_a=scope_a,
                    scope_b=scope_b,
                    detail=(
                        f"Unit '{unit}': sem_a='{_sem_a}' ({val_a} {unit}), "
                        f"sem_b='{_sem_b}' ({val_b} {unit}) — "
                        f"historical reference values represent technology progression."
                    ),
                ))
            continue

        if val_a == 0 and val_b == 0:
            continue
        denom = max(val_a, val_b)
        if denom == 0:
            continue
        diff_ratio = abs(val_a - val_b) / denom
        if diff_ratio <= 0.20:
            continue

        severity = "high" if diff_ratio >= 0.50 else "medium"
        topic = _UNIT_TO_TOPIC.get(unit, "numeric specification")

        return Contradiction(
            contradiction_id="",
            topic=topic,
            evidence_a_id=a.evidence_id or "?",
            evidence_b_id=b.evidence_id or "?",
            evidence_a_claim=a.claim,
            evidence_b_claim=b.claim,
            evidence_a_source=a.source_document,
            evidence_b_source=b.source_document,
            severity=severity,
            explanation=(
                f"Numeric conflict: A says {val_a} {unit} ({metric_type_a}), "
                f"B says {val_b} {unit} ({metric_type_b})"
            ),
            metric_type_a=metric_type_a,
            metric_type_b=metric_type_b,
            entity_a=a.entity,
            entity_b=b.entity,
            scope_a=scope_a,
            scope_b=scope_b,
        )

    return None


def _check_categorical_conflict(
    a: EvidenceItem,
    b: EvidenceItem,
    out_suppressed: list[SuppressedComparison] | None = None,
) -> Contradiction | None:
    """Return a Contradiction when A and B match opposing exclusive-pair terms.

    Applies entity (J1.6.1), scope (J1.6.2), and metric/milestone (J1.6.4/5)
    gates before flagging.
    """
    claim_a = a.claim.lower()
    claim_b = b.claim.lower()

    scope_a = a.scope if a.scope else _extract_scope(claim_a)
    scope_b = b.scope if b.scope else _extract_scope(claim_b)

    for set_x, set_y in EXCLUSIVE_PAIRS:
        for term_a_set, term_b_set in ((set_x, set_y), (set_y, set_x)):
            matched_a = _find_match(claim_a, term_a_set)
            matched_b = _find_match(claim_b, term_b_set)
            if not (matched_a and matched_b):
                continue

            mt_a = _categorical_metric_type(claim_a, matched_a)
            mt_b = _categorical_metric_type(claim_b, matched_b)

            # Gate: entity compatibility
            if not _entity_names_compatible(a.entity, b.entity):
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="entity_mismatch",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Categorical '{matched_a}' vs '{matched_b}': "
                            f"entity '{a.entity}' vs '{b.entity}' — different subjects."
                        ),
                    ))
                continue

            # Gate 3.5 (J6.5b): product / generation mismatch
            _cat_compat, _cat_reason = _product_compatibility_check(a.claim, b.claim)
            if not _cat_compat:
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason=_cat_reason,
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Categorical '{matched_a}' vs '{matched_b}': "
                            f"{_cat_reason} — different products or GPU generations."
                        ),
                    ))
                continue

            # Gate: metric-type / milestone compatibility
            if not _metric_types_compatible(mt_a, mt_b):
                if (mt_a in _LIFECYCLE_MILESTONE_TYPES
                        and mt_b in _LIFECYCLE_MILESTONE_TYPES
                        and mt_a != mt_b
                        and out_suppressed is not None):
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="milestone_progression",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"'{matched_a}' is {mt_a} and '{matched_b}' is {mt_b}. "
                            f"Different lifecycle milestones represent project "
                            f"progression, not a contradiction."
                        ),
                    ))
                continue

            # Gate: scope compatibility
            if not _scopes_compatible(scope_a, scope_b):
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="scope_mismatch",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Categorical '{matched_a}' vs '{matched_b}': "
                            f"{scope_a} scope vs {scope_b} scope — "
                            f"different subsystems can independently use "
                            f"different approaches."
                        ),
                    ))
                continue

            # Gate 6.5 (J6.5c): comparison context incompatibility
            _cat_ctx_a = _classify_comparison_context(a.claim)
            _cat_ctx_b = _classify_comparison_context(b.claim)
            _cat_ctx_fire = (
                _cat_ctx_a == "example_deployment" or _cat_ctx_b == "example_deployment"
                or (
                    _cat_ctx_a != "unknown" and _cat_ctx_b != "unknown"
                    and frozenset({_cat_ctx_a, _cat_ctx_b}) in _INCOMPATIBLE_CONTEXT_PAIRS
                )
            )
            if _cat_ctx_fire:
                if out_suppressed is not None:
                    out_suppressed.append(SuppressedComparison(
                        evidence_a_id=a.evidence_id or "?",
                        evidence_b_id=b.evidence_id or "?",
                        evidence_a_claim=a.claim,
                        evidence_b_claim=b.claim,
                        reason="context_mismatch",
                        scope_a=scope_a,
                        scope_b=scope_b,
                        detail=(
                            f"Categorical '{matched_a}' vs '{matched_b}': "
                            f"context '{_cat_ctx_a}' vs '{_cat_ctx_b}' — "
                            f"incompatible measurement contexts."
                        ),
                    ))
                continue

            severity, topic = _categorical_severity_topic(matched_a, matched_b)
            return Contradiction(
                contradiction_id="",
                topic=topic,
                evidence_a_id=a.evidence_id or "?",
                evidence_b_id=b.evidence_id or "?",
                evidence_a_claim=a.claim,
                evidence_b_claim=b.claim,
                evidence_a_source=a.source_document,
                evidence_b_source=b.source_document,
                severity=severity,
                explanation=(
                    f"Categorical conflict: A claims '{matched_a}' ({mt_a}), "
                    f"B claims '{matched_b}' ({mt_b})"
                ),
                metric_type_a=mt_a,
                metric_type_b=mt_b,
                entity_a=a.entity,
                entity_b=b.entity,
                scope_a=scope_a,
                scope_b=scope_b,
            )

    return None


def _check_duration_conflict(
    a: EvidenceItem,
    b: EvidenceItem,
    out_suppressed: list[SuppressedComparison] | None = None,
) -> Contradiction | None:
    """Detect a contradiction when both claims state a duration for the same entity."""
    claim_a = a.claim.lower()
    claim_b = b.claim.lower()

    # Fall back to inline extraction when items haven't been pre-enriched
    entity_a = a.entity or _extract_entity(a.claim)[0]
    entity_b = b.entity or _extract_entity(b.claim)[0]

    dur_a = _extract_duration(claim_a)
    dur_b = _extract_duration(claim_b)
    if dur_a is None or dur_b is None:
        return None

    lo_a, hi_a, _ = dur_a
    lo_b, hi_b, _ = dur_b

    tok_a = _entity_tokens(claim_a)
    tok_b = _entity_tokens(claim_b)
    shared = tok_a & tok_b
    if not shared:
        return None

    mt_a = _duration_metric_type(claim_a)
    mt_b = _duration_metric_type(claim_b)
    if mt_a != mt_b and "generic" not in (mt_a, mt_b):
        return None

    # Overlapping ranges → no contradiction
    if hi_a >= lo_b and hi_b >= lo_a:
        return None

    mid_a = (lo_a + hi_a) / 2
    mid_b = (lo_b + hi_b) / 2
    diff_ratio = abs(mid_a - mid_b) / max(mid_a, mid_b)
    severity: str = "high" if diff_ratio >= 0.50 else "medium"

    m_a = _DURATION_RE.search(a.claim)
    m_b = _DURATION_RE.search(b.claim)
    dur_str_a = m_a.group(0) if m_a else f"{lo_a:.0f} months"
    dur_str_b = m_b.group(0) if m_b else f"{lo_b:.0f} months"

    shared_label = ", ".join(sorted(shared)[:4])
    topic = "construction duration" if "construction" in mt_a or "construction" in mt_b else "timeline"

    return Contradiction(
        contradiction_id="",
        topic=topic,
        evidence_a_id=a.evidence_id or "?",
        evidence_b_id=b.evidence_id or "?",
        evidence_a_claim=a.claim,
        evidence_b_claim=b.claim,
        evidence_a_source=a.source_document,
        evidence_b_source=b.source_document,
        severity=severity,
        explanation=(
            f"Duration conflict (months): "
            f"A says {dur_str_a} ({lo_a:.0f}–{hi_a:.0f} mo), "
            f"B says {dur_str_b} ({lo_b:.0f}–{hi_b:.0f} mo)"
        ),
        metric_type_a=mt_a,
        metric_type_b=mt_b,
        entity_a=entity_a,
        entity_b=entity_b,
        comparison_reason=(
            f"Shared entity tokens: {shared_label}. "
            f"Duration metric: {mt_a}. "
            f"Ranges [{lo_a:.0f}, {hi_a:.0f}] and [{lo_b:.0f}, {hi_b:.0f}] months "
            f"do not overlap."
        ),
    )


# ---------------------------------------------------------------------------
# J6.5a – Suppression metrics
# ---------------------------------------------------------------------------

def compute_suppression_metrics(
    suppressed: list[SuppressedComparison],
    final_count: int,
) -> dict:
    """Return suppression metrics for the Research Object and QA trace.

    Parameters
    ----------
    suppressed:
        List collected via ``out_suppressed`` in ``detect_contradictions``.
    final_count:
        Number of confirmed contradictions (len of detect_contradictions return).
    """
    by_reason: dict[str, int] = {}
    for s in suppressed:
        by_reason[s.reason] = by_reason.get(s.reason, 0) + 1
    suppressed_count = len(suppressed)
    return {
        "candidate_count": final_count + suppressed_count,
        "suppressed_count": suppressed_count,
        "final_count": final_count,
        "by_reason": by_reason,
        "scope_filtering_present": bool(
            by_reason.get("scope_mismatch", 0)
            or by_reason.get("metric_scope_mismatch", 0)
        ),
        "entity_filtering_present": bool(by_reason.get("entity_mismatch", 0)),
        "temporal_filtering_present": bool(by_reason.get("temporal_progression", 0)),
        "product_filtering_present": bool(
            by_reason.get("product_mismatch", 0)
            or by_reason.get("generation_progression", 0)
        ),
        "range_filtering_present": bool(by_reason.get("range_average_compatible", 0)),
        "context_filtering_present": bool(by_reason.get("context_mismatch", 0)),
        # J6.5d – numeric semantic suppression flags
        "threshold_filtering_present": bool(by_reason.get("threshold_vs_measurement", 0)),
        "historical_filtering_present": bool(by_reason.get("historical_progression", 0)),
        # J6.5d – numeric semantics sub-breakdown
        "numeric_semantics": {
            "threshold_vs_measurement": by_reason.get("threshold_vs_measurement", 0),
            "historical_progression": by_reason.get("historical_progression", 0),
            "range_contains_value": by_reason.get("range_average_compatible", 0),
        },
        # J6.5c – eligibility engine summary (candidate → eligible → final)
        "eligibility_engine": {
            "candidate_pairs": final_count + suppressed_count,
            "eligible_pairs": final_count,
            "suppressed_pairs": suppressed_count,
        },
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_contradictions(
    evidence: Sequence[EvidenceItem],
    source_quality_map: dict[str, SourceQuality] | None = None,
    profile: DomainProfile | None = None,
    *,
    out_suppressed: list[SuppressedComparison] | None = None,
) -> list[Contradiction]:
    """Compare all pairs of evidence items for conflicts.

    Callers should run ``enrich_evidence_items(evidence)`` first so that
    ``entity``, ``entity_type``, and ``scope`` fields are populated on each
    item before pairwise comparison begins.

    Parameters
    ----------
    evidence:
        Evidence items to compare.  Should be pre-enriched.
    source_quality_map:
        Optional map of document name → quality score.
    profile:
        Optional domain profile for topic classification (J1.5).
    out_suppressed:
        Optional list populated with :class:`SuppressedComparison` records for
        every pair blocked by entity/scope/milestone gates.
    """
    items = list(evidence)
    contradictions: list[Contradiction] = []

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a = items[i]
            b = items[j]
            if a.evidence_id and b.evidence_id and a.evidence_id > b.evidence_id:
                a, b = b, a

            c = _check_numeric_conflict(a, b, out_suppressed)
            if c:
                contradictions.append(c)

            c = _check_categorical_conflict(a, b, out_suppressed)
            if c:
                contradictions.append(c)

            c = _check_duration_conflict(a, b, out_suppressed)
            if c:
                contradictions.append(c)

    with_ids = _assign_contradiction_ids(contradictions)
    if profile is not None:
        with_ids = _apply_profile_topics(with_ids, profile)
    if source_quality_map:
        return _annotate_quality(with_ids, source_quality_map)
    return with_ids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_profile_topics(
    contradictions: list[Contradiction],
    profile: DomainProfile,
) -> list[Contradiction]:
    result: list[Contradiction] = []
    for c in contradictions:
        topic, topic_source = profile.classify_contradiction_topic(
            c.evidence_a_claim, c.evidence_b_claim
        )
        result.append(c.model_copy(update={"topic": topic, "topic_source": topic_source}))
    return result


def _annotate_quality(
    contradictions: list[Contradiction],
    source_quality_map: dict[str, SourceQuality],
) -> list[Contradiction]:
    annotated: list[Contradiction] = []
    for c in contradictions:
        qa = source_quality_map.get(c.evidence_a_source)
        qb = source_quality_map.get(c.evidence_b_source)
        score_a = qa.source_quality_score if qa else 3
        score_b = qb.source_quality_score if qb else 3
        confidence = _contradiction_confidence(score_a, score_b)
        annotated.append(c.model_copy(update={
            "source_quality_a": score_a,
            "source_quality_b": score_b,
            "confidence": confidence,
        }))
    return annotated


def _contradiction_confidence(score_a: int, score_b: int) -> str:
    if score_a >= 4 and score_b >= 4:
        return "high"
    if abs(score_a - score_b) >= 3:
        return "low"
    return "medium"


def _categorical_metric_type(text: str, matched_term: str) -> str:
    _YEAR_TERMS: frozenset[str] = frozenset({"2025", "2026", "2027", "2028", "2029", "2030", "2031"})
    if matched_term in _YEAR_TERMS:
        return _year_context_type(text, matched_term)
    if "cool" in matched_term or "dlc" in matched_term or "liquid" in matched_term:
        return "cooling_method"
    if "phase" in matched_term:
        return "cooling_phase"
    if "gpu" in matched_term:
        return "gpu_count"
    if "nvl" in matched_term:
        return "rack_type"
    return "categorical_specification"


def _find_match(text: str, terms: set[str]) -> str | None:
    for term in terms:
        if term in text:
            return term
    return None


def _categorical_severity_topic(term_a: str, term_b: str) -> tuple[str, str]:
    combined = f"{term_a} {term_b}"
    if "cool" in combined or "liquid" in combined or "dlc" in combined or "water" in combined or "air" in combined:
        return "high", "cooling type"
    if "phase" in combined:
        return "medium", "cooling phase"
    if any(y in combined for y in ("2025", "2026", "2027", "2028", "2029", "2030")):
        return "medium", "timeline"
    if "gpu" in combined:
        return "medium", "gpu count"
    if "nvl" in combined:
        return "medium", "rack type"
    return "medium", "specification"


def _assign_contradiction_ids(contradictions: list[Contradiction]) -> list[Contradiction]:
    return [
        c.model_copy(update={"contradiction_id": f"C{i:03d}"})
        for i, c in enumerate(contradictions, start=1)
    ]
