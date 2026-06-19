"""Perspective taxonomy and diversity analysis for evidence retrieval (J3.2).

Adds a *perspective* dimension to EvidenceItem — orthogonal to evidence_type
(which describes HOW a claim is structured) and topics (which are profile-level
keywords).  A perspective is the research dimension the evidence addresses:

AI Data Center:  power | cooling | networking | operations | economics |
                 deployment | supply_chain

SMR:             licensing | construction | economics | fuel |
                 grid_integration | supply_chain | deployment |
                 public_acceptance

Two concerns are addressed here:
  1. Classify each evidence item into a perspective (J3.2.2)
  2. Select a diverse set of evidence items across perspectives (J3.2.3–4)
  3. Report perspective coverage and diversity metrics (J3.2.5–6)

Public API
----------
classify_perspective(claim, source_document) -> str
compute_perspective_coverage(items) -> dict[str, int]
compute_diversity_score(items, domain) -> float
select_diverse_evidence(items, top_n, max_per_perspective) -> list[EvidenceItem]
build_diversity_metrics(items, domain) -> dict
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import EvidenceItem


# ---------------------------------------------------------------------------
# Perspective taxonomies  (J3.2.1)
# ---------------------------------------------------------------------------

# Each entry: list of lowercase keyword strings.  An evidence claim that
# contains more of these keywords scores higher for that perspective.
PERSPECTIVES_AI_DC: dict[str, list[str]] = {
    "power": [
        "power", "watt", "kw", "mw", "tdp", "thermal design power",
        "power consumption", "power draw", "energy", "pdu",
        "power delivery", "electrical", "ampere", "volt",
        "ups", "generator", "pue", "power usage effectiveness",
    ],
    "cooling": [
        "cool", "thermal", "heat", "temperature", "cdu",
        "liquid cooling", "air cooling", "immersion", "coolant",
        "btu", "hvac", "chiller", "rear door", "warm water",
        "direct liquid", "cold plate", "heat exchanger",
    ],
    "networking": [
        "network", "nvlink", "infiniband", "pcie", "bandwidth",
        "latency", "interconnect", "switch", "fabric", "rdma",
        "ethernet", "gb/s", "tb/s", "port", "topology",
        "nvl", "c2c", "chip-to-chip", "roce", "osfp",
    ],
    "operations": [
        "deploy", "install", "setup", "maintenance", "operate",
        "rack", "cabling", "floor", "management", "software",
        "driver", "firmware", "configuration", "lifecycle",
        "integration", "rollout", "skill", "staff",
    ],
    "economics": [
        "cost", "price", "price/performance", "tco", "roi",
        "dollar", "usd", "revenue", "efficiency", "$/",
        "investment", "capex", "opex", "value", "margin",
        "payback", "return", "profitability",
    ],
    "deployment": [
        "availability", "lead time", "ship", "quarter",
        "timeline", "schedule", "release", "launch", "product",
        "ramp", "roadmap", "ga", "general availability",
        "by 20", "planned", "next-generation",
    ],
    "supply_chain": [
        "supply", "manufacture", "wafer", "tsmc", "production",
        "chip", "fabrication", "shortage", "inventory",
        "foundry", "packaging", "memory", "hbm", "yield",
    ],
}

PERSPECTIVES_SMR: dict[str, list[str]] = {
    "licensing": [
        "license", "nrc", "cnsc", "regulatory", "approval",
        "permit", "certification", "review", "iaea",
        "standard design", "design certification", "dc/col",
        "licensing process", "regulatory approval", "dcd",
    ],
    "construction": [
        "construct", "build", "concrete", "erect", "install",
        "timeline", "schedule", "first concrete", "commissioning",
        "modular", "fabricat", "assembly", "site preparation",
        "construction period", "build time", "years to build",
    ],
    "economics": [
        "cost", "capex", "lcoe", "financi", "dollar",
        "million", "billion", "price", "economics",
        "tco", "investment", "revenue", "overnight cost",
        "levelized", "capital cost", "project cost",
    ],
    "fuel": [
        "fuel", "uranium", "haleu", "enrichment", "fissile",
        "pellet", "rod", "assembly", "burnup", "cycle",
        "enriched", "u-235", "fuel cycle", "spent fuel",
        "fresh fuel", "refuel", "reload",
    ],
    "grid_integration": [
        "grid", "dispatch", "flexible", "load follow",
        "baseload", "peaking", "integration", "ancillary",
        "frequency", "capacity factor", "mwh", "mwe",
        "load following", "variable", "intermittent",
    ],
    "supply_chain": [
        "supply chain", "manufacture", "component",
        "vendor", "procurement", "forging", "pressure vessel",
        "shortage", "domestic", "international", "tier-1",
        "industrial base", "qualified supplier",
    ],
    "deployment": [
        "deploy", "site", "location", "commissioning",
        "first unit", "project", "contract", "mou", "loi",
        "commercial operation", "cod", "planned site",
    ],
    "public_acceptance": [
        "public", "community", "acceptance", "opposition",
        "sentiment", "opinion", "trust", "stakeholder",
        "local", "social licence", "consultation", "engagement",
        "perception", "support", "protest",
    ],
}

# Fallback perspective when no domain is detected
_PERSPECTIVES_FALLBACK: dict[str, list[str]] = {
    **PERSPECTIVES_AI_DC,
    **PERSPECTIVES_SMR,
}

# SMR source document signals — if any appear in the source filename, it's SMR
_SMR_DOC_SIGNALS = frozenset({
    "smr", "nuclear", "reactor", "bwrx", "nuscale", "nrc", "haleu",
    "moltex", "terrestrial", "kairos", "x-energy", "usnc", "cnsc",
    "thorcon", "newcleo", "holtec",
})


# ---------------------------------------------------------------------------
# Domain detection
# ---------------------------------------------------------------------------

def detect_domain(source_document: str) -> str:
    """Return 'smr' or 'ai_dc' based on source document filename.

    Falls back to 'ai_dc' when the filename contains no SMR signals.
    """
    doc_lower = source_document.lower()
    if any(sig in doc_lower for sig in _SMR_DOC_SIGNALS):
        return "smr"
    return "ai_dc"


def _perspectives_for_domain(domain: str) -> dict[str, list[str]]:
    if domain == "smr":
        return PERSPECTIVES_SMR
    if domain == "ai_dc":
        return PERSPECTIVES_AI_DC
    return _PERSPECTIVES_FALLBACK


# ---------------------------------------------------------------------------
# Perspective classification  (J3.2.2)
# ---------------------------------------------------------------------------

def classify_perspective(claim: str, source_document: str) -> str:
    """Classify *claim* into one perspective based on keyword density.

    Uses the domain taxonomy inferred from *source_document*.
    Returns 'general' when no keyword matches fire.
    """
    domain = detect_domain(source_document)
    perspectives = _perspectives_for_domain(domain)
    text_lower = (claim or "").lower()

    best_perspective = "general"
    best_score = 0

    for perspective, keywords in perspectives.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > best_score:
            best_score = score
            best_perspective = perspective

    return best_perspective


# ---------------------------------------------------------------------------
# Coverage and diversity metrics  (J3.2.5–6)
# ---------------------------------------------------------------------------

def compute_perspective_coverage(items: "list[EvidenceItem]") -> dict[str, int]:
    """Return a count of evidence items per perspective.

    Example: {"economics": 8, "fuel": 1, "licensing": 0}
    """
    counts: dict[str, int] = {}
    for item in items:
        p = getattr(item, "perspective", None) or "general"
        counts[p] = counts.get(p, 0) + 1
    return counts


def compute_diversity_score(
    items: "list[EvidenceItem]",
    domain: str = "",
) -> float:
    """Return a diversity score in [0, 1].

    diversity_score = unique_perspectives / total_possible_perspectives

    A score of 1.0 means every domain-relevant perspective is represented.
    'general' is excluded from the denominator (it is a fallback, not a gap).
    """
    if not items:
        return 0.0

    total_possible = len(_perspectives_for_domain(domain or "ai_dc"))
    perspectives_found = {
        getattr(item, "perspective", None) or "general"
        for item in items
        if (getattr(item, "perspective", None) or "general") != "general"
    }
    if total_possible == 0:
        return 1.0
    return round(min(len(perspectives_found) / total_possible, 1.0), 4)


def build_diversity_metrics(
    items: "list[EvidenceItem]",
    domain: str = "",
) -> dict:
    """Build the J3.2.6 diversity metrics dict for trace output.

    Keys
    ----
    unique_perspectives   : int   — number of distinct perspectives represented
    evidence_items        : int   — total evidence items in the set
    diversity_score       : float — unique_perspectives / total_possible (0-1)
    perspective_coverage  : dict  — count per perspective (J3.2.5)
    perspectives_found    : list  — sorted list of represented perspectives
    """
    coverage = compute_perspective_coverage(items)
    perspectives_found = sorted(
        p for p, cnt in coverage.items() if cnt > 0 and p != "general"
    )
    return {
        "unique_perspectives": len(perspectives_found),
        "evidence_items": len(items),
        "diversity_score": compute_diversity_score(items, domain),
        "perspective_coverage": coverage,
        "perspectives_found": perspectives_found,
    }


# ---------------------------------------------------------------------------
# Diversity-aware evidence selection  (J3.2.3–4)
# ---------------------------------------------------------------------------

def select_diverse_evidence(
    items: "list[EvidenceItem]",
    top_n: int = 50,
    max_per_perspective: int = 8,
) -> "list[EvidenceItem]":
    """Select up to *top_n* evidence items with perspective diversity.

    Items must already be sorted by score (descending) — this function
    preserves relative score ordering while capping over-represented
    perspectives.

    Algorithm (three-pass):
      Pass 1 — Seed: take the top item from each perspective (ensures every
               represented perspective gets at least one slot).
      Pass 2 — Fill: continue through sorted items, skipping any perspective
               that has already reached *max_per_perspective*.
      Pass 3 — Overflow: if slots remain (rare), fill without the cap.

    Returns a list of at most *top_n* items.
    """
    selected: list[EvidenceItem] = []
    selected_ids: set[str] = set()
    per_perspective: dict[str, int] = {}

    def _add(item: EvidenceItem) -> None:
        selected.append(item)
        selected_ids.add(item.evidence_id or item.claim)
        p = getattr(item, "perspective", None) or "general"
        per_perspective[p] = per_perspective.get(p, 0) + 1

    def _seen(item: EvidenceItem) -> bool:
        return (item.evidence_id or item.claim) in selected_ids

    # Pass 1: one item per perspective (seed)
    seeded: set[str] = set()
    for item in items:
        if len(selected) >= top_n:
            break
        p = getattr(item, "perspective", None) or "general"
        if p not in seeded:
            _add(item)
            seeded.add(p)

    # Pass 2: fill with per-perspective cap
    for item in items:
        if len(selected) >= top_n:
            break
        if _seen(item):
            continue
        p = getattr(item, "perspective", None) or "general"
        if per_perspective.get(p, 0) < max_per_perspective:
            _add(item)

    # Pass 3: overflow — fill remaining slots ignoring cap
    for item in items:
        if len(selected) >= top_n:
            break
        if not _seen(item):
            _add(item)

    return selected
