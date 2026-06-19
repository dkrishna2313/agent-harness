"""Retrieval planning layer (J3.0 / J3.0a).

Expands a user question into a set of targeted retrieval queries so that
``select_top_chunks`` can be run multiple times and the results merged.
This is pure term manipulation — no LLM calls.

Architecture
------------
Question
  ↓ RetrievalPlanner.plan()
RetrievalPlan(primary_question, planner_mode, entity_lock, metric_lock, queries=[…])
  ↓ select_top_chunks_multi() in retrieval.py
merged, deduplicated chunks

J3.0a adds planner modes:

  FACT_LOOKUP         – "How many GPUs are in NVL72?"  → 1-3 queries, entity-locked
  COMPARISON          – "Compare NVL72 vs DGX H100"    → 3-5 queries
  EXPLANATION         – "Why does GB200 need liquid cooling?" → 3-5 queries
  EXPLORATORY_RESEARCH – "What factors drive SMR LCOE?" → 5-8 queries (full expansion)

Entity locking (FACT_LOOKUP only):
  Detected product entities (e.g. "GB200 NVL72") are anchored into every
  generated query.  Topic-table expansion is SKIPPED for FACT_LOOKUP so that
  adjacent products (NVL36, DGX H100, GB300) are never pulled in by association.

Metric locking (FACT_LOOKUP only):
  Detected target metric (e.g. gpu_count, rack_power) focuses generated
  queries on that specific measurement.  Unrelated metric queries are omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profile import DomainProfile


# ---------------------------------------------------------------------------
# Query mode
# ---------------------------------------------------------------------------

class QueryMode(str, Enum):
    FACT_LOOKUP = "FACT_LOOKUP"
    COMPARISON = "COMPARISON"
    EXPLANATION = "EXPLANATION"
    EXPLORATORY_RESEARCH = "EXPLORATORY_RESEARCH"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RetrievalPlan:
    """Output of the retrieval planner."""

    primary_question: str
    queries: list[str]              # 1–8 targeted query strings
    detected_topics: list[str]      # profile topics inferred from question
    planner_mode: QueryMode = QueryMode.EXPLORATORY_RESEARCH
    entity_lock: str | None = None  # primary entity anchor (FACT_LOOKUP only)
    metric_lock: str | None = None  # detected target metric (FACT_LOOKUP only)
    expansion_source: str = "rule"  # always "rule" in J3.0

    @property
    def query_count(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict:
        return {
            "primary_question": self.primary_question,
            "planner_mode": self.planner_mode.value,
            "entity_lock": self.entity_lock,
            "metric_lock": self.metric_lock,
            "entity_locked": self.entity_lock is not None,
            "metric_locked": self.metric_lock is not None,
            "queries": self.queries,
            "detected_topics": self.detected_topics,
            "expansion_source": self.expansion_source,
            "query_count": self.query_count,
        }


# ---------------------------------------------------------------------------
# Stopwords
# ---------------------------------------------------------------------------

_PLAN_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "in", "of", "to", "for", "is", "are",
    "was", "were", "be", "been", "being", "do", "does", "did", "have",
    "has", "had", "with", "by", "on", "at", "from", "into", "about",
    "what", "why", "how", "when", "where", "which", "who", "that",
    "this", "these", "those", "can", "could", "would", "should", "will",
    "between", "among", "across", "through", "within", "single", "one",
    "integrated", "installed", "per", "its", "their", "given",
})


def _content_words(text: str) -> list[str]:
    """Return alphabetic tokens longer than 2 chars that are not stopwords."""
    tokens = re.findall(r"[a-z][a-z0-9-]{1,}", text.lower())
    return [t for t in tokens if t not in _PLAN_STOPWORDS]


# ---------------------------------------------------------------------------
# J3.0a: Question mode classifier
# ---------------------------------------------------------------------------

# Words that indicate exploratory intent — checked first because they override
# vaguer patterns like "what is" (e.g. "What factors drive LCOE?")
_EXPLORATORY_WORDS: frozenset[str] = frozenset({
    "factors", "factor", "barriers", "barrier", "challenges", "challenge",
    "implications", "implication", "trends", "trend", "landscape", "risks",
    "risk", "opportunities", "opportunity", "drivers", "driver", "overview",
    "summary", "analysis", "considerations", "consideration", "outlook",
    "prospects", "viability", "feasibility", "aspects", "aspect",
    "dimension", "dimensions", "issues", "issue", "concerns", "concern",
})

# Patterns that strongly indicate FACT_LOOKUP
_FACT_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bhow many\b",
        r"\bhow much\b",
        r"\bwhat is the\b",
        r"\bwhat are the\b(?!.+\b(?:factors|barriers|challenges|drivers|implications)\b)",
        r"\bhow (?:large|small|tall|wide|deep|heavy|fast|slow|efficient)\b",
        r"\bwhat (?:power|capacity|count|speed|bandwidth|weight|height|temperature|voltage|rating|limit|maximum|minimum|threshold)\b",
    ]
)

# Patterns that indicate COMPARISON
_COMPARISON_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bdifference[s]? between\b", r"\bcontrast\b",
        r"\bbetter than\b", r"\bworse than\b",
    ]
)

# Patterns that indicate EXPLANATION
_EXPLANATION_PATTERNS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bwhy\b", r"\bhow does\b", r"\bhow do\b",
        r"\bexplain\b", r"\bwhat causes?\b", r"\breason\b",
        r"\bmechanism\b", r"\bprinciple\b",
    ]
)


def classify_question_mode(question: str) -> QueryMode:
    """Classify a question into a retrieval mode.

    Priority order:
    1. EXPLORATORY_RESEARCH  – explicit exploratory vocabulary
    2. COMPARISON            – comparison / versus language
    3. EXPLANATION           – why / how does / explain
    4. FACT_LOOKUP           – how many / how much / what is the X
    5. Short factual default – ≤8 content words → FACT_LOOKUP
    6. Default               – EXPLORATORY_RESEARCH
    """
    # 1 – Exploratory vocabulary takes priority (prevents "what factors" → FACT)
    cwords = set(_content_words(question))
    if cwords & _EXPLORATORY_WORDS:
        return QueryMode.EXPLORATORY_RESEARCH

    # Multi-part "what are the main X" phrasing
    if re.search(r"\bwhat (?:are|were) (?:the )?(?:main |key |primary |major |top )", question, re.I):
        return QueryMode.EXPLORATORY_RESEARCH

    # 2 – Comparison
    if any(p.search(question) for p in _COMPARISON_PATTERNS):
        return QueryMode.COMPARISON

    # 3 – Explanation
    if any(p.search(question) for p in _EXPLANATION_PATTERNS):
        return QueryMode.EXPLANATION

    # 4 – Fact lookup
    if any(p.search(question) for p in _FACT_PATTERNS):
        return QueryMode.FACT_LOOKUP

    # 5 – Short question with few content words → probably factual
    if len(_content_words(question)) <= 8:
        return QueryMode.FACT_LOOKUP

    return QueryMode.EXPLORATORY_RESEARCH


# ---------------------------------------------------------------------------
# J3.0a: Entity detection
# ---------------------------------------------------------------------------

# Ordered from most-specific to least-specific.  The first match wins.
# Each tuple: (canonical_name_for_lock, [trigger_strings_in_question])
_AI_DC_ENTITY_PATTERNS: list[tuple[str, list[str]]] = [
    # Specific rack/system models — most specific first
    ("GB200 NVL72",  ["NVL72", "nvl72"]),
    ("GB200 NVL36",  ["NVL36", "nvl36"]),
    ("DGX B200",     ["DGX B200", "dgx b200"]),
    ("DGX H100",     ["DGX H100", "dgx h100"]),
    ("DGX H200",     ["DGX H200", "dgx h200"]),
    ("HGX H100",     ["HGX H100", "hgx h100"]),
    # GPU generations
    ("GB200",        ["GB200", "gb200"]),
    ("GB300",        ["GB300", "gb300"]),
    ("H100",         ["H100", "h100"]),
    ("H200",         ["H200", "h200"]),
    ("B200",         ["B200 GPU", "b200 gpu"]),
    ("A100",         ["A100", "a100"]),
    # Architectures (less specific)
    ("Blackwell",    ["Blackwell", "blackwell"]),
    ("Hopper",       ["Hopper GPU", "hopper gpu"]),
]

_SMR_ENTITY_PATTERNS: list[tuple[str, list[str]]] = [
    ("BWRX-300",     ["BWRX-300", "BWRX300", "bwrx-300", "bwrx300"]),
    ("NuScale VOYGR", ["NuScale", "VOYGR", "nuscale", "voygr"]),
    ("AP300",        ["AP300", "ap300"]),
    ("SMR-160",      ["SMR-160", "smr-160"]),
    ("NuwardSMR",    ["Nuward", "nuward"]),
]

_ALL_ENTITY_PATTERNS = _AI_DC_ENTITY_PATTERNS + _SMR_ENTITY_PATTERNS


def detect_entity_lock(
    question: str,
    profile: "DomainProfile | None" = None,
) -> str | None:
    """Return the canonical name of the most specific entity found in the question.

    When *profile* is supplied and has ``entity_patterns``, those are checked first.
    Falls back to the hardcoded ``_ALL_ENTITY_PATTERNS`` table.
    """
    if profile is not None and profile.entity_patterns:
        for entry in profile.entity_patterns:
            name = entry.get("name", "")
            signals = entry.get("signals", [])
            if any(str(s) in question for s in signals):
                return name
    for canonical, triggers in _ALL_ENTITY_PATTERNS:
        if any(t in question for t in triggers):
            return canonical
    return None


# ---------------------------------------------------------------------------
# J3.0a: Metric detection
# ---------------------------------------------------------------------------

# metric_key → (list of regex patterns that signal this metric in the question)
_METRIC_DETECTION: dict[str, list[re.Pattern]] = {
    "gpu_count": [
        re.compile(r"\bhow many\s+(?:b200s?|gpus?|accelerators?)\b", re.I),
        re.compile(r"\b(?:number|count)\s+of\s+(?:gpus?|accelerators?)\b", re.I),
        re.compile(r"\bgpus?\s+(?:integrated|installed|in|per|count)\b", re.I),
    ],
    "rack_power": [
        re.compile(r"\brack\s+power\b", re.I),
        re.compile(r"\bpower\s+(?:consumption|requirement|draw|demand|budget)\b", re.I),
        re.compile(r"\btotal\s+power\b", re.I),
        re.compile(r"\bhow\s+much\s+power\b", re.I),
        re.compile(r"\bwatt[s]?\b|\bkilo?watt[s]?\b|\bkw\b", re.I),
    ],
    "cooling_capacity": [
        re.compile(r"\bcooling\s+(?:capacity|requirement|infrastructure)\b", re.I),
        re.compile(r"\bwhy.+(?:liquid|water)\s+cool", re.I),
        re.compile(r"\bhow\s+(?:is|does).+cool\b", re.I),
    ],
    "memory_capacity": [
        re.compile(r"\b(?:gpu\s+)?memory\s+(?:capacity|size|bandwidth|per)\b", re.I),
        re.compile(r"\bhbm\b", re.I),
    ],
    "network_bandwidth": [
        re.compile(r"\bbandwidth\b", re.I),
        re.compile(r"\bnvlink\b|\binfiniband\b", re.I),
    ],
    "lcoe": [
        re.compile(r"\blcoe\b", re.I),
        re.compile(r"\blevelized\s+cost\b", re.I),
        re.compile(r"\bcost\s+of\s+electricity\b", re.I),
    ],
    "construction_cost": [
        re.compile(r"\bconstruction\s+cost\b", re.I),
        re.compile(r"\bcapital\s+cost\b", re.I),
        re.compile(r"\bevernight\s+cost\b", re.I),
    ],
    "electric_output": [
        re.compile(r"\belectric(?:al)?\s+output\b", re.I),
        re.compile(r"\bmwe\b|\bkwe\b", re.I),
        re.compile(r"\bpower\s+output\b|\boutput\s+capacity\b", re.I),
    ],
}


def detect_metric_lock(
    question: str,
    profile: "DomainProfile | None" = None,
) -> str | None:
    """Return the metric key that best matches the question, or None.

    When *profile* is supplied and has ``metric_patterns``, those regex strings
    are compiled and checked first. Falls back to the hardcoded ``_METRIC_DETECTION``.
    """
    if profile is not None and profile.metric_patterns:
        for metric_key, pattern_strings in profile.metric_patterns.items():
            compiled = [re.compile(p, re.I) for p in pattern_strings]
            if any(p.search(question) for p in compiled):
                return metric_key
    for metric_key, patterns in _METRIC_DETECTION.items():
        if any(p.search(question) for p in patterns):
            return metric_key
    return None


# ---------------------------------------------------------------------------
# J3.0a: Metric-anchored query terms for FACT_LOOKUP
# ---------------------------------------------------------------------------

# When entity + metric are both locked, these templates are filled with the
# entity name and emitted as focused queries.  {e} = entity_lock value.
_METRIC_ANCHOR_QUERIES: dict[str, list[str]] = {
    "gpu_count": [
        "{e} GPU count accelerators",
        "{e} 72 GPUs specifications",
        "{e} number of B200 GPUs",
    ],
    "rack_power": [
        "{e} total rack power kW",
        "{e} power consumption requirements",
    ],
    "cooling_capacity": [
        "{e} cooling requirements CDU",
        "{e} liquid cooling thermal",
    ],
    "memory_capacity": [
        "{e} GPU memory HBM capacity",
        "{e} memory bandwidth specifications",
    ],
    "network_bandwidth": [
        "{e} NVLink bandwidth fabric",
        "{e} interconnect bandwidth specifications",
    ],
    "lcoe": [
        "{e} levelized cost electricity LCOE",
        "{e} construction cost economics",
    ],
    "construction_cost": [
        "{e} construction capital cost overnight",
        "{e} cost schedule economics",
    ],
    "electric_output": [
        "{e} electrical output capacity MWe",
        "{e} power generation specifications",
    ],
}

# When entity is locked but no metric detected — use these generic anchored queries.
_GENERIC_ANCHOR_QUERIES: list[str] = [
    "{e} specifications overview",
    "{e} design details",
]


# ---------------------------------------------------------------------------
# Domain-specific expansion tables (EXPLORATORY / EXPLANATION / COMPARISON only)
# ---------------------------------------------------------------------------

# AI data-center / NVIDIA — note: "gb200" and "nvl72" entries are deliberately
# conservative and do NOT reference adjacent products or architectures.
_AI_DC_TOPIC_EXPANSIONS: dict[str, list[str]] = {
    "power": [
        "rack power consumption total load",
        "DC power infrastructure distribution",
        "power distribution unit PDU cooling",
        "UPS backup power capacity",
    ],
    "cooling": [
        "liquid cooling direct liquid cooling DLC",
        "coolant distribution unit CDU capacity",
        "thermal management heat dissipation",
        "cooling infrastructure facility design",
    ],
    "networking": [
        "NVLink switch fabric bandwidth",
        "InfiniBand networking topology",
        "network switch interconnect bandwidth latency",
    ],
    "rack architecture": [
        "rack density form factor design",
        "compute tray shelf architecture layout",
        "rack unit U height configuration",
    ],
    "operations": [
        "deployment commissioning operations",
        "data center monitoring observability",
        "maintenance schedule management",
    ],
    "gpu": [
        "GPU memory bandwidth specifications",
        "GPU thermal design power TDP",
        "accelerator performance cluster",
    ],
    # Note: "nvl72" and "gb200" expansions deliberately omit adjacent product names
    # so they are safe to use in EXPLANATION/COMPARISON modes but still broad enough
    # for EXPLORATORY questions.
    "nvl72": [
        "NVL72 rack system architecture",
        "NVL72 power infrastructure shelf",
        "NVL72 liquid cooling design",
        "NVL72 NVLink switch configuration",
    ],
    "gb200": [
        "GB200 NVL72 rack specifications",
        "GB200 power consumption infrastructure",
        "GB200 deployment design",
    ],
    "facility": [
        "data center facility upgrade power",
        "raised floor power density design",
        "electrical infrastructure capacity",
    ],
}

# SMR / nuclear expansions
_SMR_TOPIC_EXPANSIONS: dict[str, list[str]] = {
    "economics": [
        "SMR levelized cost electricity LCOE",
        "nuclear construction cost overrun",
        "FOAK first-of-a-kind cost premium",
        "NOAK nth-of-a-kind cost reduction",
        "economy of scale factory manufacturing",
        "nuclear financing interest rates overnight capital",
    ],
    "construction": [
        "SMR construction duration schedule timeline",
        "modular construction factory fabrication",
        "construction risk contingency workforce",
        "civil works concrete steel cost",
    ],
    "licensing": [
        "NRC design certification application",
        "nuclear regulatory approval process",
        "construction permit operating license",
        "SMR licensing pathway timeline risk",
    ],
    "reactor design": [
        "SMR core design thermal output MWe",
        "passive safety system design",
        "reactor coolant loop pressure vessel",
    ],
    "grid integration": [
        "grid flexibility load following baseload",
        "capacity factor ancillary services",
        "SMR dispatch flexibility hybrid energy",
    ],
    "fuel cycle": [
        "uranium fuel enrichment supply chain",
        "HALEU high-assay low-enriched uranium",
        "fuel cost nuclear operations",
    ],
    "deployment timeline": [
        "SMR commercial operation date commissioning",
        "deployment timeline first plant schedule",
    ],
    "safety": [
        "passive safety shutdown decay heat",
        "containment design accident scenario",
    ],
    "waste management": [
        "spent nuclear fuel storage disposal",
        "radioactive waste high-level repository",
    ],
    "bwrx": [
        "BWRX-300 design specifications GE Hitachi",
        "BWRX-300 construction cost licensing",
    ],
    "nuscale": [
        "NuScale SMR VOYGR design module",
        "NuScale licensing cost economics",
    ],
}


# ---------------------------------------------------------------------------
# Per-mode query count limits
# ---------------------------------------------------------------------------

_MODE_MAX_QUERIES: dict[QueryMode, int] = {
    QueryMode.FACT_LOOKUP: 3,
    QueryMode.COMPARISON: 5,
    QueryMode.EXPLANATION: 5,
    QueryMode.EXPLORATORY_RESEARCH: 8,
}

_MODE_MIN_QUERIES: dict[QueryMode, int] = {
    QueryMode.FACT_LOOKUP: 1,
    QueryMode.COMPARISON: 3,
    QueryMode.EXPLANATION: 3,
    QueryMode.EXPLORATORY_RESEARCH: 5,
}


# ---------------------------------------------------------------------------
# Core planner
# ---------------------------------------------------------------------------

class RetrievalPlanner:
    """Expand a question into a mode-aware multi-query retrieval plan.

    Parameters
    ----------
    profile:
        Optional domain profile.  When supplied, ``classify_question_topics``
        is used to detect relevant topics for EXPLORATORY expansion.
    max_queries:
        Optional override for the per-mode max query count.
    min_queries:
        Optional override for the per-mode min query count.
    """

    def __init__(
        self,
        profile: "DomainProfile | None" = None,
        *,
        max_queries: int | None = None,
        min_queries: int | None = None,
    ) -> None:
        self._profile = profile
        self._max_override = max_queries
        self._min_override = min_queries

    def plan(self, question: str) -> RetrievalPlan:
        """Return a ``RetrievalPlan`` for *question*."""
        mode = classify_question_mode(question)
        max_q = self._max_override or _MODE_MAX_QUERIES[mode]
        min_q = self._min_override or _MODE_MIN_QUERIES[mode]

        queries: list[str] = []
        seen: set[str] = set()

        def add(q: str) -> None:
            norm = q.strip()
            if norm and norm.lower() not in seen:
                seen.add(norm.lower())
                queries.append(norm)

        # --- Always: primary question as query 1 ---
        add(question)

        if mode == QueryMode.FACT_LOOKUP:
            entity_lock = detect_entity_lock(question, self._profile)
            metric_lock = detect_metric_lock(question, self._profile)
            self._build_fact_lookup_queries(question, entity_lock, metric_lock, add)
        else:
            entity_lock = None
            metric_lock = None
            self._build_broad_queries(question, mode, add, queries, max_q)

        # --- Ensure min_queries via content-word fallback ---
        if len(queries) < min_q:
            for fb in self._fallback_queries(question):
                add(fb)
                if len(queries) >= min_q:
                    break

        queries = queries[:max_q]
        detected_topics = self._detect_topics(question)

        return RetrievalPlan(
            primary_question=question,
            queries=queries,
            detected_topics=detected_topics,
            planner_mode=mode,
            entity_lock=entity_lock,
            metric_lock=metric_lock,
        )

    # ------------------------------------------------------------------
    # Mode-specific query builders
    # ------------------------------------------------------------------

    def _build_fact_lookup_queries(
        self,
        question: str,
        entity_lock: str | None,
        metric_lock: str | None,
        add,
    ) -> None:
        """Generate 1–3 tightly focused queries for FACT_LOOKUP.

        No topic-table expansion is used.  When an entity is detected every
        query is anchored to that entity to prevent adjacent-product bleed.
        """
        if entity_lock and metric_lock:
            e = entity_lock
            # Use profile metric_anchor_queries if available, fall back to hardcoded
            anchor_queries = (
                self._profile.metric_anchor_queries
                if self._profile is not None and self._profile.metric_anchor_queries
                else _METRIC_ANCHOR_QUERIES
            )
            for template in anchor_queries.get(metric_lock, _METRIC_ANCHOR_QUERIES.get(metric_lock, []))[:2]:
                add(template.format(e=e))
        elif entity_lock:
            e = entity_lock
            # Combine entity with the content words from the question
            cwords = _content_words(question)
            if cwords:
                add(f"{e} {' '.join(cwords[:4])}")
            for template in _GENERIC_ANCHOR_QUERIES[:1]:
                add(template.format(e=e))
        else:
            # No entity detected — compact content-word query only
            cwords = _content_words(question)
            if len(cwords) >= 2:
                add(" ".join(cwords[:6]))

    def _build_broad_queries(
        self,
        question: str,
        mode: QueryMode,
        add,
        queries: list[str],
        max_q: int,
    ) -> None:
        """Generate queries for COMPARISON, EXPLANATION, and EXPLORATORY modes."""
        # Compact content-word phrase (query 2)
        cwords = _content_words(question)
        if len(cwords) >= 2:
            add(" ".join(cwords[:6]))

        # Topic-table expansion
        expansion_table = self._pick_expansion_table()
        detected_topics = self._detect_topics(question)
        for topic in detected_topics:
            for exp_q in expansion_table.get(topic, []):
                add(exp_q)
                if len(queries) >= max_q:
                    return

    # ------------------------------------------------------------------
    # Topic detection
    # ------------------------------------------------------------------

    def _detect_topics(self, question: str) -> list[str]:
        """Return profile topics present in the question."""
        if self._profile is not None:
            return sorted(self._profile.classify_question_topics(question))

        q_lower = question.lower()
        found: list[str] = []
        for table in (_AI_DC_TOPIC_EXPANSIONS, _SMR_TOPIC_EXPANSIONS):
            for topic in table:
                trigger = topic.lower().replace(" ", "")
                if trigger in q_lower.replace(" ", "") or any(
                    w in q_lower for w in topic.lower().split()
                ):
                    if topic not in found:
                        found.append(topic)
        return found

    def _pick_expansion_table(self) -> dict[str, list[str]]:
        """Choose the expansion table based on profile name or profile topic_query_expansions."""
        if self._profile is not None and self._profile.topic_query_expansions:
            return self._profile.topic_query_expansions
        if self._profile is None:
            return {**_AI_DC_TOPIC_EXPANSIONS, **_SMR_TOPIC_EXPANSIONS}
        name = (self._profile.name or "").lower()
        if "smr" in name or "nuclear" in name:
            return _SMR_TOPIC_EXPANSIONS
        if "ai" in name or "data_center" in name or "nvidia" in name:
            return _AI_DC_TOPIC_EXPANSIONS
        return {**_AI_DC_TOPIC_EXPANSIONS, **_SMR_TOPIC_EXPANSIONS}

    def _fallback_queries(self, question: str) -> list[str]:
        """Fallback queries built from content-word pairs."""
        words = _content_words(question)
        pairs: list[str] = []
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words))):
                pairs.append(f"{words[i]} {words[j]}")
        return pairs[:4]
