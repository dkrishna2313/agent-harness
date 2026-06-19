"""Domain profile loading and validation.

A domain profile configures the harness for a specific research domain by
supplying topic keywords, gap checks, coverage topics, source quality hints,
and memo section hints.  The default profile (``ai_data_centers``) reproduces
the behaviour that was previously hard-coded across evaluator.py, coverage.py,
gap_detector.py, and agent.py.

Usage::

    from research_agent.profile import load_profile, get_default_profile

    profile = load_profile("ai_data_centers")   # by name from profiles/ dir
    profile = load_profile("./my_profile.yaml")  # by relative path
    profile = load_profile("/abs/path/to/profile.yaml")  # by absolute path

    default = get_default_profile()  # always ai_data_centers
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# Profiles directory sits alongside the project root (one level up from this file)
_PROFILES_DIR = Path(__file__).parent.parent / "profiles"
DEFAULT_PROFILE_NAME = "ai_data_centers"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class WebSearchConfig(BaseModel):
    """Configuration for optional web retrieval (K1.0)."""

    enabled: bool = False
    max_results: int = Field(default=5, ge=1, le=20)
    max_pages: int = Field(default=5, ge=1, le=10)
    timeout_seconds: int = Field(default=20, ge=5, le=120)


class GapCheck(BaseModel):
    """A single research gap subtopic to check against the evidence corpus."""

    topic: str
    keywords: list[str]
    priority: Literal["high", "medium", "low"] = "medium"
    description: str = ""
    rationale: str = ""


# ---------------------------------------------------------------------------
# Main profile model
# ---------------------------------------------------------------------------

class DomainProfile(BaseModel):
    """Configuration bundle for one research domain."""

    name: str
    description: str

    # topic name -> keywords used to detect the topic in a research question
    topic_keywords: dict[str, list[str]] = Field(default_factory=dict)

    # ordered list of topics to include in the coverage matrix
    coverage_topics: list[str] = Field(default_factory=list)

    # topic name -> list of gap subtopics to check against evidence
    research_gap_checks: dict[str, list[GapCheck]] = Field(default_factory=dict)

    # "primary" / "secondary" / "synthetic" -> filename tokens for quality hints
    source_quality_hints: dict[str, list[str]] = Field(default_factory=dict)

    # ordered list of memo section names expected from synthesis
    memo_section_hints: list[str] = Field(default_factory=list)

    # optional: topic -> evidence categories that count toward coverage.
    # When absent for a topic, keyword-based matching is used instead.
    topic_categories: dict[str, list[str]] | None = None

    # topic name -> keywords for contradiction topic classification (J1.5)
    # Checked in definition order; first matching topic wins.
    contradiction_topics: dict[str, list[str]] = Field(default_factory=dict)

    # K1.0 – optional web search config; disabled by default
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)

    # optional: terms used for relevance boosting in evidence scoring
    domain_terms: list[str] | None = None

    # optional: technical vocabulary used for specificity scoring
    specificity_terms: list[str] | None = None

    # Perspective taxonomy: perspective_name -> list of keyword strings
    # Used by perspectives.py to classify evidence items
    perspectives: dict[str, list[str]] = Field(default_factory=dict)

    # Document signals that identify this domain from a filename
    # Used by perspectives.py detect_domain()
    domain_signals: list[str] = Field(default_factory=list)

    # Entity patterns for retrieval planning (ordered most-specific first)
    # Each entry: {name: "GB200 NVL72", signals: ["NVL72", "nvl72"]}
    entity_patterns: list[dict] = Field(default_factory=list)

    # Metric detection patterns (regex strings, compiled at load time)
    # metric_key -> list of regex strings
    metric_patterns: dict[str, list[str]] = Field(default_factory=dict)

    # Metric-anchored query templates: metric_key -> list of query strings with {e} placeholder
    metric_anchor_queries: dict[str, list[str]] = Field(default_factory=dict)

    # Topic query expansion strings: topic_key -> list of query strings
    topic_query_expansions: dict[str, list[str]] = Field(default_factory=dict)

    # Source quality rules (ordered, first match wins)
    # Each entry: {signals: [...], type: "nvidia_technical", score: 5, label: "..."}
    source_quality_rules: list[dict] = Field(default_factory=list)

    # Topic terms for evaluator.py (topic -> list of keyword strings)
    # When absent, falls back to topic_keywords
    evaluator_topic_terms: dict[str, list[str]] = Field(default_factory=dict)

    # Topic section checks for evaluator.py
    # topic -> [section_key, Section Title, missing_section_code, missing_citations_code]
    topic_section_checks: dict[str, list[str]] = Field(default_factory=dict)

    # Coverage gap keywords: topic -> list of gap keyword strings
    coverage_gap_keywords: dict[str, list[str]] = Field(default_factory=dict)

    # Required topic terms for memo quality checks in agent.py
    # topic -> list of required term strings
    required_topic_terms: dict[str, list[str]] = Field(default_factory=dict)

    # resolved path to the loaded profile file (set by loader, not in YAML)
    profile_path: str = Field(default="", exclude=True)

    # ------------------------------------------------------------------
    # Convenience helpers used by pipeline stages
    # ------------------------------------------------------------------

    def classify_contradiction_topic(
        self, claim_a: str, claim_b: str
    ) -> tuple[str, str]:
        """Return ``(topic, topic_source)`` for a contradiction between two claims.

        Searches ``contradiction_topics`` in definition order; the first topic
        whose keywords appear in the combined claim text wins.  Falls back to
        ``("other", "profile:<name>:fallback")`` when no topic matches.
        """
        combined = f"{claim_a} {claim_b}".lower()
        for topic_name, keywords in self.contradiction_topics.items():
            if any(kw in combined for kw in keywords):
                return topic_name, f"profile:{self.name}"
        return "other", f"profile:{self.name}:fallback"

    def classify_question_topics(self, question: str) -> set[str]:
        """Return the set of profile topics detected in *question*."""
        lower = question.lower()
        return {
            topic
            for topic, keywords in self.topic_keywords.items()
            if any(kw in lower for kw in keywords)
        }

    def get_domain_terms(self) -> set[str]:
        """Return the domain term set (falls back to flattened topic keywords)."""
        if self.domain_terms:
            return set(self.domain_terms)
        terms: set[str] = set()
        for kws in self.topic_keywords.values():
            terms.update(kws)
        return terms

    def get_specificity_terms(self) -> set[str]:
        """Return the specificity term set."""
        return set(self.specificity_terms) if self.specificity_terms else set()

    def get_evaluator_topic_terms(self) -> dict[str, list[str]]:
        """Return topic terms for evaluator. Falls back to topic_keywords."""
        return self.evaluator_topic_terms if self.evaluator_topic_terms else self.topic_keywords

    def get_topic_categories(self, topic: str) -> frozenset[str]:
        """Return evidence categories that count toward *topic* coverage.

        Falls back to ``{topic}`` when no explicit mapping is configured.
        """
        if self.topic_categories and topic in self.topic_categories:
            return frozenset(self.topic_categories[topic])
        return frozenset({topic})


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_profile(name_or_path: str) -> DomainProfile:
    """Load a :class:`DomainProfile` by name or path.

    Args:
        name_or_path: A profile name (e.g. ``"ai_data_centers"``) looked up in
            the ``profiles/`` directory, or a path to a ``.yaml`` / ``.yml``
            file (absolute or relative to the current working directory).

    Returns:
        A validated :class:`DomainProfile` instance.

    Raises:
        FileNotFoundError: If the profile file cannot be located.
        ValueError: If the YAML content is not a mapping or is otherwise
            structurally invalid.
    """
    path = _resolve_profile_path(name_or_path)

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Profile file {path} must contain a YAML mapping, "
            f"got {type(raw).__name__}"
        )

    # Parse research_gap_checks: dict[topic, list[dict | str]] -> dict[topic, list[GapCheck]]
    parsed_gaps: dict[str, list[GapCheck]] = {}
    for topic, checks in raw.get("research_gap_checks", {}).items():
        if not isinstance(checks, list):
            parsed_gaps[topic] = []
            continue
        items: list[GapCheck] = []
        for c in checks:
            if isinstance(c, dict):
                items.append(GapCheck(**c))
            else:
                # Plain string shorthand: treat as topic label with itself as keyword
                items.append(GapCheck(topic=str(c), keywords=[str(c).lower()]))
        parsed_gaps[topic] = items

    # Parse contradiction_topics: dict[topic, {keywords: [...]} | [...]]
    raw_ct = raw.get("contradiction_topics", {})
    parsed_ct: dict[str, list[str]] = {}
    for topic, val in raw_ct.items():
        if isinstance(val, dict):
            parsed_ct[topic] = [str(k).lower() for k in val.get("keywords", [])]
        elif isinstance(val, list):
            parsed_ct[topic] = [str(k).lower() for k in val]
        else:
            parsed_ct[topic] = []

    raw_ws = raw.get("web_search", {})
    web_search_cfg = WebSearchConfig(**raw_ws) if isinstance(raw_ws, dict) and raw_ws else WebSearchConfig()

    profile = DomainProfile(
        name=raw.get("name", name_or_path),
        description=raw.get("description", ""),
        topic_keywords=raw.get("topic_keywords", {}),
        coverage_topics=raw.get("coverage_topics", []),
        research_gap_checks=parsed_gaps,
        source_quality_hints=raw.get("source_quality_hints", {}),
        memo_section_hints=raw.get("memo_section_hints", []),
        topic_categories=raw.get("topic_categories"),
        domain_terms=raw.get("domain_terms"),
        specificity_terms=raw.get("specificity_terms"),
        contradiction_topics=parsed_ct,
        web_search=web_search_cfg,
        perspectives=raw.get("perspectives", {}),
        domain_signals=raw.get("domain_signals", []),
        entity_patterns=raw.get("entity_patterns", []),
        metric_patterns=raw.get("metric_patterns", {}),
        metric_anchor_queries=raw.get("metric_anchor_queries", {}),
        topic_query_expansions=raw.get("topic_query_expansions", {}),
        source_quality_rules=raw.get("source_quality_rules", []),
        evaluator_topic_terms=raw.get("evaluator_topic_terms", {}),
        topic_section_checks=raw.get("topic_section_checks", {}),
        coverage_gap_keywords=raw.get("coverage_gap_keywords", {}),
        required_topic_terms=raw.get("required_topic_terms", {}),
        profile_path=str(path),
    )
    return profile


def list_available_profiles() -> list[str]:
    """Return the names of profiles available in the ``profiles/`` directory."""
    if not _PROFILES_DIR.exists():
        return []
    seen: set[str] = set()
    names: list[str] = []
    for ext in ("*.yaml", "*.yml"):
        for p in sorted(_PROFILES_DIR.glob(ext)):
            if p.stem not in seen:
                seen.add(p.stem)
                names.append(p.stem)
    return names


def _resolve_profile_path(name_or_path: str) -> Path:
    """Resolve a name or path string to an absolute :class:`Path`."""
    candidate = Path(name_or_path)

    # Treat as a file path if it has a YAML extension or path separators
    is_path = (
        candidate.suffix in (".yaml", ".yml")
        or "/" in name_or_path
        or "\\" in name_or_path
    )
    if is_path:
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Profile file not found: {candidate}")
        return candidate

    # Otherwise look in the profiles/ directory
    for ext in (".yaml", ".yml"):
        p = _PROFILES_DIR / f"{name_or_path}{ext}"
        if p.exists():
            return p

    available = list_available_profiles()
    raise FileNotFoundError(
        f"Profile '{name_or_path}' not found in {_PROFILES_DIR}. "
        f"Available profiles: {available}"
    )


# ---------------------------------------------------------------------------
# Cached default profile
# ---------------------------------------------------------------------------

_default_profile: DomainProfile | None = None


def get_default_profile() -> DomainProfile:
    """Return the default domain profile (``ai_data_centers``), loading once."""
    global _default_profile
    if _default_profile is None:
        _default_profile = load_profile(DEFAULT_PROFILE_NAME)
    return _default_profile
