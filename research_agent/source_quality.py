"""Deterministic source quality classification.

Classification is rule-based and filename-driven.  No LLM or network calls.

Score definitions
-----------------
5  NVIDIA primary technical: technical blogs, architecture documents,
   specification sheets, platform documentation.
4  NVIDIA press/marketing, vendor technical whitepapers, solution briefs.
3  Independent technical analysis: StorageReview, technical journalism,
   industry analysis.
2  Blogs, commentary, community content, generic unknown sources.
1  Synthetic test files, .txt files without recognised signals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .schemas import SourceQuality, SourceType

if TYPE_CHECKING:
    from .profile import DomainProfile

# ---------------------------------------------------------------------------
# Profile hint tier → (source_type, score)
# Profile YAML can declare these tier names under ``source_quality_hints``.
# Evaluation order: synthetic first (catch test files), then descending score.
# ---------------------------------------------------------------------------
_PROFILE_TIER_ORDER: list[str] = ["synthetic", "primary", "secondary", "tertiary"]
_PROFILE_TIER_MAP: dict[str, tuple[SourceType, int]] = {
    "primary":   ("authoritative_primary", 5),
    "secondary": ("industry_vendor",       4),
    "tertiary":  ("independent_technical", 3),
    "synthetic": ("synthetic",             1),
}

# ---------------------------------------------------------------------------
# Classification rule table
# ---------------------------------------------------------------------------
# Each entry: (all_tokens_required, source_type, score, rationale)
# `all_tokens_required` is a list of lowercase substrings ALL of which must be
# present in the lowercased filename for the rule to fire.
# First matching rule wins.

_RULES: list[tuple[list[str], SourceType, int, str]] = [
    # ---- Score 5: NVIDIA primary technical --------------------------------
    (["nvidia", "technical blog"],   "nvidia_technical", 5, "NVIDIA technical blog"),
    (["nvidia", "architecture"],     "nvidia_technical", 5, "NVIDIA architecture document"),
    (["nvidia", "specification"],    "nvidia_technical", 5, "NVIDIA specification sheet"),
    (["nvidia", "spec"],             "nvidia_technical", 5, "NVIDIA specification sheet"),
    (["nvidia", "platform"],         "nvidia_technical", 5, "NVIDIA platform documentation"),
    (["nvidia", "blog"],             "nvidia_technical", 5, "NVIDIA technical blog"),
    (["inside the nvidia"],          "nvidia_technical", 5, "NVIDIA architecture deep-dive"),
    (["nvidia", "inside"],           "nvidia_technical", 5, "NVIDIA architecture deep-dive"),
    (["nvidia", "rubin"],            "nvidia_technical", 5, "NVIDIA Rubin platform document"),
    (["nvidia", "blackwell"],        "nvidia_technical", 5, "NVIDIA Blackwell platform document"),
    (["nvidia", "nvl"],              "nvidia_technical", 5, "NVIDIA NVL platform document"),
    (["nvidia", "vera"],             "nvidia_technical", 5, "NVIDIA Vera platform document"),
    (["nvidia", "datasheet"],        "nvidia_technical", 5, "NVIDIA datasheet"),
    (["nvidia", "data sheet"],       "nvidia_technical", 5, "NVIDIA datasheet"),
    (["nvidia", "reference"],        "nvidia_technical", 5, "NVIDIA reference document"),
    (["nvidia", "developer"],        "nvidia_technical", 5, "NVIDIA developer documentation"),
    # ---- Score 3 (specific): rules placed BEFORE generic NVIDIA catch-all --
    # so that e.g. "StorageReview.com NVIDIA GB200 Analysis.pdf" correctly
    # scores as independent technical (3) rather than NVIDIA marketing (4).
    (["storagereview"],              "independent_technical", 3, "StorageReview independent analysis"),
    (["dissecting"],                 "independent_technical", 3, "Independent technical analysis"),
    (["microbenchmark"],             "independent_technical", 3, "Technical microbenchmark analysis"),
    (["benchmarks"],                 "independent_technical", 3, "Technical benchmark analysis"),
    # ---- Score 4: NVIDIA marketing / vendor whitepapers -------------------
    (["nvidia"],                     "nvidia_marketing", 4, "NVIDIA press or marketing content"),
    (["whitepaper"],                 "vendor_whitepaper", 4, "Vendor technical whitepaper"),
    (["white paper"],                "vendor_whitepaper", 4, "Vendor technical whitepaper"),
    (["solution brief"],             "vendor_brief",     4, "Vendor solution brief"),
    (["solution_brief"],             "vendor_brief",     4, "Vendor solution brief"),
    (["press release"],              "press_release",    4, "Press release"),
    # ---- Score 3 (generic): placed AFTER NVIDIA catch-all so "NVIDIA…
    # Analysis.pdf" still gets score 4 rather than 3 ----------------------
    (["analysis"],                   "independent_technical", 3, "Technical analysis"),
    (["review"],                     "independent_technical", 3, "Technical review"),
    # ---- Score 2: Blogs, commentary, generic sources ----------------------
    (["blog"],                       "blog", 2, "Blog or commentary content"),
    (["commentary"],                 "blog", 2, "Commentary content"),
    (["opinion"],                    "blog", 2, "Opinion content"),
    (["news"],                       "blog", 2, "News article"),
    # ---- Score 1: Synthetic / test files ----------------------------------
    (["test_"],                      "synthetic", 1, "Synthetic test file"),
    (["_test."],                     "synthetic", 1, "Synthetic test file"),
    (["synthetic"],                  "synthetic", 1, "Synthetic test file"),
    (["fixture"],                    "synthetic", 1, "Test fixture file"),
]


def classify_source_quality(document_name: str) -> SourceQuality:
    """Return a deterministic SourceQuality for *document_name*.

    Classification uses only the filename; no LLM or network calls are made.
    """
    lower = document_name.lower()

    for tokens, source_type, score, rationale in _RULES:
        if all(t in lower for t in tokens):
            return SourceQuality(
                source_document=document_name,
                source_type=source_type,
                source_quality_score=score,
                rationale=rationale,
            )

    # Extension-based fallback: bare .txt files (not matched above) are
    # most likely synthetic/test inputs.
    if lower.endswith(".txt"):
        return SourceQuality(
            source_document=document_name,
            source_type="synthetic",
            source_quality_score=1,
            rationale="Unclassified text file; treated as synthetic/low-quality source",
        )

    # Default: unknown source, generic score
    return SourceQuality(
        source_document=document_name,
        source_type="unknown",
        source_quality_score=2,
        rationale="Insufficient filename signals; default quality assumed",
    )


def classify_source_quality_with_profile(
    document_name: str,
    profile: "DomainProfile",
) -> SourceQuality:
    """Classify *document_name* using ``profile.source_quality_hints``.

    Tiers are checked in order: ``synthetic`` → ``primary`` → ``secondary``
    → ``tertiary``.  Any token in the tier's hint list that appears as a
    substring of the lowercased filename triggers a match.

    Falls back to the base ``classify_source_quality`` if no profile hint
    matches.
    """
    lower = document_name.lower()
    hints = profile.source_quality_hints

    for tier in _PROFILE_TIER_ORDER:
        tokens = hints.get(tier, [])
        for token in tokens:
            if token.lower() in lower:
                source_type, score = _PROFILE_TIER_MAP[tier]
                return SourceQuality(
                    source_document=document_name,
                    source_type=source_type,
                    source_quality_score=score,
                    rationale=(
                        f"Profile '{profile.name}' hint match "
                        f"(tier={tier}, token='{token}')"
                    ),
                )

    # No profile hint matched — fall back to domain-agnostic rules
    return classify_source_quality(document_name)


def build_source_quality_map(
    document_names: list[str],
    profile: "DomainProfile | None" = None,
) -> dict[str, SourceQuality]:
    """Return a {document_name: SourceQuality} map for *document_names*.

    When *profile* is provided its ``source_quality_hints`` are applied first;
    the base NVIDIA-specific rule table is used as a fallback (or exclusively
    when no profile is given).
    """
    if profile is not None:
        return {
            name: classify_source_quality_with_profile(name, profile)
            for name in document_names
        }
    return {name: classify_source_quality(name) for name in document_names}
