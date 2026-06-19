"""Post-extraction evidence quality filter — source-locality enforcement.

Evidence items must describe only what their own source document states.
Claims that contain cross-document comparison language (e.g. "contradicts",
"inconsistent with", "in contrast to") indicate the extractor was comparing
sources rather than summarising a single source.  Such items are rejected here
so they never reach the contradiction engine, which is the only component
authorised to compare evidence across sources.

Rationale
---------
* ``300 GW capacity target`` vs ``13 GW/year licensing throughput`` should
  NOT appear as a single claim labelled "inconsistent"; each is a valid
  standalone fact from its own source.
* ``HALEU not available from OECD suppliers`` and ``HALEU available from
  Russia/China`` are complementary facts from different sources — not a
  contradiction.
* ``FOAK historical delays`` and ``future modular construction benefits``
  come from different analytical frames and must not be merged into a
  single comparative claim.

Only the contradiction engine (``contradiction.py``) may compare items.
"""

from __future__ import annotations

import logging
import re

from .schemas import EvidenceItem

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cross-comparison language patterns
# ---------------------------------------------------------------------------
# Any of these in a claim field indicate the extractor was reasoning across
# documents rather than summarising a single source.

_CROSS_COMPARISON_RE = re.compile(
    # All inflections of "contradict"
    r"\bcontradict(?:s|ed|ing|ion|ions|ory)?\b"
    # "inconsistent" (the check deliberately matches the bare word so that
    # both "inconsistent" and "inconsistent with" are caught)
    r"|\binconsistent\b"
    # "conflicting" as a descriptor, "conflicts with" as a cross-reference
    r"|\bconflicting\b|\bconflicts?\s+with\b"
    # "in contrast to" — explicit comparative frame
    r"|\bin\s+contrast\s+to\b"
    # "unlike [other/previous/the other/earlier]"
    r"|\bunlike\s+(?:the\s+)?(?:other|previous|earlier|above|another)\b"
    # "disagree(s/d)" or "disagreement" when comparing sources
    r"|\bdisagree[sd]?\s+with\b|\bin\s+disagreement\s+with\b",
    re.IGNORECASE,
)


def is_source_local(claim: str) -> bool:
    """Return True when *claim* contains no cross-document comparison language.

    A source-local claim describes only what its own source document states.
    It does not reference, compare, or contrast with other documents.
    """
    return _CROSS_COMPARISON_RE.search(claim) is None


def sanitize_evidence_items(
    items: list[EvidenceItem],
    *,
    stage: str = "extraction",
) -> list[EvidenceItem]:
    """Return *items* with cross-comparison claims removed.

    Each rejected item is logged at WARNING level.  *stage* is included in
    the log message to indicate where in the pipeline the rejection occurred
    (e.g. ``"mock_extraction"`` or ``"claude_extraction"``).
    """
    clean: list[EvidenceItem] = []
    for item in items:
        if is_source_local(item.claim):
            clean.append(item)
        else:
            LOGGER.warning(
                "evidence_filter [%s]: rejected item from '%s' — "
                "claim contains cross-document comparison language: %r",
                stage,
                item.source_document,
                item.claim[:140],
            )
    if len(clean) < len(items):
        LOGGER.info(
            "evidence_filter [%s]: %d of %d items retained after source-locality check",
            stage,
            len(clean),
            len(items),
        )
    return clean
