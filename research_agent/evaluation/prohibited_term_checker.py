"""Context-aware prohibited term checking (J3.1c).

Current (J3.1a): if prohibited_term in answer → penalty
Target  (J3.1c): if prohibited_term in answer AND context is not negating → penalty

Motivation
----------
A benchmark answer about the Grace CPU might correctly say:
  "Grace eliminates the need for a PCIe connection."
The answer is factually correct, but the literal phrase "PCIe connection"
triggers a false-positive violation.

Similarly, an SMR construction answer might say:
  "BWRX-300 targets 3–4 years, not 6 months as sometimes mis-stated."
The phrase "6 months" appears but is explicitly negated.

Classification
--------------
CONTEXT_ALLOWED    — term found, but in a negating or contrastive sentence.
                     Penalty not applied.
HARD_PROHIBITED    — term found without negating context.
                     Penalty applied.

Note: must_not_include exact matching is still the baseline (J3.1a.6).
Context detection only exempts terms where the sentence clearly negates or
contrasts the claim, preventing false-positive penalties.

Public API
----------
check_prohibited_term(term, answer_text) → ProhibitedTermResult
build_prohibition_stats(results)          → dict   (J3.1c.5)
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Context detection patterns
# ---------------------------------------------------------------------------

# Negation words expected within _PRE_WINDOW chars BEFORE the prohibited term.
# Example: "without a PCIe connection", "not a PCIe connection"
_PRE_WINDOW = 40   # characters before the term to scan for direct negation

# Broader paragraph window for fallback check (adjacent sentences).
# When the prohibited term's own sentence has no exemption signal, the
# surrounding context (previous/next sentence) is checked as a softer gate.
_PARA_WINDOW = 350  # chars on each side of the term position

_NEGATION_PRE = re.compile(
    r"\b(no|not|cannot|can't|never|without|neither|nor|no\s+longer)\b",
    re.I,
)

# Exemption patterns across the sentence (or paragraph) containing the term.
# Priority order within the regex does not matter; any match exempts the term.
_EXEMPT = re.compile(
    r"\b("
    # --- Elimination / replacement (direct) ---
    r"eliminat\w*"          # eliminates, eliminated, eliminating
    r"|replac\w*"           # replaces, replaced, replacing
    r"|remov\w*"            # removes, removed
    r"|supersed\w*"         # supersedes, superseded
    r"|obsolet\w*"          # obsolete, obsoleted
    r"|instead\s+of"
    r"|rather\s+than"
    r"|avoid\w*"            # avoids, avoided, avoiding
    r"|prevent\w*"          # prevents, prevented
    # --- Performance comparison (alternative is better) ---
    r"|exceed\w*"           # exceeds, exceeded, exceeding — alternative exceeds PCIe
    r"|surpass\w*"          # surpasses, surpassed
    r"|outperform\w*"       # outperforms, outperformed
    r"|faster\s+than"       # NVLink is faster than PCIe connection
    r"|greater\s+than"      # greater bandwidth than PCIe
    r"|higher\s+than"       # higher bandwidth than PCIe
    r"|lower\s+than"        # lower latency than PCIe (NVLink wins)
    r"|beyond\s+what"       # "beyond what a PCIe connection delivers"
    # --- PCIe as legacy / old approach ---
    r"|legacy"              # legacy PCIe connection
    r"|conventional"        # conventional PCIe approach
    r"|traditional"         # traditional PCIe
    r"|predecessor\w*"      # PCIe as predecessor
    r"|previous\s+\w+\s+approach"  # previous PCIe approach
    r"|used\s+to\s+rel"     # "used to rely on PCIe"
    # --- Bottleneck / limitation language ---
    r"|bottleneck"          # "PCIe bottleneck" = problem with PCIe
    r"|limitation"
    r"|limited\s+to"        # "PCIe bandwidth is limited to X" (cap / inferior baseline)
    r"|constrain\w*"        # constrained by PCIe
    r"|overhead"            # PCIe overhead
    r"|plagued"             # "plagued PCIe-attached systems"
    # --- Miscellaneous exemptions ---
    r"|no\s+longer\s+need"
    r"|not\s+via"
    r"|not\s+using"
    r"|not\s+required"
    r"|not\s+necessar"      # necessary / necessitate
    r"|mis-stat\w*"         # mis-stated
    r"|incorrect\w*"        # incorrectly stated
    r"|reduc\w+\s+\w+\s+overhead"  # "reduces the power overhead"
    r")",
    re.I,
)

# Contrastive connectors anywhere in the sentence/paragraph.
# J4.5d.3: product-family terms — NVL36 alongside NVL72/product-family language
#   is a comparison context, not a hallucination.
# J4.5d.4: multi-month/year ranges — "24–36 months" near "6 months" signals
#   that "6 months" is being compared or corrected, not asserted as fact.
_CONTRASTIVE = re.compile(
    r"\b(unlike|whereas|however|by\s+contrast|in\s+contrast"
    r"|compared\s+to|versus|vs\.?|rather|contrast"
    # J4.5d.3 – NVL product-family / scaling context: the correct sibling (NVL72)
    # or product-family language signals this is a comparison, not a hallucination.
    # Never add the prohibited term itself here.
    r"|nvl72|product\s+famil\w*|lineup|portfolio|sku"
    r"|scaling\s+factor|different\s+config\w*"
    # J4.5d.4 – longer duration alongside a short one ("6 months")
    r"|\d+\s*[-–]\s*\d+\s*months?"   # "24-36 months", "36–48 months"
    r"|multi.?year\w*"                # multi-year
    r"|several\s+years?"              # several years
    r"|decade"                        # decade-long
    r"|years?\s+to\s+build"           # years to build
    r"|year\s+construction"           # year construction timeline
    r")\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Result dataclass (J3.1c.4 audit trace)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProhibitedTermResult:
    """Result of context-checking one prohibited term against an answer.

    Fields
    ------
    term            : the must_not_include string
    found           : True if the term appears in the answer at all
    context_window  : the sentence(s) containing the term (up to 200 chars)
    classification  : "hard_prohibited" | "context_allowed" | "not_found"
    penalty_applied : True only when classification == "hard_prohibited"
    reason          : human-readable explanation
    """

    term: str
    found: bool
    context_window: str
    classification: str    # "hard_prohibited" | "context_allowed" | "not_found"
    penalty_applied: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "prohibited_term": self.term,
            "found": self.found,
            "context": self.context_window,
            "classification": self.classification,
            "penalty_applied": self.penalty_applied,
            "reason": self.reason,
        }


# ---------------------------------------------------------------------------
# Core check function
# ---------------------------------------------------------------------------

def check_prohibited_term(term: str, answer_text: str) -> ProhibitedTermResult:
    """Check whether *term* appears in *answer_text* and classify the usage.

    Returns
    -------
    ProhibitedTermResult with:
      - classification = "not_found"      if the term is absent
      - classification = "context_allowed" if negating/contrastive context detected
      - classification = "hard_prohibited" if term appears without exemption context
    """
    answer_lower = answer_text.lower()
    term_lower = term.lower()

    if term_lower not in answer_lower:
        return ProhibitedTermResult(
            term=term,
            found=False,
            context_window="",
            classification="not_found",
            penalty_applied=False,
            reason="term_absent",
        )

    # Find the first occurrence position
    pos = answer_lower.find(term_lower)

    # Extract surrounding sentence (split on sentence boundaries)
    sentence = _extract_sentence(answer_text, pos)
    sentence_lower = sentence.lower()

    # 1. Direct pre-term negation: "not a PCIe connection", "no 6 months"
    pre_start = max(0, pos - _PRE_WINDOW)
    pre_window = answer_lower[pre_start:pos]
    if _NEGATION_PRE.search(pre_window):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],
            classification="context_allowed",
            penalty_applied=False,
            reason="pre_term_negation",
        )

    # 2. Sentence-level exemption patterns (eliminates, replaces, bottleneck, exceeds, legacy…)
    if _EXEMPT.search(sentence_lower):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],
            classification="context_allowed",
            penalty_applied=False,
            reason="sentence_exemption",
        )

    # 2b. Interrogative sentence — term is mentioned as a topic in a question, not asserted.
    # "Are there half-rack (NVL36) configurations?" asks about NVL36 but does not claim
    # that the answer being evaluated is about NVL36.  Generalises to any entity mentioned
    # as the subject of an open question rather than as a direct factual claim.
    if sentence.rstrip().endswith("?"):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],
            classification="context_allowed",
            penalty_applied=False,
            reason="interrogative_sentence",
        )

    # 3. Contrastive connector in sentence
    if _CONTRASTIVE.search(sentence_lower):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],
            classification="context_allowed",
            penalty_applied=False,
            reason="contrastive_context",
        )

    # 4. Paragraph-level fallback: exemption signal may be in an adjacent sentence.
    #    Widen the search window to ±_PARA_WINDOW chars around the term position.
    paragraph = _extract_paragraph(answer_text, pos)
    paragraph_lower = paragraph.lower()

    if _EXEMPT.search(paragraph_lower):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],   # audit still shows the narrow sentence
            classification="context_allowed",
            penalty_applied=False,
            reason="paragraph_exemption",
        )

    if _CONTRASTIVE.search(paragraph_lower):
        return ProhibitedTermResult(
            term=term,
            found=True,
            context_window=sentence[:200],
            classification="context_allowed",
            penalty_applied=False,
            reason="paragraph_contrastive",
        )

    # No exemption found — apply penalty
    return ProhibitedTermResult(
        term=term,
        found=True,
        context_window=sentence[:200],
        classification="hard_prohibited",
        penalty_applied=True,
        reason="no_exemption_found",
    )


def _extract_sentence(text: str, pos: int) -> str:
    """Return the sentence containing character position *pos* in *text*."""
    start = pos
    while start > 0 and text[start - 1] not in ".!?\n":
        start -= 1
    end = pos + 1
    while end < len(text) and text[end - 1] not in ".!?\n":
        end += 1
    return text[start:end].strip()


def _extract_paragraph(text: str, pos: int) -> str:
    """Return up to _PARA_WINDOW chars on each side of *pos*, trimmed to sentence edges.

    Used as a broader context window for the paragraph-level fallback check — an
    exemption signal in an adjacent sentence (e.g. "NVLink-C2C replaces this.")
    following a sentence that contains the prohibited term is still valid context.
    """
    start = max(0, pos - _PARA_WINDOW)
    end = min(len(text), pos + _PARA_WINDOW)
    # Walk outward to nearest sentence boundary so we don't cut mid-sentence
    while start > 0 and text[start - 1] not in ".!?\n":
        start -= 1
    while end < len(text) and text[end] not in ".!?\n":
        end += 1
    return text[start:end].strip()


# ---------------------------------------------------------------------------
# Batch helpers and statistics (J3.1c.5)
# ---------------------------------------------------------------------------

def check_all_prohibited_terms(
    terms: list[str],
    answer_text: str,
) -> list[ProhibitedTermResult]:
    """Run context check for every must_not_include term."""
    return [check_prohibited_term(term, answer_text) for term in terms]


def build_prohibition_stats(results: list[ProhibitedTermResult]) -> dict:
    """Return J3.1c.5 audit statistics across all prohibited term results.

    Keys
    ----
    hard_prohibited    : terms found and penalized (genuine violations)
    context_allowed    : terms found but exempted (valid contextual mention)
    not_found          : terms not present in the answer
    total_checked      : total terms checked
    """
    hard = sum(1 for r in results if r.classification == "hard_prohibited")
    allowed = sum(1 for r in results if r.classification == "context_allowed")
    absent = sum(1 for r in results if r.classification == "not_found")
    return {
        "hard_prohibited": hard,
        "context_allowed": allowed,
        "not_found": absent,
        "total_checked": len(results),
    }
