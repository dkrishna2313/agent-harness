"""Extraction Analysis — developer diagnostic utility.

Selects evidence-dense zero-yield chunks, runs the production extraction
prompt and a simplified permissive prompt against identical inputs, and
produces a side-by-side report that identifies whether the production
prompt is the cause of zero evidence yield.

Invoked via:
    python3 -m functional_agents.debug analyze-extraction [OPTIONS]

This module never modifies production extraction behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy labels
# ---------------------------------------------------------------------------

PRODUCTION = "production"
SIMPLE = "simple"

# ---------------------------------------------------------------------------
# Simple extraction prompt
# ---------------------------------------------------------------------------

_SIMPLE_PROMPT_TEMPLATE = """\
Extract every explicit factual claim from the chunk below.

Do not summarize.
Do not infer.
Ignore relevance.
Ignore duplicates.
Ignore evidence quality.

Return every concrete factual statement that could later become an EvidenceItem.
Quote supporting text where appropriate.

Respond using the extract_evidence tool with JSON matching the tool schema.

CHUNK ID: {chunk_id}
DOCUMENT: {document_name}

TEXT:
{chunk_text}
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExtractionStatus:
    """Per-call execution quality flags."""
    completed: bool = False
    truncated: bool = False
    parsed: bool = False
    validated: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {
            "completed": self.completed,
            "truncated": self.truncated,
            "parsed": self.parsed,
            "validated": self.validated,
        }


@dataclass
class StrategyResult:
    strategy: str
    evidence_count: int
    claims: list[dict[str, Any]] = field(default_factory=list)
    status: ExtractionStatus = field(default_factory=ExtractionStatus)
    raw_response: str = ""
    parser_error: str = ""
    validation_error: str = ""

    @property
    def usable(self) -> bool:
        """True when the call completed without truncation."""
        return self.status.completed and not self.status.truncated


@dataclass
class ChunkAnalysis:
    chunk_id: str
    document_name: str
    chunk_text_preview: str
    relevance_score: float
    candidate_signals: dict[str, int]
    production: StrategyResult = field(default_factory=lambda: StrategyResult(PRODUCTION, 0))
    simple: StrategyResult = field(default_factory=lambda: StrategyResult(SIMPLE, 0))
    diagnosis: str = ""


@dataclass
class AnalysisSummary:
    chunks_analyzed: int
    successful_comparisons: int
    tool_failures: int
    production_prompt_wins: int
    simple_prompt_wins: int
    equivalent: int
    production_evidence: int
    simple_evidence: int
    average_gain: float
    most_likely_failure_mode: str
    diagnosis_breakdown: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnosis constants
# ---------------------------------------------------------------------------

DIAGNOSIS_CONSERVATIVE = "PRODUCTION_PROMPT_TOO_CONSERVATIVE"
DIAGNOSIS_EMPTY = "NO_EXTRACTABLE_FACTS"
DIAGNOSIS_EQUIVALENT = "PROMPTS_EQUIVALENT"
DIAGNOSIS_TOOL_FAILURE = "TOOL_FAILURE"

_GAIN_THRESHOLD = 2  # simple must return >= this many more items to flag conservative


# ---------------------------------------------------------------------------
# Chunk selection
# ---------------------------------------------------------------------------


def select_analysis_chunks(
    chunk_diagnostics: list[dict[str, Any]],
    chunks_by_id: dict[str, Any],
    *,
    document_filter: str | None = None,
    limit: int = 10,
) -> list[tuple[dict[str, Any], Any]]:
    """Return up to *limit* evidence-dense zero-yield chunks sorted by signal strength.

    Selection criteria:
      chunk_type == evidence_dense  AND  evidence_items_created == 0

    Sorted by:
      relevance_score DESC, numeric_claim_count DESC,
      named_entity_count DESC, policy_or_standard_terms DESC
    """
    candidates = []
    for diag in chunk_diagnostics:
        if diag.get("chunk_type") != "evidence_dense":
            continue
        if diag.get("evidence_items_created", 0) != 0:
            continue
        if document_filter and document_filter not in diag.get("document_name", ""):
            continue

        chunk_id = diag.get("chunk_id", "")
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue

        signals = diag.get("candidate_signals", {})
        sort_key = (
            diag.get("relevance_score", 0.0),
            signals.get("numeric_claim_count", 0),
            signals.get("named_entity_count", 0),
            signals.get("policy_or_standard_terms", 0),
        )
        candidates.append((sort_key, diag, chunk))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [(diag, chunk) for _, diag, chunk in candidates[:limit]]




# ---------------------------------------------------------------------------
# Strategy runners
# ---------------------------------------------------------------------------


def _run_production(client: Any, question: str, chunk: Any) -> StrategyResult:
    """Run the production extraction path on a single chunk."""
    status = ExtractionStatus()
    failure: dict[str, str] = {"raw_response": "", "parser_error": "", "validation_error": ""}
    try:
        items = client.extract_evidence_from_chunks(question, [chunk])
        status.completed = True
        status.parsed = True
        status.validated = True
        claims = [
            {
                "claim": item.claim,
                "category": item.category,
                "confidence": item.confidence,
                "evidence_snippet": item.evidence_snippet[:120],
            }
            for item in items
        ]
        return StrategyResult(
            strategy=PRODUCTION,
            evidence_count=len(items),
            claims=claims,
            status=status,
        )
    except RuntimeError as exc:
        # Distinguish truncation from other errors
        msg = str(exc)
        status.completed = False
        status.truncated = "stop_reason=max_tokens" in msg
        failure["parser_error"] = msg
        LOGGER.warning("production strategy error for %s: %s", chunk.chunk_id, exc)
        return StrategyResult(
            strategy=PRODUCTION, evidence_count=0, status=status,
            parser_error=failure["parser_error"],
        )
    except Exception as exc:
        failure["parser_error"] = str(exc)
        LOGGER.warning("production strategy error for %s: %s", chunk.chunk_id, exc)
        return StrategyResult(
            strategy=PRODUCTION, evidence_count=0, status=status,
            parser_error=failure["parser_error"],
        )


def _run_simple(client: Any, question: str, chunk: Any) -> StrategyResult:
    """Run the permissive simple extraction prompt through the production path.

    Uses extract_evidence_from_chunks(prompt_override=...) so the call shares
    the exact same model, max_tokens, retry, and parsing logic as production.
    Only the prompt text differs.
    """
    prompt = _SIMPLE_PROMPT_TEMPLATE.format(
        chunk_id=chunk.chunk_id,
        document_name=chunk.document_name,
        chunk_text=chunk.text,
    )
    status = ExtractionStatus()
    try:
        items = client.extract_evidence_from_chunks(question, [chunk], prompt_override=prompt)
        status.completed = True
        status.parsed = True
        status.validated = True
        claims = [
            {
                "claim": item.claim,
                "category": item.category,
                "confidence": item.confidence,
                "evidence_snippet": item.evidence_snippet[:120],
            }
            for item in items
        ]
        return StrategyResult(strategy=SIMPLE, evidence_count=len(items), claims=claims, status=status)
    except RuntimeError as exc:
        msg = str(exc)
        status.truncated = "stop_reason=max_tokens" in msg
        LOGGER.warning("simple strategy error for %s: %s", chunk.chunk_id, exc)
        return StrategyResult(strategy=SIMPLE, evidence_count=0, status=status, parser_error=msg)
    except Exception as exc:
        LOGGER.warning("simple strategy error for %s: %s", chunk.chunk_id, exc)
        return StrategyResult(strategy=SIMPLE, evidence_count=0, status=status, parser_error=str(exc))


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


def _diagnose(analysis: ChunkAnalysis) -> str:
    if not analysis.production.usable or not analysis.simple.usable:
        return DIAGNOSIS_TOOL_FAILURE

    prod_n = analysis.production.evidence_count
    simple_n = analysis.simple.evidence_count

    if prod_n == 0 and simple_n == 0:
        return DIAGNOSIS_EMPTY
    if simple_n >= prod_n + _GAIN_THRESHOLD:
        return DIAGNOSIS_CONSERVATIVE
    return DIAGNOSIS_EQUIVALENT


# ---------------------------------------------------------------------------
# Core analysis runner
# ---------------------------------------------------------------------------


def run_analysis(
    client: Any,
    question: str,
    chunk_diagnostics: list[dict[str, Any]],
    chunks_by_id: dict[str, Any],
    *,
    document_filter: str | None = None,
    limit: int = 10,
) -> tuple[list[ChunkAnalysis], AnalysisSummary]:
    """Run the extraction analysis and return per-chunk results and a summary."""
    selected = select_analysis_chunks(
        chunk_diagnostics, chunks_by_id,
        document_filter=document_filter,
        limit=limit,
    )
    LOGGER.info("Extraction analysis: %d chunks selected", len(selected))

    analyses: list[ChunkAnalysis] = []
    for diag, chunk in selected:
        ca = ChunkAnalysis(
            chunk_id=chunk.chunk_id,
            document_name=chunk.document_name,
            chunk_text_preview=chunk.text[:300].replace("\n", " "),
            relevance_score=diag.get("relevance_score", 0.0),
            candidate_signals=diag.get("candidate_signals", {}),
        )
        LOGGER.info("  [production] %s", chunk.chunk_id)
        ca.production = _run_production(client, question, chunk)
        LOGGER.info("  [simple]     %s", chunk.chunk_id)
        ca.simple = _run_simple(client, question, chunk)
        ca.diagnosis = _diagnose(ca)
        LOGGER.info(
            "  → %s  (prod=%d, simple=%d, prod_ok=%s, simple_ok=%s)",
            ca.diagnosis,
            ca.production.evidence_count,
            ca.simple.evidence_count,
            ca.production.usable,
            ca.simple.usable,
        )
        analyses.append(ca)

    # Only count successful comparisons for aggregate stats
    successful = [ca for ca in analyses if ca.diagnosis != DIAGNOSIS_TOOL_FAILURE]
    failed = [ca for ca in analyses if ca.diagnosis == DIAGNOSIS_TOOL_FAILURE]

    prod_total = sum(ca.production.evidence_count for ca in successful)
    simple_total = sum(ca.simple.evidence_count for ca in successful)
    n = len(successful) or 1
    avg_gain = round((simple_total - prod_total) / n, 1)

    breakdown: dict[str, int] = {}
    for ca in analyses:
        breakdown[ca.diagnosis] = breakdown.get(ca.diagnosis, 0) + 1

    conservative_n = breakdown.get(DIAGNOSIS_CONSERVATIVE, 0)
    empty_n = breakdown.get(DIAGNOSIS_EMPTY, 0)
    valid_total = len(successful)

    if conservative_n >= valid_total * 0.5 and valid_total > 0:
        failure_mode = "production_prompt_too_conservative"
    elif empty_n >= valid_total * 0.5 and valid_total > 0:
        failure_mode = "chunks_genuinely_empty"
    elif len(failed) >= len(analyses) * 0.5:
        failure_mode = "tool_failures_dominate"
    else:
        failure_mode = "mixed"

    summary = AnalysisSummary(
        chunks_analyzed=len(analyses),
        successful_comparisons=len(successful),
        tool_failures=len(failed),
        production_prompt_wins=breakdown.get(DIAGNOSIS_EQUIVALENT, 0),
        simple_prompt_wins=breakdown.get(DIAGNOSIS_CONSERVATIVE, 0),
        equivalent=breakdown.get(DIAGNOSIS_EQUIVALENT, 0),
        production_evidence=prod_total,
        simple_evidence=simple_total,
        average_gain=avg_gain,
        most_likely_failure_mode=failure_mode,
        diagnosis_breakdown=breakdown,
    )
    return analyses, summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_comparison_table(analyses: list[ChunkAnalysis]) -> str:
    header = (
        f"{'Chunk':<28} {'Prod':>5} {'Simple':>7} {'Delta':>7} "
        f"{'Comp?':>5}  Diagnosis"
    )
    sep = "-" * len(header)
    rows = [header, sep]
    for ca in analyses:
        delta = ca.simple.evidence_count - ca.production.evidence_count
        delta_str = f"+{delta}" if delta >= 0 else str(delta)
        ok = "✓" if ca.diagnosis != DIAGNOSIS_TOOL_FAILURE else "✗"
        short_id = ca.chunk_id[-28:] if len(ca.chunk_id) > 28 else ca.chunk_id
        rows.append(
            f"{short_id:<28} {ca.production.evidence_count:>5} "
            f"{ca.simple.evidence_count:>7} {delta_str:>7} {ok:>5}  {ca.diagnosis}"
        )
    return "\n".join(rows)


def build_report(
    analyses: list[ChunkAnalysis],
    summary: AnalysisSummary,
    *,
    question: str = "",
    document_filter: str | None = None,
) -> str:
    lines: list[str] = []

    lines.append("# Extraction Analysis Report")
    lines.append("")
    if question:
        lines.append(f"**Question:** {question}")
    if document_filter:
        lines.append(f"**Document:** `{document_filter}`")
    lines.append("")

    # ---- Selected Chunks ---------------------------------------------------
    lines.append("## Selected Chunks")
    lines.append("")
    lines.append(
        f"{summary.chunks_analyzed} evidence-dense zero-yield chunks selected. "
        f"{summary.successful_comparisons} compared successfully, "
        f"{summary.tool_failures} tool failures."
    )
    lines.append("")
    for ca in analyses:
        signals_summary = ", ".join(
            f"{k.replace('_count','').replace('_',' ')}={v}"
            for k, v in ca.candidate_signals.items()
            if v > 0
        )
        status_icon = "✓" if ca.diagnosis != DIAGNOSIS_TOOL_FAILURE else "✗ TOOL_FAILURE"
        lines.append(
            f"- **{ca.chunk_id}** — relevance {ca.relevance_score:.3f}"
            + (f", signals: {signals_summary}" if signals_summary else "")
            + f"  [{status_icon}]"
        )
    lines.append("")

    # ---- Production Prompt Output ------------------------------------------
    lines.append("## Production Prompt Output")
    lines.append("")
    lines.append(
        f"{summary.production_evidence} evidence items extracted across "
        f"{summary.successful_comparisons} valid chunks."
    )
    lines.append("")
    for ca in analyses:
        status_str = _format_status(ca.production.status)
        lines.append(
            f"**{ca.chunk_id}** → {ca.production.evidence_count} items  {status_str}"
        )
        if ca.production.parser_error:
            lines.append(f"  - Error: `{ca.production.parser_error[:160]}`")
        for i, claim in enumerate(ca.production.claims[:4], 1):
            lines.append(f"  {i}. [{claim.get('category','?')}] {claim.get('claim','')[:120]}")
        if len(ca.production.claims) > 4:
            lines.append(f"  _(+{len(ca.production.claims) - 4} more)_")
    lines.append("")

    # ---- Simple Prompt Output ----------------------------------------------
    lines.append("## Simple Prompt Output")
    lines.append("")
    lines.append(
        f"{summary.simple_evidence} evidence items extracted across "
        f"{summary.successful_comparisons} valid chunks."
    )
    lines.append("")
    for ca in analyses:
        status_str = _format_status(ca.simple.status)
        lines.append(
            f"**{ca.chunk_id}** → {ca.simple.evidence_count} items  {status_str}"
        )
        if ca.simple.parser_error:
            lines.append(f"  - Parser error: `{ca.simple.parser_error[:160]}`")
        if ca.simple.validation_error:
            lines.append(f"  - Validation error: `{ca.simple.validation_error[:120]}`")
        if ca.simple.raw_response and ca.simple.parser_error:
            lines.append(f"  - Raw response: `{ca.simple.raw_response[:200]}`")
        for i, claim in enumerate(ca.simple.claims[:4], 1):
            lines.append(f"  {i}. [{claim.get('category','?')}] {claim.get('claim','')[:120]}")
        if len(ca.simple.claims) > 4:
            lines.append(f"  _(+{len(ca.simple.claims) - 4} more)_")
    lines.append("")

    # ---- Comparison --------------------------------------------------------
    lines.append("## Comparison")
    lines.append("")
    lines.append("```")
    lines.append(build_comparison_table(analyses))
    lines.append("```")
    lines.append("")
    lines.append("**Comparison quality summary:**")
    lines.append("")
    lines.append(f"- Successful comparisons: **{summary.successful_comparisons}** / {summary.chunks_analyzed}")
    lines.append(f"- Tool failures: **{summary.tool_failures}**")
    lines.append(f"- Average gain (simple − production): **{summary.average_gain:+.1f}** per successful chunk")
    lines.append("")
    lines.append("**Diagnosis breakdown:**")
    lines.append("")
    for diag, count in sorted(summary.diagnosis_breakdown.items(), key=lambda x: -x[1]):
        pct = 100 * count // max(summary.chunks_analyzed, 1)
        lines.append(f"- `{diag}`: {count} ({pct}%)")
    lines.append("")

    # ---- Root Cause Assessment ---------------------------------------------
    lines.append("## Root Cause Assessment")
    lines.append("")

    conservative_n = summary.diagnosis_breakdown.get(DIAGNOSIS_CONSERVATIVE, 0)
    empty_n = summary.diagnosis_breakdown.get(DIAGNOSIS_EMPTY, 0)
    valid = summary.successful_comparisons or 1

    if summary.most_likely_failure_mode == "production_prompt_too_conservative":
        pct = 100 * conservative_n // valid
        lines.append(
            f"**{conservative_n} of {valid} comparable chunks** ({pct}%) yield significantly "
            "more evidence under the simple prompt than under the production prompt."
        )
        lines.append("")
        lines.append(
            "The production prompt's relevance and quality self-filter is suppressing factual "
            "statements that are present in the source text. The LLM can extract the evidence — "
            "it is the extraction instructions that prevent it from doing so."
        )
        lines.append("")
        lines.append(
            f"Total items: production={summary.production_evidence}, "
            f"simple={summary.simple_evidence}, "
            f"average gain={summary.average_gain:+.1f} per chunk."
        )
    elif summary.most_likely_failure_mode == "chunks_genuinely_empty":
        pct = 100 * empty_n // valid
        lines.append(
            f"**{empty_n} of {valid} comparable chunks** ({pct}%) produce zero evidence "
            "under both prompts. The production prompt is not the bottleneck."
        )
    elif summary.most_likely_failure_mode == "tool_failures_dominate":
        lines.append(
            f"**{summary.tool_failures} of {summary.chunks_analyzed} chunks** failed to "
            "complete for one or both strategies. Valid comparisons are insufficient for a "
            "reliable root cause assessment. Review per-chunk errors above."
        )
    else:
        lines.append("Results are mixed. No single failure mode accounts for the majority of chunks.")
        lines.append("")
        for diag, count in sorted(summary.diagnosis_breakdown.items(), key=lambda x: -x[1]):
            pct = 100 * count // max(summary.chunks_analyzed, 1)
            lines.append(f"- `{diag}`: {count} ({pct}%)")

    lines.append("")

    # ---- Recommendations ---------------------------------------------------
    lines.append("## Recommendations")
    lines.append("")
    if summary.most_likely_failure_mode == "production_prompt_too_conservative":
        lines.append(
            "1. **Relax the relevance self-filter in the production prompt.** "
            "Extract all factual claims; apply relevance ranking post-extraction rather than "
            "instructing the LLM to pre-filter by relevance."
        )
        lines.append("")
        lines.append(
            "2. **Consider a two-pass extraction strategy:** a permissive first pass to maximise "
            "recall, followed by a scoring pass to rank and filter items by relevance."
        )
    elif summary.most_likely_failure_mode == "chunks_genuinely_empty":
        lines.append(
            "1. **Improve chunk classification** to identify and exclude non-extractable "
            "evidence-dense chunks before they enter the extraction pipeline."
        )
        lines.append("")
        lines.append(
            "2. **Review signal scoring thresholds** — chunks with high signal scores but no "
            "extractable content may indicate the classifier is over-sensitive."
        )
    else:
        lines.append(
            "Review the per-chunk results above. No dominant pattern found."
        )

    return "\n".join(lines)


def _format_status(status: ExtractionStatus) -> str:
    flags = []
    if status.completed:
        flags.append("completed=✓")
    else:
        flags.append("completed=✗")
    if status.truncated:
        flags.append("TRUNCATED")
    if status.parsed:
        flags.append("parsed=✓")
    if status.validated:
        flags.append("validated=✓")
    return f"[{', '.join(flags)}]"


# ---------------------------------------------------------------------------
# JSON artefact
# ---------------------------------------------------------------------------


def build_json_artifact(
    analyses: list[ChunkAnalysis],
    summary: AnalysisSummary,
) -> dict[str, Any]:
    return {
        "summary": {
            "chunks_analyzed": summary.chunks_analyzed,
            "successful_comparisons": summary.successful_comparisons,
            "tool_failures": summary.tool_failures,
            "production_prompt_wins": summary.production_prompt_wins,
            "simple_prompt_wins": summary.simple_prompt_wins,
            "equivalent": summary.equivalent,
            "production_evidence": summary.production_evidence,
            "simple_evidence": summary.simple_evidence,
            "average_gain": summary.average_gain,
            "most_likely_failure_mode": summary.most_likely_failure_mode,
            "diagnosis_breakdown": summary.diagnosis_breakdown,
        },
        "chunk_analyses": [
            {
                "chunk_id": ca.chunk_id,
                "document_name": ca.document_name,
                "relevance_score": ca.relevance_score,
                "candidate_signals": ca.candidate_signals,
                "diagnosis": ca.diagnosis,
                "production": {
                    "evidence_count": ca.production.evidence_count,
                    "claims": ca.production.claims,
                    "status": ca.production.status.to_dict(),
                    "parser_error": ca.production.parser_error,
                },
                "simple": {
                    "evidence_count": ca.simple.evidence_count,
                    "claims": ca.simple.claims,
                    "status": ca.simple.status.to_dict(),
                    "raw_response": ca.simple.raw_response,
                    "parser_error": ca.simple.parser_error,
                    "validation_error": ca.simple.validation_error,
                },
            }
            for ca in analyses
        ],
    }
