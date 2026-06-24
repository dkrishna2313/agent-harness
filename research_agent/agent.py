"""Research workflow orchestration."""

from __future__ import annotations

import re
import logging
from collections.abc import Iterable, Sequence

from .chunker import chunk_documents, compute_chunk_diagnostics, compute_evidence_yield_metrics
from .evidence_recovery import attribute_evidence_to_chunks, run_recovery_pass, compute_zero_yield_documents
from .extraction_diagnostics import (
    build_failure_diagnostics,
    build_failure_summary,
    compute_top_missed_chunks,
    analyze_document_failures,
)
from .claude_client import LLMClient, MockClaudeClient, aggregate_call_traces
from .contradiction import detect_contradictions, enrich_evidence_items, build_extraction_stats, compute_suppression_metrics
from .evidence_enricher import enrich_evidence_with_metadata, build_evidence_density_stats
from .perspectives import build_diversity_metrics, detect_domain, select_diverse_evidence
from .coverage import compute_coverage_matrix
from .evidence_filter import sanitize_evidence_items
from .gap_detector import detect_gaps
from .evaluator import classify_question_topics, evaluate_memo
from .profile import DomainProfile, get_default_profile
from .retrieval import DEFAULT_TOP_CHUNKS, RetrievalScore, select_top_chunks, select_top_chunks_multi
from .retrieval_planner import RetrievalPlan, RetrievalPlanner
from .schemas import Chunk, CoverageArea, EvidenceItem, ResearchMemo, ResearchPlan, SourceDocument, SourceQuality, assign_evidence_ids
from .source_quality import build_source_quality_map, classify_source_quality
from .web_search import WebDocument, web_retrieve
from .web_cache import WebPageCache

LOGGER = logging.getLogger(__name__)

DEFAULT_TOP_EVIDENCE = 50
# Hard cap on evidence items sent to synthesize_memo to keep the prompt within
# a safe token budget (~138 tokens/item × 15 = ~2 070 input tokens).
# Synthesis quality is determined by ranking; items beyond this limit add noise.
MAX_SYNTHESIS_EVIDENCE = 15

_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "center",
    "data",
    "explain",
    "for",
    "from",
    "how",
    "implications",
    "into",
    "its",
    "the",
    "their",
    "this",
    "what",
    "with",
}

_DOMAIN_TERMS = {
    "blackwell",
    "cooling",
    "cpu",
    "gpu",
    "inference",
    "networking",
    "nvidia",
    "nvl72",
    "power",
    "rack",
    "rubin",
    "thermal",
    "vera",
}

_POWER_TERMS = {"power", "watt", "kilowatt", "kw", "megawatt", "mw", "rack", "distribution"}
_COOLING_TERMS = {"cool", "cooling", "thermal", "liquid", "heat", "facility"}
_REQUIRED_TOPIC_TERMS = {
    "architecture": {
        "architecture",
        "compute",
        "gpu",
        "cpu",
        "accelerator",
        "platform",
        "system",
    },
    "power": {
        "power",
        "watt",
        "kilowatt",
        "kw",
        "megawatt",
        "mw",
        "distribution",
        "energy",
    },
    "cooling": {"cool", "cooling", "thermal", "liquid", "heat"},
    "networking": {"network", "networking", "ethernet", "infiniband", "nvlink", "switch"},
    "rack architecture": {"rack architecture", "rack-scale", "rack", "nvl72", "mgx"},
}

_SOURCE_QUALITY_SIGNALS = {
    "official": {
        "nvidia",
        "ocp",
        "open compute",
        "opencompute",
        "amd",
        "intel",
        "dell",
        "hpe",
        "supermicro",
        "spec",
        "specification",
        "datasheet",
        "whitepaper",
        "reference",
        "standard",
        "developer",
        "docs",
    },
    "technical": {
        "technical",
        "architecture",
        "manual",
        "guide",
        "design",
        "brief",
        "paper",
    },
    "generic": {
        "blog",
        "commentary",
        "opinion",
        "news",
        "press",
        "article",
        "summary",
    },
}

_SPECIFICITY_TERMS = {
    "bbu",
    "busway",
    "cabinet",
    "cdu",
    "chilled water",
    "connectx",
    "ethernet",
    "gb",
    "gpu",
    "hbm",
    "infiniband",
    "kw",
    "liquid",
    "mgx",
    "mw",
    "nvlink",
    "nvl72",
    "pdu",
    "rack",
    "spectrum",
    "switch",
    "thermal",
    "tray",
    "ups",
    "voltage",
    "watt",
}


class DcPowerAgent:
    """Simple local-documents research agent."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        top_evidence: int = DEFAULT_TOP_EVIDENCE,
        top_chunks: int = DEFAULT_TOP_CHUNKS,
        profile: DomainProfile | None = None,
    ) -> None:
        self.client = client or MockClaudeClient()
        self.top_evidence = max(1, top_evidence)
        self.top_chunks = max(1, top_chunks)
        # Use the supplied profile, or the default ai_data_centers profile.
        self.profile: DomainProfile = profile if profile is not None else get_default_profile()
        # K1.0 – shared disk cache for downloaded web pages
        self._web_cache = WebPageCache()

    def analyze(self, question: str, documents: Sequence[SourceDocument]) -> ResearchMemo:
        """Run the research workflow and return a structured memo."""

        is_mock = getattr(self.client, "is_mock", False)

        # Build source quality map once for all documents in this run
        source_quality_map = build_source_quality_map(
            [doc.path.name for doc in documents],
            profile=self.profile,
        )

        if is_mock:
            chunks = chunk_documents(documents)
            planner = RetrievalPlanner(profile=self.profile)
            retrieval_plan = planner.plan(question)
            selected_chunks, retrieval_scores, retrieval_plan_stats = select_top_chunks_multi(
                chunks, retrieval_plan.queries, top_n=self.top_chunks, source_quality_map=source_quality_map
            )
            LOGGER.debug(
                "Evidence pipeline (mock): %d chunks, %d selected by retrieval (%d queries)",
                len(chunks),
                len(selected_chunks),
                retrieval_plan.query_count,
            )
            evidence = extract_evidence(question, documents, source_quality_map=source_quality_map, profile=self.profile)
            LOGGER.debug(
                "Evidence pipeline (mock): %d total extracted and ranked",
                len(evidence),
            )
            # J3.2 — diversity-aware selection for mock path
            synthesis_evidence = select_diverse_evidence(
                evidence, top_n=self.top_evidence, max_per_perspective=8
            )
            synthesis_evidence = select_diverse_evidence(
                synthesis_evidence, top_n=MAX_SYNTHESIS_EVIDENCE, max_per_perspective=3
            )
            LOGGER.debug(
                "Evidence pipeline (mock): %d filtered to synthesis (top_evidence=%d, cap=%d, total=%d)",
                len(synthesis_evidence),
                self.top_evidence,
                MAX_SYNTHESIS_EVIDENCE,
                len(evidence),
            )
            memo = build_mock_memo(
                question=question,
                documents=documents,
                evidence=evidence,
                synthesis_evidence=synthesis_evidence,
                model_response="MOCK_LLM_RESPONSE",
                mock_llm=is_mock,
            )
            memo = _with_evidence_metadata(
                memo,
                evidence=evidence,
                synthesis_evidence=synthesis_evidence,
                top_evidence=self.top_evidence,
            )
            chunk_scores = _retrieval_scores_to_chunk_scores(chunks, retrieval_scores)
            chunk_diagnostics = compute_chunk_diagnostics(
                chunks, selected_chunks, evidence, chunk_scores
            )
            _log_chunk_diagnostics(chunk_diagnostics)
            chunk_meta = _build_chunk_metadata(chunks, selected_chunks, evidence, chunk_diagnostics)
            retrieval_meta = _build_retrieval_metadata(selected_chunks, retrieval_scores)
            # J1.6 – enrich evidence with entity/scope before detection and
            # store back on the memo so downstream outputs carry the fields.
            evidence = enrich_evidence_items(evidence)
            # J3.1 – type + topic; J3.2 – perspective enrichment
            evidence = enrich_evidence_with_metadata(evidence, self.profile)
            extraction_stats = build_extraction_stats(evidence)
            evidence_density = build_evidence_density_stats(evidence, len(chunks))
            # JH1 – post-extraction attribution: set source_chunk_id by matching snippets
            evidence = attribute_evidence_to_chunks(evidence, chunks)
            # JH1a – recovery pass: extract from high-signal zero-evidence chunks
            _recovery = run_recovery_pass(
                chunks, selected_chunks, evidence, chunk_diagnostics,
                question=question,
                source_quality_map=source_quality_map,
                profile=self.profile,
            )
            evidence = list(evidence) + _recovery.recovered_items
            zero_yield_docs = compute_zero_yield_documents(chunks, evidence, documents)
            # JH1 – evidence yield metrics (after attribution + recovery)
            evidence_yield = compute_evidence_yield_metrics(
                chunks, selected_chunks, evidence, documents_loaded=len(documents)
            )
            # JH1b – failure pipeline diagnostics
            _topic_term_sets = _build_topic_term_sets(self.profile)
            _failure_diags = build_failure_diagnostics(
                chunk_diagnostics, chunks, evidence, _topic_term_sets, is_mock=True
            )
            _failure_summary = build_failure_summary(_failure_diags)
            _top_missed = compute_top_missed_chunks(_failure_diags)
            _giant_doc_analysis = analyze_document_failures(
                "31523cfd-5bbb-4c65-a4e0-8e699e370f95.pdf",
                _failure_diags,
                chunk_diags=chunk_diagnostics,
                evidence=evidence,
            )
            # J3.2 — diversity metrics after perspective enrichment
            _mock_domain = detect_domain(documents[0].path.name if documents else "")
            retrieval_diversity_mock = build_diversity_metrics(evidence, _mock_domain)
            memo = memo.model_copy(update={"evidence": evidence})
            suppressed: list = []
            contradictions = detect_contradictions(evidence, source_quality_map, profile=self.profile, out_suppressed=suppressed)
            contradiction_metrics = compute_suppression_metrics(suppressed, len(contradictions))
            research_gaps = detect_gaps(question, evidence, self.profile)
            coverage_matrix = compute_coverage_matrix(
                question, evidence,
                research_gaps=research_gaps,
                source_quality_map=source_quality_map,
                profile=self.profile,
            )
            sq_map_serialized = {k: v.model_dump() for k, v in source_quality_map.items()}
            memo = memo.model_copy(update={
                "contradictions": contradictions,
                "research_gaps": research_gaps,
                "coverage_matrix": coverage_matrix,
                "metadata": {
                    **memo.metadata,
                    **chunk_meta,
                    **retrieval_meta,
                    "contradictions": [c.model_dump() for c in contradictions],
                    "suppressed_comparisons": [s.model_dump() for s in suppressed],
                    "contradiction_metrics": contradiction_metrics,
                    "extraction_stats": extraction_stats,
                    "evidence_density": evidence_density,
                    "evidence_yield": evidence_yield,
                    "evidence_yield_before_recovery": _recovery.yield_before,
                    "evidence_yield_after_recovery": _recovery.yield_after,
                    "evidence_recovery": _recovery.recovery_metrics,
                    "high_signal_missed_chunks": _recovery.missed_chunk_queue,
                    "category_normalization": _recovery.category_normalization,
                    "zero_yield_documents": zero_yield_docs,
                    "failure_diagnostics": _failure_diags,
                    "failure_summary": _failure_summary,
                    "top_missed_chunks": _top_missed,
                    "document_failure_analysis": _giant_doc_analysis,
                    "retrieval_diversity": retrieval_diversity_mock,
                    "research_gaps": [g.model_dump() for g in research_gaps],
                    "coverage_matrix": [a.model_dump() for a in coverage_matrix],
                    "source_quality_map": sq_map_serialized,
                    "domain_profile": _profile_to_metadata(self.profile),
                    "retrieval_plan": retrieval_plan.to_dict(),
                    "retrieval_stats": retrieval_plan_stats,
                },
            })
        else:
            memo = self._analyze_with_claude(question, documents, source_quality_map)

        warnings = evaluate_memo(memo, documents, mock_llm=is_mock)
        warning_messages = list(memo.evaluation_warnings)
        warning_messages.extend(warning.message for warning in warnings)
        warning_messages.extend(f"Claude warning: {error}" for error in memo.claude_errors)
        return memo.model_copy(update={"evaluation_warnings": _dedupe(warning_messages)})

    def _analyze_with_claude(
        self,
        question: str,
        documents: Sequence[SourceDocument],
        source_quality_map: dict[str, SourceQuality] | None = None,
    ) -> ResearchMemo:
        errors: list[str] = []
        research_plan: ResearchPlan | None = None

        try:
            research_plan = self.client.create_research_plan(question, documents)
        except Exception as exc:
            message = f"research plan failed: {exc}"
            LOGGER.error("Claude %s", message)
            errors.append(message)

        chunks = chunk_documents(documents)
        planner = RetrievalPlanner(profile=self.profile)
        retrieval_plan = planner.plan(question)
        selected_chunks, retrieval_scores, retrieval_plan_stats = select_top_chunks_multi(
            chunks, retrieval_plan.queries, top_n=self.top_chunks, source_quality_map=source_quality_map
        )
        LOGGER.debug(
            "Evidence pipeline: %d chunks from %d documents; %d selected by retrieval (%d queries)",
            len(chunks),
            len(documents),
            len(selected_chunks),
            retrieval_plan.query_count,
        )

        # K1.0 – optional web retrieval: extend the selected chunk pool
        web_search_trace: dict | None = None
        ws_cfg = self.profile.web_search
        from .log import PROGRESS
        LOGGER.log(PROGRESS,
            "[WEB SEARCH] enabled=%s  profile=%s  max_results=%d  max_pages=%d",
            ws_cfg.enabled,
            self.profile.name,
            ws_cfg.max_results,
            ws_cfg.max_pages,
        )
        if ws_cfg.enabled:
            web_docs, web_search_trace = web_retrieve(
                question,
                max_results=ws_cfg.max_results,
                max_pages=ws_cfg.max_pages,
                timeout_seconds=ws_cfg.timeout_seconds,
                cache=self._web_cache,
            )
            web_chunks = _web_docs_to_chunks(web_docs)
            web_search_trace["chunks_created"] = len(web_chunks)
            selected_chunks = list(selected_chunks) + web_chunks
            LOGGER.log(PROGRESS,
                "Web retrieval: %d web chunks added; total selected=%d",
                len(web_chunks),
                len(selected_chunks),
            )

        LOGGER.log(PROGRESS, "Starting evidence extraction from %d chunks", len(selected_chunks))
        try:
            evidence = self.client.extract_evidence_from_chunks(question, selected_chunks)
        except Exception as exc:
            message = f"evidence extraction failed: {exc}"
            LOGGER.error("Claude %s", message)
            errors.append(message)
            evidence = extract_evidence(question, documents, source_quality_map=source_quality_map, profile=self.profile)
        LOGGER.log(PROGRESS, "Starting evidence ranking")
        evidence = rank_evidence_items(
            score_evidence_items(question, assign_evidence_ids(list(evidence)), source_quality_map, self.profile)
        )
        LOGGER.log(PROGRESS,
            "Evidence: %d items extracted and ranked",
            len(evidence),
        )
        # J3.2 — diversity-aware selection: cap over-represented perspectives so
        # that the synthesis pool spans multiple research dimensions.
        synthesis_evidence = select_diverse_evidence(
            evidence,
            top_n=self.top_evidence,
            max_per_perspective=8,
        )
        synthesis_evidence = select_diverse_evidence(
            synthesis_evidence,
            top_n=MAX_SYNTHESIS_EVIDENCE,
            max_per_perspective=3,
        )
        LOGGER.debug(
            "Evidence pipeline: %d filtered to synthesis (top_evidence=%d, cap=%d, total=%d)",
            len(synthesis_evidence),
            self.top_evidence,
            MAX_SYNTHESIS_EVIDENCE,
            len(evidence),
        )
        chunk_scores = _retrieval_scores_to_chunk_scores(chunks, retrieval_scores)
        chunk_diagnostics = compute_chunk_diagnostics(
            chunks, selected_chunks, evidence, chunk_scores
        )
        _log_chunk_diagnostics(chunk_diagnostics)

        try:
            memo = self.client.synthesize_memo(question, synthesis_evidence)
        except Exception as exc:
            message = f"memo synthesis failed: {exc}"
            LOGGER.error("Claude %s", message)
            errors.append(message)
            memo = build_mock_memo(
                question=question,
                documents=documents,
                evidence=evidence,
                synthesis_evidence=synthesis_evidence,
                model_response="Claude memo synthesis failed; generated deterministic fallback memo.",
                mock_llm=False,
            )

        aggregate = aggregate_call_traces(getattr(self.client, "call_traces", []))
        memo = _with_evidence_metadata(
            memo,
            evidence=evidence,
            synthesis_evidence=synthesis_evidence,
            top_evidence=self.top_evidence,
        )
        chunk_meta = _build_chunk_metadata(chunks, selected_chunks, evidence, chunk_diagnostics)
        retrieval_meta = _build_retrieval_metadata(selected_chunks, retrieval_scores)
        # J1.6 – enrich evidence with entity/scope metadata before detection.
        evidence = enrich_evidence_items(evidence)
        # J3.1 – type + topic metadata enrichment.
        evidence = enrich_evidence_with_metadata(evidence, self.profile)
        extraction_stats = build_extraction_stats(evidence)
        evidence_density = build_evidence_density_stats(evidence, len(chunks))
        # JH1 – post-extraction attribution: set source_chunk_id by matching snippets
        evidence = attribute_evidence_to_chunks(evidence, chunks)
        # JH1a – recovery pass: extract from high-signal zero-evidence chunks
        _recovery = run_recovery_pass(
            chunks, selected_chunks, evidence, chunk_diagnostics,
            question=question,
            source_quality_map=source_quality_map,
            profile=self.profile,
        )
        evidence = list(evidence) + _recovery.recovered_items
        zero_yield_docs = compute_zero_yield_documents(chunks, evidence, documents)
        # JH1 – evidence yield metrics (after attribution + recovery)
        evidence_yield = compute_evidence_yield_metrics(
            chunks, selected_chunks, evidence, documents_loaded=len(documents)
        )
        # JH1b – failure pipeline diagnostics (real Claude path)
        _topic_term_sets = _build_topic_term_sets(self.profile)
        _failure_diags = build_failure_diagnostics(
            chunk_diagnostics, chunks, evidence, _topic_term_sets, is_mock=False
        )
        _failure_summary = build_failure_summary(_failure_diags)
        _top_missed = compute_top_missed_chunks(_failure_diags)
        _giant_doc_analysis = analyze_document_failures(
            "31523cfd-5bbb-4c65-a4e0-8e699e370f95.pdf",
            _failure_diags,
            chunk_diags=chunk_diagnostics,
            evidence=evidence,
        )
        # J3.2 — diversity metrics computed after perspective enrichment
        _diversity_domain = detect_domain(
            documents[0].path.name if documents else ""
        )
        retrieval_diversity = build_diversity_metrics(evidence, _diversity_domain)
        suppressed: list = []
        contradictions = detect_contradictions(evidence, source_quality_map, profile=self.profile, out_suppressed=suppressed)
        contradiction_metrics = compute_suppression_metrics(suppressed, len(contradictions))
        research_gaps = detect_gaps(question, evidence, self.profile)
        coverage_matrix = compute_coverage_matrix(
            question, evidence,
            research_gaps=research_gaps,
            source_quality_map=source_quality_map,
            profile=self.profile,
        )
        sq_map_serialized = {k: v.model_dump() for k, v in source_quality_map.items()}
        return memo.model_copy(
            update={
                "research_plan": research_plan,
                "evidence": evidence,
                "claude_model_name": aggregate["model_name"] or getattr(self.client, "model", None),
                "claude_request_timestamp": aggregate["request_timestamp"],
                "claude_response_success": aggregate["response_success"],
                "claude_token_usage": aggregate["token_usage"],
                "claude_call_traces": list(getattr(self.client, "call_traces", [])),
                "claude_errors": _dedupe(errors),
                "contradictions": contradictions,
                "research_gaps": research_gaps,
                "coverage_matrix": coverage_matrix,
                "metadata": {
                    **memo.metadata,
                    **chunk_meta,
                    **retrieval_meta,
                    "contradictions": [c.model_dump() for c in contradictions],
                    "suppressed_comparisons": [s.model_dump() for s in suppressed],
                    "contradiction_metrics": contradiction_metrics,
                    "extraction_stats": extraction_stats,
                    "evidence_density": evidence_density,
                    "evidence_yield": evidence_yield,
                    "evidence_yield_before_recovery": _recovery.yield_before,
                    "evidence_yield_after_recovery": _recovery.yield_after,
                    "evidence_recovery": _recovery.recovery_metrics,
                    "high_signal_missed_chunks": _recovery.missed_chunk_queue,
                    "category_normalization": _recovery.category_normalization,
                    "zero_yield_documents": zero_yield_docs,
                    "failure_diagnostics": _failure_diags,
                    "failure_summary": _failure_summary,
                    "top_missed_chunks": _top_missed,
                    "document_failure_analysis": _giant_doc_analysis,
                    "research_gaps": [g.model_dump() for g in research_gaps],
                    "coverage_matrix": [a.model_dump() for a in coverage_matrix],
                    "source_quality_map": sq_map_serialized,
                    "domain_profile": _profile_to_metadata(self.profile),
                    "retrieval_plan": retrieval_plan.to_dict(),
                    "retrieval_stats": retrieval_plan_stats,
                    **({"web_search": web_search_trace} if web_search_trace is not None else {}),
                },
            }
        )


def extract_evidence(
    question: str,
    documents: Sequence[SourceDocument],
    *,
    min_items_per_document: int = 4,
    max_items_per_document: int = 20,
    source_quality_map: dict[str, SourceQuality] | None = None,
    profile: DomainProfile | None = None,
) -> list[EvidenceItem]:
    """Create deterministic evidence notes from topic matches and early chunks.

    J3.1 improvements:
    - Profile-aware topic terms: uses profile.topic_keywords to find SMR/AI DC
      topics instead of the hard-coded AI DC-only _REQUIRED_TOPIC_TERMS.
    - Multi-claim extraction: extracts multiple sentences per topic (up to 4)
      rather than stopping at the first matching chunk.
    - Higher default max_items_per_document (20 vs 8) for better coverage.
    - Typed and topic-tagged evidence via enrich_evidence_with_metadata().
    """
    effective_domain_terms = (
        profile.get_domain_terms() if profile is not None else _DOMAIN_TERMS
    )
    query_terms = _keywords(question) | effective_domain_terms
    evidence: list[EvidenceItem] = []

    # Build the topic term sets from the profile (or fall back to AI DC defaults)
    topic_term_sets = _build_topic_term_sets(profile)

    for document in documents:
        document_items: list[EvidenceItem] = []
        seen_keys: set[tuple[str, str]] = set()

        # J3.1.2 + J3.1.7: multi-claim extraction per topic, profile-aware
        for category, terms in topic_term_sets.items():
            for chunk in _find_topic_chunks_multi(document.text, terms, max_per_topic=4):
                _append_evidence_item(
                    document_items,
                    seen_keys,
                    document=document,
                    chunk=chunk,
                    category=category,
                    question=question,
                    query_terms=query_terms,
                    max_items=max_items_per_document,
                )

        # Fallback: raw sentence extraction to reach min_items
        if len(document_items) < min_items_per_document:
            fallback_chunks = _meaningful_chunks(
                document.text,
                min_chunks=min_items_per_document,
                max_chunks=max_items_per_document,
            )
            for chunk in fallback_chunks:
                category = _category_for_snippet(chunk, profile)
                _append_evidence_item(
                    document_items,
                    seen_keys,
                    document=document,
                    chunk=chunk,
                    category=category,
                    question=question,
                    query_terms=query_terms,
                    max_items=max_items_per_document,
                )
                if len(document_items) >= max_items_per_document:
                    break

        evidence.extend(document_items[:max_items_per_document])

    # J3.1.1 + J3.1.3: type classification and topic tagging
    evidence = enrich_evidence_with_metadata(evidence, profile)

    clean = sanitize_evidence_items(evidence, stage="mock_extraction")
    return rank_evidence_items(score_evidence_items(question, assign_evidence_ids(clean), source_quality_map, profile))


def score_evidence_items(
    question: str,
    evidence_items: Sequence[EvidenceItem],
    source_quality_map: dict[str, SourceQuality] | None = None,
    profile: DomainProfile | None = None,
) -> list[EvidenceItem]:
    """Score evidence quality with simple deterministic heuristics.

    When *source_quality_map* is provided the source quality score is looked up
    from the map (which was built once for the whole document set) rather than
    re-classified per item.  Either way, ``source_quality_class`` is populated
    with the source type string (e.g. ``"nvidia_technical"``).

    When *profile* is provided, domain terms and specificity terms are taken
    from the profile instead of the hard-coded module-level sets.
    """

    question_terms = _keywords(question)
    question_topics = classify_question_topics(question, profile)
    domain_terms = profile.get_domain_terms() if profile is not None else _DOMAIN_TERMS
    specificity_terms = (
        profile.get_specificity_terms() if profile is not None else _SPECIFICITY_TERMS
    )
    scored: list[EvidenceItem] = []

    for item in evidence_items:
        relevance_score = _evidence_relevance_score(
            item, question_terms, question_topics, domain_terms
        )

        if source_quality_map and item.source_document in source_quality_map:
            sq = source_quality_map[item.source_document]
        else:
            sq = classify_source_quality(item.source_document)
        source_quality_score = sq.source_quality_score
        source_quality_class = sq.source_type

        specificity_score = _specificity_score(item, specificity_terms)
        overall_score = round(
            (relevance_score * 0.45)
            + (source_quality_score * 0.25)
            + (specificity_score * 0.30),
            2,
        )
        scored.append(
            item.model_copy(
                update={
                    "relevance_score": relevance_score,
                    "source_quality_score": source_quality_score,
                    "source_quality_class": source_quality_class,
                    "specificity_score": specificity_score,
                    "overall_score": overall_score,
                }
            )
        )

    return scored


def rank_evidence_items(evidence_items: Sequence[EvidenceItem]) -> list[EvidenceItem]:
    """Return evidence sorted from strongest to weakest."""

    return sorted(
        evidence_items,
        key=lambda item: (
            -item.overall_score,
            -item.relevance_score,
            -item.specificity_score,
            item.source_document.lower(),
            item.evidence_id or item.claim,
        ),
    )


def select_top_evidence(
    evidence_items: Sequence[EvidenceItem],
    top_n: int = DEFAULT_TOP_EVIDENCE,
) -> list[EvidenceItem]:
    """Select the highest-ranked evidence items for memo synthesis."""

    return rank_evidence_items(evidence_items)[: max(1, top_n)]


def build_mock_memo(
    *,
    question: str,
    documents: Sequence[SourceDocument],
    evidence: Sequence[EvidenceItem],
    synthesis_evidence: Sequence[EvidenceItem] | None = None,
    model_response: str,
    mock_llm: bool,
) -> ResearchMemo:
    """Build a deterministic memo while live LLM synthesis is disabled."""

    # synthesis_items drives confirmed_facts — it mirrors what the LLM would receive.
    # All other implication sections scan the full evidence pool so that limiting
    # top_evidence never silently empties a category section.
    synthesis_items = list(synthesis_evidence or evidence)
    implication_items = list(evidence) if evidence else synthesis_items
    title = f"Research Memo: {question}"
    source_count = len(documents)
    evidence_count = len(evidence)
    mode_note = "mock Claude client" if mock_llm else "Claude client"

    confirmed_facts = [
        f"{item.claim} {_citation(item)}" for item in synthesis_items[:5]
    ] or [
        f"Loaded {source_count} supported local source document(s).",
    ]

    power_implications = _implications(implication_items, _POWER_TERMS) or [
        "No direct source excerpt about power distribution was selected in the mock pass.",
    ]
    cooling_implications = _implications(implication_items, _COOLING_TERMS) or [
        "No direct source excerpt about cooling infrastructure was selected in the mock pass.",
    ]
    networking_implications = _category_implications(implication_items, "networking") or [
        "No direct source excerpt about networking infrastructure was selected in the mock pass.",
    ]
    rack_architecture_implications = _category_implications(implication_items, "rack architecture") or [
        "No direct source excerpt about rack architecture was selected in the mock pass.",
    ]

    open_questions = [
        "Which source claims should be treated as current roadmap commitments versus forward-looking statements?",
        "What rack-level power, cooling, networking, and facility assumptions need validation against primary specifications?",
    ]
    if not documents:
        open_questions.insert(0, "Which local sources should be added before drawing infrastructure conclusions?")

    inferences = [
        "Infrastructure conclusions should be treated as provisional until live LLM synthesis reviews the full source set.",
        f"The mock evidence pass selected {evidence_count} evidence note(s) from {source_count} source document(s).",
    ]
    if model_response:
        inferences.append(_compact_whitespace(model_response[:400]))

    return ResearchMemo(
        title=title,
        question=question,
        executive_summary=(
            f"This memo analyzes local AI data center infrastructure sources for: {question}. "
            f"It was generated with the {mode_note}; use the warnings section to identify review gaps."
        ),
        confirmed_facts=confirmed_facts,
        inferences=inferences,
        power_implications=power_implications,
        cooling_implications=cooling_implications,
        networking_implications=networking_implications,
        rack_architecture_implications=rack_architecture_implications,
        open_questions=open_questions,
        source_notes=list(evidence),
        evidence=list(evidence),
    )


def memo_from_model_response(
    *,
    question: str,
    documents: Sequence[SourceDocument],
    evidence: Sequence[EvidenceItem],
    model_response: str,
) -> ResearchMemo:
    """Build a memo from a live model Markdown response."""

    sections = _parse_markdown_sections(model_response)
    executive_summary = sections.get("Executive Summary", "").strip()
    if not executive_summary:
        executive_summary = _compact_whitespace(model_response[:1000])

    return ResearchMemo(
        title=f"Research Memo: {question}",
        question=question,
        executive_summary=executive_summary,
        confirmed_facts=_lines_or_bullets(sections.get("Confirmed Facts", "")),
        inferences=_lines_or_bullets(sections.get("Inferences", "")),
        power_implications=_lines_or_bullets(sections.get("Power Implications", "")),
        cooling_implications=_lines_or_bullets(sections.get("Cooling Implications", "")),
        networking_implications=_lines_or_bullets(sections.get("Networking Implications", "")),
        rack_architecture_implications=_lines_or_bullets(
            sections.get("Rack Architecture Implications", "")
        ),
        open_questions=_lines_or_bullets(sections.get("Open Questions", "")),
        source_notes=list(evidence),
        evaluation_warnings=_lines_or_bullets(sections.get("Evaluation Warnings", "")),
        evidence=list(evidence),
    )


def _with_evidence_metadata(
    memo: ResearchMemo,
    *,
    evidence: Sequence[EvidenceItem],
    synthesis_evidence: Sequence[EvidenceItem],
    top_evidence: int,
) -> ResearchMemo:
    # Estimate synthesis prompt tokens: ~138 tokens per slimmed evidence item + ~100 overhead
    estimated_synthesis_input_tokens = len(synthesis_evidence) * 138 + 100
    metadata = dict(memo.metadata)
    metadata.update(
        {
            "top_evidence_limit": top_evidence,
            "evidence_items_total": len(evidence),
            "evidence_items_used_for_synthesis": len(synthesis_evidence),
            "synthesis_input_tokens_estimate": estimated_synthesis_input_tokens,
            "evidence_passed_to_synthesis": len(synthesis_evidence),
            # contradictions and gaps are detected post-extraction, not passed to synthesize_memo
            "contradictions_passed_to_synthesis": 0,
            "research_gaps_passed_to_synthesis": 0,
        }
    )
    return memo.model_copy(
        update={
            "source_notes": list(evidence),
            "evidence": list(evidence),
            "metadata": metadata,
        }
    )


def _evidence_relevance_score(
    item: EvidenceItem,
    question_terms: set[str],
    question_topics: set[str],
    domain_terms: set[str] | None = None,
) -> int:
    effective_domain_terms = domain_terms if domain_terms is not None else _DOMAIN_TERMS
    haystack = " ".join(
        [
            item.claim,
            item.evidence_snippet,
            item.category,
            item.relevance,
        ]
    ).lower()
    matched_terms = sum(1 for term in question_terms if term in haystack)
    category_matches_topic = item.category in question_topics

    if matched_terms >= 4 or (category_matches_topic and matched_terms >= 2):
        return 5
    if matched_terms >= 2 or category_matches_topic:
        return 4
    if matched_terms == 1 or item.category != "other":
        return 3
    if _contains_any_term(haystack, effective_domain_terms):
        return 2
    return 1


def _source_quality_score(item: EvidenceItem) -> int:
    """Legacy per-item quality scorer.  Delegates to the source quality classifier."""
    return classify_source_quality(item.source_document).source_quality_score


def _specificity_score(
    item: EvidenceItem,
    specificity_terms: set[str] | None = None,
) -> int:
    effective_terms = specificity_terms if specificity_terms is not None else _SPECIFICITY_TERMS
    text = f"{item.claim} {item.evidence_snippet}".lower()
    numeric_claims = len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|kw|mw|w|v|gb|tb|gb/s|tb/s)?\b", text))
    technical_hits = sum(1 for term in effective_terms if term in text)

    if numeric_claims >= 2 or technical_hits >= 5:
        return 5
    if numeric_claims >= 1 or technical_hits >= 3:
        return 4
    if technical_hits >= 1 or len(item.evidence_snippet) >= 140:
        return 3
    if len(item.evidence_snippet) >= 60:
        return 2
    return 1


def _normalize_score_text(text: str) -> str:
    return text.lower().replace("_", " ").replace("-", " ")


def _keywords(text: str) -> set[str]:
    words = {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)}
    return {word for word in words if word not in _STOPWORDS}


def _split_sentences(text: str) -> Iterable[str]:
    normalized = _compact_whitespace(text)
    for sentence in re.split(r"(?<=[.!?])\s+", normalized):
        sentence = sentence.strip()
        if len(sentence) < 40:
            continue
        yield sentence


def _meaningful_chunks(text: str, *, min_chunks: int, max_chunks: int) -> list[str]:
    chunks: list[str] = []

    for sentence in _split_sentences(text):
        chunks.append(sentence[:500])
        if len(chunks) == max_chunks:
            return chunks

    if len(chunks) >= min_chunks:
        return chunks

    normalized = _compact_whitespace(text)
    if not normalized:
        return chunks

    for start in range(0, len(normalized), 500):
        chunk = normalized[start : start + 500].strip()
        if len(chunk) < 40 or chunk in chunks:
            continue
        chunks.append(chunk)
        if len(chunks) >= min_chunks or len(chunks) == max_chunks:
            break

    if len(chunks) < min_chunks and len(normalized) >= 40:
        window = min(len(normalized), max(40, len(normalized) * 2 // 3))
        starts = [0, max(0, len(normalized) - window)]
        for start in starts:
            chunk = normalized[start : start + window].strip()
            if chunk and chunk not in chunks:
                chunks.append(chunk)
            if len(chunks) >= min_chunks or len(chunks) == max_chunks:
                break

    return chunks[:max_chunks]


_VALID_EVIDENCE_CATEGORIES: frozenset[str] = frozenset({
    "architecture", "power", "cooling", "networking", "rack architecture",
    "operations", "gpu", "facility", "resiliency", "economics", "construction",
    "licensing", "reactor design", "grid integration", "fuel cycle",
    "deployment timeline", "safety", "waste management", "bwrx", "nuscale", "other",
})

# Normalise topic names that don't match any valid EvidenceCategory literal.
# Profile topics use human-readable names (e.g. "backup/resiliency") that differ
# from the tighter Literal set used by EvidenceItem.
_CATEGORY_NORM: dict[str, str] = {
    # Profile topic aliases → canonical EvidenceCategory
    "backup/resiliency": "resiliency",
    "backup":            "resiliency",
    "resilience":        "resiliency",
    "rack":              "rack architecture",
    "reactor":           "reactor design",
    "grid":              "grid integration",
    "deployment":        "deployment timeline",
    "fuel":              "fuel cycle",
    "waste":             "waste management",
    "nuclear safety":    "safety",
    "design":            "reactor design",
    "timeline":          "deployment timeline",
    "integration":       "grid integration",
}


def _normalize_category(topic: str) -> str:
    """Map a profile topic name to the nearest valid EvidenceCategory.

    Returns 'other' for any topic not in the Literal set and not in the
    explicit normalization table.
    """
    if topic in _VALID_EVIDENCE_CATEGORIES:
        return topic
    return _CATEGORY_NORM.get(topic, "other")


def _build_topic_term_sets(profile: DomainProfile | None) -> dict[str, set[str]]:
    """Return topic→term_set mapping for evidence extraction.

    When a profile is provided its topic_keywords drive extraction so that
    SMR topics (economics, construction, grid integration…) are properly
    targeted.  Falls back to the AI DC-oriented _REQUIRED_TOPIC_TERMS.
    """
    if profile is not None and profile.topic_keywords:
        return {
            topic: {kw.lower() for kw in keywords if kw}
            for topic, keywords in profile.topic_keywords.items()
            if keywords
        }
    if profile is not None and profile.required_topic_terms:
        return {
            topic: {kw.lower() for kw in terms if kw}
            for topic, terms in profile.required_topic_terms.items()
            if terms
        }
    return _REQUIRED_TOPIC_TERMS


def _find_topic_chunks_multi(
    text: str,
    terms: set[str],
    *,
    max_per_topic: int = 4,
) -> list[str]:
    """Return up to *max_per_topic* non-overlapping sentences from *text* that
    contain at least one of *terms*.

    J3.1.2 multi-claim extraction: instead of stopping at the first matching
    sentence, collects all qualifying sentences so that "economy of scale",
    "factory fabrication", and "load following" each become separate items
    even when they appear in different sentences of the same document.
    """
    normalized = _compact_whitespace(text)
    if not normalized:
        return []

    found: list[str] = []
    seen: set[str] = set()

    for sentence in _split_sentences(normalized):
        if _contains_any_term(sentence, terms):
            key = sentence[:100]
            if key not in seen:
                seen.add(key)
                found.append(sentence[:500])
                if len(found) >= max_per_topic:
                    return found

    # Fallback: scan 500-char windows when sentence splitting found nothing
    if not found:
        for start in range(0, len(normalized), 400):
            chunk = normalized[start : start + 500].strip()
            if chunk and _contains_any_term(chunk, terms):
                key = chunk[:100]
                if key not in seen:
                    seen.add(key)
                    found.append(chunk)
                    if len(found) >= max_per_topic:
                        break

    return found


def _find_topic_chunk(text: str, terms: set[str]) -> str | None:
    normalized = _compact_whitespace(text)
    if not normalized:
        return None

    for sentence in _split_sentences(normalized):
        if _contains_any_term(sentence, terms):
            return sentence[:500]

    for start in range(0, len(normalized), 500):
        chunk = normalized[start : start + 500].strip()
        if chunk and _contains_any_term(chunk, terms):
            return chunk

    if _contains_any_term(normalized, terms):
        return normalized[:500]
    return None


def _score_snippet(snippet: str, keywords: set[str]) -> int:
    lower = snippet.lower()
    return sum(1 for keyword in keywords if keyword in lower)


def _contains_any_term(text: str, terms: set[str]) -> bool:
    lower = text.lower()
    return any(term in lower for term in terms)


def _append_evidence_item(
    items: list[EvidenceItem],
    seen_keys: set[tuple[str, str]],
    *,
    document: SourceDocument,
    chunk: str,
    category: str,
    question: str,
    query_terms: set[str],
    max_items: int,
) -> None:
    if len(items) >= max_items:
        return

    compacted = _compact_whitespace(chunk)
    if not compacted:
        return

    key = (category, compacted)
    if key in seen_keys:
        return

    seen_keys.add(key)
    score = _score_snippet(compacted, query_terms)
    items.append(
        EvidenceItem(
            claim=_claim_from_snippet(compacted, category),
            source_document=document.path.name,
            evidence_snippet=compacted,
            category=_normalize_category(category),
            relevance=_relevance_for_snippet(compacted, question, score),
            confidence=_confidence_for_snippet(compacted, score),
        )
    )


def _implications(evidence: Sequence[EvidenceItem], keywords: set[str]) -> list[str]:
    items: list[str] = []
    for item in evidence:
        if _score_snippet(item.evidence_snippet, keywords) <= 0:
            continue
        items.append(f"{item.claim} {_citation(item)}")
        if len(items) == 3:
            break
    return items


def _category_implications(evidence: Sequence[EvidenceItem], category: str) -> list[str]:
    items: list[str] = []
    for item in evidence:
        if item.category != category:
            continue
        items.append(f"{item.claim} {_citation(item)}")
        if len(items) == 3:
            break
    return items


def _category_for_snippet(snippet: str, profile: DomainProfile | None = None) -> str:
    lower = snippet.lower()
    # Profile-aware: check profile topic_keywords first
    if profile is not None and profile.topic_keywords:
        for topic, keywords in profile.topic_keywords.items():
            if any(kw.lower() in lower for kw in keywords):
                return topic
    # AI DC fallback
    if _contains_any_term(lower, _REQUIRED_TOPIC_TERMS["rack architecture"]):
        return "rack architecture"
    if _contains_any_term(lower, _REQUIRED_TOPIC_TERMS["cooling"]):
        return "cooling"
    if _contains_any_term(lower, _REQUIRED_TOPIC_TERMS["networking"]):
        return "networking"
    if _contains_any_term(lower, _REQUIRED_TOPIC_TERMS["power"]):
        return "power"
    if _score_snippet(lower, {"network", "networking", "ethernet", "infiniband", "nvlink"}) > 0:
        return "networking"
    if _score_snippet(lower, {"gpu", "cpu", "accelerator", "compute", "inference", "training"}) > 0:
        return "architecture"
    if _score_snippet(lower, {"rack", "nvl72", "system", "platform"}) > 0:
        return "rack architecture"
    return "other"


def _citation(item: EvidenceItem) -> str:
    if not item.evidence_id:
        return f"[Source: {item.source_document}, Evidence: unknown]"
    return f"[Source: {item.source_document}, Evidence: {item.evidence_id}]"


def _claim_from_snippet(snippet: str, category: str) -> str:
    compacted = _compact_whitespace(snippet)
    if len(compacted) > 180:
        compacted = compacted[:177].rstrip() + "..."
    return f"{category.capitalize()} note: {compacted}"


def _relevance_for_snippet(snippet: str, question: str, score: int) -> str:
    question_terms = _keywords(question)
    matched_terms = sorted(term for term in question_terms if term in snippet.lower())
    if matched_terms:
        preview = ", ".join(matched_terms[:5])
        return f"Directly relevant to the question through these terms: {preview}."
    if score > 0:
        return "Relevant as infrastructure context for the question."
    return "Indirect context; review against stronger source evidence before relying on it."


def _confidence_for_snippet(snippet: str, score: int) -> str:
    if score >= 3 and len(snippet) >= 80:
        return "high"
    if score >= 1 and len(snippet) >= 60:
        return "medium"
    return "low"


def _compact_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _parse_markdown_sections(markdown: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None

    for line in markdown.splitlines():
        heading = re.match(r"^#{1,3}\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(1).strip()
            current = title
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    return {title: "\n".join(lines).strip() for title, lines in sections.items()}


def _lines_or_bullets(text: str) -> list[str]:
    items: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line).strip()
        if cleaned:
            items.append(cleaned)
    if items:
        return items
    compacted = _compact_whitespace(text)
    return [compacted] if compacted else []


def _retrieval_scores_to_chunk_scores(
    chunks: list[Chunk],
    retrieval_scores: list[RetrievalScore],
) -> dict[str, tuple[float, int]]:
    """Convert RetrievalScore list to the legacy chunk_scores format for compute_chunk_diagnostics."""
    from .chunker import _extract_question_terms, count_evidence_candidates
    score_map = {rs.chunk_id: rs.keyword_score for rs in retrieval_scores}
    result: dict[str, tuple[float, int]] = {}
    for chunk in chunks:
        rel_score = score_map.get(chunk.chunk_id, 0.0)
        # candidate count not critical here; use 0 as placeholder
        result[chunk.chunk_id] = (rel_score, 0)
    return result


def _build_retrieval_metadata(
    selected_chunks: list[Chunk],
    retrieval_scores: list[RetrievalScore],
) -> dict:
    """Build retrieval ranking metadata to merge into memo metadata."""
    selected_ids = {c.chunk_id for c in selected_chunks}
    rejected_ids = [rs.chunk_id for rs in retrieval_scores if rs.chunk_id not in selected_ids]
    return {
        "retrieval_ranking": [rs.model_dump() for rs in retrieval_scores],
        "selected_chunk_ids": [c.chunk_id for c in selected_chunks],
        "rejected_chunk_ids": rejected_ids,
    }


def _build_chunk_metadata(
    chunks: list[Chunk],
    selected_chunks: list[Chunk],
    evidence: Sequence[EvidenceItem],
    diagnostics: list,
) -> dict:
    """Build chunk statistics and diagnostics to merge into memo metadata."""
    from .schemas import ChunkDiagnostic

    chunks_per_document: dict[str, int] = {}
    for chunk in chunks:
        chunks_per_document[chunk.document_name] = chunks_per_document.get(chunk.document_name, 0) + 1

    evidence_per_chunk: dict[str, int] = {chunk.chunk_id: 0 for chunk in chunks}
    for item in evidence:
        if item.source_chunk_id and item.source_chunk_id in evidence_per_chunk:
            evidence_per_chunk[item.source_chunk_id] += 1

    avg_chunk_size = int(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0

    return {
        "chunk_count": len(chunks),
        "chunks_selected": len(selected_chunks),
        "chunks_per_document": chunks_per_document,
        "evidence_per_chunk": evidence_per_chunk,
        "avg_chunk_size": avg_chunk_size,
        "chunk_diagnostics": [
            d.model_dump() if hasattr(d, "model_dump") else d for d in diagnostics
        ],
    }


def _log_chunk_diagnostics(diagnostics: list) -> None:
    """Emit a concise DEBUG summary of chunk selection and extraction outcomes."""
    total = len(diagnostics)
    sent = sum(1 for d in diagnostics if getattr(d, "sent_to_claude", False))
    accepted = sum(1 for d in diagnostics if getattr(d, "extraction_decision", "") == "accepted")
    rejected = sum(1 for d in diagnostics if getattr(d, "extraction_decision", "") == "rejected")
    not_sent = sum(1 for d in diagnostics if getattr(d, "extraction_decision", "") == "not_sent")

    from collections import Counter
    reason_counts = Counter(
        d.rejection_reason
        for d in diagnostics
        if getattr(d, "rejection_reason", None)
    )

    LOGGER.debug(
        "Chunk diagnostics: %d total | %d sent to Claude | %d accepted | %d rejected | %d not sent",
        total, sent, accepted, rejected, not_sent,
    )
    for reason, count in reason_counts.most_common():
        LOGGER.debug("  Rejection reason: %s (%d)", reason, count)


def _profile_to_metadata(profile: DomainProfile) -> dict:
    """Return a JSON-serialisable summary of a profile for trace/metadata storage."""
    return {
        "name": profile.name,
        "description": profile.description,
        "path": profile.profile_path,
        "coverage_topics": list(profile.coverage_topics),
        "topics_available": sorted(profile.topic_keywords.keys()),
        "gap_check_topics": sorted(profile.research_gap_checks.keys()),
    }


def _url_slug(url: str) -> str:
    """Return a short display-safe slug for a URL (last path segment)."""
    parts = url.rstrip("/").split("/")
    return parts[-1] or parts[-2] if len(parts) > 1 else url[:60]


def _web_docs_to_chunks(docs: list[WebDocument]) -> list[Chunk]:
    """Convert :class:`WebDocument` objects into :class:`Chunk` objects.

    Each web document is treated as a single chunk whose ``document_name`` is
    the URL — this lets citations show the source URL naturally.
    """
    chunks: list[Chunk] = []
    for idx, doc in enumerate(docs):
        if not doc.text.strip():
            continue
        chunk_id = f"web_{idx:04d}_0000"
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_name=doc.url,
                chunk_number=0,
                text=doc.text,
                start_offset=0,
                end_offset=len(doc.text),
                source_type="web",
                source_url=doc.url,
            )
        )
    return chunks


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped
