"""EvidenceAgent – runs the research engine and organizes evidence (J5.2)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .base import FunctionalAgent
from .context import AgentContext

LOGGER = logging.getLogger(__name__)

# Coverage thresholds (evidence items per subquestion)
_STRONG = 4
_MODERATE = 2
_WEAK = 1


def _coverage_level(count: int) -> str:
    if count >= _STRONG:
        return "STRONG"
    if count >= _MODERATE:
        return "MODERATE"
    if count >= _WEAK:
        return "WEAK"
    return "NONE"


def _tokens(text: str) -> set[str]:
    """Lower-cased word tokens from text, ignoring short stop words."""
    _STOPWORDS = {"the", "a", "an", "of", "in", "is", "are", "to", "for",
                  "and", "or", "what", "how", "why", "does", "do", "can",
                  "with", "that", "this", "it", "be", "by", "at", "as"}
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _overlap_score(text: str, reference: str) -> int:
    """Count shared tokens between *text* and *reference*."""
    return len(_tokens(text) & _tokens(reference))


def _map_evidence_to_subquestions(
    evidence_items: list[Any],
    subquestions: list[str],
) -> dict[str, list[str]]:
    """Map each evidence item to its best-matching subquestion by token overlap.

    An item is assigned to the subquestion with the highest overlap score.
    Items with zero overlap on all subquestions go into "_unmapped".
    """
    mapping: dict[str, list[str]] = {sq: [] for sq in subquestions}
    mapping["_unmapped"] = []

    for item in evidence_items:
        item_text = f"{getattr(item, 'claim', '')} {getattr(item, 'evidence_snippet', '')}"
        best_sq = None
        best_score = 0
        for sq in subquestions:
            score = _overlap_score(item_text, sq)
            if score > best_score:
                best_score = score
                best_sq = sq
        eid = getattr(item, "evidence_id", "") or ""
        if best_sq and best_score > 0:
            mapping[best_sq].append(eid)
        else:
            mapping["_unmapped"].append(eid)

    return mapping


# Category → investigation area keyword mapping
_CATEGORY_TO_AREA: dict[str, str] = {
    "power": "Power",
    "cooling": "Cooling",
    "networking": "Networking",
    "rack architecture": "Rack Architecture",
    "architecture": "Architecture",
    "operations": "Operations",
    "gpu": "GPU",
    "facility": "Facility",
    "resiliency": "Resiliency",
    "economics": "Economics",
    "construction": "Construction",
    "licensing": "Regulation",
    "reactor design": "SMR Technology",
    "grid integration": "Grid Integration",
    "fuel cycle": "Fuel Cycle",
    "deployment timeline": "Deployment Timeline",
    "safety": "Safety",
    "waste management": "Waste Management",
    "bwrx": "SMR Technology",
    "nuscale": "SMR Technology",
    "other": "Other",
}


def _map_evidence_to_areas(
    evidence_items: list[Any],
    investigation_areas: list[str],
) -> dict[str, list[str]]:
    """Map each evidence item to matching investigation areas.

    An item can appear in multiple areas (one-to-many).
    Matching uses the item's category and topics fields.
    """
    mapping: dict[str, list[str]] = {area: [] for area in investigation_areas}

    for item in evidence_items:
        eid = getattr(item, "evidence_id", "") or ""
        category = getattr(item, "category", "") or ""
        topics = getattr(item, "topics", []) or []

        # Signals: category name → canonical area + topics text
        signal_texts = [category] + list(topics)
        combined = " ".join(signal_texts).lower()

        for area in investigation_areas:
            area_tokens = _tokens(area)
            if area_tokens & _tokens(combined):
                mapping[area].append(eid)
            # Also check via _CATEGORY_TO_AREA lookup
            elif _CATEGORY_TO_AREA.get(category) == area:
                mapping[area].append(eid)

    return mapping


def _compute_coverage(
    evidence_by_subquestion: dict[str, list[str]],
    subquestions: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute coverage level per subquestion."""
    return {
        sq: {
            "evidence_count": len(evidence_by_subquestion.get(sq, [])),
            "coverage": _coverage_level(len(evidence_by_subquestion.get(sq, []))),
        }
        for sq in subquestions
    }


def _build_evidence_summary(
    evidence_items: list[Any],
    evidence_by_subquestion: dict[str, list[str]],
    evidence_by_area: dict[str, list[str]],
    subquestions: list[str],
) -> dict[str, Any]:
    coverage = _compute_coverage(evidence_by_subquestion, subquestions)
    covered = sum(1 for sq in subquestions if coverage[sq]["coverage"] != "NONE")
    return {
        "total_evidence_items": len(evidence_items),
        "subquestions_with_evidence": covered,
        "subquestions_without_evidence": len(subquestions) - covered,
        "investigation_areas_with_evidence": sum(
            1 for items in evidence_by_area.values() if items
        ),
        "coverage_distribution": {
            level: sum(1 for sq in subquestions if coverage[sq]["coverage"] == level)
            for level in ("STRONG", "MODERATE", "WEAK", "NONE")
        },
    }


def _build_profile_term_sets(domain_profiles: list[Any]) -> dict[str, set[str]]:
    """Build a keyword set per profile for attribution scoring (J5.6)."""
    result: dict[str, set[str]] = {}
    for p in domain_profiles:
        terms: set[str] = set()
        for t in (getattr(p, "domain_terms", None) or []):
            terms.add(t.lower())
        for kw_list in getattr(p, "topic_keywords", {}).values():
            for kw in kw_list:
                terms.add(kw.lower())
        result[p.name] = terms
    return result


def _attribute_evidence_profiles(
    items: list[dict],
    domain_profiles: list[Any],
    fallback_profile: str,
) -> dict[str, dict]:
    """Assign source_profile to each item dict and return per-profile coverage.

    Attribution uses keyword overlap between each item's claim + topics text
    and each profile's domain_terms + topic_keywords.  Single-profile runs
    assign all items without scoring.

    Mutates *items* in-place; returns profile_coverage_by_profile dict.
    """
    if not domain_profiles:
        for item in items:
            item["source_profile"] = fallback_profile
        return {}

    if len(domain_profiles) == 1:
        name = domain_profiles[0].name
        for item in items:
            item["source_profile"] = name
        count = len(items)
        level = "STRONG" if count >= 10 else "MODERATE" if count >= 3 else "WEAK" if count else "NONE"
        return {name: {"evidence_count": count, "coverage_level": level}}

    term_sets = _build_profile_term_sets(domain_profiles)
    profile_names = [p.name for p in domain_profiles]

    for item in items:
        text = " ".join([
            item.get("claim", ""),
            " ".join(item.get("topics", [])),
        ]).lower()
        scores = {
            pname: sum(1 for t in terms if t in text)
            for pname, terms in term_sets.items()
        }
        best_score = max(scores.values())
        best = max(scores, key=lambda k: scores[k]) if best_score > 0 else fallback_profile
        item["source_profile"] = best

    coverage: dict[str, dict] = {}
    for pname in profile_names:
        attributed = sum(1 for e in items if e.get("source_profile") == pname)
        level = "STRONG" if attributed >= 10 else "MODERATE" if attributed >= 3 else "WEAK" if attributed else "NONE"
        coverage[pname] = {"evidence_count": attributed, "coverage_level": level}

    return coverage


_KB_ETYPE_TO_CATEGORY: dict[str, str] = {
    "STRATEGIC": "other",
    "TECHNICAL": "other",
    "ECONOMIC": "economics",
    "RISK": "other",
    "REGULATORY": "licensing",
    "OPERATIONAL": "operations",
}

_VALID_CATEGORIES: frozenset[str] = frozenset([
    "architecture", "power", "cooling", "networking", "rack architecture",
    "operations", "gpu", "facility", "resiliency", "economics", "construction",
    "licensing", "reactor design", "grid integration", "fuel cycle",
    "deployment timeline", "safety", "waste management", "bwrx", "nuscale", "other",
])


def _safe_category(raw: str) -> str:
    """Return a valid EvidenceCategory string, falling back to 'other'."""
    s = (raw or "").lower().strip()
    return s if s in _VALID_CATEGORIES else _KB_ETYPE_TO_CATEGORY.get(raw.upper(), "other")


def _score_to_5(score: float) -> int:
    """Map a [0, 1] retrieval score to a 1–5 integer relevance_score."""
    return max(1, min(5, round(score * 5)))


class EvidenceAgent(FunctionalAgent):
    """Runs the research engine and organizes evidence around the research plan (J5.2).

    Responsibilities:
    - Retrieve evidence via EvidenceRetriever (J8.6 Knowledge Layer path) or
      the legacy extraction pipeline (fallback)
    - Map evidence to subquestions from PlannerAgent
    - Map evidence to investigation areas
    - Compute per-subquestion coverage levels
    - Build evidence summary
    - Write all structured evidence data into context and Research Object

    When `retriever` is provided, the Knowledge Layer path is used exclusively.
    No document loaders, source files, or extraction pipeline are invoked.
    """

    def __init__(
        self,
        *,
        sources_dir: str | Path = "sources",
        client: Any = None,
        top_evidence: int = 50,
        top_chunks: int = 20,
        domain_profile: Any = None,
        domain_profiles: list[Any] | None = None,
        retriever: Any = None,
        use_reranker: bool = False,
        rerank_candidates: int = 40,
    ) -> None:
        self._sources_dir = Path(sources_dir)
        self._client = client
        self._top_evidence = top_evidence
        self._top_chunks = top_chunks
        self._domain_profile = domain_profile
        self._domain_profiles: list[Any] = (
            domain_profiles if domain_profiles is not None
            else ([domain_profile] if domain_profile is not None else [])
        )
        self._retriever = retriever
        self._use_reranker = use_reranker
        self._rerank_candidates = rerank_candidates

    def _execute(self, context: AgentContext) -> AgentContext:
        """J10.5 — collect evidence per Decision Domain; primary flows downstream.

        The PRIMARY domain (domain_plans[0], whose plan == context.plan) runs on
        the real context, leaving context.evidence_notes and all downstream state
        byte-identical to J10.4. Secondary domains run on isolated scratch
        contexts so the real context is untouched; their evidence/mapping/coverage
        is captured into context.domain_evidence. Goal/question mode has a single
        plan → single collection, unchanged.
        """
        plans = list(context.domain_plans) if context.domain_plans else []

        # Primary run on the real context (byte-identical to prior behaviour).
        self._execute_single(context)
        primary_plan = plans[0] if plans else context.plan
        domain_evidence = [self._capture_domain_evidence(context, primary_plan)]

        # Secondary domains on isolated scratch contexts (organizational only).
        for plan in plans[1:]:
            scratch = self._scratch_context(context, plan)
            try:
                self._execute_single(scratch)
                domain_evidence.append(self._capture_domain_evidence(scratch, plan))
            except Exception as exc:  # a secondary domain must never fail the run
                LOGGER.warning(
                    "[EvidenceAgent] secondary domain evidence failed (%s: %s) — skipping.",
                    type(exc).__name__, exc,
                )

        context.domain_evidence = domain_evidence

        primary_domain = (
            primary_plan.get("decision_domain_title")
            or primary_plan.get("question", "")
            if isinstance(primary_plan, dict) else ""
        )
        context.trace["_evidence_reasoning"] = {
            "plans_received": len(plans),
            "evidence_sets_generated": len(domain_evidence),
            "evidence_sets_executed": 1 if domain_evidence else 0,
            "primary_domain": primary_domain,
        }
        return context

    def _execute_single(self, context: AgentContext) -> AgentContext:
        if self._retriever is not None:
            return self._execute_kb(context)
        return self._execute_legacy(context)

    # ------------------------------------------------------------------
    # J10.5 — multi-domain helpers
    # ------------------------------------------------------------------

    _PLAN_KEYS = ("question", "research_type", "subquestions",
                  "investigation_areas", "profiles_used", "reasoning")

    def _scratch_context(self, context: AgentContext, plan: dict) -> AgentContext:
        """Build an isolated context for a secondary domain's evidence pass.

        Shares read-only collaborators (client/retriever live on the agent or in
        the trace) but isolates everything _execute_single mutates: plan,
        question, research_object, trace, evidence_notes, agent_history.
        """
        import copy

        scratch = copy.copy(context)
        scratch.plan = {k: plan.get(k) for k in self._PLAN_KEYS}
        scratch.question = plan.get("question", context.question)
        scratch.research_object = copy.deepcopy(context.research_object) if context.research_object else {}
        # Fresh trace: keep the client for the legacy path, drop the perf tracker
        # so secondary passes don't pollute primary performance accounting.
        scratch.trace = {"_client": context.trace.get("_client")}
        scratch.evidence_notes = []
        scratch.agent_history = []
        return scratch

    @staticmethod
    def _capture_domain_evidence(context: AgentContext, plan: dict) -> dict:
        """Extract one Decision Domain's evidence collection + stats (J10.5)."""
        note = context.evidence_notes[0] if context.evidence_notes else {}
        plan = plan if isinstance(plan, dict) else {}
        return {
            "decision_domain_id": plan.get("decision_domain_id"),
            "decision_domain_title": plan.get("decision_domain_title"),
            "evidence": note.get("evidence_items", []),
            "mapping": {
                "evidence_by_subquestion": note.get("evidence_by_subquestion", {}),
                "evidence_by_area": note.get("evidence_by_area", {}),
            },
            "coverage": {
                "coverage_by_subquestion": note.get("coverage_by_subquestion", {}),
                "evidence_summary": note.get("evidence_summary", {}),
            },
        }

    # ------------------------------------------------------------------
    # Knowledge Layer path (J8.6)
    # ------------------------------------------------------------------

    def _execute_kb(self, context: AgentContext) -> AgentContext:
        """Retrieve evidence from the Knowledge Layer (no document loading)."""
        import time as _time
        import types
        from research_agent.log import PROGRESS
        from knowledge.retriever import RETRIEVAL_MODE_HYBRID, RETRIEVAL_MODE_LEXICAL

        _tracker = context.trace.get("_perf_tracker")

        subquestions: list[str] = context.plan.get("subquestions", [])
        investigation_areas: list[str] = context.plan.get("investigation_areas", [])

        # Determine primary query and mode
        primary_query = context.question or (subquestions[0] if subquestions else "")
        has_embeddings = getattr(self._retriever, "provider", None) is not None
        mode = RETRIEVAL_MODE_HYBRID if has_embeddings else RETRIEVAL_MODE_LEXICAL

        # Retrieve — larger pool when reranking
        fetch_k = self._rerank_candidates if self._use_reranker else self._top_evidence

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent:kb] query=%r  mode=%s  fetch_k=%d  subquestions=%d",
            (primary_query or "")[:80], mode, fetch_k, len(subquestions),
        )

        _t_retrieval = _time.monotonic()
        result = self._retriever.retrieve(
            primary_query,
            mode=mode,
            top_k=fetch_k,
        )
        candidates = result.items
        _retrieval_ms = (_time.monotonic() - _t_retrieval) * 1000

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent:kb] primary retrieval: total_scanned=%d  matched=%d  returned=%d",
            result.total_candidates, result.matched_candidates, len(candidates),
        )
        if _tracker is not None:
            _tracker.add_sub_phase(
                "evidence:retrieval_query",
                _retrieval_ms,
                mode=mode,
                fetch_k=fetch_k,
                scanned=result.total_candidates,
                matched=result.matched_candidates,
                returned=len(candidates),
            )

        # Also issue per-subquestion retrieval to improve coverage
        seen_ids: set[str] = {c.evidence.evidence_id for c in candidates}
        sq_added = 0
        _t_sq = _time.monotonic()
        for sq in subquestions:
            if len(candidates) >= self._top_evidence * 3:
                break
            sq_result = self._retriever.retrieve(sq, mode=mode, top_k=10)
            for item in sq_result.items:
                eid = item.evidence.evidence_id
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    candidates.append(item)
                    sq_added += 1
        _sq_ms = (_time.monotonic() - _t_sq) * 1000

        if sq_added:
            LOGGER.log(
                PROGRESS,
                "[EvidenceAgent:kb] subquestion expansion: +%d items  total=%d",
                sq_added, len(candidates),
            )
        if _tracker is not None:
            _tracker.add_sub_phase(
                "evidence:subquestion_expansion",
                _sq_ms,
                subquestions=len(subquestions),
                added=sq_added,
                total_after=len(candidates),
            )

        pre_rerank_count = len(candidates)

        # Optional LLM reranking
        _rerank_ms = 0.0
        if self._use_reranker and candidates:
            from knowledge.reranker import LLMReranker
            reranker = LLMReranker()
            rerank_result = reranker.rerank(primary_query, candidates, top_k=self._top_evidence)
            reranked = [r.candidate for r in rerank_result.items]
            _rerank_ms = rerank_result.latency_ms
            # PH1 — surface LLM-output normalization diagnostics into the trace.
            # PH1a — accumulate as a list so multiple LLM boundaries can report.
            if getattr(rerank_result, "normalization", None):
                context.trace.setdefault("_llm_normalization", []).append(rerank_result.normalization)

            LOGGER.log(
                PROGRESS,
                "[EvidenceAgent:kb] reranker: candidates_in=%d  selected=%d  latency=%.0fms",
                pre_rerank_count, len(reranked), rerank_result.latency_ms,
            )

            if not reranked and pre_rerank_count > 0:
                # Reranker returned 0 — use retrieval-order fallback rather than triggering
                # the 0-evidence legacy fallback. This handles LLM hallucinated-ID failures.
                LOGGER.warning(
                    "[EvidenceAgent:kb] Reranker returned 0 items (likely hallucinated evidence_ids). "
                    "Falling back to retrieval-order selection. pre_rerank_count=%d",
                    pre_rerank_count,
                )
                candidates = candidates[:self._top_evidence]
            else:
                candidates = reranked
        else:
            candidates = candidates[:self._top_evidence]

        if _tracker is not None:
            _tracker.add_sub_phase(
                "evidence:reranking",
                _rerank_ms,
                enabled=self._use_reranker,
                candidates_in=pre_rerank_count,
                candidates_out=len(candidates),
            )

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent:kb] retrieved %d items (mode=%s)",
            len(candidates), mode,
        )

        if not candidates:
            LOGGER.warning(
                "[EvidenceAgent:kb] Knowledge Layer returned 0 evidence items — "
                "falling back to legacy document retrieval. "
                "Check that the knowledge store is fully indexed (run: python3 -m knowledge build)."
            )
            return self._execute_legacy(context)

        # Convert to legacy dict format
        items_dicts = [
            {
                "evidence_id": c.evidence.evidence_id,
                "claim": c.statement,
                "category": _safe_category(c.evidence.category or c.evidence_type),
                "topics": [],
                "relevance_score": _score_to_5(c.score),
                "source_document": (
                    c.evidence.supporting_source_ids[0]
                    if c.evidence.supporting_source_ids else "knowledge_store"
                ),
            }
            for c in candidates
        ]

        # Build SimpleNamespace proxies for the mapping functions (which use getattr)
        def _proxy(d: dict) -> object:
            ns = types.SimpleNamespace()
            ns.claim = d["claim"]
            ns.evidence_snippet = d["claim"]
            ns.evidence_id = d["evidence_id"]
            ns.category = d["category"]
            ns.topics = d["topics"]
            return ns

        proxies = [_proxy(d) for d in items_dicts]

        # Map to subquestions and areas
        _t_mapping = _time.monotonic()
        evidence_by_subquestion: dict[str, list[str]] = {}
        evidence_by_area: dict[str, list[str]] = {}
        if subquestions:
            evidence_by_subquestion = _map_evidence_to_subquestions(proxies, subquestions)
        if investigation_areas:
            evidence_by_area = _map_evidence_to_areas(proxies, investigation_areas)

        # Coverage and summary
        coverage_by_subquestion = _compute_coverage(evidence_by_subquestion, subquestions)
        evidence_summary = _build_evidence_summary(
            proxies, evidence_by_subquestion, evidence_by_area, subquestions
        )

        covered = evidence_summary["subquestions_with_evidence"]
        uncovered = evidence_summary["subquestions_without_evidence"]
        mapped_areas = evidence_summary["investigation_areas_with_evidence"]
        _mapping_ms = (_time.monotonic() - _t_mapping) * 1000

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent:kb] mapped: subquestions=%d/%d covered, areas=%d/%d covered",
            covered, len(subquestions), mapped_areas, len(investigation_areas),
        )
        if _tracker is not None:
            _tracker.add_sub_phase(
                "evidence:mapping_coverage",
                _mapping_ms,
                subquestions_covered=covered,
                subquestions_total=len(subquestions),
                areas_covered=mapped_areas,
                areas_total=len(investigation_areas),
            )

        # Profile attribution
        _t_assembly = _time.monotonic()
        profile_coverage_by_profile = _attribute_evidence_profiles(
            items_dicts,
            self._domain_profiles,
            fallback_profile=context.execution_profile,
        )
        for item in items_dicts:
            item["profile"] = item.get("source_profile", context.execution_profile)

        profiles_requested = [p.name for p in self._domain_profiles] if self._domain_profiles else context.profiles
        profiles_contributing = [p for p, v in profile_coverage_by_profile.items() if v.get("evidence_count", 0) > 0]
        profiles_missing = [p for p, v in profile_coverage_by_profile.items() if v.get("evidence_count", 0) == 0]

        context.evidence_notes = [
            {
                "evidence_items": items_dicts,
                "evidence_by_subquestion": evidence_by_subquestion,
                "evidence_by_area": evidence_by_area,
                "coverage_by_subquestion": coverage_by_subquestion,
                "evidence_summary": evidence_summary,
                "profile_coverage_by_profile": profile_coverage_by_profile,
                "profiles_requested": profiles_requested,
                "profiles_contributing": profiles_contributing,
                "profiles_missing": profiles_missing,
            }
        ]

        # Build synthetic ResearchMemo for downstream compatibility (ReportAgent)
        memo = self._build_synthetic_memo(context.question, candidates)
        context.trace["_memo"] = memo
        context.trace["_documents"] = []
        _assembly_ms = (_time.monotonic() - _t_assembly) * 1000
        if _tracker is not None:
            _tracker.add_sub_phase(
                "evidence:evidence_assembly",
                _assembly_ms,
                evidence_count=len(candidates),
                profiles_contributing=len(profiles_contributing),
            )

        # Research Object update
        if context.research_object:
            ro = context.research_object
            ro["evidence_by_subquestion"] = {
                sq: ids for sq, ids in evidence_by_subquestion.items() if sq != "_unmapped"
            }
            ro["evidence_by_area"] = evidence_by_area
            ro["coverage_by_subquestion"] = coverage_by_subquestion
            ro["evidence_summary"] = evidence_summary
            ro["validated_contradictions"] = []
            ro["contradiction_metrics"] = {}
            ro["suppressed_contradictions"] = []

        self._record(
            context,
            status="success",
            summary=(
                f"Retrieved {len(candidates)} evidence items from Knowledge Layer (mode={mode}). "
                f"Mapped {covered}/{len(subquestions)} subquestions, "
                f"{mapped_areas}/{len(investigation_areas)} investigation areas."
            ),
            evidence_count=len(candidates),
            confirmed_facts=0,
            mapped_subquestions=covered,
            mapped_areas=mapped_areas,
            uncovered_subquestions=uncovered,
        )
        return context

    def _build_synthetic_memo(self, question: str, candidates: list) -> Any:
        """Construct a minimal ResearchMemo from Knowledge Layer Evidence for ReportAgent compat."""
        from research_agent.schemas import EvidenceItem, ResearchMemo

        ev_items = []
        for c in candidates:
            ev_items.append(EvidenceItem(
                evidence_id=c.evidence.evidence_id,
                claim=c.statement,
                source_document=(
                    c.evidence.supporting_source_ids[0]
                    if c.evidence.supporting_source_ids else "knowledge_store"
                ),
                evidence_snippet=c.statement[:300],
                category=_safe_category(c.evidence.category or c.evidence_type),
                relevance="relevant",
                confidence="medium",
                relevance_score=_score_to_5(c.score),
            ))

        confirmed_facts = [c.statement for c in candidates[:10]]
        return ResearchMemo(
            title=f"Knowledge Layer Evidence: {question[:80]}",
            question=question,
            executive_summary=f"Retrieved {len(candidates)} evidence items from the Knowledge Base.",
            confirmed_facts=confirmed_facts,
            source_notes=ev_items,
            metadata={
                "retrieval_source": "knowledge_layer",
                "evidence_count": len(candidates),
                "top_evidence_limit": len(candidates),
                "evidence_passed_to_synthesis": len(candidates),
            },
        )

    # ------------------------------------------------------------------
    # Legacy extraction path (unchanged)
    # ------------------------------------------------------------------

    def _execute_legacy(self, context: AgentContext) -> AgentContext:
        from research_agent.log import PROGRESS
        from research_agent.agent import DcPowerAgent
        from research_agent.loaders import load_sources

        # --- 1. Run extraction pipeline (unchanged) ---
        collection = load_sources(self._sources_dir)
        if collection.errors:
            for err in collection.errors:
                LOGGER.warning("Source load error: %s — %s", err.path.name, err.message)

        agent = DcPowerAgent(
            client=self._client,
            top_evidence=self._top_evidence,
            top_chunks=self._top_chunks,
            profile=self._domain_profile,
        )
        memo = agent.analyze(context.question, collection.documents)
        evidence_items = memo.source_notes or memo.evidence

        LOGGER.log(PROGRESS, "[EvidenceAgent] extracted %d items", len(evidence_items))

        # J8.7 Runtime Guardrail — explicit "insufficient evidence" path
        if not evidence_items:
            LOGGER.warning(
                "[EvidenceAgent] Legacy retrieval also returned 0 evidence items. "
                "The report will reflect insufficient evidence coverage. "
                "Ensure source documents are present in %s and the Knowledge Store is built.",
                self._sources_dir,
            )
            context.trace["_insufficient_evidence"] = True

        # --- 2. Read planner outputs ---
        subquestions: list[str] = context.plan.get("subquestions", [])
        investigation_areas: list[str] = context.plan.get("investigation_areas", [])

        # --- 3. Map evidence to subquestions and investigation areas ---
        evidence_by_subquestion: dict[str, list[str]] = {}
        evidence_by_area: dict[str, list[str]] = {}

        if subquestions:
            evidence_by_subquestion = _map_evidence_to_subquestions(
                evidence_items, subquestions
            )
        if investigation_areas:
            evidence_by_area = _map_evidence_to_areas(
                evidence_items, investigation_areas
            )

        # --- 4. Coverage and summary ---
        coverage_by_subquestion = _compute_coverage(evidence_by_subquestion, subquestions)
        evidence_summary = _build_evidence_summary(
            evidence_items, evidence_by_subquestion, evidence_by_area, subquestions
        )

        covered = evidence_summary["subquestions_with_evidence"]
        uncovered = evidence_summary["subquestions_without_evidence"]
        mapped_areas = evidence_summary["investigation_areas_with_evidence"]

        LOGGER.log(
            PROGRESS,
            "[EvidenceAgent] mapped: subquestions=%d/%d covered, areas=%d/%d covered",
            covered,
            len(subquestions),
            mapped_areas,
            len(investigation_areas),
        )

        # --- 5. Serialize evidence items and attribute to profiles (J5.6 / J5.6a) ---
        items_dicts = [
            {
                "evidence_id": getattr(e, "evidence_id", ""),
                "claim": getattr(e, "claim", ""),
                "category": getattr(e, "category", ""),
                "topics": getattr(e, "topics", []),
                "relevance_score": getattr(e, "relevance_score", 0),
                "source_document": getattr(e, "source_document", ""),
            }
            for e in evidence_items
        ]
        profile_coverage_by_profile = _attribute_evidence_profiles(
            items_dicts,
            self._domain_profiles,
            fallback_profile=context.execution_profile,
        )
        # Add spec-required "profile" field (alias for source_profile) to every item
        for item in items_dicts:
            item["profile"] = item.get("source_profile", context.execution_profile)

        # Derive requested / contributing / missing profile sets (J5.6a)
        profiles_requested = [p.name for p in self._domain_profiles] if self._domain_profiles else context.profiles
        profiles_contributing = [
            p for p, v in profile_coverage_by_profile.items() if v.get("evidence_count", 0) > 0
        ]
        profiles_missing = [
            p for p, v in profile_coverage_by_profile.items() if v.get("evidence_count", 0) == 0
        ]

        if len(self._domain_profiles) > 1:
            LOGGER.log(
                PROGRESS,
                "[EvidenceAgent] profile attribution: contributing=%s missing=%s",
                profiles_contributing,
                profiles_missing,
            )

        context.evidence_notes = [
            {
                "evidence_items": items_dicts,
                "evidence_by_subquestion": evidence_by_subquestion,
                "evidence_by_area": evidence_by_area,
                "coverage_by_subquestion": coverage_by_subquestion,
                "evidence_summary": evidence_summary,
                "profile_coverage_by_profile": profile_coverage_by_profile,
                "profiles_requested": profiles_requested,
                "profiles_contributing": profiles_contributing,
                "profiles_missing": profiles_missing,
            }
        ]

        # --- 5b. Contradiction hardening metadata (J6.5a) ---
        memo_meta = getattr(memo, "metadata", {}) or {}
        validated_contradictions: list[dict] = memo_meta.get("contradictions", [])
        contradiction_metrics: dict = memo_meta.get("contradiction_metrics", {})
        context.validated_contradictions = validated_contradictions
        context.contradiction_metrics = contradiction_metrics

        # --- 6. Update Research Object (J5.2.6) ---
        if context.research_object:
            ro = context.research_object
            ro["evidence_by_subquestion"] = {
                sq: ids for sq, ids in evidence_by_subquestion.items() if sq != "_unmapped"
            }
            ro["evidence_by_area"] = evidence_by_area
            ro["coverage_by_subquestion"] = coverage_by_subquestion
            ro["evidence_summary"] = evidence_summary
            ro["validated_contradictions"] = validated_contradictions
            ro["contradiction_metrics"] = contradiction_metrics
            ro["suppressed_contradictions"] = memo_meta.get("suppressed_comparisons", [])

        # --- 7. Agent history (J5.2.8) ---
        self._record(
            context,
            status="success",
            summary=(
                f"Retrieved {len(evidence_items)} evidence items, confirmed "
                f"{len(memo.confirmed_facts or [])} facts. "
                f"Mapped {covered}/{len(subquestions)} subquestions, "
                f"{mapped_areas}/{len(investigation_areas)} investigation areas."
            ),
            evidence_count=len(evidence_items),
            confirmed_facts=len(memo.confirmed_facts or []),
            mapped_subquestions=covered,
            mapped_areas=mapped_areas,
            uncovered_subquestions=uncovered,
        )

        # Stash memo and documents for downstream agents
        context.trace["_memo"] = memo
        context.trace["_documents"] = collection.documents
        return context
