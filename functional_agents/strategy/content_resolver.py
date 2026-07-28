"""PH12.2 — ContentResolver: theory-specific content assignment orchestrator.

Resolves theory-specific assumptions, risks, opportunities, recommendations,
evidence, and success conditions for each TheoryOfWinning using relationship-first
assignment from ContentGraph, supplemented by posture-relevance scoring.

Priority order (per spec):
  1. Explicit canonical relationships (option → assumption/risk/opportunity/rec)
  2. Mapped-option relationships
  3. Recommendation relationships
  4. Assumption/risk/opportunity relationships
  5. Normalized posture relevance (PostureRelevance)
  6. Controlled keyword inference (choice metadata)
  7. Symmetric fallback (when coverage is insufficient)

No LLM calls. No external services. Fully deterministic. Input-order invariant.
"""

from __future__ import annotations

import logging
from typing import Any

from .content_graph import ContentGraph
from .posture_normalizer import PostureNormalizer
from .posture_relevance import _obj_text, score_item
from .theory_content import (
    ContentConfidence,
    ContentCoverage,
    ContentLineageEntry,
    EvidenceLineageEntry,
    SuccessConditionEntry,
    TheoryContent,
)
from .strategic_position import TheoryOfWinning

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Assignment type bonus weights (spec §8)
# ---------------------------------------------------------------------------
_BONUS_OPTION_LINK     = 1.00
_BONUS_REC_LINK        = 0.80
_BONUS_ASSUMPTION_LINK = 0.70
_BONUS_POSTURE_MATCH   = 0.50
_BONUS_CATEGORY_MATCH  = 0.15
_BONUS_GENERIC         = 0.05
_MIN_RELEVANCE         = 0.20   # default minimum; overridden by ContentConfig

# ---------------------------------------------------------------------------
# Default limits (overridden by ContentConfig)
# ---------------------------------------------------------------------------
_MAX_ASSUMPTIONS    = 5
_MAX_RISKS          = 5
_MAX_OPPORTUNITIES  = 5
_MAX_RECOMMENDATIONS = 5
_MAX_EVIDENCE       = 12


class ContentResolver:
    """Resolves theory-specific content for a single TheoryOfWinning.

    Instantiate once, call resolve() per theory.
    """

    def __init__(
        self,
        graph: ContentGraph,
        content_config: Any = None,   # optional ContentConfig
    ) -> None:
        self._graph = graph
        self._cfg = content_config
        self._normalizer = PostureNormalizer()

        # Read limits from config or fall back to defaults
        cfg = self._cfg
        self._min_relevance  = float(getattr(cfg, "minimum_relevance_score", _MIN_RELEVANCE))
        self._max_assumptions    = int(getattr(cfg, "maximum_assumptions_per_theory", _MAX_ASSUMPTIONS))
        self._max_risks          = int(getattr(cfg, "maximum_risks_per_theory", _MAX_RISKS))
        self._max_opportunities  = int(getattr(cfg, "maximum_opportunities_per_theory", _MAX_OPPORTUNITIES))
        self._max_recommendations= int(getattr(cfg, "maximum_recommendations_per_theory", _MAX_RECOMMENDATIONS))
        self._max_evidence       = int(getattr(cfg, "maximum_evidence_per_theory", _MAX_EVIDENCE))
        self._allow_fallback     = bool(getattr(cfg, "allow_symmetric_fallback", True))

    def resolve(
        self,
        theory: TheoryOfWinning,
        mapped_option_id: str | None,
        mapping_confidence: str,
    ) -> TheoryContent:
        """Produce a TheoryContent for the given theory and its mapped option.

        Parameters
        ----------
        theory:
            The TheoryOfWinning to produce content for.
        mapped_option_id:
            Canonical option ID from the OptionMapper (may be None).
        mapping_confidence:
            Mapping confidence level from OptionMapper.
        """
        g = self._graph
        theory_id = theory.theory_id

        # Extract theory postures for relevance scoring
        choices_as_dicts = [c for c in (theory.strategic_choices or []) if isinstance(c, dict)]
        theory_postures = self._normalizer.theory_postures(choices_as_dicts)

        diagnostics: list[dict[str, Any]] = []

        # ----------------------------------------------------------------
        # 1. Recommendations
        # ----------------------------------------------------------------
        rec_ids, rec_lineage = self._assign_recommendations(
            theory_id, mapped_option_id, theory_postures, diagnostics
        )

        # ----------------------------------------------------------------
        # 2. Assumptions
        # ----------------------------------------------------------------
        assumption_ids, assumption_lineage = self._assign_assumptions(
            theory_id, mapped_option_id, rec_ids, theory_postures, diagnostics
        )

        # ----------------------------------------------------------------
        # 3. Risks
        # ----------------------------------------------------------------
        risk_ids, risk_lineage = self._assign_risks(
            theory_id, mapped_option_id, rec_ids, assumption_ids, theory_postures, diagnostics
        )

        # ----------------------------------------------------------------
        # 4. Opportunities
        # ----------------------------------------------------------------
        opp_ids, opp_lineage = self._assign_opportunities(
            theory_id, mapped_option_id, rec_ids, assumption_ids, theory_postures, diagnostics
        )

        # ----------------------------------------------------------------
        # 5. Evidence
        # ----------------------------------------------------------------
        evidence_lineage_entries = self._assign_evidence(
            theory_id, assumption_ids, risk_ids, opp_ids, rec_ids,
            mapped_option_id, theory_postures, diagnostics
        )
        evidence_ids = sorted({e.evidence_id for e in evidence_lineage_entries if e.evidence_id})

        # ----------------------------------------------------------------
        # 6. Success conditions
        # ----------------------------------------------------------------
        success_conditions = self._build_success_conditions(
            theory_id, mapped_option_id, opp_ids, rec_ids, theory, theory_postures
        )

        # ----------------------------------------------------------------
        # 7. Coverage and confidence
        # ----------------------------------------------------------------
        all_a = len(g.all_assumption_ids)
        all_r = len(g.all_risk_ids)
        all_o = len(g.all_opportunity_ids)
        all_rec = len(g.all_recommendation_ids)
        all_ev = len(g.all_evidence_ids)

        explicit_count = sum(
            1 for e in (assumption_lineage + risk_lineage + opp_lineage + rec_lineage)
            if e.assignment_type in (
                "option_link", "recommendation_link", "risk_link",
                "opportunity_link", "assumption_link"
            )
        )
        posture_count = sum(
            1 for e in (assumption_lineage + risk_lineage + opp_lineage + rec_lineage)
            if e.assignment_type == "posture_match"
        )
        fallback_count = sum(
            1 for e in (assumption_lineage + risk_lineage + opp_lineage + rec_lineage)
            if e.assignment_type == "symmetric_fallback"
        )

        coverage = ContentCoverage.compute(
            total_assumptions=all_a, total_risks=all_r,
            total_opportunities=all_o, total_recommendations=all_rec,
            total_evidence=all_ev,
            assigned_assumptions=len(assumption_ids),
            assigned_risks=len(risk_ids),
            assigned_opportunities=len(opp_ids),
            assigned_recommendations=len(rec_ids),
            assigned_evidence=len(evidence_ids),
            assigned_success_conditions=len(success_conditions),
            explicit_count=explicit_count,
            fallback_count=fallback_count,
        )

        ev_cov = coverage.evidence
        confidence = ContentConfidence.compute(
            explicit_count=explicit_count,
            fallback_count=fallback_count,
            posture_match_count=posture_count,
            contradiction_count=0,
            mapping_confidence=mapping_confidence,
            evidence_coverage=ev_cov,
        )

        # Fallback log
        content_fallbacks: list[dict[str, Any]] = []
        for e in (assumption_lineage + risk_lineage + opp_lineage + rec_lineage):
            if e.assignment_type == "symmetric_fallback":
                content_fallbacks.append({
                    "theory_id": theory_id,
                    "source_id": e.source_id,
                    "fallback_reason": "symmetric_fallback",
                    "stage": "content_assignment",
                })

        # Diagnostics: no explicit links
        if not g.has_explicit_links and all_a + all_r + all_o > 0:
            diagnostics.append({
                "theory_id": theory_id,
                "source_type": "graph",
                "source_id": "",
                "assignment_stage": "build",
                "fallback_reason": "no_explicit_links",
                "message": "ContentGraph has no explicit canonical relationship links; "
                           "all assignment uses posture relevance and symmetric fallback.",
            })

        content_lineage = {
            "assumptions":       assumption_lineage,
            "risks":             risk_lineage,
            "opportunities":     opp_lineage,
            "recommendations":   rec_lineage,
            "evidence":          [],
            "success_conditions": [],
        }

        LOGGER.debug(
            "[ContentResolver] theory=%s assumptions=%d risks=%d opps=%d recs=%d ev=%d "
            "coverage=%s confidence=%s",
            theory_id, len(assumption_ids), len(risk_ids), len(opp_ids),
            len(rec_ids), len(evidence_ids),
            coverage.status, confidence.level,
        )

        return TheoryContent(
            theory_id=theory_id,
            mapped_option_id=mapped_option_id,
            mapping_confidence=mapping_confidence,
            recommendation_ids=sorted(rec_ids),
            assumption_ids=sorted(assumption_ids),
            risk_ids=sorted(risk_ids),
            opportunity_ids=sorted(opp_ids),
            evidence_ids=evidence_ids,
            success_conditions=success_conditions,
            content_lineage=content_lineage,
            evidence_lineage=evidence_lineage_entries,
            coverage=coverage,
            confidence=confidence,
            content_fallbacks=content_fallbacks,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # Recommendation assignment (§ Mapped-Option-First)
    # ------------------------------------------------------------------

    def _assign_recommendations(
        self,
        theory_id: str,
        mapped_option_id: str | None,
        theory_postures: dict[str, str],
        diagnostics: list,
    ) -> tuple[list[str], list[ContentLineageEntry]]:
        g = self._graph
        assigned: dict[str, ContentLineageEntry] = {}

        # Tier 1: explicit option link
        if mapped_option_id:
            for rid in sorted(g.recommendation_ids_for_option(mapped_option_id)):
                if len(assigned) >= self._max_recommendations:
                    break
                if rid not in assigned:
                    assigned[rid] = ContentLineageEntry(
                        source_id=rid,
                        assignment_type="option_link",
                        via_ids=[mapped_option_id],
                        relevance_score=_BONUS_OPTION_LINK,
                        rationale=f"Explicitly linked to mapped option {mapped_option_id!r}.",
                    )

        # Tier 2: posture relevance for remaining
        if len(assigned) < self._max_recommendations:
            scored = self._score_all(
                g.all_recommendation_ids, assigned, theory_postures,
                _BONUS_POSTURE_MATCH, "posture_match",
                "Matched by theory posture.",
            )
            for rid, entry in scored:
                if len(assigned) >= self._max_recommendations:
                    break
                assigned[rid] = entry

        # Symmetric fallback
        if not assigned and self._allow_fallback:
            for rid in sorted(g.all_recommendation_ids)[:self._max_recommendations]:
                assigned[rid] = ContentLineageEntry(
                    source_id=rid,
                    assignment_type="symmetric_fallback",
                    via_ids=[],
                    relevance_score=0.0,
                    rationale="Symmetric fallback: no explicit option links.",
                )
            if assigned:
                diagnostics.append({
                    "theory_id": theory_id, "source_type": "recommendation",
                    "source_id": "", "assignment_stage": "recommendation_assignment",
                    "fallback_reason": "no_explicit_links",
                    "message": "No recommendation links found; using symmetric fallback.",
                })

        return list(assigned.keys()), list(assigned.values())

    # ------------------------------------------------------------------
    # Assumption assignment (§10)
    # ------------------------------------------------------------------

    def _assign_assumptions(
        self,
        theory_id: str,
        mapped_option_id: str | None,
        rec_ids: list[str],
        theory_postures: dict[str, str],
        diagnostics: list,
    ) -> tuple[list[str], list[ContentLineageEntry]]:
        g = self._graph
        assigned: dict[str, ContentLineageEntry] = {}

        # Tier 1: mapped option → assumptions
        if mapped_option_id:
            for aid in sorted(g.assumption_ids_for_option(mapped_option_id)):
                if len(assigned) >= self._max_assumptions:
                    break
                if aid not in assigned:
                    assigned[aid] = ContentLineageEntry(
                        source_id=aid,
                        assignment_type="option_link",
                        via_ids=[mapped_option_id],
                        relevance_score=_BONUS_OPTION_LINK,
                        rationale=f"Explicitly linked to mapped option {mapped_option_id!r}.",
                    )

        # Tier 2: recommendations → assumptions
        if len(assigned) < self._max_assumptions:
            for rec_id in rec_ids:
                for aid in sorted(g.assumption_ids_for_recommendation(rec_id)):
                    if aid not in assigned and len(assigned) < self._max_assumptions:
                        assigned[aid] = ContentLineageEntry(
                            source_id=aid,
                            assignment_type="recommendation_link",
                            via_ids=[rec_id],
                            relevance_score=_BONUS_REC_LINK,
                            rationale=f"Linked through recommendation {rec_id!r}.",
                        )

        # Tier 3: posture relevance
        if len(assigned) < self._max_assumptions:
            scored = self._score_all(
                g.all_assumption_ids, assigned, theory_postures,
                _BONUS_POSTURE_MATCH, "posture_match",
                "Matched by theory posture.",
            )
            for aid, entry in scored:
                if len(assigned) >= self._max_assumptions:
                    break
                assigned[aid] = entry

        # Symmetric fallback
        if not assigned and self._allow_fallback:
            for aid in sorted(g.all_assumption_ids)[:self._max_assumptions]:
                assigned[aid] = ContentLineageEntry(
                    source_id=aid,
                    assignment_type="symmetric_fallback",
                    via_ids=[],
                    relevance_score=0.0,
                    rationale="Symmetric fallback: no explicit links.",
                )
            if assigned:
                diagnostics.append({
                    "theory_id": theory_id, "source_type": "assumption",
                    "source_id": "", "assignment_stage": "assumption_assignment",
                    "fallback_reason": "no_explicit_links",
                    "message": "No explicit assumption links; using symmetric fallback.",
                })
        elif not assigned:
            diagnostics.append({
                "theory_id": theory_id, "source_type": "assumption",
                "source_id": "", "assignment_stage": "assumption_assignment",
                "fallback_reason": "no_canonical_evidence",
                "message": "Theory has no explicit linked assumptions.",
            })

        return list(assigned.keys()), list(assigned.values())

    # ------------------------------------------------------------------
    # Risk assignment (§11)
    # ------------------------------------------------------------------

    def _assign_risks(
        self,
        theory_id: str,
        mapped_option_id: str | None,
        rec_ids: list[str],
        assumption_ids: list[str],
        theory_postures: dict[str, str],
        diagnostics: list,
    ) -> tuple[list[str], list[ContentLineageEntry]]:
        g = self._graph
        assigned: dict[str, ContentLineageEntry] = {}

        # Tier 1: mapped option
        if mapped_option_id:
            for rid in sorted(g.risk_ids_for_option(mapped_option_id)):
                if len(assigned) >= self._max_risks:
                    break
                if rid not in assigned:
                    assigned[rid] = ContentLineageEntry(
                        source_id=rid,
                        assignment_type="option_link",
                        via_ids=[mapped_option_id],
                        relevance_score=_BONUS_OPTION_LINK,
                        rationale=f"Explicitly linked to mapped option {mapped_option_id!r}.",
                    )

        # Tier 2: assigned assumptions → risks
        if len(assigned) < self._max_risks:
            for aid in assumption_ids:
                for rid in sorted(g.risk_ids_for_assumption(aid)):
                    if rid not in assigned and len(assigned) < self._max_risks:
                        assigned[rid] = ContentLineageEntry(
                            source_id=rid,
                            assignment_type="assumption_link",
                            via_ids=[aid],
                            relevance_score=_BONUS_ASSUMPTION_LINK,
                            rationale=f"Risk linked to assigned assumption {aid!r}.",
                        )

        # Tier 3: posture relevance
        if len(assigned) < self._max_risks:
            scored = self._score_all(
                g.all_risk_ids, assigned, theory_postures,
                _BONUS_POSTURE_MATCH, "posture_match",
                "Matched by theory posture.",
            )
            for rid, entry in scored:
                if len(assigned) >= self._max_risks:
                    break
                assigned[rid] = entry

        # Symmetric fallback
        if not assigned and self._allow_fallback:
            for rid in sorted(g.all_risk_ids)[:self._max_risks]:
                assigned[rid] = ContentLineageEntry(
                    source_id=rid,
                    assignment_type="symmetric_fallback",
                    via_ids=[],
                    relevance_score=0.0,
                    rationale="Symmetric fallback: no explicit risk links.",
                )
            if assigned:
                diagnostics.append({
                    "theory_id": theory_id, "source_type": "risk",
                    "source_id": "", "assignment_stage": "risk_assignment",
                    "fallback_reason": "no_explicit_links",
                    "message": "No explicit risk links; using symmetric fallback.",
                })
        elif not assigned:
            diagnostics.append({
                "theory_id": theory_id, "source_type": "risk",
                "source_id": "", "assignment_stage": "risk_assignment",
                "fallback_reason": "no_canonical_evidence",
                "message": "Theory has no explicit linked risks.",
            })

        return list(assigned.keys()), list(assigned.values())

    # ------------------------------------------------------------------
    # Opportunity assignment (§12)
    # ------------------------------------------------------------------

    def _assign_opportunities(
        self,
        theory_id: str,
        mapped_option_id: str | None,
        rec_ids: list[str],
        assumption_ids: list[str],
        theory_postures: dict[str, str],
        diagnostics: list,
    ) -> tuple[list[str], list[ContentLineageEntry]]:
        g = self._graph
        assigned: dict[str, ContentLineageEntry] = {}

        # Tier 1: mapped option
        if mapped_option_id:
            for oid in sorted(g.opportunity_ids_for_option(mapped_option_id)):
                if len(assigned) >= self._max_opportunities:
                    break
                if oid not in assigned:
                    assigned[oid] = ContentLineageEntry(
                        source_id=oid,
                        assignment_type="option_link",
                        via_ids=[mapped_option_id],
                        relevance_score=_BONUS_OPTION_LINK,
                        rationale=f"Explicitly linked to mapped option {mapped_option_id!r}.",
                    )

        # Tier 2: assigned assumptions → opportunities
        if len(assigned) < self._max_opportunities:
            for aid in assumption_ids:
                for oid in sorted(g.opportunity_ids_for_assumption(aid)):
                    if oid not in assigned and len(assigned) < self._max_opportunities:
                        assigned[oid] = ContentLineageEntry(
                            source_id=oid,
                            assignment_type="assumption_link",
                            via_ids=[aid],
                            relevance_score=_BONUS_ASSUMPTION_LINK,
                            rationale=f"Opportunity linked to assigned assumption {aid!r}.",
                        )

        # Tier 3: posture relevance
        if len(assigned) < self._max_opportunities:
            scored = self._score_all(
                g.all_opportunity_ids, assigned, theory_postures,
                _BONUS_POSTURE_MATCH, "posture_match",
                "Matched by theory posture.",
            )
            for oid, entry in scored:
                if len(assigned) >= self._max_opportunities:
                    break
                assigned[oid] = entry

        # Symmetric fallback
        if not assigned and self._allow_fallback:
            for oid in sorted(g.all_opportunity_ids)[:self._max_opportunities]:
                assigned[oid] = ContentLineageEntry(
                    source_id=oid,
                    assignment_type="symmetric_fallback",
                    via_ids=[],
                    relevance_score=0.0,
                    rationale="Symmetric fallback: no explicit opportunity links.",
                )
            if assigned:
                diagnostics.append({
                    "theory_id": theory_id, "source_type": "opportunity",
                    "source_id": "", "assignment_stage": "opportunity_assignment",
                    "fallback_reason": "no_explicit_links",
                    "message": "No explicit opportunity links; using symmetric fallback.",
                })
        elif not assigned:
            diagnostics.append({
                "theory_id": theory_id, "source_type": "opportunity",
                "source_id": "", "assignment_stage": "opportunity_assignment",
                "fallback_reason": "no_canonical_evidence",
                "message": "Theory has no explicit linked opportunities.",
            })

        return list(assigned.keys()), list(assigned.values())

    # ------------------------------------------------------------------
    # Evidence assignment (§14/§15)
    # ------------------------------------------------------------------

    def _assign_evidence(
        self,
        theory_id: str,
        assumption_ids: list[str],
        risk_ids: list[str],
        opp_ids: list[str],
        rec_ids: list[str],
        mapped_option_id: str | None,
        theory_postures: dict[str, str],
        diagnostics: list,
    ) -> list[EvidenceLineageEntry]:
        g = self._graph
        # evidence_id → list of lineage paths
        ev_paths: dict[str, list[dict[str, str]]] = {}
        ev_scores: dict[str, float] = {}

        def _add(eid: str, source_type: str, source_id: str, score: float) -> None:
            if not eid:
                return
            ev_paths.setdefault(eid, [])
            path = {"source_type": source_type, "source_id": source_id}
            if path not in ev_paths[eid]:
                ev_paths[eid].append(path)
            ev_scores[eid] = max(ev_scores.get(eid, 0.0), score)

        # Assumption evidence
        for aid in assumption_ids:
            for eid in g.evidence_ids_for_assumption(aid):
                _add(eid, "assumption", aid, _BONUS_ASSUMPTION_LINK)

        # Risk evidence
        for rid in risk_ids:
            for eid in g.evidence_ids_for_risk(rid):
                _add(eid, "risk", rid, _BONUS_ASSUMPTION_LINK)

        # Opportunity evidence
        for oid in opp_ids:
            for eid in g.evidence_ids_for_opportunity(oid):
                _add(eid, "opportunity", oid, _BONUS_POSTURE_MATCH)

        # Recommendation evidence
        for rec_id in rec_ids:
            for eid in g.evidence_ids_for_recommendation(rec_id):
                _add(eid, "recommendation", rec_id, _BONUS_REC_LINK)

        # Symmetric fallback: use all evidence up to limit when none found
        if not ev_paths and self._allow_fallback:
            for eid in sorted(g.all_evidence_ids)[:self._max_evidence]:
                _add(eid, "fallback", "", _BONUS_GENERIC)
            if ev_paths:
                diagnostics.append({
                    "theory_id": theory_id, "source_type": "evidence",
                    "source_id": "", "assignment_stage": "evidence_assignment",
                    "fallback_reason": "no_canonical_evidence",
                    "message": "Evidence comes entirely from symmetric fallback.",
                })

        # Build EvidenceLineageEntry list, capped and sorted deterministically
        entries = []
        for eid in sorted(ev_paths.keys())[:self._max_evidence]:
            entries.append(EvidenceLineageEntry(
                evidence_id=eid,
                assignment_type=(
                    "explicit" if any(p["source_type"] != "fallback" for p in ev_paths[eid])
                    else "symmetric_fallback"
                ),
                relevance_score=round(ev_scores.get(eid, 0.0), 4),
                rationale=f"Evidence {eid!r} via {', '.join(p['source_type'] for p in ev_paths[eid][:2])}.",
                lineage_paths=ev_paths[eid],
            ))
        return entries

    # ------------------------------------------------------------------
    # Success conditions (§13)
    # ------------------------------------------------------------------

    def _build_success_conditions(
        self,
        theory_id: str,
        mapped_option_id: str | None,
        opp_ids: list[str],
        rec_ids: list[str],
        theory: TheoryOfWinning,
        theory_postures: dict[str, str],
    ) -> list[SuccessConditionEntry]:
        g = self._graph
        conditions: list[SuccessConditionEntry] = []

        # 1. Mapped option expected_outcomes
        if mapped_option_id:
            opt = g.option(mapped_option_id)
            outcomes = opt.get("expected_outcomes", [])
            if isinstance(outcomes, str):
                outcomes = [outcomes]
            for text in (outcomes or []):
                if isinstance(text, str) and text.strip():
                    conditions.append(SuccessConditionEntry(
                        text=text.strip(),
                        source_type="option",
                        source_ids=[mapped_option_id],
                        assignment_type="option_link",
                        relevance_score=_BONUS_OPTION_LINK,
                    ))

        # 2. Assigned opportunities (statement/description)
        for oid in opp_ids:
            opp = g.get_opportunity(oid)
            text = (
                opp.get("statement") or opp.get("title") or opp.get("description") or ""
            ).strip()
            if text and not any(c.text == text for c in conditions):
                conditions.append(SuccessConditionEntry(
                    text=text,
                    source_type="opportunity",
                    source_ids=[oid],
                    assignment_type="opportunity_link",
                    relevance_score=_BONUS_ASSUMPTION_LINK,
                ))

        # 3. Assigned recommendations (action/statement)
        for rec_id in rec_ids:
            rec = g.get_recommendation(rec_id)
            text = (rec.get("action") or rec.get("statement") or rec.get("title") or "").strip()
            if text and not any(c.text == text for c in conditions):
                conditions.append(SuccessConditionEntry(
                    text=text,
                    source_type="recommendation",
                    source_ids=[rec_id],
                    assignment_type="recommendation_link",
                    relevance_score=_BONUS_REC_LINK,
                ))

        # 4. Theory choices → posture-specific condition
        posture_condition = self._posture_success_condition(theory_postures)
        if posture_condition and not any(c.text == posture_condition for c in conditions):
            choices_ids = [
                c.get("selected_value", "")
                for c in (theory.strategic_choices or [])
                if isinstance(c, dict)
            ]
            conditions.append(SuccessConditionEntry(
                text=posture_condition,
                source_type="choice",
                source_ids=[i for i in choices_ids if i],
                assignment_type="posture_match",
                relevance_score=_BONUS_POSTURE_MATCH,
            ))

        return conditions

    @staticmethod
    def _posture_success_condition(theory_postures: dict[str, str]) -> str:
        """Generate one deterministic posture-specific success condition."""
        geo = theory_postures.get("geographic", "")
        pwr = theory_postures.get("power", "")
        tmg = theory_postures.get("timing", "")

        parts = []
        if geo == "diversified":
            parts.append("viable pathways secured across multiple RTOs")
        elif geo == "concentrated":
            parts.append("dominant queue position secured in target market")
        elif geo == "staged":
            parts.append("initial-market milestones met before portfolio expansion")

        if pwr == "btm_first":
            parts.append("BTM permitting, fuel supply, and generation economics validated")
        elif pwr == "grid_first":
            parts.append("interconnection queue position confirmed within target window")
        elif pwr == "hybrid":
            parts.append("dual-pathway grid and BTM commitments coordinated")

        if tmg == "milestone_gated":
            parts.append("capital released only after defined gates are cleared")
        elif tmg == "accelerate":
            parts.append("first-mover queue position secured before competitive window closes")
        elif tmg == "wait_and_monitor":
            parts.append("market signals monitored before committing capital")

        if not parts:
            return ""
        return " | ".join(parts).capitalize() + "."

    # ------------------------------------------------------------------
    # Generic relevance scoring helper
    # ------------------------------------------------------------------

    def _score_all(
        self,
        candidate_ids: frozenset[str],
        already_assigned: dict[str, Any],
        theory_postures: dict[str, str],
        bonus: float,
        assignment_type: str,
        rationale_template: str,
    ) -> list[tuple[str, ContentLineageEntry]]:
        """Score all candidates not yet assigned and return sorted descending list."""
        if not theory_postures:
            return []
        g = self._graph
        scored: list[tuple[float, str]] = []
        for cid in candidate_ids:
            if cid in already_assigned:
                continue
            obj = g._assumptions.get(cid) or g._risks.get(cid) or g._opportunities.get(cid) or g._recommendations.get(cid) or {}
            text = _obj_text(obj) if obj else ""
            if not text:
                continue
            from .posture_relevance import posture_relevance_score, contradiction_score
            ps, _ = posture_relevance_score(text, theory_postures)
            contra = contradiction_score(text, theory_postures)
            sc = round(bonus + ps - contra, 4)
            if sc >= self._min_relevance:
                scored.append((sc, cid))

        # Sort descending by score, then alphabetically for stability
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [
            (cid, ContentLineageEntry(
                source_id=cid,
                assignment_type=assignment_type,
                via_ids=[],
                relevance_score=sc,
                rationale=rationale_template,
            ))
            for sc, cid in scored
        ]
