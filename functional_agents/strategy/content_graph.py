"""PH12.2 — ContentGraph: deterministic in-memory relationship graph for Strategy content.

Indexes canonical relationships between strategic options, recommendations,
assumptions, risks, opportunities, and evidence. Used by the theory-content
resolver to assign content items through explicit relationship chains rather
than raw keyword matching.

No LLM calls. No external services. Fully deterministic. Input-order invariant.
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def _ids(obj: dict[str, Any], *fields: str) -> list[str]:
    """Collect non-empty strings from any of the given fields of a dict."""
    result: list[str] = []
    for f in fields:
        val = obj.get(f)
        if isinstance(val, str) and val:
            result.append(val)
        elif isinstance(val, list):
            result.extend(s for s in val if isinstance(s, str) and s)
    return result


class ContentGraph:
    """Deterministic in-memory relationship graph for strategy content.

    Build once from research context, then query per-theory during assignment.

    Index keys (all sets of canonical string IDs):
        options:              option_id → {supporting_assumption_ids, associated_risk_ids,
                                           associated_opportunity_ids, supporting_recommendation_ids}
        assumptions:          assumption_id → {evidence_ids, supported_recommendation_ids}
        risks:                risk_id → {related_assumption_ids, affected_recommendation_ids, evidence_ids}
        opportunities:        opportunity_id → {related_assumption_ids, evidence_ids}
        recommendations:      recommendation_id → {supported_assumption_ids, evidence_ids}
        evidence → objects:   evidence_id → set of source_ids

    Reverse indexes are derived deterministically from forward indexes.
    """

    def __init__(self) -> None:
        # Forward indexes: id → frozenset of related ids
        self._option_assumptions:       dict[str, frozenset[str]] = {}
        self._option_risks:             dict[str, frozenset[str]] = {}
        self._option_opportunities:     dict[str, frozenset[str]] = {}
        self._option_recommendations:   dict[str, frozenset[str]] = {}

        self._assumption_evidence:      dict[str, frozenset[str]] = {}
        self._assumption_recommendations: dict[str, frozenset[str]] = {}

        self._risk_assumptions:         dict[str, frozenset[str]] = {}
        self._risk_recommendations:     dict[str, frozenset[str]] = {}
        self._risk_evidence:            dict[str, frozenset[str]] = {}

        self._opp_assumptions:          dict[str, frozenset[str]] = {}
        self._opp_evidence:             dict[str, frozenset[str]] = {}

        self._rec_assumptions:          dict[str, frozenset[str]] = {}
        self._rec_evidence:             dict[str, frozenset[str]] = {}

        # Reverse: assumption_id → set of option_ids that list it
        self._assumption_options:       dict[str, frozenset[str]] = {}

        # Object catalogs keyed by canonical ID
        self._options:       dict[str, dict[str, Any]] = {}
        self._assumptions:   dict[str, dict[str, Any]] = {}
        self._risks:         dict[str, dict[str, Any]] = {}
        self._opportunities: dict[str, dict[str, Any]] = {}
        self._recommendations: dict[str, dict[str, Any]] = {}
        self._evidence:      dict[str, dict[str, Any]] = {}

        # Diagnostics
        self.diagnostics: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def build(self, research: Any) -> "ContentGraph":
        """Populate the graph from the research context.

        Accepts an AgentContext or any object with list attributes:
        strategic_options, assumptions, risks, opportunities, recommendations,
        evidence_notes (or evidence).
        """
        options       = list(getattr(research, "strategic_options", None) or [])
        assumptions   = list(getattr(research, "assumptions", None) or [])
        risks         = list(getattr(research, "risks", None) or [])
        opportunities = list(getattr(research, "opportunities", None) or [])
        recommendations = list(getattr(research, "recommendations", None) or [])
        evidence_raw  = (
            list(getattr(research, "evidence_notes", None) or [])
            or list(getattr(research, "evidence", None) or [])
        )

        self._index_options(options)
        self._index_assumptions(assumptions)
        self._index_risks(risks)
        self._index_opportunities(opportunities)
        self._index_recommendations(recommendations)
        self._index_evidence(evidence_raw)
        self._build_reverse_indexes()

        LOGGER.debug(
            "[ContentGraph] built: options=%d assumptions=%d risks=%d "
            "opportunities=%d recommendations=%d evidence=%d",
            len(self._options), len(self._assumptions), len(self._risks),
            len(self._opportunities), len(self._recommendations), len(self._evidence),
        )
        return self

    def _index_options(self, items: list[dict[str, Any]]) -> None:
        for opt in items:
            if not isinstance(opt, dict):
                continue
            oid = opt.get("option_id") or opt.get("id") or ""
            if not oid:
                continue
            self._options[oid] = opt
            self._option_assumptions[oid] = frozenset(_ids(opt,
                "supporting_assumption_ids", "assumption_ids"))
            self._option_risks[oid] = frozenset(_ids(opt,
                "associated_risk_ids", "risk_ids"))
            self._option_opportunities[oid] = frozenset(_ids(opt,
                "associated_opportunity_ids", "opportunity_ids"))
            self._option_recommendations[oid] = frozenset(_ids(opt,
                "supporting_recommendation_ids", "recommendation_ids"))

    def _index_assumptions(self, items: list[dict[str, Any]]) -> None:
        for a in items:
            if not isinstance(a, dict):
                continue
            aid = a.get("assumption_id") or a.get("id") or ""
            if not aid:
                self._emit_diagnostic(
                    "assumption", "", "index", "missing_id",
                    "Assumption object missing assumption_id — excluded from graph"
                )
                continue
            self._assumptions[aid] = a
            self._assumption_evidence[aid] = frozenset(_ids(a, "evidence_ids"))
            self._assumption_recommendations[aid] = frozenset(_ids(a,
                "supported_recommendation_ids", "recommendation_ids"))

    def _index_risks(self, items: list[dict[str, Any]]) -> None:
        for r in items:
            if not isinstance(r, dict):
                continue
            rid = r.get("risk_id") or r.get("id") or ""
            if not rid:
                continue
            self._risks[rid] = r
            self._risk_assumptions[rid] = frozenset(_ids(r, "related_assumption_ids"))
            self._risk_recommendations[rid] = frozenset(_ids(r, "affected_recommendation_ids"))
            self._risk_evidence[rid] = frozenset(_ids(r, "evidence_ids"))

    def _index_opportunities(self, items: list[dict[str, Any]]) -> None:
        for o in items:
            if not isinstance(o, dict):
                continue
            oid = o.get("opportunity_id") or o.get("id") or ""
            if not oid:
                continue
            self._opportunities[oid] = o
            self._opp_assumptions[oid] = frozenset(_ids(o, "related_assumption_ids"))
            self._opp_evidence[oid] = frozenset(_ids(o, "evidence_ids"))

    def _index_recommendations(self, items: list[dict[str, Any]]) -> None:
        for rec in items:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("recommendation_id") or rec.get("id") or ""
            if not rid:
                continue
            self._recommendations[rid] = rec
            self._rec_assumptions[rid] = frozenset(_ids(rec,
                "supported_assumption_ids", "assumption_ids"))
            self._rec_evidence[rid] = frozenset(_ids(rec,
                "supporting_evidence", "evidence_ids"))

    def _index_evidence(self, items: list[dict[str, Any]]) -> None:
        for ev in items:
            if not isinstance(ev, dict):
                continue
            eid = ev.get("evidence_id") or ev.get("id") or ""
            if not eid:
                continue
            self._evidence[eid] = ev

    def _build_reverse_indexes(self) -> None:
        """Build reverse indexes from forward ones. Deterministic, order-independent."""
        # assumption → set of option_ids
        rev: dict[str, set[str]] = {}
        for opt_id, aid_set in self._option_assumptions.items():
            for aid in aid_set:
                rev.setdefault(aid, set()).add(opt_id)
        self._assumption_options = {aid: frozenset(opts) for aid, opts in rev.items()}

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def option(self, option_id: str) -> dict[str, Any]:
        return self._options.get(option_id, {})

    def assumption_ids_for_option(self, option_id: str) -> frozenset[str]:
        return self._option_assumptions.get(option_id, frozenset())

    def risk_ids_for_option(self, option_id: str) -> frozenset[str]:
        return self._option_risks.get(option_id, frozenset())

    def opportunity_ids_for_option(self, option_id: str) -> frozenset[str]:
        return self._option_opportunities.get(option_id, frozenset())

    def recommendation_ids_for_option(self, option_id: str) -> frozenset[str]:
        return self._option_recommendations.get(option_id, frozenset())

    def assumption_ids_for_recommendation(self, rec_id: str) -> frozenset[str]:
        return self._rec_assumptions.get(rec_id, frozenset())

    def evidence_ids_for_assumption(self, assumption_id: str) -> frozenset[str]:
        return self._assumption_evidence.get(assumption_id, frozenset())

    def evidence_ids_for_risk(self, risk_id: str) -> frozenset[str]:
        return self._risk_evidence.get(risk_id, frozenset())

    def evidence_ids_for_recommendation(self, rec_id: str) -> frozenset[str]:
        return self._rec_evidence.get(rec_id, frozenset())

    def evidence_ids_for_opportunity(self, opp_id: str) -> frozenset[str]:
        return self._opp_evidence.get(opp_id, frozenset())

    def risk_ids_for_assumption(self, assumption_id: str) -> frozenset[str]:
        """Reverse: risks that list this assumption in related_assumption_ids."""
        result: set[str] = set()
        for rid, aids in self._risk_assumptions.items():
            if assumption_id in aids:
                result.add(rid)
        return frozenset(result)

    def opportunity_ids_for_assumption(self, assumption_id: str) -> frozenset[str]:
        """Reverse: opportunities that list this assumption."""
        result: set[str] = set()
        for oid, aids in self._opp_assumptions.items():
            if assumption_id in aids:
                result.add(oid)
        return frozenset(result)

    def recommendation_ids_for_assumption(self, assumption_id: str) -> frozenset[str]:
        return self._assumption_recommendations.get(assumption_id, frozenset())

    def assumption_ids_for_risk(self, risk_id: str) -> frozenset[str]:
        return self._risk_assumptions.get(risk_id, frozenset())

    def assumption_ids_for_opportunity(self, opp_id: str) -> frozenset[str]:
        return self._opp_assumptions.get(opp_id, frozenset())

    def get_assumption(self, assumption_id: str) -> dict[str, Any]:
        obj = self._assumptions.get(assumption_id)
        if obj is None:
            self._emit_diagnostic("assumption", assumption_id, "lookup", "unknown_id",
                                  f"Unknown assumption_id: {assumption_id!r}")
        return obj or {}

    def get_risk(self, risk_id: str) -> dict[str, Any]:
        obj = self._risks.get(risk_id)
        if obj is None:
            self._emit_diagnostic("risk", risk_id, "lookup", "unknown_id",
                                  f"Unknown risk_id: {risk_id!r}")
        return obj or {}

    def get_opportunity(self, opp_id: str) -> dict[str, Any]:
        obj = self._opportunities.get(opp_id)
        if obj is None:
            self._emit_diagnostic("opportunity", opp_id, "lookup", "unknown_id",
                                  f"Unknown opportunity_id: {opp_id!r}")
        return obj or {}

    def get_recommendation(self, rec_id: str) -> dict[str, Any]:
        obj = self._recommendations.get(rec_id)
        if obj is None:
            self._emit_diagnostic("recommendation", rec_id, "lookup", "unknown_id",
                                  f"Unknown recommendation_id: {rec_id!r}")
        return obj or {}

    def get_evidence(self, evidence_id: str) -> dict[str, Any]:
        return self._evidence.get(evidence_id, {})

    @property
    def all_assumption_ids(self) -> frozenset[str]:
        return frozenset(self._assumptions)

    @property
    def all_risk_ids(self) -> frozenset[str]:
        return frozenset(self._risks)

    @property
    def all_opportunity_ids(self) -> frozenset[str]:
        return frozenset(self._opportunities)

    @property
    def all_recommendation_ids(self) -> frozenset[str]:
        return frozenset(self._recommendations)

    @property
    def all_evidence_ids(self) -> frozenset[str]:
        return frozenset(self._evidence)

    @property
    def has_explicit_links(self) -> bool:
        """True if any canonical relationships exist in the graph."""
        return bool(
            any(v for v in self._option_assumptions.values())
            or any(v for v in self._option_risks.values())
            or any(v for v in self._option_opportunities.values())
            or any(v for v in self._option_recommendations.values())
            or any(v for v in self._risk_assumptions.values())
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _emit_diagnostic(
        self,
        source_type: str,
        source_id: str,
        assignment_stage: str,
        fallback_reason: str,
        message: str,
        theory_id: str = "",
    ) -> None:
        entry = {
            "theory_id": theory_id,
            "source_type": source_type,
            "source_id": source_id,
            "assignment_stage": assignment_stage,
            "fallback_reason": fallback_reason,
            "message": message,
        }
        self.diagnostics.append(entry)
        LOGGER.debug("[ContentGraph] diagnostic: %s", message)
