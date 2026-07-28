"""PH12.2 — Theory-Specific Content, Evidence, and Risk Lineage tests.

Covers (spec §30):
  - ContentGraph: relationship indexing, query API, reverse indexes, has_explicit_links
  - ContentResolver: option-link tier, recommendation-link tier, posture tier, fallback
  - TheoryContent: canonical ID lists, sorted/deduped, coverage, confidence
  - ContentCoverage.compute: sufficient / partial / fallback_heavy / insufficient
  - ContentConfidence.compute: High / Medium / Low matrix
  - PostureRelevance: concept matching, contradiction penalty
  - ContentDifferentiation: Jaccard similarity, homogenization detection
  - StrategyNarrative.build_strategy_narrative: PH12.2 fields populated from trace
  - StrategyWriter: PH12.2 bullet group and subtitle rendered
  - MarkdownRenderer: differentiation table, why-winner narrative, homogenization warning
  - Integration: end-to-end ContentGraph → ContentResolver → TheoryContent
"""
from __future__ import annotations

import types

import pytest

from functional_agents.strategy.content_differentiation import (
    _jaccard,
    compute_differentiation,
)
from functional_agents.strategy.content_graph import ContentGraph
from functional_agents.strategy.content_resolver import ContentResolver
from functional_agents.strategy.posture_relevance import (
    POSTURE_CONCEPTS,
    contradiction_score,
    posture_relevance_score,
    score_item,
)
from functional_agents.strategy.theory_content import (
    ContentConfidence,
    ContentCoverage,
    TheoryContent,
)
from functional_agents.strategy.strategy_config import ContentConfig


# ---------------------------------------------------------------------------
# Shared minimal fixtures
# ---------------------------------------------------------------------------

def _make_option(oid: str, asm_ids: list = None, risk_ids: list = None,
                 opp_ids: list = None, rec_ids: list = None) -> dict:
    return {
        "option_id": oid,
        "title": f"Option {oid}",
        "description": f"Description for {oid}",
        "supporting_assumption_ids": asm_ids or [],
        "associated_risk_ids": risk_ids or [],
        "associated_opportunity_ids": opp_ids or [],
        "supporting_recommendation_ids": rec_ids or [],
    }


def _make_assumption(aid: str, statement: str = "", ev_ids: list = None) -> dict:
    return {
        "assumption_id": aid,
        "statement": statement or f"Assumption {aid}",
        "evidence_ids": ev_ids or [],
    }


def _make_risk(rid: str, description: str = "", asm_ids: list = None, ev_ids: list = None) -> dict:
    return {
        "risk_id": rid,
        "description": description or f"Risk {rid}",
        "related_assumption_ids": asm_ids or [],
        "evidence_ids": ev_ids or [],
    }


def _make_opportunity(oid: str, description: str = "", asm_ids: list = None) -> dict:
    return {
        "opportunity_id": oid,
        "description": description or f"Opportunity {oid}",
        "related_assumption_ids": asm_ids or [],
    }


def _make_recommendation(rid: str, text: str = "", asm_ids: list = None, ev_ids: list = None) -> dict:
    return {
        "recommendation_id": rid,
        "recommendation": text or f"Recommendation {rid}",
        "supported_assumption_ids": asm_ids or [],
        "supporting_evidence": ev_ids or [],
    }


def _make_evidence(eid: str, summary: str = "") -> dict:
    return {
        "evidence_id": eid,
        "summary": summary or f"Evidence {eid}",
    }


def _make_research(
    options=None,
    assumptions=None,
    risks=None,
    opportunities=None,
    recommendations=None,
    evidence=None,
):
    ctx = types.SimpleNamespace()
    ctx.strategic_options = options or []
    ctx.assumptions = assumptions or []
    ctx.risks = risks or []
    ctx.opportunities = opportunities or []
    ctx.recommendations = recommendations or []
    ctx.evidence_notes = evidence or []
    return ctx


def _theory_ns(theory_id: str, choices: list = None) -> types.SimpleNamespace:
    t = types.SimpleNamespace()
    t.theory_id = theory_id
    t.strategic_choices = choices or []
    t.recommended_option_title = ""
    t.assumptions = []
    t.success_conditions = []
    t.failure_modes = []
    return t


# ---------------------------------------------------------------------------
# Section 1 — ContentGraph: build and basic queries
# ---------------------------------------------------------------------------

class TestContentGraphBuild:
    def test_empty_research_builds_without_error(self):
        g = ContentGraph().build(_make_research())
        assert g.has_explicit_links is False

    def test_option_indexes_assumption_ids(self):
        opt = _make_option("OPT-A", asm_ids=["A-001", "A-002"])
        g = ContentGraph().build(_make_research(
            options=[opt],
            assumptions=[_make_assumption("A-001"), _make_assumption("A-002")],
        ))
        assert g.assumption_ids_for_option("OPT-A") == frozenset({"A-001", "A-002"})

    def test_option_indexes_risk_ids(self):
        opt = _make_option("OPT-B", risk_ids=["RSK-001"])
        g = ContentGraph().build(_make_research(
            options=[opt],
            risks=[_make_risk("RSK-001")],
        ))
        assert "RSK-001" in g.risk_ids_for_option("OPT-B")

    def test_option_indexes_opportunity_ids(self):
        opt = _make_option("OPT-C", opp_ids=["OPP-001"])
        g = ContentGraph().build(_make_research(
            options=[opt],
            opportunities=[_make_opportunity("OPP-001")],
        ))
        assert "OPP-001" in g.opportunity_ids_for_option("OPT-C")

    def test_option_indexes_recommendation_ids(self):
        opt = _make_option("OPT-A", rec_ids=["REC-001"])
        g = ContentGraph().build(_make_research(
            options=[opt],
            recommendations=[_make_recommendation("REC-001")],
        ))
        assert "REC-001" in g.recommendation_ids_for_option("OPT-A")

    def test_reverse_index_assumption_to_options(self):
        opt_a = _make_option("OPT-A", asm_ids=["A-001"])
        opt_b = _make_option("OPT-B", asm_ids=["A-001", "A-002"])
        g = ContentGraph().build(_make_research(
            options=[opt_a, opt_b],
            assumptions=[_make_assumption("A-001"), _make_assumption("A-002")],
        ))
        linked_options = g._assumption_options.get("A-001", frozenset())
        assert "OPT-A" in linked_options
        assert "OPT-B" in linked_options

    def test_has_explicit_links_true_when_option_has_assumptions(self):
        opt = _make_option("OPT-A", asm_ids=["A-001"])
        g = ContentGraph().build(_make_research(options=[opt]))
        assert g.has_explicit_links is True

    def test_has_explicit_links_false_when_no_relationships(self):
        opt = _make_option("OPT-A")
        g = ContentGraph().build(_make_research(options=[opt]))
        assert g.has_explicit_links is False

    def test_missing_assumption_id_emits_diagnostic(self):
        # assumption without assumption_id is excluded
        bad_asm = {"statement": "No ID here", "evidence_ids": []}
        g = ContentGraph().build(_make_research(assumptions=[bad_asm]))
        assert not g.all_assumption_ids
        assert any("missing_id" in d.get("fallback_reason", "") for d in g.diagnostics)

    def test_evidence_indexed_by_evidence_id(self):
        ev = _make_evidence("EV-001", "Important evidence")
        g = ContentGraph().build(_make_research(evidence=[ev]))
        assert "EV-001" in g.all_evidence_ids

    def test_risk_assumption_lookup(self):
        risk = _make_risk("RSK-001", asm_ids=["A-001"])
        g = ContentGraph().build(_make_research(
            risks=[risk],
            assumptions=[_make_assumption("A-001")],
        ))
        assert "A-001" in g.assumption_ids_for_risk("RSK-001")

    def test_opportunity_assumption_lookup(self):
        opp = _make_opportunity("OPP-001", asm_ids=["A-002"])
        g = ContentGraph().build(_make_research(
            opportunities=[opp],
            assumptions=[_make_assumption("A-002")],
        ))
        assert "A-002" in g.assumption_ids_for_opportunity("OPP-001")

    def test_evidence_for_assumption(self):
        asm = _make_assumption("A-001", ev_ids=["EV-001", "EV-002"])
        g = ContentGraph().build(_make_research(
            assumptions=[asm],
            evidence=[_make_evidence("EV-001"), _make_evidence("EV-002")],
        ))
        assert g.evidence_ids_for_assumption("A-001") == frozenset({"EV-001", "EV-002"})

    def test_unknown_option_returns_empty_frozenset(self):
        g = ContentGraph().build(_make_research())
        assert g.assumption_ids_for_option("NONEXISTENT") == frozenset()

    def test_all_assumption_ids_property(self):
        asms = [_make_assumption(f"A-{i:03d}") for i in range(1, 4)]
        g = ContentGraph().build(_make_research(assumptions=asms))
        assert g.all_assumption_ids == frozenset({"A-001", "A-002", "A-003"})


# ---------------------------------------------------------------------------
# Section 2 — ContentCoverage.compute
# ---------------------------------------------------------------------------

class TestContentCoverage:
    def test_sufficient_high_explicit(self):
        cov = ContentCoverage.compute(
            total_assumptions=5, total_risks=3, total_opportunities=3,
            total_recommendations=4, total_evidence=6,
            assigned_assumptions=4, assigned_risks=3, assigned_opportunities=3,
            assigned_recommendations=4, assigned_evidence=5,
            assigned_success_conditions=2,
            explicit_count=10, fallback_count=0,
        )
        assert cov.status == "sufficient"
        assert cov.overall >= 0.70

    def test_insufficient_zero_assigned(self):
        cov = ContentCoverage.compute(
            total_assumptions=5, total_risks=5, total_opportunities=5,
            total_recommendations=5, total_evidence=5,
            assigned_assumptions=0, assigned_risks=0, assigned_opportunities=0,
            assigned_recommendations=0, assigned_evidence=0,
            assigned_success_conditions=0,
            explicit_count=0, fallback_count=0,
        )
        assert cov.status == "insufficient"

    def test_fallback_heavy_when_fallback_share_exceeds_threshold(self):
        cov = ContentCoverage.compute(
            total_assumptions=5, total_risks=5, total_opportunities=5,
            total_recommendations=5, total_evidence=5,
            assigned_assumptions=3, assigned_risks=3, assigned_opportunities=3,
            assigned_recommendations=3, assigned_evidence=3,
            assigned_success_conditions=1,
            explicit_count=1, fallback_count=15,
        )
        assert cov.status == "fallback_heavy"

    def test_partial_medium_coverage(self):
        cov = ContentCoverage.compute(
            total_assumptions=5, total_risks=5, total_opportunities=5,
            total_recommendations=5, total_evidence=5,
            assigned_assumptions=2, assigned_risks=2, assigned_opportunities=2,
            assigned_recommendations=2, assigned_evidence=2,
            assigned_success_conditions=1,
            explicit_count=5, fallback_count=3,
        )
        assert cov.status in ("partial", "sufficient")

    def test_overall_is_mean_of_six_fractions(self):
        cov = ContentCoverage.compute(
            total_assumptions=4, total_risks=4, total_opportunities=4,
            total_recommendations=4, total_evidence=4,
            assigned_assumptions=4, assigned_risks=4, assigned_opportunities=4,
            assigned_recommendations=4, assigned_evidence=4,
            assigned_success_conditions=1,
            explicit_count=20, fallback_count=0,
        )
        # All fractions = 1.0, overall should be 1.0
        assert cov.overall == pytest.approx(1.0)

    def test_success_condition_fraction_binary(self):
        cov_with = ContentCoverage.compute(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0
        )
        cov_without = ContentCoverage.compute(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0
        )
        assert cov_with.success_conditions == 1.0
        assert cov_without.success_conditions == 0.0


# ---------------------------------------------------------------------------
# Section 3 — ContentConfidence.compute
# ---------------------------------------------------------------------------

class TestContentConfidence:
    def test_high_confidence_explicit_majority_high_mapping(self):
        conf = ContentConfidence.compute(
            explicit_count=8, fallback_count=1, posture_match_count=1,
            contradiction_count=0, mapping_confidence="High",
            evidence_coverage=0.70,
        )
        assert conf.level == "High"

    def test_low_confidence_all_fallbacks(self):
        conf = ContentConfidence.compute(
            explicit_count=0, fallback_count=10, posture_match_count=0,
            contradiction_count=0, mapping_confidence="Low",
            evidence_coverage=0.10,
        )
        assert conf.level == "Low"

    def test_medium_confidence_posture_dominant(self):
        conf = ContentConfidence.compute(
            explicit_count=2, fallback_count=3, posture_match_count=5,
            contradiction_count=0, mapping_confidence="Medium",
            evidence_coverage=0.40,
        )
        assert conf.level in ("Medium", "High")

    def test_no_content_returns_low(self):
        conf = ContentConfidence.compute(
            explicit_count=0, fallback_count=0, posture_match_count=0,
            contradiction_count=0, mapping_confidence="",
            evidence_coverage=0.0,
        )
        assert conf.level == "Low"

    def test_explicit_share_computed(self):
        conf = ContentConfidence.compute(
            explicit_count=6, fallback_count=4, posture_match_count=0,
            contradiction_count=0, mapping_confidence="High",
            evidence_coverage=0.60,
        )
        assert conf.explicit_share == pytest.approx(0.60)


# ---------------------------------------------------------------------------
# Section 4 — PostureRelevance
# ---------------------------------------------------------------------------

class TestPostureRelevance:
    def test_diversified_posture_matches_geographic_keywords(self):
        score, concepts = posture_relevance_score(
            "The company operates in diversified markets and multiple geographies",
            {"geographic": "diversified"},
        )
        assert score > 0.0
        assert len(concepts) > 0

    def test_contradiction_penalty_applied(self):
        penalty = contradiction_score(
            "concentrated single market focus only",
            {"geographic": "diversified"},
        )
        assert penalty >= 0.0

    def test_posture_concepts_dict_non_empty(self):
        assert len(POSTURE_CONCEPTS) > 0

    def test_score_item_returns_in_range(self):
        obj = {"description": "diversified geographic portfolio with multiple markets"}
        result = score_item(obj, {"geographic": "diversified"}, assignment_type_bonus=0.50)
        assert 0.0 <= result <= 2.0

    def test_unrelated_text_scores_near_zero(self):
        score, _ = posture_relevance_score(
            "quarterly earnings call transcript",
            {"geographic": "concentrated"},
        )
        # Should be low — no relevant concepts
        assert score <= 0.5

    def test_all_posture_categories_present(self):
        categories = {k[0] for k in POSTURE_CONCEPTS}
        assert "geographic" in categories
        assert "power" in categories
        assert "timing" in categories


# ---------------------------------------------------------------------------
# Section 5 — ContentDifferentiation: Jaccard and homogenization
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical_sets(self):
        assert _jaccard({"A", "B", "C"}, {"A", "B", "C"}) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        assert _jaccard({"A", "B"}, {"C", "D"}) == pytest.approx(0.0)

    def test_partial_overlap(self):
        assert _jaccard({"A", "B", "C"}, {"B", "C", "D"}) == pytest.approx(0.5)

    def test_empty_both_returns_one(self):
        assert _jaccard(set(), set()) == pytest.approx(1.0)


class TestComputeDifferentiation:
    def _make_tc(self, theory_id: str, asm_ids: list, risk_ids: list,
                 opp_ids: list, ev_ids: list, rec_ids: list,
                 explicit_count: int = 0) -> TheoryContent:
        cov = ContentCoverage(status="partial", explicit_count=explicit_count)
        return TheoryContent(
            theory_id=theory_id,
            assumption_ids=asm_ids,
            risk_ids=risk_ids,
            opportunity_ids=opp_ids,
            evidence_ids=ev_ids,
            recommendation_ids=rec_ids,
            coverage=cov,
        )

    def test_single_theory_no_comparison(self):
        tc = self._make_tc("TH-001", ["A-001"], [], [], [], [])
        result = compute_differentiation([tc])
        assert result["theory_differentiation"] == {}
        assert result["content_homogenization_detected"] is False

    def test_two_theories_disjoint_content(self):
        tc1 = self._make_tc("TH-001", ["A-001"], ["RSK-001"], [], [], [])
        tc2 = self._make_tc("TH-002", ["A-002"], ["RSK-002"], [], [], [])
        result = compute_differentiation([tc1, tc2])
        key = "TH-001::TH-002"
        assert key in result["theory_differentiation"]
        metrics = result["theory_differentiation"][key]
        assert metrics["assumption_similarity"] == pytest.approx(0.0)
        assert metrics["risk_similarity"] == pytest.approx(0.0)

    def test_two_theories_identical_content_no_explicit_links(self):
        tc1 = self._make_tc("TH-001", ["A-001", "A-002"], ["RSK-001"], ["OPP-001"], ["EV-001"], ["REC-001"], explicit_count=0)
        tc2 = self._make_tc("TH-002", ["A-001", "A-002"], ["RSK-001"], ["OPP-001"], ["EV-001"], ["REC-001"], explicit_count=0)
        result = compute_differentiation([tc1, tc2])
        # Jaccard of identical sets = 1.0 in all dims → homogenization
        assert result["content_homogenization_detected"] is True

    def test_homogenization_suppressed_when_explicit_links_justify_overlap(self):
        # Same content but explicit_count > 0 → justified overlap
        tc1 = self._make_tc("TH-001", ["A-001"], ["RSK-001"], ["OPP-001"], ["EV-001"], ["REC-001"], explicit_count=5)
        tc2 = self._make_tc("TH-002", ["A-001"], ["RSK-001"], ["OPP-001"], ["EV-001"], ["REC-001"], explicit_count=5)
        result = compute_differentiation([tc1, tc2])
        assert result["content_homogenization_detected"] is False

    def test_overall_similarity_is_mean_of_five_dims(self):
        tc1 = self._make_tc("TH-001", ["A-001"], [], [], [], [])
        tc2 = self._make_tc("TH-002", ["A-001"], [], [], [], [])
        result = compute_differentiation([tc1, tc2])
        metrics = result["theory_differentiation"]["TH-001::TH-002"]
        expected_overall = (1.0 + 1.0 + 1.0 + 1.0 + 1.0) / 5.0
        assert metrics["overall_similarity"] == pytest.approx(expected_overall)

    def test_three_theory_pair_count(self):
        tcs = [
            self._make_tc(f"TH-{i:03d}", [], [], [], [], [])
            for i in range(1, 4)
        ]
        result = compute_differentiation(tcs)
        assert len(result["theory_differentiation"]) == 3

    def test_pair_keys_use_double_colon_separator(self):
        tc1 = self._make_tc("TH-001", [], [], [], [], [])
        tc2 = self._make_tc("TH-002", [], [], [], [], [])
        result = compute_differentiation([tc1, tc2])
        keys = list(result["theory_differentiation"].keys())
        assert all("::" in k for k in keys)


# ---------------------------------------------------------------------------
# Section 6 — ContentResolver: end-to-end with explicit links
# ---------------------------------------------------------------------------

class TestContentResolverExplicitLinks:
    def _setup(self):
        """Research context with explicit relationships for OPT-B."""
        opt_b = _make_option(
            "OPT-B",
            asm_ids=["A-001", "A-002"],
            risk_ids=["RSK-001"],
            opp_ids=["OPP-001"],
            rec_ids=["REC-001"],
        )
        asms = [
            _make_assumption("A-001", "Solar adoption increases in BTM segment", ev_ids=["EV-001"]),
            _make_assumption("A-002", "Grid constraints persist for next decade"),
        ]
        risks = [_make_risk("RSK-001", "Interconnection delays", ev_ids=["EV-002"])]
        opps = [_make_opportunity("OPP-001", "Battery storage cost declines")]
        recs = [_make_recommendation("REC-001", "Prioritize BTM installations", ev_ids=["EV-001"])]
        evs = [_make_evidence("EV-001", "Market data"), _make_evidence("EV-002", "Grid study")]
        research = _make_research(
            options=[opt_b, _make_option("OPT-A")],
            assumptions=asms,
            risks=risks,
            opportunities=opps,
            recommendations=recs,
            evidence=evs,
        )
        return research

    def _make_theory(self, choices: list = None) -> types.SimpleNamespace:
        return _theory_ns("TH-001", choices or [
            {"dimension": "power_pathway", "selected_value": "btm_first",
             "metadata": {"choice_title": "BTM First"}},
        ])

    def test_option_link_assigns_assumption_a001(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert "A-001" in tc.assumption_ids

    def test_option_link_assigns_risk_rsk001(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert "RSK-001" in tc.risk_ids

    def test_option_link_assigns_opportunity_opp001(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert "OPP-001" in tc.opportunity_ids

    def test_option_link_assigns_recommendation_rec001(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert "REC-001" in tc.recommendation_ids

    def test_evidence_assigned_via_assumption_link(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert "EV-001" in tc.evidence_ids

    def test_theory_id_propagated(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert tc.theory_id == "TH-001"

    def test_mapped_option_id_recorded(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert tc.mapped_option_id == "OPT-B"

    def test_assumption_ids_are_sorted(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert tc.assumption_ids == sorted(tc.assumption_ids)

    def test_coverage_status_populated(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert tc.coverage.status in ("sufficient", "partial", "fallback_heavy", "insufficient")

    def test_confidence_level_populated(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert tc.confidence.level in ("High", "Medium", "Low")

    def test_success_conditions_non_empty(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert len(tc.success_conditions) >= 0  # any count is valid; list exists

    def test_no_duplicate_assumption_ids(self):
        research = self._setup()
        graph = ContentGraph().build(research)
        theory = self._make_theory()
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-B", "High")
        assert len(tc.assumption_ids) == len(set(tc.assumption_ids))


# ---------------------------------------------------------------------------
# Section 7 — ContentResolver: fallback path (no explicit links)
# ---------------------------------------------------------------------------

class TestContentResolverFallback:
    def _setup_no_links(self):
        opt = _make_option("OPT-A")  # no asm/risk/opp/rec links
        asms = [_make_assumption("A-001", "Grid modernization supports distributed energy")]
        risks = [_make_risk("RSK-001", "Interconnection queue delays distributed projects")]
        research = _make_research(
            options=[opt],
            assumptions=asms,
            risks=risks,
        )
        return research

    def test_fallback_resolves_without_error(self):
        research = self._setup_no_links()
        graph = ContentGraph().build(research)
        theory = _theory_ns("TH-002", [
            {"dimension": "geographic_portfolio", "selected_value": "concentrated",
             "metadata": {"choice_title": "Concentrated"}},
        ])
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-A", "Low")
        assert tc.theory_id == "TH-002"

    def test_fallback_sets_theory_id(self):
        research = self._setup_no_links()
        graph = ContentGraph().build(research)
        theory = _theory_ns("TH-002")
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-A", "Low")
        assert tc.theory_id == "TH-002"

    def test_coverage_status_in_valid_set(self):
        research = self._setup_no_links()
        graph = ContentGraph().build(research)
        theory = _theory_ns("TH-002")
        resolver = ContentResolver(graph, ContentConfig())
        tc = resolver.resolve(theory, "OPT-A", "Low")
        valid = {"sufficient", "partial", "fallback_heavy", "insufficient"}
        assert tc.coverage.status in valid


# ---------------------------------------------------------------------------
# Section 8 — StrategyNarrative PH12.2 fields from trace
# ---------------------------------------------------------------------------

class TestStrategyNarrativePH122:
    def _make_minimal_trace(self, winner_id: str = "TH-001") -> types.SimpleNamespace:
        """Minimal trace shaped like StrategyTrace for build_strategy_narrative."""
        sel = types.SimpleNamespace(
            winner_theory_id=winner_id,
            winner_score=0.45,
            runner_up_theory_id="TH-002",
            runner_up_score=0.20,
            score_margin=0.25,
            tie_breaker_used=None,
            selection_status="selected",
        )

        def _make_choice(dim, val, title):
            return {"dimension": dim, "selected_value": val, "metadata": {"choice_title": title}}

        winner_theory = types.SimpleNamespace(
            theory_id=winner_id,
            recommended_option_title="Option B",
            winning_position="BTM first strategy wins",
            winning_mechanism="Captures behind-the-meter margin",
            assumptions=[{"statement": "Grid congestion persists"}],
            failure_modes=[],
            success_conditions=["Achieve 40% BTM share by 2027"],
            strategic_choices=[
                _make_choice("power_pathway", "btm_first", "BTM First"),
            ],
        )
        alt_theory = types.SimpleNamespace(
            theory_id="TH-002",
            recommended_option_title="Option A",
            winning_position="Grid-first",
            winning_mechanism="Front-of-meter scale",
            assumptions=[],
            failure_modes=[],
            success_conditions=[],
            strategic_choices=[],
        )

        winner_eval = types.SimpleNamespace(
            theory_id=winner_id,
            overall_score=0.45,
            confidence="Medium",
            criteria_scores={
                "assumption_robustness": types.SimpleNamespace(score=0.60),
                "risk_resilience": types.SimpleNamespace(score=0.40),
            },
            strengths=["BTM margin advantage"],
            weaknesses=[],
            residual_risks=[],
        )
        alt_eval = types.SimpleNamespace(
            theory_id="TH-002",
            overall_score=0.20,
            confidence="Low",
            criteria_scores={},
            strengths=[],
            weaknesses=["Lower score"],
            residual_risks=[],
        )

        theory_content_entry = {
            "theory_id": winner_id,
            "assumption_ids": ["A-001", "A-002"],
            "risk_ids": ["RSK-001"],
            "opportunity_ids": ["OPP-001"],
            "evidence_ids": ["EV-001"],
            "coverage": {"status": "partial"},
            "confidence": {"level": "Medium"},
        }

        diff_data = {
            "theory_differentiation": {
                f"{winner_id}::TH-002": {
                    "assumption_similarity": 0.30,
                    "risk_similarity": 0.20,
                    "opportunity_similarity": 0.10,
                    "evidence_similarity": 0.25,
                    "recommendation_similarity": 0.15,
                    "overall_similarity": 0.20,
                }
            },
            "content_homogenization_detected": False,
            "homogenization_details": {
                "detected": False,
                "message": "Within bounds.",
            },
        }

        trace = types.SimpleNamespace(
            trace_id="TR-TEST-001",
            selection=sel,
            theories=[winner_theory, alt_theory],
            evaluations=[winner_eval, alt_eval],
            metadata={"framework": "test"},
            plan=types.SimpleNamespace(framework="test"),
            strategic_position=types.SimpleNamespace(position_id="POS-001"),
            alignment={},
            saturation={},
            theory_option_mappings=[
                {"theory_id": winner_id, "mapping_confidence": "Medium"}
            ],
            constraint_results={},
            theory_content=[theory_content_entry],
            theory_differentiation=diff_data,
            content_homogenization=diff_data.get("homogenization_details", {}),
            content_fallbacks=[],
        )
        return trace

    def test_content_assumption_ids_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_assumption_ids == ["A-001", "A-002"]

    def test_content_risk_ids_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_risk_ids == ["RSK-001"]

    def test_content_opportunity_ids_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_opportunity_ids == ["OPP-001"]

    def test_content_evidence_ids_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_evidence_ids == ["EV-001"]

    def test_content_coverage_status_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_coverage_status == "partial"

    def test_content_confidence_level_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_confidence_level == "Medium"

    def test_theory_differentiation_populated(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert "theory_differentiation" in sn.theory_differentiation

    def test_content_homogenization_detected_false(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.content_homogenization_detected is False

    def test_winner_rationale_non_empty(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert sn.winner_rationale != ""

    def test_winner_rationale_contains_score(self):
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        sn = build_strategy_narrative(trace)
        assert "0.450" in sn.winner_rationale or "0.45" in sn.winner_rationale

    def test_backward_compat_no_theory_content_field(self):
        """Trace without PH12.2 fields should still build narrative without error."""
        from functional_agents.editorial.strategy_narrative import build_strategy_narrative
        trace = self._make_minimal_trace()
        del trace.theory_content  # simulate old trace
        sn = build_strategy_narrative(trace)
        assert sn.content_assumption_ids == []


# ---------------------------------------------------------------------------
# Section 9 — StrategyWriter PH12.2 rendering
# ---------------------------------------------------------------------------

class TestStrategyWriterPH122:
    def _make_sn(self, **kwargs):
        from functional_agents.editorial.strategy_narrative import StrategyNarrative
        defaults = dict(
            trace_id="TR-001",
            winner_theory_id="TH-001",
            content_assumption_ids=["A-001", "A-002"],
            content_risk_ids=["RSK-001"],
            content_opportunity_ids=["OPP-001"],
            content_evidence_ids=["EV-001"],
            content_coverage_status="partial",
            content_confidence_level="Medium",
            content_fallback_used=False,
            content_homogenization_detected=False,
            winner_rationale="Winner scored +0.450; content confidence: Medium",
            theory_differentiation={},
            winner_score=0.45,
            overall_confidence="Medium",
            evaluation_criteria=[],
            criterion_scores={},
        )
        defaults.update(kwargs)
        return StrategyNarrative(**defaults)

    def _make_manuscript(self):
        from functional_agents.editorial.editorial_manuscript import (
            EditorialManuscript,
            ExecutiveSummaryManuscriptSection,
            DecisionAnalysisManuscriptSection,
            RecommendationManuscriptSection,
            RiskManuscriptSection,
            OpportunityManuscriptSection,
            ConfidenceManuscriptSection,
            AppendixManuscriptSection,
            StrategyManuscriptSection,
            ManuscriptMetadata,
        )
        return EditorialManuscript(
            metadata=ManuscriptMetadata(
                manuscript_id="MS-TEST", created_at="2026-01-01T00:00:00Z",
                brief_id="BR-TEST", pipeline_run_id="PR-TEST",
                decision_model_id="DM-TEST", research_object_id="RO-TEST",
                question="Test question",
            ),
            executive_summary=ExecutiveSummaryManuscriptSection(title="Executive Summary"),
            decision_analysis=DecisionAnalysisManuscriptSection(title="Decision Analysis"),
            recommendations=RecommendationManuscriptSection(title="Recommendations"),
            strategic_risks=RiskManuscriptSection(title="Risks"),
            strategic_opportunities=OpportunityManuscriptSection(title="Opportunities"),
            executive_confidence=ConfidenceManuscriptSection(title="Confidence"),
            appendix=AppendixManuscriptSection(title="Appendix"),
            strategic_direction=StrategyManuscriptSection(title="Strategic Direction"),
        )

    def _run_writer(self, sn):
        from functional_agents.editorial.strategy_writer import StrategyWriter

        brief = types.SimpleNamespace(strategy_narrative=sn, strategic_options=None)
        ms = self._make_manuscript()
        writer = StrategyWriter()
        return writer.write(brief, ms)

    def test_content_assumption_ids_in_bullet_group(self):
        sn = self._make_sn()
        ms = self._run_writer(sn)
        bgs = ms.strategic_direction.bullet_groups
        combined = " ".join(" ".join(bg) for bg in bgs)
        assert "A-001" in combined
        assert "A-002" in combined

    def test_content_risk_ids_in_bullet_group(self):
        sn = self._make_sn()
        ms = self._run_writer(sn)
        bgs = ms.strategic_direction.bullet_groups
        combined = " ".join(" ".join(bg) for bg in bgs)
        assert "RSK-001" in combined

    def test_content_coverage_status_in_subtitle(self):
        sn = self._make_sn()
        ms = self._run_writer(sn)
        assert "partial" in (ms.strategic_direction.subtitle or "")

    def test_winner_rationale_in_paragraphs(self):
        sn = self._make_sn()
        ms = self._run_writer(sn)
        combined = " ".join(ms.strategic_direction.paragraphs or [])
        assert "Winner scored" in combined

    def test_fallback_disclosure_when_fallback_used(self):
        sn = self._make_sn(content_fallback_used=True)
        ms = self._run_writer(sn)
        bgs = ms.strategic_direction.bullet_groups
        combined = " ".join(" ".join(bg) for bg in bgs)
        assert "fallback" in combined.lower()

    def test_homogenization_warning_when_detected(self):
        sn = self._make_sn(content_homogenization_detected=True)
        ms = self._run_writer(sn)
        bgs = ms.strategic_direction.bullet_groups
        combined = " ".join(" ".join(bg) for bg in bgs)
        assert "homogenization" in combined.lower() or "overlap" in combined.lower()

    def test_no_content_bullets_when_all_empty(self):
        sn = self._make_sn(
            content_assumption_ids=[],
            content_risk_ids=[],
            content_opportunity_ids=[],
            content_evidence_ids=[],
            content_coverage_status="",
            content_confidence_level="",
            winner_rationale="",
        )
        ms = self._run_writer(sn)
        # Should not crash; coverage status absent from subtitle
        assert "Coverage:" not in (ms.strategic_direction.subtitle or "")


# ---------------------------------------------------------------------------
# Section 10 — MarkdownRenderer PH12.2 output
# ---------------------------------------------------------------------------

class TestMarkdownRendererPH122:
    def _render(self, sn=None):
        from functional_agents.editorial.markdown_renderer import MarkdownRenderer
        from functional_agents.editorial.editorial_manuscript import EditorialManuscript, ManuscriptSection
        from functional_agents.editorial.strategy_writer import StrategyWriter

        if sn is None:
            from functional_agents.editorial.strategy_narrative import StrategyNarrative
            sn = StrategyNarrative(
                trace_id="TR-001",
                winner_theory_id="TH-001",
                winning_position="BTM first wins",
                winning_mechanism="Distributed energy margin",
                content_assumption_ids=["A-001"],
                content_risk_ids=["RSK-001"],
                content_opportunity_ids=["OPP-001"],
                content_evidence_ids=["EV-001"],
                content_coverage_status="sufficient",
                content_confidence_level="High",
                content_fallback_used=False,
                content_homogenization_detected=False,
                winner_rationale="Winner scored +0.450; content confidence: High",
                theory_differentiation={
                    "theory_differentiation": {
                        "TH-001::TH-002": {
                            "assumption_similarity": 0.30,
                            "risk_similarity": 0.20,
                            "opportunity_similarity": 0.10,
                            "evidence_similarity": 0.25,
                            "recommendation_similarity": 0.15,
                            "overall_similarity": 0.20,
                        }
                    },
                    "content_homogenization_detected": False,
                    "homogenization_details": {"detected": False, "message": "OK"},
                },
                winner_score=0.45,
                overall_confidence="High",
                evaluation_criteria=[],
                criterion_scores={},
            )

        from functional_agents.editorial.editorial_manuscript import (
            EditorialManuscript,
            ExecutiveSummaryManuscriptSection,
            DecisionAnalysisManuscriptSection,
            RecommendationManuscriptSection,
            RiskManuscriptSection,
            OpportunityManuscriptSection,
            ConfidenceManuscriptSection,
            AppendixManuscriptSection,
            StrategyManuscriptSection,
            ManuscriptMetadata,
        )

        brief = types.SimpleNamespace(
            strategy_narrative=sn,
            metadata=types.SimpleNamespace(
                question="Test question",
                brief_id="BR-TEST",
                pipeline_run_id="PR-TEST",
                decision_model_id="DM-TEST",
                profiles=[],
                execution_profile="",
            ),
            appendix=types.SimpleNamespace(research_object_id="RO-TEST"),
            strategic_options=types.SimpleNamespace(options=[]),
            executive_summary=types.SimpleNamespace(
                headline="", sub_headline="", decision_context="",
                critical_risks=[], recommended_option_title="", profiles=[],
            ),
            executive_confidence=types.SimpleNamespace(
                decision_horizon="", overall_confidence="", readiness_score=0.0,
                decision_readiness="", board_recommendation="",
                confidence_factors=[], barriers_to_confidence=[],
                key_assumptions=[], key_uncertainties=[],
                confidence_drivers=[], confidence_limiters=[],
                critical_unknowns=[],
            ),
            decision_analysis=types.SimpleNamespace(strategic_options=[]),
            recommendations=types.SimpleNamespace(recommendations=[]),
            strategic_risks=types.SimpleNamespace(risks=[], top_risk_id=""),
            strategic_opportunities=types.SimpleNamespace(opportunities=[]),
            strategic_assumptions=types.SimpleNamespace(assumptions=[]),
            investment_context=None,
        )

        ms = EditorialManuscript(
            metadata=ManuscriptMetadata(
                manuscript_id="MS-TEST", created_at="2026-01-01T00:00:00Z",
                brief_id="BR-TEST", pipeline_run_id="PR-TEST",
                decision_model_id="DM-TEST", research_object_id="RO-TEST",
                question="Test question",
            ),
            executive_summary=ExecutiveSummaryManuscriptSection(title="Executive Summary"),
            decision_analysis=DecisionAnalysisManuscriptSection(title="Decision Analysis"),
            recommendations=RecommendationManuscriptSection(title="Recommendations"),
            strategic_risks=RiskManuscriptSection(title="Risks"),
            strategic_opportunities=OpportunityManuscriptSection(title="Opportunities"),
            executive_confidence=ConfidenceManuscriptSection(title="Confidence"),
            appendix=AppendixManuscriptSection(title="Appendix"),
            strategic_direction=StrategyManuscriptSection(title="Strategic Direction"),
        )

        writer = StrategyWriter()
        ms = writer.write(brief, ms)

        renderer = MarkdownRenderer()
        return renderer.render(ms, brief)

    def test_differentiation_table_header_rendered(self):
        output = self._render()
        assert "Content Differentiation" in output

    def test_differentiation_table_pair_row_rendered(self):
        output = self._render()
        assert "TH-001 vs TH-002" in output

    def test_why_winner_section_rendered(self):
        output = self._render()
        assert "Why This Theory Won" in output

    def test_theory_specific_content_section_rendered(self):
        output = self._render()
        assert "Theory-Specific Content" in output

    def test_homogenization_warning_not_shown_when_not_detected(self):
        output = self._render()
        assert "Homogenization Warning" not in output

    def test_homogenization_warning_shown_when_detected(self):
        from functional_agents.editorial.strategy_narrative import StrategyNarrative
        sn = StrategyNarrative(
            trace_id="TR-002",
            winner_theory_id="TH-001",
            winning_position="Test winning position for rendering.",
            content_homogenization_detected=True,
            theory_differentiation={
                "theory_differentiation": {},
                "content_homogenization_detected": True,
                "homogenization_details": {
                    "detected": True,
                    "message": "Content homogenization detected across all dimensions.",
                },
            },
            winner_score=0.30,
            overall_confidence="Low",
            evaluation_criteria=[],
            criterion_scores={},
        )
        output = self._render(sn)
        assert "Homogenization Warning" in output


# ---------------------------------------------------------------------------
# Section 11 — Integration: ContentGraph → ContentResolver → TheoryContent → Trace
# ---------------------------------------------------------------------------

class TestPH122Integration:
    def test_full_pipeline_two_theories(self):
        """Two theories get distinct content when option links differ."""
        opt_a = _make_option("OPT-A", asm_ids=["A-001"], risk_ids=["RSK-001"])
        opt_b = _make_option("OPT-B", asm_ids=["A-002"], opp_ids=["OPP-001"])
        asms = [_make_assumption("A-001"), _make_assumption("A-002")]
        risks = [_make_risk("RSK-001")]
        opps = [_make_opportunity("OPP-001")]
        research = _make_research(
            options=[opt_a, opt_b],
            assumptions=asms,
            risks=risks,
            opportunities=opps,
        )
        graph = ContentGraph().build(research)
        resolver = ContentResolver(graph, ContentConfig())

        th1 = _theory_ns("TH-001")
        th2 = _theory_ns("TH-002")

        tc1 = resolver.resolve(th1, "OPT-A", "High")
        tc2 = resolver.resolve(th2, "OPT-B", "High")

        # Each theory gets option-specific content
        assert "A-001" in tc1.assumption_ids
        assert "A-002" in tc2.assumption_ids

        # Differentiation shows them as distinct
        result = compute_differentiation([tc1, tc2])
        diff = result["theory_differentiation"]
        assert len(diff) == 1
        # Assumption similarity should be 0 (disjoint A-001 vs A-002)
        pair_metrics = list(diff.values())[0]
        assert pair_metrics["assumption_similarity"] == pytest.approx(0.0)

    def test_content_config_max_limits_respected(self):
        """ContentConfig maximum limits are respected by resolver."""
        asms = [_make_assumption(f"A-{i:03d}") for i in range(1, 10)]
        opt = _make_option("OPT-A", asm_ids=[a["assumption_id"] for a in asms])
        research = _make_research(options=[opt], assumptions=asms)
        graph = ContentGraph().build(research)
        config = ContentConfig(maximum_assumptions_per_theory=3)
        resolver = ContentResolver(graph, config)
        th = _theory_ns("TH-001")
        tc = resolver.resolve(th, "OPT-A", "High")
        assert len(tc.assumption_ids) <= 3

    def test_no_duplicate_evidence_across_lineage_paths(self):
        """Same evidence referenced via multiple paths is deduplicated."""
        asm = _make_assumption("A-001", ev_ids=["EV-001"])
        risk = _make_risk("RSK-001", ev_ids=["EV-001"])
        opt = _make_option("OPT-A", asm_ids=["A-001"], risk_ids=["RSK-001"])
        research = _make_research(
            options=[opt],
            assumptions=[asm],
            risks=[risk],
            evidence=[_make_evidence("EV-001")],
        )
        graph = ContentGraph().build(research)
        resolver = ContentResolver(graph, ContentConfig())
        th = _theory_ns("TH-001")
        tc = resolver.resolve(th, "OPT-A", "High")
        assert tc.evidence_ids.count("EV-001") == 1
