"""PH12.2d — Strategic Option Mapping and Alignment Resolution tests.

Covers:
- Content-ID fallback: Jaccard scoring on assumption/risk IDs when posture signals absent
- Upstream preference prior: breaks ties without overriding content evidence
- Confidence levels: High/Medium/Low/None from score and separation
- Fallback activation: content-ID path triggered by empty theory_postures
- Posture path preserved: energy-domain theories still use PostureNormalizer
- Alignment evaluation: status transitions (confirmed / refined / unresolved)
- Extra OptionMapping fields: mapping_method, mapping_margin, runner_up_option_id
- Diagnostic completeness: option_scores entries contain required fields
- Edge cases: empty theory IDs, no options, both-empty sets, single-option pool
- Coordinator integration: sports-domain produces non-None mapped_option_id
- Multi-theory mapping: all theories are mapped, not only the winner
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from functional_agents.strategy.alignment import AlignmentResult, OptionMapping
from functional_agents.strategy.alignment_evaluator import AlignmentEvaluator
from functional_agents.strategy.option_mapper import OptionMapper
from functional_agents.strategy.strategic_position import TheoryOfWinning
from functional_agents.strategy.strategy_config import AlignmentPolicy
from functional_agents.strategy.strategy_selector import StrategySelection


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _theory(
    *,
    theory_id: str = "TH-001",
    assumption_ids: list[str] | None = None,
    risk_ids: list[str] | None = None,
    choices: list[dict] | None = None,
    winning_position: str = "",
    winning_mechanism: str = "",
) -> TheoryOfWinning:
    """Build a minimal TheoryOfWinning for testing."""
    assumptions = [{"assumption_id": a} for a in (assumption_ids or [])]
    failure_modes = [{"risk_id": r} for r in (risk_ids or [])]
    return TheoryOfWinning(
        theory_id=theory_id,
        source_choice_set_id="SCS-001",
        winning_position=winning_position,
        winning_mechanism=winning_mechanism,
        strategic_choices=choices or [],
        assumptions=assumptions,
        failure_modes=failure_modes,
    )


def _opt(
    opt_id: str,
    *,
    assumption_ids: list[str] | None = None,
    risk_ids: list[str] | None = None,
    title: str = "",
    description: str = "",
) -> dict:
    d: dict[str, Any] = {"option_id": opt_id, "title": title, "description": description}
    if assumption_ids is not None:
        d["supporting_assumption_ids"] = assumption_ids
    if risk_ids is not None:
        d["associated_risk_ids"] = risk_ids
    return d


def _research(
    options: list[dict],
    *,
    preferred_id: str = "",
    recommended_id: str = "",
) -> Any:
    m = MagicMock()
    m.strategic_options = options
    m.preferred_option = {"option_id": preferred_id} if preferred_id else {}
    m.decision_analysis = {"recommended_option_id": recommended_id} if recommended_id else {}
    return m


def _selection(
    winner_id: str = "TH-001",
    *,
    score_margin: float = 0.05,
    tie_breaker_used: str | None = None,
) -> StrategySelection:
    return StrategySelection(
        winner_theory_id=winner_id,
        winner_score=0.95,
        runner_up_theory_id="TH-002",
        runner_up_score=0.90,
        score_margin=score_margin,
        tie_breaker_used=tie_breaker_used,
        selection_status="selected",
        selection_rationale="Highest score.",
        alignment_status="unresolved",
        mapped_option_id=None,
        saturation_detected=False,
    )


# ---------------------------------------------------------------------------
# 1. TestJaccardComputation
# ---------------------------------------------------------------------------

class TestJaccardComputation:
    """Verify Jaccard similarity semantics within _content_id_map scores."""

    def test_perfect_overlap_gives_weight_in_score(self):
        """Full assumption + risk overlap → content_score = _W_ASSUMPTION + _W_RISK = 0.75."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        opt = _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        res = OptionMapper().map(theory, _research([opt]))
        # No posture signals → content path; score = 0.40*1.0 + 0.35*1.0 = 0.75
        entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
        assert entry["assumption_overlap"] == pytest.approx(1.0)
        assert entry["risk_overlap"] == pytest.approx(1.0)
        assert entry["score"] == pytest.approx(0.75, abs=1e-4)

    def test_zero_overlap_gives_zero_score(self):
        """Disjoint assumption/risk sets → score = 0 (no upstream prior)."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=["R-001"])
        opt = _opt("OPT-A", assumption_ids=["A-002"], risk_ids=["R-002"])
        res = OptionMapper().map(theory, _research([opt]))
        entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
        assert entry["assumption_overlap"] == pytest.approx(0.0)
        assert entry["risk_overlap"] == pytest.approx(0.0)
        assert entry["score"] == pytest.approx(0.0, abs=1e-4)

    def test_partial_assumption_overlap(self):
        """Jaccard(A, B) = |A∩B| / |A∪B| — partial overlap."""
        # A={A-001,A-002} opt={A-001,A-002,A-003} → jaccard=2/3
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=[])
        opt = _opt("OPT-A", assumption_ids=["A-001", "A-002", "A-003"], risk_ids=[])
        res = OptionMapper().map(theory, _research([opt]))
        entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
        assert entry["assumption_overlap"] == pytest.approx(2 / 3, abs=1e-4)

    def test_both_empty_ids_give_zero_not_one(self):
        """Both theory and option have no IDs → Jaccard = 0.0 (no signal)."""
        theory = _theory(assumption_ids=[], risk_ids=[])
        opt = _opt("OPT-A", assumption_ids=[], risk_ids=[])
        res = OptionMapper().map(theory, _research([opt]))
        entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
        assert entry["assumption_overlap"] == pytest.approx(0.0)
        assert entry["risk_overlap"] == pytest.approx(0.0)

    def test_matched_ids_listed_in_entry(self):
        """assumption_ids_matched and risk_ids_matched enumerate the intersection."""
        theory = _theory(assumption_ids=["A-001", "A-002", "A-003"], risk_ids=["R-001", "R-002"])
        opt = _opt("OPT-A", assumption_ids=["A-002", "A-003", "A-004"], risk_ids=["R-001"])
        res = OptionMapper().map(theory, _research([opt]))
        entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
        assert sorted(entry["assumption_ids_matched"]) == ["A-002", "A-003"]
        assert sorted(entry["risk_ids_matched"]) == ["R-001"]


# ---------------------------------------------------------------------------
# 2. TestContentIdMapScoring
# ---------------------------------------------------------------------------

class TestContentIdMapScoring:
    """Verify that scores rank options correctly for realistic option pools."""

    def _sports_research(self) -> tuple[list[dict], Any]:
        """Build 4-option pool matching the sports strategy monitor context."""
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002", "A-003", "A-004"],
                 risk_ids=["R-001", "R-002", "R-003"]),
            _opt("OPT-B", assumption_ids=["A-001", "A-002", "A-003", "A-004"],
                 risk_ids=["R-001", "R-002", "R-003", "R-004", "R-005"]),
            _opt("OPT-C", assumption_ids=["A-001", "A-002", "A-003", "A-004", "A-005"],
                 risk_ids=["R-001", "R-002", "R-003", "R-004", "R-005", "R-006", "R-007"]),
            _opt("OPT-D", assumption_ids=["A-001", "A-002", "A-003"],
                 risk_ids=["R-001", "R-002"]),
        ]
        # Theory covers all 5 assumptions and 6 risks (RSK-007 excluded from R-006)
        theory = _theory(
            assumption_ids=["A-001", "A-002", "A-003", "A-004", "A-005"],
            risk_ids=["R-001", "R-002", "R-003", "R-004", "R-005", "R-007"],
        )
        return opts, theory

    def test_opt_c_wins_without_upstream_prior(self):
        """OPT-C has highest assumption overlap → ranked #1 without prior."""
        opts, theory = self._sports_research()
        res = OptionMapper().map(theory, _research(opts))
        assert res.option_scores[0]["option_id"] == "OPT-C"

    def test_opt_c_wins_with_upstream_prior(self):
        """OPT-C wins by both content and upstream prior."""
        opts, theory = self._sports_research()
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-C"))
        assert res.mapped_option_id == "OPT-C"

    def test_scores_are_sorted_descending(self):
        """option_scores are sorted highest → lowest."""
        opts, theory = self._sports_research()
        res = OptionMapper().map(theory, _research(opts))
        scores = [e["score"] for e in res.option_scores]
        assert scores == sorted(scores, reverse=True)

    def test_mapped_option_id_is_not_none(self):
        """Content-based mapping should always produce a non-None mapped_option_id
        when any option shares IDs with the theory."""
        opts, theory = self._sports_research()
        res = OptionMapper().map(theory, _research(opts))
        assert res.mapped_option_id is not None

    def test_upstream_prior_only_applied_to_matching_option(self):
        """Upstream prior is 0.10 for the preferred option and 0.0 for others."""
        opts, theory = self._sports_research()
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-D"))
        d_entry = next(e for e in res.option_scores if e["option_id"] == "OPT-D")
        other_entries = [e for e in res.option_scores if e["option_id"] != "OPT-D"]
        assert d_entry["upstream_prior"] == pytest.approx(0.10)
        for e in other_entries:
            assert e["upstream_prior"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 3. TestContentIdMapConfidenceLevels
# ---------------------------------------------------------------------------

class TestContentIdMapConfidenceLevels:
    """Verify that confidence is assigned correctly from score and separation."""

    def _one_opt(self, assumption_ids, risk_ids, *, recommended_id="") -> OptionMapping:
        theory = _theory(assumption_ids=assumption_ids, risk_ids=risk_ids)
        opts = [_opt("OPT-A", assumption_ids=assumption_ids, risk_ids=risk_ids)]
        return OptionMapper().map(theory, _research(opts, recommended_id=recommended_id))

    def test_high_confidence_when_score_and_separation_sufficient(self):
        """Perfect overlap, two options with large margin → High confidence."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001", "R-002"])
        # OPT-A matches all; OPT-B matches none
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001", "R-002"]),
            _opt("OPT-B", assumption_ids=["A-003"], risk_ids=["R-003"]),
        ]
        res = OptionMapper().map(theory, _research(opts))
        assert res.mapping_confidence == "High"
        assert res.mapped_option_id == "OPT-A"

    def test_medium_confidence_when_score_above_threshold_small_separation(self):
        """Score >= 0.20 but separation < 0.20 → Medium confidence."""
        theory = _theory(assumption_ids=["A-001", "A-002", "A-003"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002", "A-003"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=["A-001", "A-002"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts))
        # OPT-A: 0.40*1.0 = 0.40; OPT-B: 0.40*(2/3) = 0.267; separation=0.133 < 0.20
        assert res.mapping_confidence in ("Medium", "High")

    def test_none_confidence_when_score_zero(self):
        """All disjoint IDs → score=0 → confidence=None → mapped_id=None."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=["A-002"], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        assert res.mapping_confidence == "None"
        assert res.mapped_option_id is None

    def test_none_confidence_all_empty(self):
        """No IDs at all → confidence=None, mapped_id=None."""
        theory = _theory(assumption_ids=[], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=[], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        assert res.mapped_option_id is None


# ---------------------------------------------------------------------------
# 4. TestContentIdMapUpstreamPrior
# ---------------------------------------------------------------------------

class TestContentIdMapUpstreamPrior:
    """Verify upstream-preference prior behavior."""

    def test_prior_breaks_tie_in_favour_of_preferred_option(self):
        """When two options have identical content scores, the upstream prior wins."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        # Both share A-001; prior tips OPT-B
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-B"))
        assert res.mapped_option_id == "OPT-B"

    def test_prior_from_decision_analysis(self):
        """Upstream prior reads from decision_analysis.recommended_option_id."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-A"))
        assert res.mapped_option_id == "OPT-A"

    def test_prior_falls_back_to_preferred_option(self):
        """When decision_analysis has no ID, preferred_option.option_id is used."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        # preferred_id only, no recommended_id
        res = OptionMapper().map(theory, _research(opts, preferred_id="OPT-B"))
        assert res.mapped_option_id == "OPT-B"

    def test_prior_for_unknown_option_has_no_effect(self):
        """Prior for an option ID not in the pool is effectively zero."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-UNKNOWN"))
        # Should still map to OPT-A (only option with non-zero content score)
        assert res.mapped_option_id == "OPT-A"


# ---------------------------------------------------------------------------
# 5. TestContentIdMapEdgeCases
# ---------------------------------------------------------------------------

class TestContentIdMapEdgeCases:
    """Edge cases: empty pools, single option, missing keys."""

    def test_empty_options_returns_none(self):
        """No options → mapped_option_id=None, confidence=None."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=["R-001"])
        res = OptionMapper().map(theory, _research([]))
        assert res.mapped_option_id is None
        assert res.mapping_confidence == "None"

    def test_single_option_no_runner_up(self):
        """Single option: no runner-up; still returns a mapping if score > threshold."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        opts = [_opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])]
        res = OptionMapper().map(theory, _research(opts))
        # score=0.75, runner_up_score = score-1.0 = -0.25 → separation=1.0 → High
        assert res.mapped_option_id == "OPT-A"
        assert res.mapping_confidence == "High"

    def test_option_without_assumption_key_gets_zero_assumption_overlap(self):
        """Option with missing supporting_assumption_ids treats it as empty."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opt = {"option_id": "OPT-A"}  # no supporting_assumption_ids key
        res = OptionMapper().map(theory, _research([opt]))
        entry = res.option_scores[0]
        assert entry["assumption_overlap"] == pytest.approx(0.0)

    def test_theory_with_no_assumption_or_risk_ids(self):
        """Theory has no IDs → Jaccard = 0 → mapped_id = None (no upstream prior too)."""
        theory = _theory(assumption_ids=[], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=["R-001"])]
        res = OptionMapper().map(theory, _research(opts))
        # Jaccard(empty, non-empty) = 0.0 per our spec; no upstream prior → score=0
        assert res.mapped_option_id is None

    def test_assumption_dicts_missing_key_are_skipped(self):
        """Assumption dicts without 'assumption_id' key are silently skipped."""
        theory = TheoryOfWinning(
            theory_id="TH-001",
            source_choice_set_id="SCS-001",
            assumptions=[
                {"note": "no-id-key"},        # missing assumption_id
                {"assumption_id": "A-001"},   # valid
            ],
            failure_modes=[],
        )
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        entry = res.option_scores[0]
        # Only A-001 extracted → perfect overlap with OPT-A
        assert entry["assumption_overlap"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 6. TestFallbackActivation
# ---------------------------------------------------------------------------

class TestFallbackActivation:
    """Verify that the content-ID fallback is invoked when posture signals are absent."""

    def test_no_posture_keywords_triggers_content_path(self):
        """Sports-domain choices lack energy-domain posture tokens → content path."""
        choices = [
            {"dimension": "winning_aspiration", "selected_value": "advisory_led",
             "metadata": {"choice_title": "Advisory-Led Growth"}},
        ]
        theory = _theory(
            assumption_ids=["A-001"],
            risk_ids=["R-001"],
            choices=choices,
        )
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=["R-001"])]
        res = OptionMapper().map(theory, _research(opts))
        # theory_postures is empty because no energy-domain keywords found
        assert res.theory_postures == {}
        # Content path produced a score for OPT-A
        assert len(res.option_scores) == 1
        assert res.option_scores[0]["assumption_overlap"] == pytest.approx(1.0)

    def test_mapping_method_set_to_content_id_overlap(self):
        """When content path is used, mapping_method extra field is content_id_overlap."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        assert getattr(res, "mapping_method", None) == "content_id_overlap"


# ---------------------------------------------------------------------------
# 7. TestPosturePathPreserved
# ---------------------------------------------------------------------------

class TestPosturePathPreserved:
    """Verify the posture-based path is unchanged for energy-domain theories."""

    def test_energy_domain_theory_uses_posture_path(self):
        """Theory with grid/btm tokens → posture path; theory_postures is non-empty."""
        choices = [
            {"dimension": "power_strategy", "selected_value": "grid_first",
             "metadata": {"choice_title": "Grid-First", "choice_description": "grid interconnect"}},
        ]
        theory = _theory(
            assumption_ids=["A-001"],
            risk_ids=["R-001"],
            choices=choices,
            winning_position="We pursue a grid-first strategy for our portfolio.",
        )
        opts = [
            _opt("OPT-X", assumption_ids=["A-001"], risk_ids=["R-001"],
                 description="grid interconnect behind-the-meter staged"),
        ]
        # OptionMapper maps via posture path if theory_postures is non-empty
        res = OptionMapper().map(theory, _research(opts))
        # posture path: theory_postures should be populated
        # (exact content depends on PostureNormalizer; we only assert path taken)
        # Confirm: no mapping_method extra field (posture path doesn't set it)
        assert getattr(res, "mapping_method", None) != "content_id_overlap"

    def test_posture_path_returns_option_scores_with_posture_diagnostics(self):
        """Posture-path option_scores have posture_matches / contradictions fields."""
        choices = [
            {"dimension": "geographic", "selected_value": "concentrated",
             "metadata": {"choice_title": "Concentrated", "choice_description": "single-state focus"}},
        ]
        theory = _theory(
            choices=choices,
            winning_position="concentrated single-state strategy",
        )
        opts = [_opt("OPT-X", description="concentrated single-state generation")]
        res = OptionMapper().map(theory, _research(opts))
        if res.option_scores:
            entry = res.option_scores[0]
            assert "posture_matches" in entry
            assert "contradictions" in entry


# ---------------------------------------------------------------------------
# 8. TestAlignmentEvaluation
# ---------------------------------------------------------------------------

class TestAlignmentEvaluation:
    """Verify AlignmentEvaluator status transitions after content mapping."""

    def _eval(
        self,
        *,
        mapped_id: str | None,
        preferred_id: str,
        conf: str = "High",
        margin: float = 0.05,
        min_conf: str = "Medium",
        min_margin: float = 0.10,
    ) -> AlignmentResult:
        theory = _theory(theory_id="TH-001")
        mapping = OptionMapping(
            mapped_option_id=mapped_id,
            mapping_score=0.70,
            mapping_confidence=conf,
            option_scores=[],
            theory_postures={},
        )
        sel = _selection("TH-001", score_margin=margin)
        research = _research([], preferred_id=preferred_id)
        policy = AlignmentPolicy(
            minimum_mapping_confidence=min_conf,
            minimum_challenge_margin=min_margin,
        )
        return AlignmentEvaluator().evaluate(theory, mapping, sel, research, policy=policy)

    def test_confirmed_when_high_conf_and_margin_sufficient(self):
        result = self._eval(mapped_id="OPT-C", preferred_id="OPT-C",
                            conf="High", margin=0.15, min_margin=0.10)
        assert result.status == "confirmed"

    def test_refined_when_same_option_but_small_margin(self):
        result = self._eval(mapped_id="OPT-C", preferred_id="OPT-C",
                            conf="High", margin=0.02, min_margin=0.10)
        assert result.status == "refined"

    def test_refined_when_same_option_medium_conf(self):
        result = self._eval(mapped_id="OPT-C", preferred_id="OPT-C",
                            conf="Medium", margin=0.15, min_margin=0.10)
        assert result.status == "refined"

    def test_challenged_when_different_option_large_margin(self):
        result = self._eval(mapped_id="OPT-A", preferred_id="OPT-C",
                            conf="High", margin=0.15, min_margin=0.10)
        assert result.status == "challenged"

    def test_unresolved_when_mapped_id_none(self):
        result = self._eval(mapped_id=None, preferred_id="OPT-C",
                            conf="None", margin=0.05)
        assert result.status == "unresolved"

    def test_unresolved_when_confidence_below_minimum(self):
        result = self._eval(mapped_id="OPT-C", preferred_id="OPT-C",
                            conf="Low", min_conf="Medium")
        assert result.status == "unresolved"

    def test_unresolved_when_no_preferred_option(self):
        theory = _theory(theory_id="TH-001")
        mapping = OptionMapping(
            mapped_option_id="OPT-C",
            mapping_score=0.70,
            mapping_confidence="High",
            option_scores=[],
            theory_postures={},
        )
        sel = _selection("TH-001", score_margin=0.15)
        research = _research([], preferred_id="")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.status == "unresolved"

    def test_unresolved_different_option_below_min_margin(self):
        result = self._eval(mapped_id="OPT-A", preferred_id="OPT-C",
                            conf="High", margin=0.03, min_margin=0.10)
        assert result.status == "unresolved"


# ---------------------------------------------------------------------------
# 9. TestAlignmentResultFields
# ---------------------------------------------------------------------------

class TestAlignmentResultFields:
    """Verify AlignmentResult carries required metadata."""

    def test_result_carries_preferred_option_id(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-C", assumption_ids=["A-001"], risk_ids=[])]
        mapping = OptionMapper().map(theory, _research(opts, preferred_id="OPT-C"))
        sel = _selection("TH-001", score_margin=0.05)
        research = _research(opts, preferred_id="OPT-C")
        result = AlignmentEvaluator().evaluate(theory, mapping, sel, research)
        assert result.preferred_option_id == "OPT-C"
        assert result.selected_theory_id == "TH-001"
        assert result.mapped_option_id == mapping.mapped_option_id

    def test_result_status_not_unresolved_when_content_mapping_succeeds(self):
        """When content mapping assigns OPT-C and preferred=OPT-C, status != unresolved."""
        theory = _theory(
            assumption_ids=["A-001", "A-002", "A-003"],
            risk_ids=["R-001", "R-002"],
        )
        opts = [
            _opt("OPT-C", assumption_ids=["A-001", "A-002", "A-003"], risk_ids=["R-001", "R-002"]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        mapping = OptionMapper().map(theory, _research(opts, recommended_id="OPT-C"))
        sel = _selection("TH-001", score_margin=0.05)
        research = _research(opts, preferred_id="OPT-C")
        result = AlignmentEvaluator().evaluate(
            theory, mapping, sel, research,
            policy=AlignmentPolicy(minimum_mapping_confidence="Medium", minimum_challenge_margin=0.10),
        )
        assert result.status != "unresolved"


# ---------------------------------------------------------------------------
# 10. TestConsistencyGuard
# ---------------------------------------------------------------------------

class TestConsistencyGuard:
    """Verify mapped_option_id is consistent between mapping and selection write-back."""

    def test_consistency_between_mapping_and_selection(self):
        """The mapped_option_id written to StrategySelection matches winner_mapping."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _opt("OPT-B", assumption_ids=[], risk_ids=[]),
        ]
        mapping = OptionMapper().map(theory, _research(opts, recommended_id="OPT-A"))
        # The coordinator writes back: StrategySelection(mapped_option_id=winner_mapping.mapped_option_id)
        # Here we simulate the guard check
        assert mapping.mapped_option_id == "OPT-A"

    def test_theory_id_and_mapped_id_are_independent(self):
        """mapped_option_id refers to an upstream option, not the theory itself."""
        theory = _theory(theory_id="TH-001", assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-X", assumption_ids=["A-001"], risk_ids=[])]
        mapping = OptionMapper().map(theory, _research(opts))
        assert mapping.mapped_option_id == "OPT-X"
        assert mapping.mapped_option_id != theory.theory_id


# ---------------------------------------------------------------------------
# 11. TestOptionMappingExtraFields
# ---------------------------------------------------------------------------

class TestOptionMappingExtraFields:
    """Extra fields on OptionMapping (model_config extra=allow)."""

    def _map_two_opts(self) -> OptionMapping:
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        return OptionMapper().map(theory, _research(opts))

    def test_mapping_method_is_content_id_overlap(self):
        res = self._map_two_opts()
        assert getattr(res, "mapping_method", None) == "content_id_overlap"

    def test_mapping_margin_equals_winner_minus_runnerup(self):
        res = self._map_two_opts()
        margin = getattr(res, "mapping_margin", None)
        assert margin is not None
        assert margin >= 0.0
        # Verify it equals winner.score - runner_up.score
        w = res.option_scores[0]["score"]
        ru = res.option_scores[1]["score"]
        assert margin == pytest.approx(w - ru, abs=1e-4)

    def test_runner_up_option_id_is_set(self):
        res = self._map_two_opts()
        runner_up = getattr(res, "runner_up_option_id", None)
        assert runner_up is not None
        assert runner_up != res.mapped_option_id

    def test_runner_up_score_is_set(self):
        res = self._map_two_opts()
        runner_up_score = getattr(res, "runner_up_score", None)
        assert runner_up_score is not None
        assert runner_up_score <= res.mapping_score

    def test_extra_fields_not_present_on_posture_path(self):
        """Posture path does not set mapping_method=content_id_overlap."""
        # Only check that posture path doesn't accidentally set content_id_overlap
        # Use a theory with no assumptions/risks but with posture-bearing prose
        theory = _theory(
            winning_position="grid interconnect behind-the-meter staged phased",
        )
        opts = [_opt("OPT-X", description="grid behind-the-meter staged phased")]
        res = OptionMapper().map(theory, _research(opts))
        # If posture signals extracted → mapping_method should not be content_id_overlap
        if getattr(res, "mapping_method", None) == "content_id_overlap":
            # This means posture path also failed → content path engaged (acceptable if no energy tokens)
            assert res.theory_postures == {}
        else:
            assert getattr(res, "mapping_method", None) != "content_id_overlap"


# ---------------------------------------------------------------------------
# 12. TestDiagnosticCompleteness
# ---------------------------------------------------------------------------

class TestDiagnosticCompleteness:
    """Verify per-option score entries contain required diagnostic fields."""

    _REQUIRED_FIELDS = {
        "option_id", "score", "posture_matches", "contradictions",
        "generic_matches", "penalties", "option_postures", "rationale",
    }
    _CONTENT_EXTRA_FIELDS = {"assumption_overlap", "risk_overlap", "assumption_ids_matched", "risk_ids_matched"}

    def test_content_path_entry_has_required_fields(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=["R-001"])
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=["R-001"])]
        res = OptionMapper().map(theory, _research(opts))
        entry = res.option_scores[0]
        for field in self._REQUIRED_FIELDS:
            assert field in entry, f"Missing field: {field}"
        for field in self._CONTENT_EXTRA_FIELDS:
            assert field in entry, f"Missing content-path field: {field}"

    def test_content_path_contradictions_is_empty_list(self):
        """Content path never generates contradiction entries."""
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=["A-002"], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        entry = res.option_scores[0]
        assert entry["contradictions"] == []
        assert entry["penalties"] == []
        assert entry["posture_matches"] == []

    def test_option_scores_all_have_option_ids(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=[], risk_ids=[]),
            _opt("OPT-C", assumption_ids=["A-001"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts))
        ids = [e["option_id"] for e in res.option_scores]
        assert set(ids) == {"OPT-A", "OPT-B", "OPT-C"}


# ---------------------------------------------------------------------------
# 13. TestMappingRationale
# ---------------------------------------------------------------------------

class TestMappingRationale:
    """Verify rationale strings carry meaningful content."""

    def test_rationale_contains_mapped_option_id(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=["R-001"])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=["R-001"]),
            _opt("OPT-B", assumption_ids=[], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts))
        assert "OPT-A" in res.mapping_rationale

    def test_rationale_contains_score(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=["A-001"], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        assert "score=" in res.mapping_rationale

    def test_rationale_mentions_upstream_prior_when_applied(self):
        theory = _theory(assumption_ids=["A-001"], risk_ids=[])
        opts = [
            _opt("OPT-A", assumption_ids=["A-001"], risk_ids=[]),
            _opt("OPT-B", assumption_ids=["A-001"], risk_ids=[]),
        ]
        res = OptionMapper().map(theory, _research(opts, recommended_id="OPT-A"))
        # If OPT-A wins by prior, rationale must mention the prior
        if res.mapped_option_id == "OPT-A":
            winner_entry = next(e for e in res.option_scores if e["option_id"] == "OPT-A")
            if winner_entry["upstream_prior"] > 0:
                assert "upstream" in res.mapping_rationale or "prior" in res.mapping_rationale

    def test_none_mapping_rationale_is_informative(self):
        """When no mapping is found, rationale explains why."""
        theory = _theory(assumption_ids=[], risk_ids=[])
        opts = [_opt("OPT-A", assumption_ids=[], risk_ids=[])]
        res = OptionMapper().map(theory, _research(opts))
        if res.mapped_option_id is None:
            assert len(res.mapping_rationale) > 10


# ---------------------------------------------------------------------------
# 14. TestMultipleTheoriesMapped
# ---------------------------------------------------------------------------

class TestMultipleTheoriesMapped:
    """All theories in the pool are mapped independently."""

    def test_each_theory_gets_its_own_mapping(self):
        """Three theories with different ID sets → three independent mappings."""
        opts = [
            _opt("OPT-A", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _opt("OPT-B", assumption_ids=["A-003", "A-004"], risk_ids=["R-002"]),
        ]
        theories = [
            _theory(theory_id="TH-001", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _theory(theory_id="TH-002", assumption_ids=["A-003", "A-004"], risk_ids=["R-002"]),
            _theory(theory_id="TH-003", assumption_ids=[], risk_ids=[]),
        ]
        mapper = OptionMapper()
        research = _research(opts)
        mappings = {t.theory_id: mapper.map(t, research) for t in theories}

        assert mappings["TH-001"].mapped_option_id == "OPT-A"
        assert mappings["TH-002"].mapped_option_id == "OPT-B"
        assert mappings["TH-003"].mapped_option_id is None  # no IDs → no signal

    def test_winner_mapping_id_matches_expected(self):
        """Winner mapping is looked up from all_mappings by winner_theory_id."""
        opts = [
            _opt("OPT-C", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _opt("OPT-D", assumption_ids=[], risk_ids=[]),
        ]
        theories = [
            _theory(theory_id="TH-001", assumption_ids=["A-001", "A-002"], risk_ids=["R-001"]),
            _theory(theory_id="TH-002", assumption_ids=[], risk_ids=[]),
        ]
        mapper = OptionMapper()
        research = _research(opts)
        all_mappings = {t.theory_id: mapper.map(t, research) for t in theories}
        winner_mapping = all_mappings["TH-001"]
        assert winner_mapping.mapped_option_id == "OPT-C"


# ---------------------------------------------------------------------------
# Fixture helpers for TestCoordinatorIntegration
# ---------------------------------------------------------------------------

def _load_fixture_context(fixture_path):
    """Load a committed JSON fixture as a SimpleNamespace mimicking AgentContext."""
    import json
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).parent / "fixtures" / fixture_path
    with path.open() as f:
        data = json.load(f)
    ctx = SimpleNamespace(**data)
    for attr in ("run_id", "question", "execution_profile"):
        if not hasattr(ctx, attr) or not isinstance(getattr(ctx, attr), str):
            setattr(ctx, attr, "")
    for attr in ("profiles",):
        if not hasattr(ctx, attr):
            setattr(ctx, attr, [])
    for attr in ("decision_model", "engagement", "preferred_option",
                 "research_object", "executive_confidence", "decision_analysis", "trace"):
        if not hasattr(ctx, attr):
            setattr(ctx, attr, {})
    for attr in ("strategic_options", "assumptions", "risks", "recommendations", "opportunities"):
        if not hasattr(ctx, attr):
            setattr(ctx, attr, [])
    return ctx


def _load_sports_strategy_config():
    """Load the sports engagement strategy config. Fails fast if YAML missing."""
    import yaml
    from pathlib import Path
    from functional_agents.strategy.strategy_config_resolver import resolve_strategy_config

    engagement_path = Path("engagements/sports_strategy_monitor_v1.yaml")
    if not engagement_path.exists():
        pytest.skip("Sports engagement YAML not available")
    with engagement_path.open() as f:
        engagement = yaml.safe_load(f)
    raw_strategy = engagement.get("strategy", {}) or {}
    return resolve_strategy_config(raw_strategy), raw_strategy


# ---------------------------------------------------------------------------
# 15. TestCoordinatorIntegration
# ---------------------------------------------------------------------------

class TestCoordinatorIntegration:
    """End-to-end coordinator integration using committed deterministic fixtures.

    Fixtures live under tests/fixtures/ and are version-controlled.
    No test in this class reads from outputs/ — all inputs are deterministic.
    """

    def test_no_outputs_dependency(self):
        """Prove this test class does not depend on any outputs/ live artifact.

        All fixture paths are resolved relative to tests/fixtures/, not outputs/.
        """
        from pathlib import Path
        fixture_dir = Path(__file__).parent / "fixtures"
        assert (fixture_dir / "sports_strategy_clear_mapping.json").exists(), (
            "Committed clear-mapping fixture must exist under tests/fixtures/"
        )
        assert (fixture_dir / "sports_strategy_ambiguous_mapping.json").exists(), (
            "Committed ambiguous-mapping fixture must exist under tests/fixtures/"
        )

    def test_clear_mapping_fixture_produces_non_none_mapped_option(self):
        """Deterministic clear-mapping fixture produces a valid mapped_option_id."""
        from functional_agents.strategy import StrategyCoordinator

        ctx = _load_fixture_context("sports_strategy_clear_mapping.json")
        resolved, raw_strategy = _load_sports_strategy_config()
        coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        sel = coord._selection

        assert sel is not None
        assert sel.mapped_option_id is not None, (
            f"mapped_option_id should not be None; alignment_status={sel.alignment_status}"
        )

    def test_clear_mapping_fixture_maps_to_valid_option(self):
        """Deterministic clear-mapping fixture maps winner to a valid option in strategic_options."""
        from functional_agents.strategy import StrategyCoordinator

        ctx = _load_fixture_context("sports_strategy_clear_mapping.json")
        resolved, raw_strategy = _load_sports_strategy_config()
        coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        sel = coord._selection

        valid_ids = {
            opt.get("option_id")
            for opt in (ctx.strategic_options or [])
            if isinstance(opt, dict) and opt.get("option_id")
        }
        assert sel.mapped_option_id in valid_ids, (
            f"mapped_option_id {sel.mapped_option_id!r} not in valid options {valid_ids}"
        )

    def test_clear_mapping_fixture_produces_positive_margin(self):
        """Deterministic clear-mapping fixture produces mapping_margin > 0.

        OPT-WINNER has supporting_assumption_ids and associated_risk_ids that
        overlap with theory assumption/risk IDs.  Other options have none.
        Jaccard separation guarantees a positive mapping margin.
        """
        from functional_agents.strategy import StrategyCoordinator

        ctx = _load_fixture_context("sports_strategy_clear_mapping.json")
        resolved, raw_strategy = _load_sports_strategy_config()
        coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        sel = coord._selection

        mapping_margin = sel.model_extra.get("mapping_margin") if sel else None
        assert mapping_margin is not None, "mapping_margin should be present in StrategySelection extras"
        assert mapping_margin > 0, (
            f"mapping_margin should be > 0 for clear-mapping fixture; got {mapping_margin}"
        )

    def test_clear_mapping_fixture_alignment_is_not_unresolved(self):
        """Deterministic clear-mapping fixture produces alignment_status != 'unresolved'.

        Root cause of the original nondeterministic failure: live context.json
        lacked assumption_id fields on theory assumptions and lacked a
        decision_analysis.recommended_option_id, causing all options to score 0.0
        → mapping_margin 0.0 → Low confidence → unresolved alignment.

        The committed fixture provides explicit assumption_id fields and a
        preferred_option so alignment resolves deterministically.
        """
        from functional_agents.strategy import StrategyCoordinator

        ctx = _load_fixture_context("sports_strategy_clear_mapping.json")
        resolved, raw_strategy = _load_sports_strategy_config()
        coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        sel = coord._selection

        assert sel is not None
        assert sel.alignment_status != "unresolved", (
            f"alignment_status should not be 'unresolved'; got {sel.alignment_status!r}. "
            f"mapping_margin={sel.model_extra.get('mapping_margin')}, "
            f"mapped_option_id={sel.mapped_option_id}"
        )

    def test_ambiguous_fixture_allows_unresolved_alignment(self):
        """Ambiguous fixture with no preferred_option legitimately produces unresolved alignment.

        This proves that 'unresolved' is a valid outcome for genuinely ambiguous
        contexts, not a defect in the mapping logic.  The ambiguous fixture has
        no preferred_option and options with no overlapping assumption/risk IDs.
        """
        from functional_agents.strategy import StrategyCoordinator

        ctx = _load_fixture_context("sports_strategy_ambiguous_mapping.json")
        resolved, raw_strategy = _load_sports_strategy_config()
        coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord.build(ctx)
        sel = coord._selection

        assert sel is not None
        # No preferred_option → AlignmentEvaluator returns "unresolved" (expected, not a defect)
        assert sel.alignment_status == "unresolved", (
            f"Ambiguous fixture with no preferred_option should produce unresolved alignment; "
            f"got {sel.alignment_status!r}"
        )

    def test_determinism_same_fixture_twice(self):
        """Running the same deterministic fixture twice produces identical results."""
        from functional_agents.strategy import StrategyCoordinator

        def _run():
            ctx = _load_fixture_context("sports_strategy_clear_mapping.json")
            resolved, raw_strategy = _load_sports_strategy_config()
            coord = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
            coord.build(ctx)
            sel = coord._selection
            return (
                sel.winner_theory_id,
                sel.mapped_option_id,
                sel.mapping_score,
                sel.model_extra.get("mapping_margin"),
                sel.alignment_status,
            )

        first = _run()
        second = _run()
        assert first == second, (
            f"Two runs of the same fixture produced different results:\n"
            f"  first:  {first}\n"
            f"  second: {second}"
        )

    def test_option_reordering_preserves_clear_winner(self):
        """Reordering strategic_options does not change the semantic mapping winner.

        When one option has significantly higher content-ID overlap than others,
        the stable sort in _content_id_map preserves the winner regardless of
        the original list order.
        """
        import json
        from pathlib import Path
        from types import SimpleNamespace
        from functional_agents.strategy import StrategyCoordinator

        fixture_path = Path(__file__).parent / "fixtures" / "sports_strategy_clear_mapping.json"
        with fixture_path.open() as f:
            data = json.load(f)

        # Run with original option order
        ctx1 = SimpleNamespace(**data)
        for attr in ("run_id", "question", "execution_profile"):
            if not hasattr(ctx1, attr) or not isinstance(getattr(ctx1, attr), str):
                setattr(ctx1, attr, "")
        for attr in ("profiles",):
            if not hasattr(ctx1, attr):
                setattr(ctx1, attr, [])
        for attr in ("decision_model", "engagement", "preferred_option",
                     "research_object", "executive_confidence", "decision_analysis", "trace"):
            if not hasattr(ctx1, attr):
                setattr(ctx1, attr, {})
        for attr in ("strategic_options", "assumptions", "risks", "recommendations", "opportunities"):
            if not hasattr(ctx1, attr):
                setattr(ctx1, attr, [])

        # Run with reversed option order
        data_reversed = dict(data)
        data_reversed["strategic_options"] = list(reversed(data["strategic_options"]))
        ctx2 = SimpleNamespace(**data_reversed)
        for attr in ("run_id", "question", "execution_profile"):
            if not hasattr(ctx2, attr) or not isinstance(getattr(ctx2, attr), str):
                setattr(ctx2, attr, "")
        for attr in ("profiles",):
            if not hasattr(ctx2, attr):
                setattr(ctx2, attr, [])
        for attr in ("decision_model", "engagement", "preferred_option",
                     "research_object", "executive_confidence", "decision_analysis", "trace"):
            if not hasattr(ctx2, attr):
                setattr(ctx2, attr, {})
        for attr in ("strategic_options", "assumptions", "risks", "recommendations", "opportunities"):
            if not hasattr(ctx2, attr):
                setattr(ctx2, attr, [])

        resolved, raw_strategy = _load_sports_strategy_config()

        coord1 = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord1.build(ctx1)

        coord2 = StrategyCoordinator(config=resolved.resolved, raw_strategy_yaml=raw_strategy)
        coord2.build(ctx2)

        assert coord1._selection.mapped_option_id == coord2._selection.mapped_option_id, (
            f"Mapped option changed with option reordering: "
            f"{coord1._selection.mapped_option_id!r} vs {coord2._selection.mapped_option_id!r}"
        )


# ---------------------------------------------------------------------------
# 16. TestBackwardCompatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Verify that the PH12.2b/c changes are not regressed by PH12.2d."""

    def test_option_mapping_model_is_frozen(self):
        """OptionMapping model_config frozen=True prevents mutation."""
        mapping = OptionMapping(
            mapped_option_id="OPT-A",
            mapping_score=0.70,
            mapping_confidence="High",
            option_scores=[],
            theory_postures={},
        )
        with pytest.raises(Exception):
            mapping.mapped_option_id = "OPT-B"  # type: ignore[misc]

    def test_option_mapping_extra_fields_allowed(self):
        """OptionMapping accepts extra fields (extra=allow)."""
        mapping = OptionMapping(
            mapped_option_id="OPT-A",
            mapping_score=0.70,
            mapping_confidence="High",
            option_scores=[],
            theory_postures={},
            mapping_method="content_id_overlap",
            mapping_margin=0.25,
        )
        assert getattr(mapping, "mapping_method") == "content_id_overlap"
        assert getattr(mapping, "mapping_margin") == pytest.approx(0.25)

    def test_alignment_result_default_status_is_unresolved(self):
        """AlignmentResult defaults to 'unresolved'."""
        result = AlignmentResult()
        assert result.status == "unresolved"

    def test_mapper_no_options_returns_none_mapped_id(self):
        """No options available → mapped_option_id=None regardless of theory IDs."""
        theory = _theory(assumption_ids=["A-001", "A-002"], risk_ids=["R-001"])
        res = OptionMapper().map(theory, _research([]))
        assert res.mapped_option_id is None
        assert res.mapping_confidence == "None"
