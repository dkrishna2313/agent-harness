"""Regression suite for evidence source-locality enforcement (J1.5).

This file tests two complementary concerns:

1. **Extraction sanitizer** (``evidence_filter.py``)
   Evidence items whose ``claim`` field contains cross-document comparison
   language must be rejected before they reach the contradiction engine.

2. **Contradiction engine outcome** (end-to-end with clean evidence)
   When extraction is correct (source-local claims), the contradiction engine
   must produce the right result for each corpus:

   Case                             | Expected outcome
   ---------------------------------|------------------------------
   Same-metric duration conflict    | contradiction detected
   300 GW target vs 13 GW/year rate | NO contradiction (J1.4)
   OECD vs Russia/China HALEU       | NO contradiction
   FOAK history vs modular future   | NO contradiction
"""

from __future__ import annotations

import pytest

from research_agent.contradiction import detect_contradictions
from research_agent.evidence_filter import is_source_local, sanitize_evidence_items
from research_agent.schemas import EvidenceItem, assign_evidence_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ev(
    claim: str,
    source: str = "doc.pdf",
    category: str = "other",
    snippet: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source_document=source,
        evidence_snippet=snippet if snippet is not None else claim,
        category=category,
        relevance="Relevant to the research question.",
        confidence="high",
        relevance_score=4,
        source_quality_score=4,
        specificity_score=4,
        overall_score=4.0,
    )


# ---------------------------------------------------------------------------
# 1. is_source_local — unit tests
# ---------------------------------------------------------------------------


class TestIsSourceLocal:

    # ---- word forms that must be rejected -----------------------------------

    def test_contradicts_rejected(self):
        assert not is_source_local("This contradicts claims from other documents.")

    def test_contradict_bare_rejected(self):
        assert not is_source_local("These figures contradict the earlier estimate.")

    def test_contradicting_rejected(self):
        assert not is_source_local("The two values are contradicting each other.")

    def test_contradicted_rejected(self):
        assert not is_source_local("This was contradicted by newer data.")

    def test_contradiction_noun_rejected(self):
        assert not is_source_local("There is a contradiction between the two sources.")

    def test_contradictions_plural_rejected(self):
        assert not is_source_local("These contradictions make comparison difficult.")

    def test_contradictory_rejected(self):
        assert not is_source_local("The figures are contradictory.")

    def test_inconsistent_rejected(self):
        assert not is_source_local("These values are inconsistent with each other.")

    def test_inconsistent_standalone_rejected(self):
        assert not is_source_local("The data is inconsistent.")

    def test_conflicting_adjective_rejected(self):
        assert not is_source_local("These conflicting estimates complicate the analysis.")

    def test_conflicts_with_rejected(self):
        assert not is_source_local("This conflicts with the NRC estimate.")

    def test_in_contrast_to_rejected(self):
        assert not is_source_local("In contrast to the vendor claim, the regulator says 5 GW/year.")

    def test_unlike_other_rejected(self):
        assert not is_source_local("Unlike other sources, this document claims 300 MW.")

    def test_unlike_previous_rejected(self):
        assert not is_source_local("Unlike the previous report, this study shows lower costs.")

    def test_disagrees_with_rejected(self):
        assert not is_source_local("This disagrees with the NRC assessment.")

    def test_in_disagreement_with_rejected(self):
        assert not is_source_local("The figure is in disagreement with earlier studies.")

    def test_case_insensitive_rejection(self):
        assert not is_source_local("INCONSISTENT findings were noted.")
        assert not is_source_local("This CONTRADICTS the estimate.")

    # ---- clean claims that must pass ----------------------------------------

    def test_clean_capacity_target_passes(self):
        assert is_source_local(
            "Scaling US nuclear capacity to 300 GW by 2050 requires sustained investment."
        )

    def test_clean_rate_claim_passes(self):
        assert is_source_local(
            "The NRC must license 13 GW per year to reach deployment targets."
        )

    def test_clean_haleu_oecd_passes(self):
        assert is_source_local(
            "HALEU fuel is not commercially available from OECD member country suppliers."
        )

    def test_clean_haleu_russia_passes(self):
        assert is_source_local(
            "HALEU fuel can be obtained from Russia (Rosatom) and potentially China."
        )

    def test_clean_foak_delay_passes(self):
        assert is_source_local(
            "First-of-a-kind nuclear plants historically experienced 50–200% cost overruns."
        )

    def test_clean_modular_benefit_passes(self):
        assert is_source_local(
            "Modular factory construction is projected to reduce build time by 40%."
        )

    def test_clean_construction_months_passes(self):
        assert is_source_local(
            "The BWRX-300 is designed for construction in 24–36 months using modular techniques."
        )

    def test_word_conflict_in_unrelated_context_passes(self):
        """'conflict' as a standalone noun (not 'conflicts with') should pass."""
        assert is_source_local(
            "The Russia-Ukraine conflict disrupted fuel supply chains in 2022."
        )


# ---------------------------------------------------------------------------
# 2. sanitize_evidence_items — unit tests
# ---------------------------------------------------------------------------


class TestSanitizeEvidenceItems:

    def test_removes_contradicts_item(self):
        items = [
            _ev("HALEU is not available from OECD suppliers.", "doc_a.pdf"),
            _ev("This contradicts claims of global HALEU availability.", "doc_b.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 1
        assert clean[0].claim == "HALEU is not available from OECD suppliers."

    def test_removes_inconsistent_item(self):
        items = [
            _ev("The 300 GW target requires 13 GW/year of licensing."),
            _ev("These figures are inconsistent.", "doc_b.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 1

    def test_removes_contradicting_item(self):
        items = [
            _ev("HALEU availability is limited in OECD countries."),
            _ev("Russia and China offer HALEU, contradicting OECD scarcity claims.", "doc_b.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 1

    def test_removes_contradict_item(self):
        items = [
            _ev("FOAK projects historically take longer and cost more."),
            _ev("Future modular benefits contradict the FOAK cost-overrun record.", "doc_b.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 1

    def test_all_clean_items_pass_through(self):
        items = [
            _ev("300 GW nuclear capacity target by 2050.", "doc_a.pdf"),
            _ev("NRC licensing throughput: 13 GW per year.", "doc_b.pdf"),
            _ev("HALEU not available from OECD suppliers.", "doc_c.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 3

    def test_empty_input_returns_empty(self):
        assert sanitize_evidence_items([]) == []

    def test_stage_label_in_log(self, caplog):
        import logging
        items = [_ev("This contradicts everything.", "doc.pdf")]
        with caplog.at_level(logging.WARNING, logger="research_agent.evidence_filter"):
            sanitize_evidence_items(items, stage="test_stage")
        assert "test_stage" in caplog.text

    def test_rejected_item_logged(self, caplog):
        import logging
        bad_claim = "This contradicts the earlier estimate."
        items = [_ev(bad_claim, "bad_doc.pdf")]
        with caplog.at_level(logging.WARNING, logger="research_agent.evidence_filter"):
            sanitize_evidence_items(items)
        assert "bad_doc.pdf" in caplog.text


# ---------------------------------------------------------------------------
# 3. Contradiction engine regression tests — source-local claims
# ---------------------------------------------------------------------------
#
# These tests verify the EXPECTED OUTCOMES from the requirements:
#
#   Case                               | Expected
#   -----------------------------------|------------------
#   Same-metric deployment year (2026  | contradiction
#   vs 2027 commercial operation)      | detected
#   300 GW target vs 13 GW/year rate   | NO contradiction
#   OECD HALEU vs Russia/China HALEU   | NO contradiction
#   FOAK history vs modular future     | NO contradiction


class TestSourceLocalCorpusOutcomes:
    """End-to-end regression: properly extracted source-local claims produce
    correct contradiction outcomes."""

    # ---- Positive control: same-metric conflict must fire -------------------

    def test_same_metric_deployment_year_conflict_detected(self):
        """
        POSITIVE CONTROL — same-metric duration conflict: DETECTED.

        Two sources disagree on the commercial-operation year for the same
        project.  Both are ``year_deployment`` context → compatible metric
        types → contradiction engine fires.
        """
        items = assign_evidence_ids([
            _ev(
                "The NuScale VOYGR plant targets commercial operation in 2026.",
                "nuscale_schedule.pdf",
            ),
            _ev(
                "The NRC approval timeline puts commercial operation in 2027.",
                "nrc_review_schedule.pdf",
            ),
        ])
        result = detect_contradictions(items)
        timeline = [c for c in result if c.topic == "timeline"]
        assert timeline, (
            "Expected a timeline contradiction for 2026 vs 2027 deployment years. "
            f"Got: {result}"
        )

    def test_same_metric_gw_rate_conflict_detected(self):
        """
        POSITIVE CONTROL — same-metric rate conflict: DETECTED.

        Two sources disagree on the NRC's licensing throughput rate (both
        are ``gw_rate``) → compatible → contradiction engine fires.
        """
        items = assign_evidence_ids([
            _ev(
                "Current NRC capacity is 5 GW per year under existing review resources.",
                "nrc_capacity_report.pdf",
            ),
            _ev(
                "NRC throughput would need to reach 13 GW per year to meet targets.",
                "doe_liftoff_report.pdf",
            ),
        ])
        result = detect_contradictions(items)
        assert any(c.topic == "rack power" for c in result), (
            "Expected a GW rate contradiction (5 vs 13 GW/year, same metric). "
            f"Got: {result}"
        )

    # ---- Negative controls: cross-metric / cross-context pairs must NOT fire -

    def test_300gw_target_vs_13gw_per_year_no_contradiction(self):
        """
        NEGATIVE CONTROL — 300 GW capacity target vs 13 GW/year throughput rate.

        Source A: total deployment goal (gw_target).
        Source B: annual licensing rate   (gw_rate).
        Different metric types → J1.4 gate suppresses the comparison.
        """
        items = assign_evidence_ids([
            _ev(
                "Scaling US nuclear capacity to 300 GW by 2050 is the stated policy goal.",
                "doe_policy.pdf",
            ),
            _ev(
                "Reaching the target requires the NRC to license 13 GW per year.",
                "doe_liftoff_report.pdf",
            ),
        ])
        result = detect_contradictions(items)
        gw_conflicts = [c for c in result if c.topic == "rack power"]
        assert not gw_conflicts, (
            "300 GW capacity target vs 13 GW/year throughput must NOT produce a "
            f"GW contradiction. Got: {gw_conflicts}"
        )

    def test_haleu_oecd_vs_russia_china_no_contradiction(self):
        """
        NEGATIVE CONTROL — HALEU availability: OECD suppliers vs Russia/China.

        Source A: HALEU not commercially available from OECD suppliers.
        Source B: HALEU obtainable from Russia (Rosatom) and potentially China.

        These are complementary facts about DIFFERENT suppliers — not
        contradictory.  No shared numeric units, no categorical exclusive pairs.
        """
        items = assign_evidence_ids([
            _ev(
                "HALEU fuel is not commercially available from OECD member country suppliers.",
                "iaea_fuel_report.pdf",
            ),
            _ev(
                "HALEU fuel can be obtained from Russia (Rosatom) and potentially China.",
                "world_nuclear_assoc.pdf",
            ),
        ])
        result = detect_contradictions(items)
        assert not result, (
            "OECD HALEU unavailability and Russia/China availability are complementary "
            f"facts, not contradictions. Got: {result}"
        )

    def test_foak_history_vs_modular_future_no_contradiction(self):
        """
        NEGATIVE CONTROL — FOAK historical delays vs future modular benefits.

        Source A: historical FOAK cost overruns (50–200%).
        Source B: projected future reduction in construction time (40%).

        Different analytical frames (historical vs prospective) with no shared
        numeric units — not a contradiction.
        """
        items = assign_evidence_ids([
            _ev(
                "First-of-a-kind nuclear plants historically experienced 50–200% cost overruns "
                "and multi-year schedule delays.",
                "nea_cost_study.pdf",
            ),
            _ev(
                "Modular factory construction is projected to reduce build time by 40% "
                "compared to conventional site-built approaches.",
                "inl_modular_report.pdf",
            ),
        ])
        result = detect_contradictions(items)
        assert not result, (
            "FOAK historical delays and future modular benefits are not contradictory. "
            f"Got: {result}"
        )



# ---------------------------------------------------------------------------
# 4. Synthetic corpus — extracted claims with injected comparison language
#    must be rejected BEFORE reaching the contradiction engine
# ---------------------------------------------------------------------------


class TestSyntheticCorpusRejection:
    """Verify that bad claims from the synthetic regression corpus are
    rejected by the sanitizer, so the contradiction engine never sees them."""

    def test_300gw_inconsistent_claim_rejected(self):
        """The claim '300 GW and 13 GW/year are inconsistent' is rejected."""
        bad = _ev(
            "The 300 GW target and 13 GW/year licensing rate are inconsistent.",
            "synth_doc.pdf",
        )
        assert not is_source_local(bad.claim)
        clean = sanitize_evidence_items([bad])
        assert clean == []

    def test_haleu_contradicting_claim_rejected(self):
        """The claim 'HALEU availability from Russia contradicting OECD claims' is rejected."""
        bad = _ev(
            "HALEU availability from Russia and China, contradicting OECD supplier constraints.",
            "synth_doc.pdf",
        )
        assert not is_source_local(bad.claim)
        clean = sanitize_evidence_items([bad])
        assert clean == []

    def test_foak_contradict_claim_rejected(self):
        """The claim 'modular benefits contradict FOAK delays' is rejected."""
        bad = _ev(
            "Future modular construction benefits contradict the FOAK delay record.",
            "synth_doc.pdf",
        )
        assert not is_source_local(bad.claim)
        clean = sanitize_evidence_items([bad])
        assert clean == []

    def test_valid_source_local_claim_survives(self):
        """Valid source-local claims must survive the sanitizer in the same corpus."""
        good_claims = [
            "300 GW by 2050 is the US nuclear policy target.",
            "The NRC must license 13 GW per year to reach the target.",
            "HALEU fuel is not commercially available from OECD suppliers.",
            "Rosatom (Russia) is the primary current supplier of HALEU.",
            "FOAK plants historically experienced 50–200% cost overruns.",
            "Modular construction is projected to reduce build time by 40%.",
        ]
        items = [_ev(c) for c in good_claims]
        clean = sanitize_evidence_items(items)
        assert len(clean) == len(items), (
            f"Expected all {len(items)} clean claims to survive. "
            f"Got {len(clean)}: {[i.claim for i in clean]}"
        )

    def test_mixed_corpus_only_bad_rejected(self):
        """Mix of valid and invalid claims: only the bad ones are removed."""
        items = [
            _ev("300 GW nuclear target by 2050.", "doc_a.pdf"),
            _ev("The 300 GW and 13 GW/year figures are inconsistent.", "synth.pdf"),
            _ev("NRC licenses 13 GW per year under the proposal.", "doc_b.pdf"),
            _ev("This contradicts the OECD scarcity narrative.", "synth.pdf"),
            _ev("HALEU not available from OECD suppliers.", "doc_c.pdf"),
        ]
        clean = sanitize_evidence_items(items)
        assert len(clean) == 3
        sources = {i.source_document for i in clean}
        assert "synth.pdf" not in sources


# ---------------------------------------------------------------------------
# 5. Synthetic contradiction regression suite (J1.6)
# ---------------------------------------------------------------------------
#
# Four canonical test scenarios specified in the requirements:
#
#   Scenario                                   | Expected contradictions
#   -------------------------------------------|------------------------
#   Reactor Alpha: 24–36 months vs 8–12 years  | 1 (duration conflict)
#   HALEU scope: OECD vs Russia/China           | 0
#   Capacity target: 300 GW vs 13 GW/year      | 0
#   FOAK history vs modular future benefits     | 0


class TestSyntheticContradictionRegression:
    """End-to-end regression: the duration contradiction engine fires exactly
    where specified and stays silent everywhere else."""

    # ---- Positive control ---------------------------------------------------

    def test_reactor_alpha_duration_conflict_detected(self):
        """
        POSITIVE CONTROL — same entity, same metric, non-overlapping ranges.

        Source A: 24–36 months  →  [24, 36] months
        Source B: 8–12 years    →  [96, 144] months
        Ranges do not overlap   →  contradiction expected.
        """
        items = assign_evidence_ids([
            _ev(
                "Reactor Alpha construction duration is 24-36 months.",
                "vendor_spec_alpha.pdf",
            ),
            _ev(
                "Reactor Alpha construction duration is 8-12 years.",
                "independent_review.pdf",
            ),
        ])
        result = detect_contradictions(items)
        duration_conflicts = [
            c for c in result if c.topic == "construction duration"
        ]
        assert len(duration_conflicts) == 1, (
            "Expected exactly 1 construction-duration contradiction for "
            f"Reactor Alpha (24–36 months vs 8–12 years). Got: {result}"
        )
        c = duration_conflicts[0]
        assert c.metric_type_a == "duration_construction"
        assert c.metric_type_b == "duration_construction"
        assert c.entity_a != ""
        assert c.comparison_reason != ""

    # ---- Negative controls --------------------------------------------------

    def test_haleu_scope_no_contradiction(self):
        """
        NEGATIVE CONTROL — HALEU availability differs by geographic scope.

        Source A: OECD suppliers do not offer HALEU commercially.
        Source B: HALEU is obtainable from Russia/China.

        No duration values in either claim → duration checker returns None.
        No shared numeric units       → numeric checker returns None.
        No exclusive-pair matches     → categorical checker returns None.
        Expected: 0 contradictions.
        """
        items = assign_evidence_ids([
            _ev(
                "HALEU fuel is not commercially available from OECD member country suppliers.",
                "iaea_fuel_supply.pdf",
            ),
            _ev(
                "HALEU fuel can be obtained from Russia (Rosatom) and potentially China.",
                "world_nuclear_assoc.pdf",
            ),
        ])
        result = detect_contradictions(items)
        assert not result, (
            "OECD HALEU unavailability and Russia/China availability are "
            f"complementary facts — expected 0 contradictions. Got: {result}"
        )

    def test_capacity_target_vs_rate_no_contradiction(self):
        """
        NEGATIVE CONTROL — 300 GW capacity target vs 13 GW/year licensing rate.

        Source A: total deployment goal (gw_target).
        Source B: annual licensing throughput (gw_rate).
        Different metric types → J1.4 gate suppresses the comparison.
        No duration values     → J1.6 duration checker also returns None.
        Expected: 0 contradictions.
        """
        items = assign_evidence_ids([
            _ev(
                "Scaling US nuclear capacity to 300 GW by 2050 is the stated policy goal.",
                "doe_policy.pdf",
            ),
            _ev(
                "Reaching the target requires the NRC to license 13 GW per year.",
                "doe_liftoff_report.pdf",
            ),
        ])
        result = detect_contradictions(items)
        gw_conflicts = [c for c in result if c.topic == "rack power"]
        assert not gw_conflicts, (
            "300 GW capacity target vs 13 GW/year throughput must NOT produce a "
            f"GW contradiction. Got: {gw_conflicts}"
        )

    def test_foak_history_vs_modular_future_no_contradiction(self):
        """
        NEGATIVE CONTROL — FOAK historical delays vs modular future benefits.

        Source A: historical cost overruns (50–200%) and "multi-year" delays.
        Source B: projected 40% build-time reduction.

        '50–200%' and '40%': the '%' word-boundary rule means the numeric
        checker cannot extract a value for '%', so no numeric conflict fires.
        'multi-year' contains no leading digit → duration regex does not match.
        '40%' also has no months/years unit → duration checker silent.
        Expected: 0 contradictions.
        """
        items = assign_evidence_ids([
            _ev(
                "First-of-a-kind nuclear plants historically experienced 50–200% cost overruns "
                "and multi-year schedule delays.",
                "nea_cost_study.pdf",
            ),
            _ev(
                "Modular factory construction is projected to reduce build time by 40% "
                "compared to conventional site-built approaches.",
                "inl_modular_report.pdf",
            ),
        ])
        result = detect_contradictions(items)
        assert not result, (
            "FOAK historical delays and future modular benefits are not "
            f"contradictory — expected 0 contradictions. Got: {result}"
        )
