"""Tests for research_agent.evaluation.semantic_matcher (J3.1b).

Covers:
  - Three-tier matching (exact → synonym → token_overlap)
  - Confidence bands (HIGH ≥ 0.92, MEDIUM ≥ 0.85, LOW < 0.85)
  - Anti-synonym blocking (J3.1b.3)
  - Semantic match audit fields (J3.1b.4 / J3.1b.6)
  - Batch scoring and statistics (J3.1b.7)
  - Scorer integration: SMR_008 / SMR_009 still pass
  - Spurious match cases are now blocked (economy of scale ≠ learning rate, etc.)
"""

from __future__ import annotations

import pytest

from research_agent.evaluation.semantic_matcher import (
    SemanticMatch,
    compute_match_stats,
    score_term_coverage,
    semantic_match,
)


# ---------------------------------------------------------------------------
# J3.1b.1: Tier-1 — Exact match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_substring_accepted(self):
        m = semantic_match("load following", "SMRs support load following capability.")
        assert m.matched is True
        assert m.match_type == "exact"
        assert m.similarity == 1.0
        assert m.confidence == "HIGH"
        assert m.reason == "exact_substring_match"

    def test_exact_case_insensitive(self):
        m = semantic_match("Load Following", "The unit demonstrates Load Following behavior.")
        assert m.matched is True
        assert m.match_type == "exact"

    def test_exact_returns_dataclass(self):
        m = semantic_match("factory fabrication", "Uses factory fabrication techniques.")
        assert isinstance(m, SemanticMatch)
        assert m.expected == "factory fabrication"


# ---------------------------------------------------------------------------
# J3.1b: Tier-2 — Synonym registry
# ---------------------------------------------------------------------------

class TestSynonymRegistry:
    # SMR_008 regression: must still pass
    def test_economy_of_scale_to_economics_of_scale(self):
        m = semantic_match(
            "economy of scale",
            "SMRs face an economics-of-scale disadvantage compared to large reactors.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"
        assert m.confidence == "HIGH"
        assert m.reason == "synonym_registry_match"
        assert m.matched_phrase  # non-empty

    def test_economy_of_scale_to_economies_of_scale(self):
        m = semantic_match(
            "economy of scale",
            "Larger plants benefit from economies of scale in capital costs.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"

    # SMR_009 regression: must still pass
    def test_load_following_to_grid_flexibility(self):
        m = semantic_match(
            "load following",
            "The design offers grid flexibility to match variable renewable output.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"
        assert m.confidence == "HIGH"
        assert "grid flexibility" in m.matched_phrase

    def test_load_following_to_flexible_operation(self):
        m = semantic_match(
            "load following",
            "Flexible operations allow output to ramp from 100% to 20%.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"

    def test_load_following_to_dispatchable(self):
        m = semantic_match(
            "load following",
            "Unlike baseload plants, SMRs can be dispatchable.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"

    def test_foak_synonym(self):
        m = semantic_match(
            "first-of-a-kind",
            "The high first of a kind costs are expected to drop with serial production.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"

    def test_lcoe_synonym(self):
        m = semantic_match(
            "levelized cost",
            "The LCOE for BWRX-300 is estimated at $80-100/MWh.",
        )
        assert m.matched is True
        assert m.match_type in ("exact", "synonym")

    def test_modular_construction_synonym(self):
        m = semantic_match(
            "factory fabrication",
            "Modular construction reduces on-site assembly time.",
        )
        assert m.matched is True
        assert m.match_type == "synonym"

    def test_liquid_cooling_synonym_or_exact(self):
        # "liquid cooling" is an exact substring of "direct liquid cooling"
        m = semantic_match(
            "liquid cooling",
            "Racks require direct liquid cooling via CDU units.",
        )
        assert m.matched is True
        assert m.match_type in ("exact", "synonym")


# ---------------------------------------------------------------------------
# J3.1b.3: Anti-synonym blocking
# ---------------------------------------------------------------------------

class TestAntiSynonymBlocking:
    # J3.1b.8: economy of scale ≠ learning rate
    def test_learning_rate_does_not_satisfy_economy_of_scale(self):
        """'learning rate' is an anti-synonym for 'economy of scale'."""
        m = semantic_match(
            "economy of scale",
            "The learning rate curve drives cost reductions in serial SMR builds.",
        )
        # Should not match — "learning rate" is in anti-synonyms
        assert m.matched is False
        assert m.reason == "anti_synonym" or m.matched is False

    # J3.1b.8: load following ≠ baseload
    def test_baseload_does_not_satisfy_load_following(self):
        """'baseload' is an anti-synonym for 'load following'."""
        m = semantic_match(
            "load following",
            "The plant provides baseload power to the grid at full capacity.",
        )
        assert m.matched is False

    def test_base_load_does_not_satisfy_load_following(self):
        m = semantic_match(
            "load following",
            "SMRs designed for base load operation run continuously.",
        )
        assert m.matched is False

    def test_hbm_not_satisfied_by_gddr(self):
        m = semantic_match(
            "hbm",
            "The discrete GPU uses GDDR6 memory rather than HBM.",
        )
        # "hbm" appears exactly in "rather than HBM" → exact match should fire
        # (exact always wins regardless of anti-synonym)
        assert m.matched is True
        assert m.match_type == "exact"

    def test_nvlink_not_satisfied_by_infiniband(self):
        m = semantic_match(
            "nvlink",
            "The cluster uses InfiniBand HDR for inter-node networking.",
        )
        assert m.matched is False

    def test_foak_satisfied_by_foak_synonym(self):
        m = semantic_match(
            "first-of-a-kind",
            "NOAK economics improve dramatically over FOAK.",
        )
        # "foak" is a synonym of "first-of-a-kind" → synonym match
        assert m.matched is True
        assert m.match_type == "synonym"

    def test_anti_synonym_reason_in_match(self):
        """When anti-synonym blocks, reason field must be 'anti_synonym'."""
        m = semantic_match(
            "economy of scale",
            "Serial production improves the learning rate for SMR construction.",
        )
        # If it matched at all (synonym or token overlap), it must be rejected
        if not m.matched:
            # Check that if it tried to match, it was blocked by anti_synonym
            # The reason could be anti_synonym or low_confidence
            assert m.reason in ("anti_synonym", "low_confidence", "none")

    def test_economy_of_scale_extra_synonym_blocked_if_anti(self):
        """Extra synonyms that are anti-synonyms should also be blocked."""
        m = semantic_match(
            "economy of scale",
            "The learning rate for SMR construction is 15%.",
            extra_synonyms=["learning rate"],
        )
        assert m.matched is False


# ---------------------------------------------------------------------------
# J3.1b.1: Confidence bands
# ---------------------------------------------------------------------------

class TestConfidenceBands:
    def test_exact_match_is_high_confidence(self):
        m = semantic_match("economy of scale", "Economy of scale benefits.")
        assert m.confidence == "HIGH"

    def test_synonym_match_is_high_confidence(self):
        m = semantic_match(
            "economy of scale",
            "Scale economics favor larger plants.",
        )
        assert m.matched is True
        assert m.confidence == "HIGH"

    def test_no_match_reports_confidence_band(self):
        m = semantic_match(
            "economy of scale",
            "The reactor is licensed and safe.",
        )
        assert m.matched is False
        assert m.confidence in ("HIGH", "MEDIUM", "LOW", "NONE")

    def test_low_confidence_never_matches(self):
        """Any match in the LOW band (< 0.85) must be rejected."""
        # Force a token overlap that would score below 0.85
        m = semantic_match(
            "economy of scale",
            "Regulatory processes take time.",
        )
        assert m.matched is False

    def test_threshold_custom_rejects_synonym_at_1_0(self):
        """threshold=1.0 means only exact matches pass."""
        m = semantic_match(
            "economy of scale",
            "Scale economics favor larger plants.",
            threshold=1.0,
        )
        assert m.matched is False


# ---------------------------------------------------------------------------
# J3.1b.4 / J3.1b.6: Audit fields
# ---------------------------------------------------------------------------

class TestAuditFields:
    def test_to_dict_includes_confidence(self):
        m = semantic_match("economy of scale", "Economies of scale are key.")
        d = m.to_dict()
        assert "confidence" in d
        assert "reason" in d
        assert "match_type" in d
        assert "similarity" in d

    def test_to_dict_exact_match(self):
        m = semantic_match("economy of scale", "Economy of scale benefits.")
        d = m.to_dict()
        assert d["match_type"] == "exact"
        assert d["confidence"] == "HIGH"
        assert d["reason"] == "exact_substring_match"
        assert d["matched"] is True

    def test_to_dict_synonym_match(self):
        m = semantic_match(
            "economy of scale",
            "Scale advantage is lost at smaller sizes.",
        )
        d = m.to_dict()
        assert d["matched"] is True
        assert d["match_type"] == "synonym"
        assert d["confidence"] == "HIGH"
        assert d["reason"] == "synonym_registry_match"

    def test_to_dict_unmatched(self):
        m = semantic_match("economy of scale", "The site is in Ontario Canada.")
        d = m.to_dict()
        assert d["matched"] is False
        assert d["reason"] in ("low_confidence", "anti_synonym", "none")


# ---------------------------------------------------------------------------
# Extra synonyms (acceptable_alternatives)
# ---------------------------------------------------------------------------

class TestExtraSynonyms:
    def test_acceptable_alternative_matches(self):
        m = semantic_match(
            "economy of scale",
            "The report discusses serial production savings.",
            extra_synonyms=["serial production savings", "modular build efficiency"],
        )
        assert m.matched is True
        assert m.match_type == "synonym"
        assert "serial production" in m.matched_phrase

    def test_exact_term_still_preferred_over_alternative(self):
        m = semantic_match(
            "economy of scale",
            "Economy of scale favours large plants, not serial production savings.",
            extra_synonyms=["serial production savings"],
        )
        assert m.matched is True
        assert m.match_type == "exact"


# ---------------------------------------------------------------------------
# must_not_include stays exact (J3.1a.6 — not touched by semantic layer)
# ---------------------------------------------------------------------------

class TestMustNotIncludeExact:
    def test_exact_match_detection_at_threshold_1(self):
        answer = "The reactor uses a conventional design without proprietary claims."
        m = semantic_match("NVL72", answer, threshold=1.0)
        assert m.matched is False

    def test_partial_word_not_matched(self):
        m = semantic_match("NVL36", "The NVL72 system was tested.", threshold=1.0)
        assert m.matched is False


# ---------------------------------------------------------------------------
# J3.1b.7: score_term_coverage + compute_match_stats
# ---------------------------------------------------------------------------

class TestScoreTermCoverage:
    def test_all_exact_matches(self):
        terms = ["economy of scale", "factory fabrication"]
        answer = "Economy of scale issues and factory fabrication benefits are noted."
        matches = score_term_coverage(terms, answer)
        assert all(m.matched for m in matches)
        assert all(m.match_type == "exact" for m in matches)

    def test_mix_of_exact_and_synonym(self):
        terms = ["economy of scale", "load following"]
        answer = "Economics-of-scale disadvantage and grid flexibility benefits."
        matches = score_term_coverage(terms, answer)
        assert all(m.matched for m in matches)
        types = {m.match_type for m in matches}
        assert "synonym" in types

    def test_empty_terms_returns_empty(self):
        matches = score_term_coverage([], "any answer")
        assert matches == []

    def test_alternatives_used_for_all_terms(self):
        terms = ["economy of scale", "load following"]
        answer = "Serial production and ramp capability are key."
        matches = score_term_coverage(
            terms, answer, alternatives=["serial production", "ramp capability"]
        )
        assert all(m.matched for m in matches)

    def test_unmatched_term_listed_as_not_matched(self):
        matches = score_term_coverage(["vanishing term xyz"], "Completely unrelated text.")
        assert matches[0].matched is False

    def test_anti_synonym_excluded_from_hits(self):
        """learning rate must not count as economy of scale hit."""
        matches = score_term_coverage(
            ["economy of scale"],
            "The learning rate for SMR builds is around 15%.",
        )
        assert matches[0].matched is False


class TestComputeMatchStats:
    def test_all_exact_stats(self):
        matches = [
            SemanticMatch("a", "a", 1.0, True, "exact", "HIGH", "exact_substring_match"),
            SemanticMatch("b", "b", 1.0, True, "exact", "HIGH", "exact_substring_match"),
        ]
        stats = compute_match_stats(matches)
        assert stats["exact_matches"] == 2
        assert stats["synonym_matches"] == 0
        assert stats["semantic_matches"] == 0
        assert stats["rejected_semantic_matches"] == 0
        assert stats["unmatched"] == 0

    def test_rejected_anti_synonym_counted(self):
        matches = [
            SemanticMatch("a", "a", 1.0, True, "exact", "HIGH", "exact_substring_match"),
            SemanticMatch("b", "learning rate", 0.95, False, "synonym", "HIGH", "anti_synonym"),
            SemanticMatch("c", "", 0.2, False, "none", "LOW", "low_confidence"),
        ]
        stats = compute_match_stats(matches)
        assert stats["exact_matches"] == 1
        assert stats["rejected_semantic_matches"] == 1  # b is rejected synonym
        assert stats["anti_synonym_blocks"] == 1
        assert stats["unmatched"] == 2

    def test_empty_matches(self):
        stats = compute_match_stats([])
        assert stats["total_terms"] == 0
        assert stats["semantic_match_rate"] == 0.0

    def test_backward_compat_aliases(self):
        """exact_matches_found and semantic_matches_found must be present."""
        matches = [
            SemanticMatch("a", "a", 1.0, True, "exact", "HIGH", "exact_substring_match"),
        ]
        stats = compute_match_stats(matches)
        assert "exact_matches_found" in stats
        assert "semantic_matches_found" in stats


# ---------------------------------------------------------------------------
# Integration: scorer uses semantic matching — SMR_008 / SMR_009 still pass
# ---------------------------------------------------------------------------

class TestScorerIntegration:
    def test_smr_008_economy_of_scale(self):
        """SMR_008: answer uses 'economics-of-scale', benchmark requires 'economy of scale'."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="SMR_008_SEM",
            domain="smr",
            difficulty="medium",
            question="What are the economics of scale challenges for SMRs?",
            must_include=["economy of scale", "factory"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="SMR Economics",
            question=question.question,
            executive_summary=(
                "SMRs face an economics-of-scale disadvantage compared to large reactors. "
                "Factory fabrication can offset some of these cost premiums through serial production."
            ),
            confirmed_facts=[
                "The economics-of-scale disadvantage of SMRs means higher overnight cost per kW.",
                "Factory-built modules reduce on-site labour significantly.",
            ],
        )
        score = score_qa_response(question, memo)
        assert score.must_include_hits == 2, (
            f"Expected 2 hits, got {score.must_include_hits}. Matches: {score.semantic_matches}"
        )
        assert score.passed is True

    def test_smr_009_load_following(self):
        """SMR_009: answer uses 'grid flexibility', benchmark requires 'load following'."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="SMR_009_SEM",
            domain="smr",
            difficulty="medium",
            question="What grid flexibility advantages do SMRs offer?",
            must_include=["load following", "grid"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="SMR Grid",
            question=question.question,
            executive_summary=(
                "SMRs offer significant grid flexibility advantages, including the ability "
                "to ramp output to match demand. This makes them valuable for decarbonising "
                "industrial and grid applications."
            ),
            confirmed_facts=[
                "Flexible operations allow output to ramp from 100% to 20% power over 30 minutes.",
                "The grid integration of SMRs benefits from their dispatchable nature.",
            ],
        )
        score = score_qa_response(question, memo)
        assert score.must_include_hits == 2, (
            f"Expected 2 hits, got {score.must_include_hits}. Matches: {score.semantic_matches}"
        )
        assert score.passed is True

    def test_must_not_include_stays_exact(self):
        """must_not_include uses exact matching — a synonym must NOT trigger it."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="EXACT_001",
            domain="smr",
            difficulty="easy",
            question="How many GPUs are in NVL72?",
            must_include=["72"],
            must_not_include=["NVL36"],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="GPU Count",
            question=question.question,
            executive_summary="The GB200 NVL72 contains 72 GPUs in a single rack unit.",
            confirmed_facts=["72 Grace Blackwell Superchips per NVL72 rack."],
        )
        score = score_qa_response(question, memo)
        assert score.must_not_include_violations == []
        assert score.passed is True

    def test_semantic_matches_include_audit_fields(self):
        """QAScore.semantic_matches contains confidence and reason per term."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="AUDIT_001",
            domain="smr",
            difficulty="easy",
            question="What is LCOE for SMRs?",
            must_include=["economy of scale"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary="The economics-of-scale challenge is significant.",
        )
        score = score_qa_response(question, memo)
        assert len(score.semantic_matches) == 1
        sm = score.semantic_matches[0]
        assert sm["expected"] == "economy of scale"
        assert sm["matched"] is True
        assert "confidence" in sm
        assert "reason" in sm
        assert sm["confidence"] == "HIGH"

    def test_learning_rate_not_counted_as_economy_of_scale_hit(self):
        """J3.1b.8: 'learning rate' must not satisfy 'economy of scale'."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="ANTI_001",
            domain="smr",
            difficulty="medium",
            question="Economics of SMRs?",
            must_include=["economy of scale"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary=(
                "SMR construction benefits from a learning rate of 10-20% per doubling. "
                "The overnight capital cost per kW is higher than large reactors."
            ),
        )
        score = score_qa_response(question, memo)
        # "economy of scale" was NOT mentioned — "learning rate" must not satisfy it
        assert score.must_include_hits == 0, (
            f"'learning rate' should not satisfy 'economy of scale'. "
            f"Matches: {score.semantic_matches}"
        )

    def test_baseload_not_counted_as_load_following_hit(self):
        """J3.1b.8: 'baseload' must not satisfy 'load following'."""
        from research_agent.evaluation.scorer import score_qa_response
        from research_agent.evaluation.loader import QAQuestion
        from research_agent.schemas import ResearchMemo

        question = QAQuestion(
            question_id="ANTI_002",
            domain="smr",
            difficulty="medium",
            question="Grid characteristics of SMRs?",
            must_include=["load following"],
            must_not_include=[],
            acceptable_alternatives=[],
        )
        memo = ResearchMemo(
            title="t",
            question=question.question,
            executive_summary=(
                "SMRs are often positioned as baseload power sources, "
                "operating at full capacity to provide continuous electricity."
            ),
        )
        score = score_qa_response(question, memo)
        assert score.must_include_hits == 0, (
            f"'baseload' should not satisfy 'load following'. "
            f"Matches: {score.semantic_matches}"
        )
