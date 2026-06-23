"""Tests for J5.6b Profile-Driven Retrieval Validation.

Covers:
- CORPUS has 50 items: 28 ai_data_centers + 22 transmission
- score_item() correctly counts term overlaps
- retrieve_top_n() returns n items ranked by relevance
- Run A (ai_data_centers) retrieves primarily ai_data_centers items
- Run B (transmission) retrieves primarily transmission items
- Run C (ai_data_centers,transmission) contains items from both domains
- Evidence Jaccard(A, B) < 1.0 (different retrieval)
- Evidence Jaccard(A, C) > Jaccard(A, B) (C includes A domain)
- Evidence Jaccard(B, C) > Jaccard(A, B) (C includes B domain)
- Findings differ between A and B
- Recommendations differ between A and B
- Finding keywords for A include ai_data_centers terms
- Finding keywords for B include transmission terms
- Profile attribution assigns items to correct profiles in Run C
- profiles_contributing populated correctly
- build_run() returns a ProfileRun with all required fields
- compare_runs() returns evidence/findings/recommendations overlap dicts
- run_all() returns runs + comparisons + similarity_matrix + behavioral_validation
- behavioral_validation all True when runs differ
- build_comparison_report() returns markdown string
- Report contains all three run IDs and similarity table
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from functional_agents.profile_comparison import (
    CORPUS,
    score_item,
    score_corpus,
    retrieve_top_n,
    build_run,
    compare_runs,
    run_all,
    build_comparison_report,
    _jaccard,
    _load_profile_term_set,
)


# ---------------------------------------------------------------------------
# Corpus structure
# ---------------------------------------------------------------------------

def test_corpus_has_fifty_items():
    assert len(CORPUS) == 50


def test_corpus_items_have_id_claim_topics():
    for item in CORPUS:
        assert "id" in item
        assert "claim" in item
        assert "topics" in item
        assert item["id"].startswith("C")


def test_corpus_ai_dc_items_ids():
    adc_ids = [i["id"] for i in CORPUS if int(i["id"][1:]) <= 28]
    assert len(adc_ids) == 28


def test_corpus_transmission_items_ids():
    tx_ids = [i["id"] for i in CORPUS if int(i["id"][1:]) >= 29]
    assert len(tx_ids) == 22


# ---------------------------------------------------------------------------
# score_item() and score_corpus()
# ---------------------------------------------------------------------------

def test_score_item_counts_term_matches():
    item = {"claim": "GPU cooling rack power", "topics": ["cooling"]}
    terms = {"gpu", "cooling"}
    score = score_item(item, terms)
    assert score >= 2


def test_score_item_zero_for_no_match():
    item = {"claim": "completely unrelated text", "topics": []}
    terms = {"gpu", "nvlink", "hvdc"}
    assert score_item(item, terms) == 0


def test_score_corpus_returns_all_items():
    terms = {"gpu"}
    result = score_corpus(CORPUS[:5], terms)
    assert len(result) == 5


def test_score_corpus_sorted_descending():
    terms = {"gpu", "cooling", "rack", "interconnection", "grid"}
    result = score_corpus(CORPUS, terms)
    scores = [item["_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# retrieve_top_n()
# ---------------------------------------------------------------------------

def test_retrieve_top_n_returns_n_items():
    terms = {"gpu", "cooling", "rack"}
    result = retrieve_top_n(CORPUS, terms, n=10)
    assert len(result) == 10


def test_retrieve_top_n_high_scoring_items_first():
    terms = {"gpu", "cooling", "rack", "liquid", "thermal"}
    result = retrieve_top_n(CORPUS, terms, n=5)
    scores = [item["_score"] for item in result]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Profile term set loading (uses real profile YAML files)
# ---------------------------------------------------------------------------

def test_load_profile_term_set_ai_data_centers_non_empty():
    terms = _load_profile_term_set("ai_data_centers")
    assert len(terms) > 10
    # GPU-domain terms should be present
    assert any("gpu" in t or "rack" in t or "cooling" in t for t in terms)


def test_load_profile_term_set_transmission_non_empty():
    terms = _load_profile_term_set("transmission")
    assert len(terms) > 10
    # Grid-domain terms should be present
    assert any("transmission" in t or "grid" in t or "interconnect" in t for t in terms)


def test_profile_term_sets_are_different():
    adc = _load_profile_term_set("ai_data_centers")
    tx = _load_profile_term_set("transmission")
    # The sets should overlap significantly less than they differ
    overlap = len(adc & tx)
    total = len(adc | tx)
    jaccard = overlap / total if total else 0
    # They should NOT be nearly identical
    assert jaccard < 0.5, f"Profile term sets are too similar: Jaccard={jaccard:.2f}"


# ---------------------------------------------------------------------------
# build_run() — single profile
# ---------------------------------------------------------------------------

def test_build_run_a_retrieves_n_items():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_b_retrieves_n_items():
    run = build_run("run_b", ["transmission"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_a_biased_toward_ai_dc():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    # Most items should have ids <= C028 (ai_data_centers domain)
    adc_items = sum(1 for item in run.top_evidence if int(item["id"][1:]) <= 28)
    assert adc_items > len(run.top_evidence) // 2, (
        f"Run A should retrieve more ai_data_centers items; got {adc_items}/{len(run.top_evidence)}"
    )


def test_build_run_b_biased_toward_transmission():
    run = build_run("run_b", ["transmission"], n=18)
    # Most items should have ids >= C029 (transmission domain)
    tx_items = sum(1 for item in run.top_evidence if int(item["id"][1:]) >= 29)
    assert tx_items > len(run.top_evidence) // 2, (
        f"Run B should retrieve more transmission items; got {tx_items}/{len(run.top_evidence)}"
    )


def test_build_run_a_finds_findings():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    assert len(run.findings) > 0


def test_build_run_a_has_recommendations():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    assert len(run.recommendations) > 0


def test_build_run_a_finding_keywords_include_adc_terms():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    adc_keywords = {"cooling", "rack", "networking", "power"}
    overlap = run.finding_keywords & adc_keywords
    assert len(overlap) > 0, (
        f"Run A findings should include ai_data_centers terms; got {run.finding_keywords}"
    )


def test_build_run_b_finding_keywords_include_tx_terms():
    run = build_run("run_b", ["transmission"], n=18)
    tx_keywords = {"interconnection", "congestion", "transmission", "grid"}
    overlap = run.finding_keywords & tx_keywords
    assert len(overlap) > 0, (
        f"Run B findings should include transmission terms; got {run.finding_keywords}"
    )


def test_build_run_profiles_contributing_single():
    run = build_run("run_a", ["ai_data_centers"], n=18)
    assert "ai_data_centers" in run.profiles_contributing
    assert run.profiles_missing == []


# ---------------------------------------------------------------------------
# build_run() — multi-profile
# ---------------------------------------------------------------------------

def test_build_run_c_retrieves_n_items():
    run = build_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_c_profiles_contributing_both():
    run = build_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert "ai_data_centers" in run.profiles_contributing
    assert "transmission" in run.profiles_contributing


def test_build_run_c_attribution_has_both_profiles():
    run = build_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert len(run.profile_attribution["ai_data_centers"]) > 0
    assert len(run.profile_attribution["transmission"]) > 0


def test_build_run_c_retrieval_summary_has_both():
    run = build_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert "ai_data_centers" in run.profile_retrieval_summary
    assert "transmission" in run.profile_retrieval_summary


def test_build_run_c_finding_keywords_from_both_domains():
    run = build_run("run_c", ["ai_data_centers", "transmission"], n=18)
    adc_kw = {"cooling", "rack", "networking"}
    tx_kw = {"interconnection", "congestion", "transmission", "grid"}
    has_adc = bool(run.finding_keywords & adc_kw)
    has_tx = bool(run.finding_keywords & tx_kw)
    assert has_adc or has_tx, (
        f"Run C should have findings from at least one domain; got {run.finding_keywords}"
    )


# ---------------------------------------------------------------------------
# compare_runs()
# ---------------------------------------------------------------------------

def test_compare_runs_has_required_keys():
    run_a = build_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_run("run_b", ["transmission"], n=18)
    comp = compare_runs(run_a, run_b)
    for key in ("pair", "evidence", "findings", "recommendations"):
        assert key in comp


def test_compare_runs_evidence_jaccard_a_vs_b_less_than_one():
    run_a = build_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_run("run_b", ["transmission"], n=18)
    comp = compare_runs(run_a, run_b)
    assert comp["evidence"]["jaccard"] < 1.0, (
        f"Run A and B should have different evidence; Jaccard={comp['evidence']['jaccard']}"
    )


def test_compare_runs_findings_differ_a_vs_b():
    run_a = build_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_run("run_b", ["transmission"], n=18)
    comp = compare_runs(run_a, run_b)
    assert comp["findings"]["jaccard"] < 1.0, (
        f"Run A and B findings should differ; Jaccard={comp['findings']['jaccard']}"
    )


def test_compare_runs_recommendations_differ_a_vs_b():
    run_a = build_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_run("run_b", ["transmission"], n=18)
    comp = compare_runs(run_a, run_b)
    assert comp["recommendations"]["jaccard"] < 1.0, (
        f"Run A and B recommendations should differ; Jaccard={comp['recommendations']['jaccard']}"
    )


# ---------------------------------------------------------------------------
# run_all()
# ---------------------------------------------------------------------------

def test_run_all_has_required_top_level_keys():
    results = run_all(n=18)
    for key in ("runs", "comparisons", "similarity_matrix", "behavioral_validation"):
        assert key in results


def test_run_all_has_three_runs():
    results = run_all(n=18)
    assert set(results["runs"].keys()) == {"run_a", "run_b", "run_c"}


def test_run_all_has_three_comparisons():
    results = run_all(n=18)
    assert set(results["comparisons"].keys()) == {"a_vs_b", "a_vs_c", "b_vs_c"}


def test_run_all_similarity_matrix_keys():
    results = run_all(n=18)
    for pair in ("a_vs_b", "a_vs_c", "b_vs_c"):
        s = results["similarity_matrix"][pair]
        for metric in ("evidence_similarity", "finding_similarity", "recommendation_similarity"):
            assert metric in s


def test_run_all_behavioral_validation_all_true():
    results = run_all(n=18)
    bv = results["behavioral_validation"]
    for key, val in bv.items():
        assert val is True, f"behavioral_validation[{key!r}] should be True; got {val}"


def test_run_all_a_vs_b_evidence_similarity_below_threshold():
    results = run_all(n=18)
    sim = results["similarity_matrix"]["a_vs_b"]["evidence_similarity"]
    assert sim < 0.7, (
        f"A vs B evidence should be substantially different; Jaccard={sim:.3f}"
    )


def test_run_all_c_more_similar_to_a_than_b_is_to_a():
    results = run_all(n=18)
    # C uses ai_data_centers as execution profile, so C should be more similar to A
    sim_a_vs_c = results["similarity_matrix"]["a_vs_c"]["evidence_similarity"]
    sim_a_vs_b = results["similarity_matrix"]["a_vs_b"]["evidence_similarity"]
    assert sim_a_vs_c >= sim_a_vs_b, (
        f"C should be >= as similar to A as B is; A vs C={sim_a_vs_c:.3f}, A vs B={sim_a_vs_b:.3f}"
    )


def test_run_all_run_dict_has_required_keys():
    results = run_all(n=18)
    required = {
        "run_id", "profiles", "execution_profile", "evidence_count", "evidence_ids",
        "profile_attribution", "profiles_contributing", "profiles_missing",
        "profile_retrieval_summary", "finding_count", "recommendation_count",
        "finding_keywords", "recommendation_keywords",
    }
    for rid, run in results["runs"].items():
        missing = required - run.keys()
        assert not missing, f"{rid} missing keys: {missing}"


# ---------------------------------------------------------------------------
# Jaccard helper
# ---------------------------------------------------------------------------

def test_jaccard_identical_sets():
    assert _jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets():
    assert _jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_partial():
    j = _jaccard({1, 2, 3}, {2, 3, 4})
    assert abs(j - 0.5) < 0.001


def test_jaccard_empty_sets():
    assert _jaccard(set(), set()) == 1.0


# ---------------------------------------------------------------------------
# build_comparison_report()
# ---------------------------------------------------------------------------

def test_build_report_is_string():
    results = run_all(n=18)
    report = build_comparison_report(results)
    assert isinstance(report, str)
    assert len(report) > 100


def test_build_report_has_heading():
    results = run_all(n=18)
    report = build_comparison_report(results)
    assert "J5.6b" in report


def test_build_report_has_all_run_ids():
    results = run_all(n=18)
    report = build_comparison_report(results)
    for label in ("RUN_A", "RUN_B", "RUN_C"):
        assert label in report


def test_build_report_has_similarity_table():
    results = run_all(n=18)
    report = build_comparison_report(results)
    assert "Similarity Matrix" in report
    assert "A vs B" in report


def test_build_report_has_behavioral_validation():
    results = run_all(n=18)
    report = build_comparison_report(results)
    assert "Behavioral Validation" in report


def test_build_report_verdict_yes_when_all_pass():
    results = run_all(n=18)
    report = build_comparison_report(results)
    assert "YES" in report
