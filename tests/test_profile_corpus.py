"""Tests for J5.6c Profile Corpus Validation.

Covers:
- SOURCE_CORPUS has 60 items: 32 ai_data_centers + 28 transmission
- Every corpus item has id, claim, topics, source_document, source_type
- build_source_run() returns SourceAttributedRun with all required fields
- Single-profile run A (ai_data_centers) retrieves NVIDIA/ASHRAE sources
- Single-profile run B (transmission) retrieves PJM/MISO/ERCOT sources
- Multi-profile run C populates both ai_data_centers and transmission source pools
- profile_retrieval_summary contains evidence_sources and source_types per profile
- profile_source_pools keys match the run's profiles
- compare_source_pools() computes Jaccard on publisher sets
- Source Jaccard(A vs B) < 1.0 (different publisher pools)
- Source pools for A and B are disjoint or near-disjoint
- run_corpus_validation() returns runs + source_comparison + similarity_matrix + behavioral_validation
- behavioral_validation passes: retrieval_changed, source_pools_differ, etc.
- run_b_retrieves_grid_sources is True
- run_a_retrieves_ai_dc_sources is True
- multi_profile_combines_both is True
- all_profiles_contributing_c is True
- build_corpus_report() returns markdown string with expected sections
- write_corpus_artifacts() produces j56c files
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from functional_agents.profile_corpus_validator import (
    SOURCE_CORPUS,
    SourceAttributedRun,
    build_source_run,
    compare_source_pools,
    run_corpus_validation,
    build_corpus_report,
    write_corpus_artifacts,
    _AI_DC_SOURCES,
    _TRANSMISSION_SOURCES,
)


# ---------------------------------------------------------------------------
# SOURCE_CORPUS structure
# ---------------------------------------------------------------------------

def test_source_corpus_has_sixty_items():
    assert len(SOURCE_CORPUS) == 60


def test_source_corpus_items_have_required_fields():
    for item in SOURCE_CORPUS:
        assert "id" in item, f"Missing id: {item}"
        assert "claim" in item
        assert "topics" in item
        assert "source_document" in item
        assert "source_type" in item
        assert item["id"].startswith("S")


def test_source_corpus_ai_dc_items_count():
    ids = [item for item in SOURCE_CORPUS if int(item["id"][1:]) <= 32]
    assert len(ids) == 32


def test_source_corpus_transmission_items_count():
    ids = [item for item in SOURCE_CORPUS if int(item["id"][1:]) >= 33]
    assert len(ids) == 28


def test_source_corpus_ai_dc_has_nvidia_source():
    nvidia_items = [i for i in SOURCE_CORPUS if i["source_document"] == "NVIDIA"]
    assert len(nvidia_items) >= 2


def test_source_corpus_transmission_has_pjm_source():
    pjm_items = [i for i in SOURCE_CORPUS if i["source_document"] == "PJM"]
    assert len(pjm_items) >= 2


def test_source_corpus_transmission_has_miso_source():
    miso_items = [i for i in SOURCE_CORPUS if i["source_document"] == "MISO"]
    assert len(miso_items) >= 2


def test_source_corpus_transmission_has_ercot_source():
    ercot_items = [i for i in SOURCE_CORPUS if i["source_document"] == "ERCOT"]
    assert len(ercot_items) >= 2


def test_source_corpus_transmission_has_ferc_source():
    ferc_items = [i for i in SOURCE_CORPUS if i["source_document"] == "FERC"]
    assert len(ferc_items) >= 2


def test_source_corpus_ai_dc_has_ashrae_source():
    items = [i for i in SOURCE_CORPUS if i["source_document"] == "ASHRAE"]
    assert len(items) >= 1


def test_source_corpus_source_types_are_strings():
    for item in SOURCE_CORPUS:
        assert isinstance(item["source_type"], str)
        assert len(item["source_type"]) > 0


# ---------------------------------------------------------------------------
# Source set constants
# ---------------------------------------------------------------------------

def test_ai_dc_sources_contains_expected():
    for expected in ("NVIDIA", "ASHRAE", "Google", "Meta", "AWS"):
        assert expected in _AI_DC_SOURCES


def test_transmission_sources_contains_expected():
    for expected in ("PJM", "MISO", "ERCOT", "FERC", "NERC"):
        assert expected in _TRANSMISSION_SOURCES


def test_ai_dc_and_transmission_sources_are_mostly_disjoint():
    overlap = _AI_DC_SOURCES & _TRANSMISSION_SOURCES
    # IEA and LBNL appear in both, but the sets should be mostly distinct
    assert len(overlap) < 5


# ---------------------------------------------------------------------------
# build_source_run() – single profile
# ---------------------------------------------------------------------------

def test_build_run_a_returns_source_attributed_run():
    run = build_source_run("run_a", ["ai_data_centers"])
    assert isinstance(run, SourceAttributedRun)


def test_build_run_a_retrieves_n_items():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_b_retrieves_n_items():
    run = build_source_run("run_b", ["transmission"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_a_source_pool_contains_ai_dc_sources():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    sources = run.profile_source_pools.get("ai_data_centers", set())
    overlap = sources & _AI_DC_SOURCES
    assert len(overlap) > 0, f"Run A sources should include AI DC publishers; got {sources}"


def test_build_run_b_source_pool_contains_grid_sources():
    run = build_source_run("run_b", ["transmission"], n=18)
    sources = run.profile_source_pools.get("transmission", set())
    overlap = sources & _TRANSMISSION_SOURCES
    assert len(overlap) > 0, f"Run B sources should include grid publishers; got {sources}"


def test_build_run_a_profiles_contributing():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    assert "ai_data_centers" in run.profiles_contributing
    assert run.profiles_missing == []


def test_build_run_b_profiles_contributing():
    run = build_source_run("run_b", ["transmission"], n=18)
    assert "transmission" in run.profiles_contributing
    assert run.profiles_missing == []


# ---------------------------------------------------------------------------
# build_source_run() – multi-profile
# ---------------------------------------------------------------------------

def test_build_run_c_retrieves_n_items():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert len(run.top_evidence) == 18


def test_build_run_c_profile_source_pools_has_both_keys():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert "ai_data_centers" in run.profile_source_pools
    assert "transmission" in run.profile_source_pools


def test_build_run_c_both_profiles_contributing():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert "ai_data_centers" in run.profiles_contributing
    assert "transmission" in run.profiles_contributing


def test_build_run_c_profile_source_pools_both_non_empty():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert len(run.profile_source_pools["ai_data_centers"]) > 0
    assert len(run.profile_source_pools["transmission"]) > 0


def test_build_run_c_ai_sources_in_pool():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    ai_pool = run.profile_source_pools.get("ai_data_centers", set())
    overlap = ai_pool & _AI_DC_SOURCES
    assert len(overlap) > 0


def test_build_run_c_tx_sources_in_pool():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    tx_pool = run.profile_source_pools.get("transmission", set())
    overlap = tx_pool & _TRANSMISSION_SOURCES
    assert len(overlap) > 0


# ---------------------------------------------------------------------------
# profile_retrieval_summary structure
# ---------------------------------------------------------------------------

def test_retrieval_summary_has_evidence_sources_key():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    prs = run.profile_retrieval_summary.get("ai_data_centers", {})
    assert "evidence_sources" in prs


def test_retrieval_summary_has_source_types_key():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    prs = run.profile_retrieval_summary.get("ai_data_centers", {})
    assert "source_types" in prs


def test_retrieval_summary_evidence_sources_is_list():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    prs = run.profile_retrieval_summary["ai_data_centers"]
    assert isinstance(prs["evidence_sources"], list)


def test_retrieval_summary_source_types_is_list():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    prs = run.profile_retrieval_summary["ai_data_centers"]
    assert isinstance(prs["source_types"], list)


def test_retrieval_summary_evidence_count_matches():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    prs = run.profile_retrieval_summary["ai_data_centers"]
    assert prs["evidence_count"] == len(run.profile_attribution["ai_data_centers"])


def test_retrieval_summary_c_has_both_profiles():
    run = build_source_run("run_c", ["ai_data_centers", "transmission"], n=18)
    assert "ai_data_centers" in run.profile_retrieval_summary
    assert "transmission" in run.profile_retrieval_summary


# ---------------------------------------------------------------------------
# to_dict()
# ---------------------------------------------------------------------------

def test_to_dict_has_required_keys():
    run = build_source_run("run_a", ["ai_data_centers"], n=18)
    d = run.to_dict()
    required = {
        "run_id", "profiles", "execution_profile", "evidence_count", "evidence_ids",
        "profile_source_pools", "profile_attribution",
        "profiles_contributing", "profiles_missing",
        "profile_retrieval_summary",
        "finding_count", "recommendation_count",
        "finding_keywords", "recommendation_keywords",
    }
    assert required <= d.keys()


# ---------------------------------------------------------------------------
# compare_source_pools()
# ---------------------------------------------------------------------------

def test_compare_source_pools_returns_dict():
    run_a = build_source_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_source_run("run_b", ["transmission"], n=18)
    comp = compare_source_pools(run_a, run_b)
    assert isinstance(comp, dict)


def test_compare_source_pools_has_jaccard():
    run_a = build_source_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_source_run("run_b", ["transmission"], n=18)
    comp = compare_source_pools(run_a, run_b)
    assert "jaccard" in comp
    assert 0.0 <= comp["jaccard"] <= 1.0


def test_compare_source_pools_a_vs_b_near_disjoint():
    run_a = build_source_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_source_run("run_b", ["transmission"], n=18)
    comp = compare_source_pools(run_a, run_b)
    assert comp["jaccard"] < 0.5, (
        f"Source pools for A and B should be mostly disjoint; Jaccard={comp['jaccard']:.3f}"
    )


def test_compare_source_pools_has_pair_key():
    run_a = build_source_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_source_run("run_b", ["transmission"], n=18)
    comp = compare_source_pools(run_a, run_b)
    assert "pair" in comp


def test_compare_source_pools_has_shared_sources():
    run_a = build_source_run("run_a", ["ai_data_centers"], n=18)
    run_b = build_source_run("run_b", ["transmission"], n=18)
    comp = compare_source_pools(run_a, run_b)
    assert "shared_sources" in comp
    assert isinstance(comp["shared_sources"], list)


# ---------------------------------------------------------------------------
# run_corpus_validation()
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_results():
    return run_corpus_validation(n=18)


def test_run_validation_has_required_keys(corpus_results):
    for key in ("runs", "source_comparison", "similarity_matrix", "behavioral_validation"):
        assert key in corpus_results


def test_run_validation_has_three_runs(corpus_results):
    assert set(corpus_results["runs"].keys()) == {"run_a", "run_b", "run_c"}


def test_run_validation_source_comparison_has_three_pairs(corpus_results):
    assert set(corpus_results["source_comparison"].keys()) == {"a_vs_b", "a_vs_c", "b_vs_c"}


def test_run_validation_similarity_matrix_has_source_similarity(corpus_results):
    for pair in ("a_vs_b", "a_vs_c", "b_vs_c"):
        s = corpus_results["similarity_matrix"][pair]
        assert "source_similarity" in s


def test_run_validation_similarity_matrix_has_evidence_similarity(corpus_results):
    for pair in ("a_vs_b", "a_vs_c", "b_vs_c"):
        s = corpus_results["similarity_matrix"][pair]
        assert "evidence_similarity" in s


def test_run_validation_behavioral_validation_retrieval_changed(corpus_results):
    assert corpus_results["behavioral_validation"]["retrieval_changed"] is True


def test_run_validation_behavioral_validation_source_pools_differ(corpus_results):
    assert corpus_results["behavioral_validation"]["source_pools_differ"] is True


def test_run_validation_behavioral_validation_run_b_grid_sources(corpus_results):
    assert corpus_results["behavioral_validation"]["run_b_retrieves_grid_sources"] is True


def test_run_validation_behavioral_validation_run_a_ai_dc_sources(corpus_results):
    assert corpus_results["behavioral_validation"]["run_a_retrieves_ai_dc_sources"] is True


def test_run_validation_behavioral_validation_multi_profile_combines(corpus_results):
    assert corpus_results["behavioral_validation"]["multi_profile_combines_both"] is True


def test_run_validation_behavioral_validation_all_profiles_contributing_c(corpus_results):
    assert corpus_results["behavioral_validation"]["all_profiles_contributing_c"] is True


def test_run_validation_behavioral_validation_coverage_sufficient(corpus_results):
    assert corpus_results["behavioral_validation"]["coverage_status_sufficient"] is True


def test_run_validation_all_behavioral_validations_pass(corpus_results):
    bv = corpus_results["behavioral_validation"]
    for key, val in bv.items():
        assert val is True, f"behavioral_validation[{key!r}] should be True; got {val}"


# ---------------------------------------------------------------------------
# build_corpus_report()
# ---------------------------------------------------------------------------

def test_build_corpus_report_is_string(corpus_results):
    report = build_corpus_report(corpus_results)
    assert isinstance(report, str)
    assert len(report) > 200


def test_build_corpus_report_has_heading(corpus_results):
    report = build_corpus_report(corpus_results)
    assert "J5.6c" in report


def test_build_corpus_report_has_three_run_ids(corpus_results):
    report = build_corpus_report(corpus_results)
    for label in ("RUN_A", "RUN_B", "RUN_C"):
        assert label in report


def test_build_corpus_report_has_profile_retrieval_section(corpus_results):
    report = build_corpus_report(corpus_results)
    assert "Profile Retrieval Summary" in report


def test_build_corpus_report_mentions_key_sources(corpus_results):
    report = build_corpus_report(corpus_results)
    # At least one expected AI DC or grid publisher should appear
    publishers = ["NVIDIA", "PJM", "MISO", "ERCOT", "ASHRAE"]
    matched = [p for p in publishers if p in report]
    assert len(matched) > 0, f"Report should mention domain-specific sources; checked {publishers}"


def test_build_corpus_report_has_similarity_matrix(corpus_results):
    report = build_corpus_report(corpus_results)
    assert "Similarity Matrix" in report


def test_build_corpus_report_has_behavioral_validation(corpus_results):
    report = build_corpus_report(corpus_results)
    assert "Behavioral Validation" in report


def test_build_corpus_report_has_verdict(corpus_results):
    report = build_corpus_report(corpus_results)
    assert "Verdict" in report
    assert "YES" in report


# ---------------------------------------------------------------------------
# write_corpus_artifacts()
# ---------------------------------------------------------------------------

def test_write_corpus_artifacts_creates_files(corpus_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        write_corpus_artifacts(corpus_results, out)
        assert (out / "j56c_profile_corpus_report.md").exists()
        assert (out / "j56c_profile_corpus.json").exists()


def test_write_corpus_artifacts_json_is_valid(corpus_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        write_corpus_artifacts(corpus_results, out)
        data = json.loads((out / "j56c_profile_corpus.json").read_text())
        assert "runs" in data
        assert "behavioral_validation" in data


def test_write_corpus_artifacts_report_has_content(corpus_results):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir)
        write_corpus_artifacts(corpus_results, out)
        text = (out / "j56c_profile_corpus_report.md").read_text()
        assert "J5.6c" in text
        assert len(text) > 200
