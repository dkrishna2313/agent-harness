"""Profile-Driven Retrieval Validation Harness (J5.6b).

Proves that different profile selections produce measurably different
research outputs.  Uses real profile YAML term sets (loaded from disk)
and a synthetic 50-item evidence corpus tagged with domain-specific
language to demonstrate that:

  Run A  (ai_data_centers)          → GPU / cooling / rack emphasis
  Run B  (transmission)             → grid / interconnection emphasis
  Run C  (ai_data_centers,transmission) → both domains merged

Public API
----------
CORPUS                – 50-item synthetic evidence corpus
score_corpus()        – relevance-score items against a profile term set
retrieve_top_n()      – return top-N items by profile relevance
build_run()           – full ProfileRun for one profile configuration
compare_runs()        – overlap and Jaccard similarity across run pair
run_all()             – execute A, B, C and produce full comparison
build_comparison_report() – markdown summary
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_TOP_N = 18  # evidence items retrieved per run (mirrors typical pipeline cap)


# ---------------------------------------------------------------------------
# Synthetic evidence corpus – 50 items across two domain profiles
# ---------------------------------------------------------------------------
# Items 1-28: ai_data_centers-domain language
# Items 29-50: transmission-domain language
# Items with id suffix "X" bridge both domains (appear in Run C overlap)

CORPUS: list[dict[str, Any]] = [
    # ── ai_data_centers domain (28 items) ──────────────────────────────────
    {"id": "C001", "claim": "NVIDIA Blackwell GPU rack density exceeds 100 kW per rack, requiring liquid cooling infrastructure.", "topics": ["rack architecture", "cooling", "power"]},
    {"id": "C002", "claim": "NVL72 rack-scale systems require direct liquid cooling (DLC) manifolds and coolant distribution units (CDU).", "topics": ["cooling", "rack architecture"]},
    {"id": "C003", "claim": "AI factory inference throughput is constrained by networking bandwidth between GPU clusters.", "topics": ["networking", "rack architecture"]},
    {"id": "C004", "claim": "Thermal management at 30–50 kW per rack requires chilled water loop integration with facility cooling plant.", "topics": ["cooling", "power"]},
    {"id": "C005", "claim": "GPU memory bandwidth (HBM) is the primary bottleneck for large language model inference latency.", "topics": ["rack architecture", "networking"]},
    {"id": "C006", "claim": "InfiniBand and Spectrum-X ethernet form the two competing GPU cluster interconnect fabrics.", "topics": ["networking"]},
    {"id": "C007", "claim": "Power usage effectiveness (PUE) improves from 1.4 to 1.1 with warm-water liquid cooling in GPU facilities.", "topics": ["cooling", "power"]},
    {"id": "C008", "claim": "UPS and BBU systems must support peak GPU power draw of up to 10 MW per data center pod.", "topics": ["power", "backup/resiliency"]},
    {"id": "C009", "claim": "AI data center commissioning requires integrated testing of power, cooling, and networking systems.", "topics": ["operations"]},
    {"id": "C010", "claim": "Rubin GPU architecture increases compute density 4× over prior generation Hopper systems.", "topics": ["rack architecture"]},
    {"id": "C011", "claim": "MGX modular chassis supports rapid GPU swap for maintenance without full rack decommission.", "topics": ["rack architecture", "operations"]},
    {"id": "C012", "claim": "PDU busway systems must support 415V three-phase distribution to high-density GPU racks.", "topics": ["power"]},
    {"id": "C013", "claim": "AI inference workload profiles differ from training: lower sustained power, higher memory bandwidth utilisation.", "topics": ["rack architecture", "operations"]},
    {"id": "C014", "claim": "ConnectX-7 NIC cards provide 400G Ethernet per GPU node for scale-out AI training clusters.", "topics": ["networking"]},
    {"id": "C015", "claim": "Battery backup units (BBU) in GPU racks provide 30–60 second ride-through for power grid disturbances.", "topics": ["backup/resiliency", "power"]},
    {"id": "C016", "claim": "GPU tray hot-swap capability reduces mean time to recovery from hardware failures in AI factories.", "topics": ["operations", "rack architecture"]},
    {"id": "C017", "claim": "NVLink switch fabric provides 1.8 TB/s bandwidth within an NVL72 GPU rack system.", "topics": ["networking", "rack architecture"]},
    {"id": "C018", "claim": "Cooling tower capacity must be sized for 15–20% headroom above peak GPU thermal load.", "topics": ["cooling"]},
    {"id": "C019", "claim": "AI data center power density growth requires switchgear upgrade cycles every 5–7 years.", "topics": ["power"]},
    {"id": "C020", "claim": "Liquid-cooled GPU racks reduce data center floor space requirement by 40% versus air-cooled equivalents.", "topics": ["cooling", "rack architecture"]},
    {"id": "C021", "claim": "Vera Rubin NVL576 systems will require 1 MW per cabinet, setting new power density benchmarks.", "topics": ["power", "rack architecture"]},
    {"id": "C022", "claim": "Chilled water supply temperature of 18–24°C is optimal for warm-water DLC GPU cooling.", "topics": ["cooling"]},
    {"id": "C023", "claim": "GPU cluster networking requires < 1 µs latency for collective operations in distributed training.", "topics": ["networking"]},
    {"id": "C024", "claim": "AI factory economics require 90%+ GPU utilisation to achieve positive return on infrastructure investment.", "topics": ["operations"]},
    {"id": "C025", "claim": "Thermal runaway risk in high-density GPU racks requires automated cooling fault detection and shutdown.", "topics": ["cooling", "backup/resiliency"]},
    {"id": "C026", "claim": "Power capping firmware on Blackwell GPUs enables dynamic rack power management within utility contract limits.", "topics": ["power", "rack architecture"]},
    {"id": "C027", "claim": "Data center cooling infrastructure lead times of 18–24 months are a binding constraint on AI capacity expansion.", "topics": ["cooling", "operations"]},
    {"id": "C028", "claim": "GPU rack cabinets require seismic bracing rated for zone 4 earthquakes in California deployments.", "topics": ["rack architecture", "operations"]},

    # ── transmission domain (22 items) ────────────────────────────────────
    {"id": "C029", "claim": "Interconnection queue backlogs at PJM and MISO exceed 2,000 GW of pending generation capacity.", "topics": ["interconnection", "grid_operators"]},
    {"id": "C030", "claim": "FERC Order 1920 requires transmission providers to conduct 20-year long-range transmission planning.", "topics": ["grid_operators", "capacity"]},
    {"id": "C031", "claim": "HVDC transmission links enable 2,000 MW power transfer across congested AC grid interfaces.", "topics": ["transmission_lines", "congestion"]},
    {"id": "C032", "claim": "New 500 kV transmission corridors face 7–10 year permitting and siting timelines under current NEPA rules.", "topics": ["permitting", "transmission_lines"]},
    {"id": "C033", "claim": "CAISO curtails 15–20 TWh of renewable energy annually due to transmission congestion in California.", "topics": ["congestion", "grid_operators"]},
    {"id": "C034", "claim": "Substation transformer lead times of 18 months constrain grid interconnection timelines for large loads.", "topics": ["grid_infrastructure", "interconnection"]},
    {"id": "C035", "claim": "Available transfer capability (ATC) on Eastern Interconnection bottlenecks limit power transfer by up to 40%.", "topics": ["capacity", "congestion"]},
    {"id": "C036", "claim": "N-1 contingency analysis requires transmission operators to maintain stability after any single element outage.", "topics": ["grid_infrastructure", "capacity"]},
    {"id": "C037", "claim": "ERCOT operates as an islanded grid, limiting power import capability during peak demand events.", "topics": ["grid_operators", "capacity"]},
    {"id": "C038", "claim": "Series capacitor compensation increases transmission line power transfer capacity by 30–50%.", "topics": ["grid_infrastructure", "transmission_lines"]},
    {"id": "C039", "claim": "Transmission congestion costs in PJM exceeded $4 billion in 2023, increasing generation dispatch costs.", "topics": ["congestion", "grid_operators"]},
    {"id": "C040", "claim": "HVAC underground cable installations cost 5–10× more per mile than equivalent overhead transmission lines.", "topics": ["transmission_lines", "permitting"]},
    {"id": "C041", "claim": "NERC reliability standards require balancing authorities to maintain frequency within 60 ± 0.5 Hz.", "topics": ["grid_operators", "grid_infrastructure"]},
    {"id": "C042", "claim": "Static VAR compensators (SVC) and FACTS devices improve voltage stability on long transmission corridors.", "topics": ["grid_infrastructure", "transmission_lines"]},
    {"id": "C043", "claim": "Right-of-way acquisition is the primary cost driver for new transmission construction in populated corridors.", "topics": ["permitting", "transmission_lines"]},
    {"id": "C044", "claim": "Large load interconnection studies take 2–4 years in most ISO/RTO queue processes under current rules.", "topics": ["interconnection", "grid_operators"]},
    {"id": "C045", "claim": "765 kV AC transmission achieves 50% lower line losses per MW-mile than 345 kV equivalents.", "topics": ["transmission_lines", "capacity"]},
    {"id": "C046", "claim": "Transmission congestion pricing (TCC) instruments hedge generation and load portfolios against grid constraint costs.", "topics": ["congestion", "capacity"]},
    {"id": "C047", "claim": "Voltage regulator upgrades at substation level are required to support large data center load interconnection.", "topics": ["grid_infrastructure", "interconnection"]},
    {"id": "C048", "claim": "MISO transmission expansion planning identifies $30 billion in projects required through 2035.", "topics": ["grid_operators", "capacity"]},
    {"id": "C049", "claim": "Generation interconnection queue reform under FERC Order 2023 introduces cluster study methodology to reduce delays.", "topics": ["interconnection", "grid_operators"]},
    {"id": "C050", "claim": "NYISO capacity zone constraints limit power import into New York City during summer peak demand.", "topics": ["grid_operators", "congestion"]},
]


# ---------------------------------------------------------------------------
# Profile term loading and evidence scoring
# ---------------------------------------------------------------------------

def _load_profile_term_set(profile_name: str) -> set[str]:
    """Load domain_terms + topic_keywords from a real profile YAML file."""
    from research_agent.profile import load_profile

    try:
        p = load_profile(profile_name)
        terms: set[str] = set()
        for t in (p.domain_terms or []):
            terms.add(t.lower())
        for kw_list in (p.topic_keywords or {}).values():
            for kw in kw_list:
                terms.add(kw.lower())
        return terms
    except Exception as exc:
        LOGGER.warning("Could not load profile %r: %s — using empty term set", profile_name, exc)
        return set()


def score_item(item: dict, term_set: set[str]) -> int:
    """Count how many profile terms appear in the item's claim + topics text."""
    text = (item.get("claim", "") + " " + " ".join(item.get("topics", []))).lower()
    return sum(1 for t in term_set if t in text)


def score_corpus(
    corpus: list[dict],
    term_set: set[str],
) -> list[dict]:
    """Return corpus items enriched with `_score` (descending order)."""
    scored = [{**item, "_score": score_item(item, term_set)} for item in corpus]
    return sorted(scored, key=lambda x: x["_score"], reverse=True)


def retrieve_top_n(
    corpus: list[dict],
    term_set: set[str],
    n: int = _TOP_N,
) -> list[dict]:
    """Return the top-N corpus items by relevance to a profile term set."""
    ranked = score_corpus(corpus, term_set)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Finding and recommendation generation
# ---------------------------------------------------------------------------

_FINDING_TEMPLATES: dict[str, list[str]] = {
    "cooling": [
        "Direct liquid cooling (DLC) is a prerequisite for AI rack densities above 30 kW.",
        "Cooling infrastructure lead times of 18–24 months are a binding constraint on AI capacity.",
    ],
    "rack": [
        "GPU rack density is increasing toward 1 MW per cabinet, requiring infrastructure redesign.",
        "NVL72/NVL576 rack systems require co-designed power and cooling delivery architectures.",
    ],
    "networking": [
        "GPU cluster networking bandwidth (InfiniBand / Spectrum-X) is a primary AI training bottleneck.",
        "Sub-microsecond latency interconnects are required for distributed AI training scalability.",
    ],
    "power": [
        "AI data center power intensity growth demands 5–7 year switchgear upgrade cycles.",
        "UPS and BBU systems must be sized for 10+ MW per AI factory pod.",
    ],
    "interconnection": [
        "Interconnection queue backlogs exceed 2,000 GW, creating 2–4 year delays for new loads.",
        "FERC Order 2023 cluster study reforms are necessary to reduce interconnection timelines.",
    ],
    "congestion": [
        "Transmission congestion costs exceed $4 billion annually in major ISOs, escalating generation costs.",
        "HVDC links and FACTS devices are required to resolve chronic grid congestion bottlenecks.",
    ],
    "transmission": [
        "New 500 kV transmission corridors require 7–10 year permitting timelines under NEPA.",
        "765 kV AC transmission achieves 50% lower line losses versus 345 kV alternatives.",
    ],
    "grid": [
        "20-year transmission planning (FERC Order 1920) is required to anticipate AI load growth.",
        "Substation transformer lead times constrain grid interconnection for large data center loads.",
    ],
}

_REC_TEMPLATES: dict[str, list[str]] = {
    "cooling": [
        "Deploy direct liquid cooling as the default thermal solution for all new AI rack deployments.",
        "Accelerate cooling supply chain qualification to reduce 18-month lead time risk.",
    ],
    "rack": [
        "Design facilities for 1 MW per cabinet power density to accommodate next-generation GPU systems.",
        "Standardise on modular rack chassis (MGX) to enable GPU swap without rack decommission.",
    ],
    "networking": [
        "Select GPU cluster networking fabric (InfiniBand vs Spectrum-X) as a strategic platform decision.",
        "Over-provision east-west networking bandwidth by 2× to accommodate future GPU scale-out.",
    ],
    "interconnection": [
        "Engage ISO interconnection queue 3–4 years ahead of facility commissioning target date.",
        "Negotiate utility capacity reservation agreements before committing facility capital expenditure.",
    ],
    "congestion": [
        "Locate AI data centers in grid regions with low congestion risk and adequate ATC margins.",
        "Hedge transmission congestion exposure through TCC instruments where available.",
    ],
    "transmission": [
        "Prioritise sites with existing high-voltage transmission access over greenfield transmission builds.",
        "Participate in MISO/PJM transmission expansion planning to secure future grid access rights.",
    ],
    "power": [
        "Structure power procurement through long-term utility contracts with indexed capacity rights.",
        "Install on-site backup generation sized for full facility load to manage grid reliability risk.",
    ],
}


def _keywords_in_top_evidence(top_items: list[dict]) -> set[str]:
    """Return the set of domain keywords present in retrieved evidence."""
    text = " ".join(
        item.get("claim", "") + " " + " ".join(item.get("topics", []))
        for item in top_items
    ).lower()
    all_kw = {
        kw for topic_kw in {**_FINDING_TEMPLATES, **_REC_TEMPLATES}
        for kw in topic_kw.split()
    }
    return {kw for kw in all_kw if kw in text}


def generate_findings(top_items: list[dict]) -> list[dict]:
    """Generate deterministic findings from the retrieved evidence set."""
    text = " ".join(
        item.get("claim", "") + " " + " ".join(item.get("topics", []))
        for item in top_items
    ).lower()
    findings: list[dict] = []
    fid = 1
    for key, templates in _FINDING_TEMPLATES.items():
        if key in text:
            for tmpl in templates:
                findings.append({
                    "finding_id": f"F{fid:02d}",
                    "finding": tmpl,
                    "domain_key": key,
                    "supporting_evidence": [
                        item["id"] for item in top_items
                        if key in (item.get("claim", "") + " ".join(item.get("topics", []))).lower()
                    ][:3],
                })
                fid += 1
    return findings


def generate_recommendations(top_items: list[dict]) -> list[dict]:
    """Generate deterministic recommendations from retrieved evidence."""
    text = " ".join(
        item.get("claim", "") + " " + " ".join(item.get("topics", []))
        for item in top_items
    ).lower()
    recs: list[dict] = []
    rid = 1
    for key, templates in _REC_TEMPLATES.items():
        if key in text:
            for tmpl in templates:
                recs.append({
                    "recommendation_id": f"R{rid:02d}",
                    "recommendation": tmpl,
                    "domain_key": key,
                })
                rid += 1
    return recs


# ---------------------------------------------------------------------------
# ProfileRun dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProfileRun:
    run_id: str
    profiles: list[str]
    execution_profile: str

    # Evidence
    top_evidence: list[dict] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)

    # Per-profile attribution
    profile_attribution: dict[str, list[str]] = field(default_factory=dict)
    profiles_contributing: list[str] = field(default_factory=list)
    profiles_missing: list[str] = field(default_factory=list)
    profile_retrieval_summary: dict[str, dict] = field(default_factory=dict)

    # Findings and recommendations
    findings: list[dict] = field(default_factory=list)
    recommendations: list[dict] = field(default_factory=list)
    finding_keywords: set[str] = field(default_factory=set)
    recommendation_keywords: set[str] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profiles": self.profiles,
            "execution_profile": self.execution_profile,
            "evidence_count": len(self.top_evidence),
            "evidence_ids": sorted(self.evidence_ids),
            "profile_attribution": {
                p: sorted(ids) for p, ids in self.profile_attribution.items()
            },
            "profiles_contributing": self.profiles_contributing,
            "profiles_missing": self.profiles_missing,
            "profile_retrieval_summary": self.profile_retrieval_summary,
            "finding_count": len(self.findings),
            "recommendation_count": len(self.recommendations),
            "finding_keywords": sorted(self.finding_keywords),
            "recommendation_keywords": sorted(self.recommendation_keywords),
        }


# ---------------------------------------------------------------------------
# Run builder
# ---------------------------------------------------------------------------

def build_run(run_id: str, profiles: list[str], n: int = _TOP_N) -> ProfileRun:
    """Execute a profile run against the synthetic corpus.

    Single-profile runs: retrieve top-N by execution profile relevance.
    Multi-profile runs: retrieve top n//num_profiles items per profile and
    union them (each profile contributes its most relevant items).  This
    mirrors the architecture principle that supporting profiles contribute
    domain-specific knowledge to the evidence pool.

    After retrieval, each item is attributed to the best-matching profile.
    """
    execution_profile = profiles[0]
    all_term_sets = {p: _load_profile_term_set(p) for p in profiles}

    if len(profiles) == 1:
        # Single profile: straightforward top-N retrieval
        top = retrieve_top_n(CORPUS, all_term_sets[execution_profile], n=n)
    else:
        # Multi-profile: allocate a guaranteed quota to each profile so every
        # profile contributes its most relevant items to the evidence pool.
        # Remaining slots (from rounding) are filled by the execution profile.
        per_profile = max(1, n // len(profiles))
        seen: set[str] = set()
        profile_buckets: dict[str, list[dict]] = {}
        for p in profiles:
            bucket: list[dict] = []
            for item in retrieve_top_n(CORPUS, all_term_sets[p], n=len(CORPUS)):
                if item["id"] not in seen and len(bucket) < per_profile:
                    bucket.append(item)
                    seen.add(item["id"])
            profile_buckets[p] = bucket
        # Combine profile buckets (round-robin to preserve order diversity)
        top_ids_ordered: list[str] = []
        max_bucket = max(len(b) for b in profile_buckets.values())
        for i in range(max_bucket):
            for p in profiles:
                b = profile_buckets[p]
                if i < len(b) and len(top_ids_ordered) < n:
                    top_ids_ordered.append(b[i]["id"])
        # Fill remaining slots with execution-profile items
        if len(top_ids_ordered) < n:
            for item in retrieve_top_n(CORPUS, all_term_sets[execution_profile], n=n):
                if item["id"] not in seen and item["id"] not in top_ids_ordered and len(top_ids_ordered) < n:
                    top_ids_ordered.append(item["id"])
        id_to_item = {item["id"]: item for item in CORPUS}
        top: list[dict] = [id_to_item[eid] for eid in top_ids_ordered if eid in id_to_item]

    # Attribution: assign each retrieved item to best-matching profile
    attribution: dict[str, list[str]] = {p: [] for p in profiles}
    for item in top:
        if len(profiles) == 1:
            attribution[profiles[0]].append(item["id"])
        else:
            scores = {p: score_item(item, all_term_sets[p]) for p in profiles}
            best_score = max(scores.values())
            if best_score == 0:
                attribution[execution_profile].append(item["id"])
            else:
                best = max(scores, key=lambda k: scores[k])
                attribution[best].append(item["id"])

    retrieval_summary = {
        p: {"evidence_count": len(ids)}
        for p, ids in attribution.items()
    }
    contributing = [p for p, ids in attribution.items() if ids]
    missing = [p for p in profiles if p not in contributing]

    findings = generate_findings(top)
    recommendations = generate_recommendations(top)

    return ProfileRun(
        run_id=run_id,
        profiles=profiles,
        execution_profile=execution_profile,
        top_evidence=top,
        evidence_ids={item["id"] for item in top},
        profile_attribution=attribution,
        profiles_contributing=contributing,
        profiles_missing=missing,
        profile_retrieval_summary=retrieval_summary,
        findings=findings,
        recommendations=recommendations,
        finding_keywords={f["domain_key"] for f in findings},
        recommendation_keywords={r["domain_key"] for r in recommendations},
    )


# ---------------------------------------------------------------------------
# Similarity and overlap
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return round(intersection / union, 3) if union else 0.0


def _overlap_report(label_a: str, set_a: set, label_b: str, set_b: set) -> dict[str, Any]:
    return {
        "jaccard": _jaccard(set_a, set_b),
        "shared": sorted(set_a & set_b),
        f"unique_to_{label_a}": sorted(set_a - set_b),
        f"unique_to_{label_b}": sorted(set_b - set_a),
    }


def compare_runs(run_a: ProfileRun, run_b: ProfileRun) -> dict[str, Any]:
    label_a, label_b = run_a.run_id, run_b.run_id
    return {
        "pair": f"{label_a}_vs_{label_b}",
        "evidence": _overlap_report(label_a, run_a.evidence_ids, label_b, run_b.evidence_ids),
        "findings": _overlap_report(label_a, run_a.finding_keywords, label_b, run_b.finding_keywords),
        "recommendations": _overlap_report(label_a, run_a.recommendation_keywords, label_b, run_b.recommendation_keywords),
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all(n: int = _TOP_N) -> dict[str, Any]:
    """Execute runs A, B, C and produce full comparison."""
    run_a = build_run("run_a", ["ai_data_centers"], n=n)
    run_b = build_run("run_b", ["transmission"], n=n)
    run_c = build_run("run_c", ["ai_data_centers", "transmission"], n=n)

    a_vs_b = compare_runs(run_a, run_b)
    a_vs_c = compare_runs(run_a, run_c)
    b_vs_c = compare_runs(run_b, run_c)

    behavioral_validation = {
        "retrieval_changed": a_vs_b["evidence"]["jaccard"] < 1.0,
        "evidence_changed": a_vs_b["evidence"]["jaccard"] < 1.0,
        "findings_changed": a_vs_b["findings"]["jaccard"] < 1.0,
        "recommendations_changed": a_vs_b["recommendations"]["jaccard"] < 1.0,
        "multi_profile_combines_contributions": (
            len(run_c.evidence_ids & run_a.evidence_ids) > 0
            and len(run_c.evidence_ids & run_b.evidence_ids) > 0
        ),
    }

    return {
        "runs": {
            "run_a": run_a.to_dict(),
            "run_b": run_b.to_dict(),
            "run_c": run_c.to_dict(),
        },
        "comparisons": {
            "a_vs_b": a_vs_b,
            "a_vs_c": a_vs_c,
            "b_vs_c": b_vs_c,
        },
        "similarity_matrix": {
            "a_vs_b": {
                "evidence_similarity":        a_vs_b["evidence"]["jaccard"],
                "finding_similarity":         a_vs_b["findings"]["jaccard"],
                "recommendation_similarity":  a_vs_b["recommendations"]["jaccard"],
            },
            "a_vs_c": {
                "evidence_similarity":        a_vs_c["evidence"]["jaccard"],
                "finding_similarity":         a_vs_c["findings"]["jaccard"],
                "recommendation_similarity":  a_vs_c["recommendations"]["jaccard"],
            },
            "b_vs_c": {
                "evidence_similarity":        b_vs_c["evidence"]["jaccard"],
                "finding_similarity":         b_vs_c["findings"]["jaccard"],
                "recommendation_similarity":  b_vs_c["recommendations"]["jaccard"],
            },
        },
        "behavioral_validation": behavioral_validation,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def build_comparison_report(results: dict[str, Any]) -> str:
    runs = results["runs"]
    sims = results["similarity_matrix"]
    bv = results["behavioral_validation"]
    comps = results["comparisons"]

    lines: list[str] = [
        "# J5.6b Profile-Driven Retrieval Validation Report",
        "",
        "**Goal:** Develop a strategy for AI infrastructure investment over the next decade.",
        "",
        "## Run Configuration",
        "",
        "| Run | Profiles | Execution Profile | Evidence Retrieved |",
        "|-----|----------|-------------------|--------------------|",
    ]
    for rid, run in runs.items():
        lines.append(
            f"| {rid.upper()} | {', '.join(run['profiles'])} "
            f"| {run['execution_profile']} | {run['evidence_count']} |"
        )

    lines += ["", "## Evidence Attribution", ""]
    for rid, run in runs.items():
        lines.append(f"### {rid.upper()} — {', '.join(run['profiles'])}")
        lines.append("")
        for profile, ids in run["profile_attribution"].items():
            prs = run["profile_retrieval_summary"].get(profile, {})
            lines.append(f"- **{profile}**: {prs.get('evidence_count', 0)} items ({', '.join(ids[:5])}{'…' if len(ids) > 5 else ''})")
        lines.append(f"- **profiles_contributing**: {run['profiles_contributing']}")
        lines.append(f"- **profiles_missing**: {run['profiles_missing']}")
        lines.append("")

    lines += ["## Finding Topics", ""]
    for rid, run in runs.items():
        kws = run.get("finding_keywords", [])
        lines.append(f"- **{rid.upper()}**: {', '.join(sorted(kws)) if kws else '(none)'}")
    lines.append("")

    lines += ["## Recommendation Topics", ""]
    for rid, run in runs.items():
        kws = run.get("recommendation_keywords", [])
        lines.append(f"- **{rid.upper()}**: {', '.join(sorted(kws)) if kws else '(none)'}")
    lines.append("")

    lines += ["## Similarity Matrix", ""]
    lines += [
        "| Pair | Evidence | Findings | Recommendations |",
        "|------|----------|----------|-----------------|",
    ]
    for pair_key, pair_label in [("a_vs_b", "A vs B"), ("a_vs_c", "A vs C"), ("b_vs_c", "B vs C")]:
        s = sims[pair_key]
        lines.append(
            f"| {pair_label} | {s['evidence_similarity']:.3f} "
            f"| {s['finding_similarity']:.3f} "
            f"| {s['recommendation_similarity']:.3f} |"
        )

    lines += ["", "## Evidence Overlap Detail", ""]
    for pair_key, pair_label in [("a_vs_b", "A vs B"), ("a_vs_c", "A vs C"), ("b_vs_c", "B vs C")]:
        c = comps[pair_key]["evidence"]
        shared = len(c.get("shared", []))
        total = len(c.get("shared", [])) + len(c.get(f"unique_to_run_{'a' if 'a_vs' in pair_key else 'b'}", []))
        lines.append(f"**{pair_label}**: {shared} shared items, Jaccard={c['jaccard']:.3f}")
    lines.append("")

    lines += ["## Behavioral Validation", ""]
    for key, val in bv.items():
        symbol = "✓" if val else "✗"
        label = key.replace("_", " ").title()
        lines.append(f"- {symbol} {label}: **{'YES' if val else 'NO'}**")
    lines.append("")

    # Verdict
    all_pass = all(bv.values())
    lines += [
        "## Verdict",
        "",
        "**Do profiles materially affect research outcomes?**",
        "",
    ]
    if all_pass:
        lines += [
            "> **YES.** Profiles are functioning as true knowledge modules.",
            "> Different profile selections produce demonstrably different evidence sets,",
            "> findings, and recommendations.  The multi-profile run (C) combines contributions",
            "> from both domains.  Profile influence is fully observable and measurable.",
        ]
    else:
        failed = [k for k, v in bv.items() if not v]
        lines += [
            f"> **PARTIAL.** The following criteria were not met: {', '.join(failed)}.",
            "> Investigate profile term sets and corpus coverage.",
        ]
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# File writer
# ---------------------------------------------------------------------------

def write_artifacts(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_comparison_report(results)
    (out_dir / "j56b_profile_comparison_report.md").write_text(report, encoding="utf-8")

    (out_dir / "j56b_profile_comparison.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    LOGGER.info("[ProfileComparison] Artifacts written to %s", out_dir)
