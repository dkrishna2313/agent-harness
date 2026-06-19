Implement J1: Domain Profiles.

Goal:
Make the harness reusable across research domains by moving domain-specific topic logic, gap heuristics, coverage topics, and source classification hints into configurable domain profiles.

Current issue:
The harness currently contains AI data center / NVIDIA-specific concepts such as:
- power
- cooling
- rack architecture
- UPS
- PDU
- heat rejection

These are useful for AI infrastructure research but should not be hard-coded into the general harness.

Desired architecture:

Question
→ Domain Profile
→ Topic Detection
→ Retrieval
→ Evidence Extraction
→ Contradiction Detection
→ Research Gap Detection
→ Coverage Matrix
→ Evidence Ranking
→ Memo Synthesis

Requirements:

1. Create domain profile configuration files.

Add directory:

profiles/

Create initial profiles:

profiles/ai_data_centers.yaml
profiles/smr.yaml

2. Domain profile schema.

Each profile should include:

- name
- description
- topic_keywords
- coverage_topics
- research_gap_checks
- source_quality_hints
- memo_section_hints

Example structure:

name: ai_data_centers
description: AI data center infrastructure and rack-scale computing research

topic_keywords:
  power:
    - power
    - megawatt
    - grid
    - UPS
    - PDU
    - BESS
  cooling:
    - cooling
    - liquid cooling
    - coolant
    - heat rejection
  rack architecture:
    - rack
    - NVL72
    - MGX
    - tray

coverage_topics:
  - power
  - cooling
  - rack architecture
  - operations
  - resiliency

research_gap_checks:
  power:
    - rack power consumption
    - PDU topology
    - UPS requirements
    - utility interconnect
  cooling:
    - heat rejection load
    - coolant supply temperature
    - flow rate
    - CDU requirements

source_quality_hints:
  primary:
    - NVIDIA Technical Blog
    - NVIDIA architecture
    - specification
  secondary:
    - StorageReview
    - analyst
  synthetic:
    - test_

memo_section_hints:
  - Power Implications
  - Cooling Implications
  - Rack Architecture Implications
  - Research Gaps
  - Coverage Matrix

3. Add SMR profile.

profiles/smr.yaml should include topics such as:

- reactor design
- licensing
- safety
- economics
- construction
- grid integration
- fuel cycle
- waste management
- cooling
- deployment timeline

Example research_gap_checks:

licensing:
  - NRC licensing path
  - construction permit
  - operating license
  - regulatory timeline

economics:
  - overnight capital cost
  - levelized cost of electricity
  - financing model
  - cost overrun risk

fuel cycle:
  - fuel availability
  - HALEU supply
  - enrichment
  - refueling interval

waste management:
  - spent fuel handling
  - interim storage
  - disposal pathway

grid integration:
  - interconnection requirements
  - load following
  - transmission constraints
  - capacity factor

cooling:
  - water requirements
  - heat rejection
  - cooling technology

4. CLI support.

Add CLI option:

--profile PROFILE_NAME_OR_PATH

Examples:

python3 -m research_agent.cli \
  "What are the DC power and cooling implications of NVIDIA Rubin NVL72 racks?" \
  --sources ./sources \
  --profile ai_data_centers \
  --out ./outputs/rubin.md \
  --debug

python3 -m research_agent.cli \
  "What are the deployment challenges and infrastructure implications of SMRs?" \
  --sources ./smr_sources \
  --profile smr \
  --out ./outputs/smr.md \
  --debug

Default behavior:
If no profile is provided, use ai_data_centers for backward compatibility.

5. Topic detection changes.

Replace hard-coded topic detection with profile-driven topic detection.

Use:
- profile.topic_keywords
- question text
- optionally evidence categories

The detected topics should be written to trace as before.

6. Research gap detection changes.

Replace hard-coded gap checks with profile.research_gap_checks.

The gap detector should:
- inspect detected topics
- inspect evidence items
- identify missing required concepts from the profile

7. Coverage matrix changes.

Use profile.coverage_topics instead of hard-coded topics.

Coverage scoring should remain domain-independent:
- evidence_count
- source_count
- source_quality

8. Memo section hints.

Use profile.memo_section_hints to guide memo structure.

Do not require every profile to use identical memo sections.

9. Trace support.

Add to trace:

domain_profile:
  name
  path
  description

profile_topics_available

profile_gap_checks_available

10. Tests.

Add tests for:

- loading ai_data_centers profile
- loading smr profile
- CLI --profile option
- topic detection from profile
- research gap checks from profile
- coverage matrix topics from profile
- backward compatibility when no profile is supplied

11. Preserve existing behavior.

The current NVIDIA Rubin test should still work with:

--profile ai_data_centers

and should also work with no --profile argument.

Do not add:
- internet search
- embeddings
- web downloading

This is configuration separation only.