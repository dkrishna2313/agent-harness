Implement J1.3: Profile-Aware Source Classification.

Problem:

The SMR profile is loading correctly.

However all SMR documents are being classified as:

source_quality_score = 2
source_type = unknown

Examples:

DOE Liftoff Report Advanced Nuclear.pdf
IAEA SMR Catalogue 2024.pdf
NRC Licensing Guidance.pdf
NEA Small Modular Reactor Dashboard.pdf
INL Small Reactors in Microgrids.pdf

These should be treated as primary authoritative sources.

Requirements:

Use profile.source_quality_hints.

Add deterministic filename matching.

SMR mappings:

Score 5:

- DOE
- Department of Energy
- IAEA
- NRC
- Nuclear Regulatory Commission
- OECD
- NEA
- INL
- Idaho National Laboratory
- International Energy Agency
- IEA

Score 4:

- World Nuclear Association
- NuScale
- TerraPower
- GE Hitachi
- BWRX

Score 3:

- industry analysis
- consultant reports

Score 2:

- unknown

Score 1:

- synthetic test files

Update:

source_quality_map
source_quality_details
evidence source_quality_score
retrieval source_quality_score

Success criteria:

DOE, IAEA, NRC, INL and NEA appear as score 5 in the trace.