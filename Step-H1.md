Implement H1: Source Quality Weighting.

Goal:
Improve research quality by explicitly weighting sources based on credibility, authority, and document type.

Current workflow:

Question
→ Retrieval
→ Evidence Extraction
→ Contradiction Detection
→ Research Gap Detection
→ Evidence Ranking
→ Memo Synthesis

Current issue:

Most sources receive similar document_priority_score values.
The harness does not sufficiently distinguish between:
- NVIDIA primary technical documents
- NVIDIA marketing content
- independent technical analysis
- secondary reporting
- synthetic test files

New behavior:

Every source should receive a source quality classification.

This classification should influence:

1. Retrieval ranking
2. Evidence ranking
3. Contradiction severity assessment

Requirements:

1. Create SourceQuality schema.

Fields:

- source_document
- source_type
- source_quality_score
- rationale

2. Define initial source classes.

Score 5:
- NVIDIA technical blogs
- NVIDIA architecture documents
- NVIDIA specification sheets
- NVIDIA platform documentation

Score 4:
- NVIDIA press releases
- NVIDIA solution briefs
- vendor technical whitepapers

Score 3:
- StorageReview
- technical journalism
- industry analysis

Score 2:
- blogs
- commentary
- community content

Score 1:
- synthetic test files
- unknown sources

3. Add source classification.

Use:
- filename
- source metadata
- configurable mapping table

Do not use LLM classification.

Keep deterministic.

4. Retrieval changes.

In retrieval scoring:

overall_retrieval_score should incorporate:

- keyword_score
- topic_match_score
- source_quality_score

Ensure high-quality sources rank higher when relevance is similar.

5. Evidence ranking changes.

Current:

overall_score =
relevance +
source_quality +
specificity

Enhance:

source quality should contribute meaningfully.

Example:

Two equivalent claims:

NVIDIA technical spec
vs
test_power_a.txt

The NVIDIA source should rank higher.

6. Contradiction improvements.

When contradictions are detected:

Record:

- source_quality_a
- source_quality_b

Add confidence assessment.

Example:

Source A:
score 5

Source B:
score 1

Result:

Lower confidence in contradiction.

Potential explanation:
lower-quality source conflicts with higher-quality source.

7. Trace enhancements.

Add:

source_quality_map

Example:

{
  "Inside the NVIDIA Vera Rubin Platform.pdf": 5,
  "StorageReview.com.pdf": 3,
  "test_power_a.txt": 1
}

Add to evidence:

source_quality_class

8. Debug output.

Show:

Source Quality Summary

Score 5:
- 4 documents

Score 3:
- 1 document

Score 1:
- 2 documents

9. Tests.

Verify:

- source classification
- retrieval weighting
- evidence ranking weighting
- contradiction confidence assessment
- trace output

Preserve:

- chunking
- retrieval
- contradiction detection
- research gap detection
- memo synthesis

Do not add:

- web lookups
- reputation APIs
- embeddings

Use deterministic rules only.