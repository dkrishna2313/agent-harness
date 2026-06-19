Implement G3: Research Gap Detection.

Goal:
Identify important unanswered questions and missing evidence within the current source corpus.

Current workflow:

Question
→ Retrieval
→ Evidence Extraction
→ Contradiction Detection
→ Evidence Ranking
→ Memo Synthesis

New workflow:

Question
→ Retrieval
→ Evidence Extraction
→ Contradiction Detection
→ Research Gap Detection
→ Evidence Ranking
→ Memo Synthesis

Requirements:

1. Create ResearchGap schema.

Fields:

- gap_id
- topic
- priority
- description
- rationale

Priority values:

- high
- medium
- low

2. Gap detection should analyze:

Question topics
+
Evidence extracted
+
Source coverage

Determine whether important topics are insufficiently supported.

3. Initial heuristics.

For power-related questions:

Look for evidence regarding:

- rack power
- power delivery
- UPS
- batteries
- generators
- utility interconnect
- power quality

For cooling-related questions:

Look for:

- cooling technology
- CDU requirements
- water temperature
- flow rate
- heat rejection
- facility integration

For networking-related questions:

Look for:

- bandwidth
- topology
- optics
- switch architecture

For operations-related questions:

Look for:

- commissioning
- monitoring
- maintenance
- resiliency

4. Gap generation rules.

Example:

Question:
Power implications of Rubin NVL72

Evidence:
No explicit rack power figure

Gap:

Topic:
Rack Power

Priority:
High

Description:
No explicit NVL72 rack power consumption figure found.

Rationale:
Power planning requires a rack-level power target.

5. Add trace support.

Add:

research_gaps

Each gap should include:

- gap_id
- topic
- priority
- description
- rationale

6. Add memo section.

Insert:

Research Gaps

Example:

Research Gaps

High Priority
-------------
- No explicit NVL72 rack power figure found.

Medium Priority
---------------
- No CDU sizing guidance found.

Low Priority
------------
- No UPS topology recommendations found.

7. Add evaluator integration.

Add informational metrics:

- gap_count
- high_priority_gap_count

Do not create warnings simply because gaps exist.

8. Add tests.

Verify:

- gap generation
- trace output
- memo rendering
- prioritization

Keep implementation simple.

Do not add:
- web search
- external lookups
- embeddings

Only use currently available evidence.