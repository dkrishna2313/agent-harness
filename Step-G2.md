Implement G2: Contradiction Detection.

Goal:
Identify potentially conflicting evidence before memo synthesis.

Current workflow:

Question
→ Retrieval
→ Evidence Extraction
→ Evidence Ranking
→ Memo Synthesis

New workflow:

Question
→ Retrieval
→ Evidence Extraction
→ Contradiction Detection
→ Evidence Ranking
→ Memo Synthesis

Requirements:

1. Add contradiction analysis.

Compare extracted EvidenceItems.

Look for:

* conflicting numbers
* conflicting performance claims
* conflicting power values
* conflicting cooling requirements
* conflicting rack specifications
* conflicting timelines
* conflicting architecture descriptions

2. Create schema:

Contradiction

Fields:

* contradiction_id
* topic
* evidence_a_id
* evidence_b_id
* evidence_a_claim
* evidence_b_claim
* severity
* explanation

Severity:

* low
* medium
* high

3. Detection rules (v1)

Start simple:

Detect:

* numeric disagreements
* mutually exclusive statements
* incompatible categorical claims

Examples:

Power:
120 kW vs 180 kW

Cooling:
air cooled vs liquid cooled

Timeline:
2026 vs 2027

Architecture:
72 GPUs vs 144 GPUs

4. Trace support.

Add:

contradictions_detected

Each contradiction should include:

* evidence ids
* source documents
* severity

5. Debug output.

Show:

Contradictions detected: N

Examples:

Topic: rack power
Severity: medium
Evidence: E014 vs E021

6. Memo support.

Add section:

Potential Contradictions

Include:

* topic
* conflicting claims
* sources involved

If no contradictions:

Potential Contradictions

No significant contradictions detected.

7. Evaluator.

Add warning:

High-severity contradiction detected.

Do not block output.

8. Tests.

Add tests for:

* numeric contradictions
* categorical contradictions
* no contradiction case
* trace generation
* memo rendering

Keep implementation simple.

Do not add:

* embeddings
* web search
* external fact checking

Only compare evidence already extracted.
