Implement evidence ranking.

Goal:
Improve memo quality by scoring evidence items before synthesis.

Requirements:
1. Add evidence scoring fields to EvidenceItem:
   - relevance_score: int 1-5
   - source_quality_score: int 1-5
   - specificity_score: int 1-5
   - overall_score: float

2. Implement scoring logic:
   - relevance_score: how directly the evidence answers the user question
   - source_quality_score: official vendor/OCP/technical docs score higher than generic commentary
   - specificity_score: concrete technical claims score higher than vague marketing language

3. Sort evidence by overall_score before memo synthesis.

4. Limit synthesis input to top N evidence items.
   - Default top_n = 50
   - Add CLI flag: --top-evidence 50

5. Include evidence ranking in trace JSON.

6. Add debug output:
   - total evidence items
   - evidence items used for synthesis
   - top 5 evidence items with scores

7. Add evaluator warning:
   - warn if fewer than 10 high-quality evidence items are available
   - high-quality = overall_score >= 3.5

8. Add tests for:
   - evidence scoring
   - evidence sorting
   - top-N filtering
   - trace includes scores
   - debug output includes top evidence summary

Keep it simple.
Do not add chunking yet.
Do not add vector databases.
Do not add web search.