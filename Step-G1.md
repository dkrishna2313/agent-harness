Implement G1: Question-Aware Retrieval.

Goal:
Improve research quality and scalability by selecting the most relevant chunks before evidence extraction.

Current behavior:

* Chunks are created.
* Chunks are sent to Claude primarily based on character budget.
* Many chunks are processed even when they are weakly relevant.

New behavior:
Question
→ Chunk documents
→ Score chunk relevance
→ Select top chunks
→ Extract evidence
→ Rank evidence
→ Synthesize memo

Requirements:

1. Add retrieval scoring.

For every chunk compute:

* keyword_score
* topic_match_score
* document_priority_score
* overall_retrieval_score

Definitions:

keyword_score:

* overlap between question terms and chunk terms

topic_match_score:

* overlap with detected question topics
* power
* cooling
* networking
* rack architecture
* operations
* resiliency

document_priority_score:

* configurable score by source type

Initial source priority:

5:

* NVIDIA primary sources
* NVIDIA technical blogs

4:

* NVIDIA enterprise documents
* rack architecture documents

3:

* independent technical analysis

2:

* general commentary

overall_retrieval_score:
weighted combination of the above

2. Retrieval selection.

Add CLI option:

--top-chunks N

Default:
15

Only the top-N chunks should be sent to evidence extraction.

3. Trace enhancements.

Add:

retrieval_ranking:

* chunk_id
* document_name
* retrieval_score
* keyword_score
* topic_match_score
* document_priority_score

selected_chunk_ids

rejected_chunk_ids

4. Debug output.

Show:

Top chunks selected:

chunk_id
retrieval_score
document_name

Example:

Top chunks:

1. Inside_the_NVIDIA_Vera_R_C013 score=0.91

2. Inside_the_NVIDIA_Vera_R_C012 score=0.88

3. Rack_Scale_Agentic_AI_Su_C001 score=0.85

4. Evaluator.

Add informational metrics:

* chunk_count
* chunks_selected
* retrieval_ratio

Do not create warnings for rejected chunks.

6. Tests.

Add tests for:

* retrieval scoring
* top chunk selection
* trace output
* debug output
* top-chunks filtering

Preserve:

* chunking
* evidence extraction
* evidence ranking
* citation system
* trace generation
* regression suite

Do not add:

* embeddings
* vector databases
* web search

Keep implementation simple and deterministic.
