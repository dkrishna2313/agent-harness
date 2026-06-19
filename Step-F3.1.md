Review the F3 chunking implementation.

Current observations from memo.trace.json:

- chunk_count = 47
- evidence_items_used_for_synthesis = 13
- 41 chunks produced zero evidence items
- fewer than 3 source documents have evidence items

Chunking itself appears to be working:
- chunks are being created
- source_chunk_id is being recorded
- evidence extraction, ranking, and synthesis are functioning

However, chunk-level evidence yield appears low and several source documents that previously produced evidence now produce none.

Goal:
Determine whether this is expected behavior or whether evidence extraction is overly selective.

Tasks:

1. Add chunk-level diagnostics to the trace.

For every chunk record:
- chunk_id
- document_name
- chunk_size
- relevance_to_question
- evidence_candidate_count
- evidence_items_created
- extraction_decision:
    accepted
    rejected
- rejection_reason

Examples:
- not relevant to question
- marketing content
- duplicate evidence
- insufficient specificity
- parsing failure
- extraction threshold not met

2. Add debug output summary.

Examples:
- total chunks
- accepted chunks
- rejected chunks
- top rejection reasons
- evidence items per document
- evidence items per chunk

3. Analyze the current extraction prompt.

Determine whether:
- chunking is functioning correctly
- evidence extraction is too restrictive
- document-level filtering is accidentally occurring

4. Add tests.

Verify:
- chunk diagnostics are written to trace
- rejection reasons are captured
- accepted/rejected counts are correct

Do not add retrieval yet.
Do not add embeddings.
Do not add vector databases.

First explain your findings.
Then implement improvements.