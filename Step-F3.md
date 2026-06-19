Implement F3: document chunking.

Goal:
Improve scalability and evidence extraction quality for large document sets.

Requirements:

1. Add document chunking before evidence extraction.

Chunk size:
- Target 6000-8000 characters.
- Preserve sentence boundaries when possible.

2. Create Chunk schema.

Fields:
- chunk_id
- document_name
- chunk_number
- text
- start_offset
- end_offset

3. Update extraction workflow.

Current:
documents -> evidence

New:
documents -> chunks -> evidence

4. Evidence items must record:
- source_document
- source_chunk_id

5. Add trace output:
- chunk_count
- chunks_per_document
- evidence_per_chunk

6. Add debug output:
- documents loaded
- chunks generated
- average chunk size
- evidence extracted per chunk

7. Add evaluator warnings:
- document generated zero chunks
- chunk generated zero evidence
- chunk extraction failure

8. Add tests:
- chunk generation
- chunk boundaries
- evidence linked to chunk IDs
- trace includes chunk metadata

Do not add embeddings.
Do not add vector databases.
Do not add retrieval yet.

Preserve all existing functionality.