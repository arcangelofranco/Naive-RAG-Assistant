from psycopg import sql

UPSERT_DOCUMENT = """
    INSERT INTO documents (id, title, category, source_path, raw_content)
    VALUES (%s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title,
        category = EXCLUDED.category,
        source_path = EXCLUDED.source_path,
        raw_content = EXCLUDED.raw_content
"""
"""Insert a document, or refresh it in place when the id already exists.

Updating by id rather than inserting blindly is half of what makes
re-ingestion idempotent.
"""

DELETE_DOCUMENT_CHUNKS = "DELETE FROM chunks WHERE document_id = %s"
"""Remove every chunk of a document before writing the new ones.

The other half of idempotency. Vectors go with them through the cascade on
``chunk_embeddings``, and the index on ``document_id`` is what keeps both
the delete and the cascade off a sequential scan.
"""

INSERT_CHUNK = """
    INSERT INTO chunks (document_id, chunk_index, section, content, token_count)
    VALUES (%s, %s, %s, %s, %s)
"""
"""Insert one chunk. Executed as a batch for all chunks of a document."""

CHUNK_IDS_BY_INDEX = "SELECT chunk_index, id FROM chunks WHERE document_id = %s"
"""Read back the generated ids, mapped by position within the document.

Chunk ids come from a sequence, so they are only known after insertion.
This is what lets each vector be attached to the right chunk row.
"""

CHUNKS_TO_EMBED = """
    SELECT c.id AS chunk_id, d.title, c.section, c.content
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    ORDER BY c.id
"""
"""List every stored chunk with the fields needed to rebuild its embedding text."""

INSERT_EMBEDDING = sql.SQL("INSERT INTO {} (chunk_id, model, embedding) VALUES (%s, %s, %s)")
"""Insert a vector for a freshly written chunk.

A plain insert is safe here because the chunk rows were just recreated, so
no vector can exist for them yet.
"""

UPSERT_EMBEDDING = sql.SQL("""
    INSERT INTO {} (chunk_id, model, embedding) VALUES (%s, %s, %s)
    ON CONFLICT (chunk_id, model) DO UPDATE SET embedding = EXCLUDED.embedding
""")
"""Write or replace a vector for a chunk that already exists.

Used by the backfill path, where the same chunk may be re-embedded more
than once and the ``(chunk_id, model)`` primary key would otherwise clash.
"""

SEARCH_SIMILAR = sql.SQL("""
    SELECT d.id AS document_id, d.title, c.section, c.chunk_index, c.content,
           1 - (e.embedding <=> %s::vector) AS similarity
    FROM {} e
    JOIN chunks c ON c.id = e.chunk_id
    JOIN documents d ON d.id = c.document_id
    WHERE e.model = %s
    ORDER BY e.embedding <=> %s::vector
    LIMIT %s
""")
"""Rank chunks by closeness to a query vector.

``<=>`` is pgvector's cosine distance operator, so the projection converts
it into a similarity with ``1 - distance``. Filtering on ``model`` keeps
vectors from different embedding models from being compared against each
other. With no vector index in place this is an exact scan, which returns
the true nearest neighbors rather than an approximation.
"""
