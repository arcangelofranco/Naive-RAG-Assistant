from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    """Reported state of the service and its dependencies.

    Attributes:
        status: ``ok`` when the database answered, otherwise
            ``database_unreachable``.
        database: Whether the database accepted a query.
        chunks_indexed: Whether the index holds at least one chunk. A fresh
            database answers ``false`` here, which distinguishes "not
            ingested yet" from "not reachable".
        embedding_model_loaded: Whether the model finished loading during
            startup.
    """

    status: str
    database: bool
    chunks_indexed: bool
    embedding_model_loaded: bool

class RetrievedChunkResponse(BaseModel):
    """A retrieval result as returned by the debug endpoint.

    Unlike ``SourceResponse`` this one includes the chunk text, because
    inspecting what was actually retrieved is the point of that endpoint.

    Attributes:
        document_id: Identifier of the source document.
        title: Title of the source document.
        section: Heading of the section the chunk came from.
        chunk_index: Zero-based position of the chunk in its document.
        content: The chunk text.
        similarity: Cosine similarity against the query vector.
    """

    document_id: str
    title: str
    section: str
    chunk_index: int
    content: str
    similarity: float

class DebugRetrieveResponse(BaseModel):
    """Raw retrieval output for a query, with no generation involved.

    Attributes:
        question: The query as received.
        results: The retrieved chunks, ordered by decreasing similarity and
            not filtered by the threshold, so results below the cutoff stay
            visible for inspection.
    """

    question: str
    results: list[RetrievedChunkResponse]

class AskRequest(BaseModel):
    """A question submitted to the full pipeline.

    Attributes:
        question: The user question. Bounded on both ends, so an empty body
            and an oversized one are both rejected with ``422`` before any
            embedding or database work happens.
        top_k: Per-request override of the configured retrieval width, or
            ``None`` to use the default.
    """

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)

class SourceResponse(BaseModel):
    """A passage supporting an answer.

    Attributes:
        document_id: Identifier of the source document.
        title: Title of the source document.
        section: Heading of the section the passage came from.
        chunk_index: Zero-based position of the chunk in its document.
        similarity: Cosine similarity against the query vector.
    """

    document_id: str
    title: str
    section: str
    chunk_index: int
    similarity: float

class AskResponse(BaseModel):
    """The answer to a question, or the refusal.

    Attributes:
        answer: The grounded answer with inline ``[n]`` citations, or the
            exact refusal string.
        sources: The passages backing the answer, empty on a refusal. Their
            order matches the citation numbers used in ``answer``.
        grounded: Whether the answer is supported by retrieved context. It
            is ``false`` both when nothing cleared the threshold and when
            citation validation rejected the generated text.
    """

    answer: str
    sources: list[SourceResponse]
    grounded: bool
