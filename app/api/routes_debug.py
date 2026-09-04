from fastapi import APIRouter, Depends, Query

from app.api.schemas import DebugRetrieveResponse, RetrievedChunkResponse
from app.deps import get_query_service
from app.services.query_service import QueryService

from typing import Annotated

router = APIRouter()

@router.get("/debug/retrieve", response_model=DebugRetrieveResponse)
def debug_retrieve(
    query_service: Annotated[QueryService, Depends(get_query_service)],
    q: str = Query(..., min_length=1, max_length=1000),
    top_k: int | None = Query(default=None, ge=1, le=20)
):
    """Return the chunks nearest to a query, scores included.

    Results are deliberately not filtered by the similarity threshold, so
    the scores that fell below the cutoff stay visible. Seeing them is what
    makes it possible to tell a genuinely out-of-domain question from one
    the threshold rejected by mistake.

    Args:
        query_service: The pipeline, injected per request.
        q: The query text.
        top_k: Number of chunks to retrieve, or ``None`` for the configured
            default.

    Returns:
        The query and its retrieval results, ordered by decreasing
        similarity.
    """
    results = query_service.retrive(q, top_k)
    return DebugRetrieveResponse(
        question=q,
        results=[RetrievedChunkResponse(**vars(r)) for r in results]
    )
