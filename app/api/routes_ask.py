from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import AskRequest, AskResponse, SourceResponse
from app.deps import get_query_service
from app.services.query_service import QueryService
from app.providers.llm.llm_exceptions import LLMRateLimitError, LLMUnavailableError

from typing import Annotated

router = APIRouter()

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, query_service: Annotated[QueryService, Depends(get_query_service)]):
    """Answer a question from the knowledge base, or refuse.

    A refusal is not an error: it comes back as ``200`` with ``grounded``
    set to false and no sources, because the system declining to answer is a
    valid outcome rather than a failure.

    Both error responses carry an empty ``detail`` deliberately. Naming the
    provider or echoing its message would leak which backend is configured,
    so the client learns only that the condition is transient.

    Note:
        This non-disclosure behavior is not covered by automated tests.

    Args:
        req: The validated request body.
        query_service: The pipeline, injected per request.

    Returns:
        The answer with its supporting sources, or the refusal with none.

    Raises:
        HTTPException: ``429`` when the provider quota stayed exhausted
            across every retry, forwarding ``Retry-After`` when the provider
            declared one, or ``503`` when the provider stayed unreachable.
    """
    try:
        res = query_service.ask(req.question, req.top_k)
    except LLMRateLimitError as e:
        headers = {
            "Retry-After": str(e.retry_interval_sec)
        } if e.retry_interval_sec else None
        raise HTTPException(
            status_code=429,
            detail="",
            headers=headers
        ) from e
    except LLMUnavailableError as e:
        raise HTTPException(
            status_code=503,
            detail=""
        ) from e

    return AskResponse(
        answer=res.answer,
        sources=[
            SourceResponse(
                document_id=s.document_id,
                title=s.title,
                section=s.section,
                chunk_index=s.chunk_index,
                similarity=s.similarity
            ) for s in res.sources
        ],
        grounded=res.grounded
    )
