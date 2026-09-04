from fastapi import Depends, Request
from psycopg_pool import ConnectionPool

from app.composition import build_llm_provider, build_query_service
from app.config.config import Settings, get_settings
from app.providers.embedding.base import EmbeddingProvider
from app.providers.llm.base import LLMProvider
from app.services.query_service import QueryService

from typing import Annotated

def get_database_pool(req: Request) -> ConnectionPool:
    """Return the connection pool opened during application startup.

    Args:
        req: The incoming request, used to reach application state.

    Returns:
        The process-wide pool.
    """
    return req.app.state.db_pool

def get_embedding_provider(req: Request) -> EmbeddingProvider:
    """Return the embedding provider loaded during application startup.

    It is built in the lifespan hook because loading the model takes several
    seconds, which would be unacceptable per request.

    Args:
        req: The incoming request, used to reach application state.

    Returns:
        The process-wide embedding provider.
    """
    return req.app.state.embedding_provider

def get_llm_provider(req: Request, settings: Annotated[Settings, Depends(get_settings)]) -> LLMProvider:
    """Return the LLM stack, building it on first use.

    Construction is deferred rather than done at startup because it needs an
    API key. Building it eagerly would stop the application from booting
    without credentials, and both ingestion and the retrieval debug endpoint
    are meant to stay usable in that state.

    Memoizing it in application state is also what makes the response cache
    effective, since the cache lives inside the stack.

    Args:
        req: The incoming request, used to reach application state.
        settings: Application settings, injected by FastAPI.

    Returns:
        The process-wide LLM stack: cache over retry over the adapter.

    Raises:
        ValueError: If the provider is unknown or a required credential is
            missing. It surfaces on the first request that needs the model.
    """
    provider = getattr(req.app.state, "llm_provider", None)
    if provider is None:
        provider = build_llm_provider(settings)
        req.app.state.llm_provider = provider
    return provider

def get_query_service(
    embedding_provider: Annotated[EmbeddingProvider, Depends(get_embedding_provider)],
    pool: Annotated[ConnectionPool, Depends(get_database_pool)],
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
    settings: Annotated[Settings, Depends(get_settings)]
) -> QueryService:
    """Assemble the query service for a request.

    The service itself is cheap to build, since it only holds references to
    collaborators that already exist. Only the repository is constructed per
    request, wrapping the shared pool.

    Args:
        embedding_provider: The process-wide embedding provider.
        pool: The process-wide connection pool.
        llm_provider: The process-wide LLM stack.
        settings: Application settings.

    Returns:
        A query service ready to answer one request.
    """
    return build_query_service(
        settings=settings,
        pool=pool,
        embedding_provider=embedding_provider,
        llm_provider=llm_provider
    )
