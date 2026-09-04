from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from psycopg_pool import ConnectionPool

from app.api.schemas import HealthResponse
from app.deps import get_database_pool

from typing import Annotated

router = APIRouter()

@router.get("/health")
def health(req: Request, pool: Annotated[ConnectionPool, Depends(get_database_pool)]):
    """Report whether the service can actually serve questions.

    The check goes beyond "the process is up". It runs a real query, so a
    pool that exists but cannot reach PostgreSQL is reported as unhealthy,
    and it distinguishes an empty index from an unreachable database, since
    a freshly created volume needs ingestion rather than repair.

    The database probe catches broadly on purpose: any failure to query is
    an unhealthy database as far as a health check is concerned, and this
    route must answer rather than propagate.

    The LLM stack is not probed, because it is built lazily on first use and
    contacting it would spend a real API call on every check.

    Args:
        req: The incoming request, used to inspect application state.
        pool: The connection pool, injected per request.

    Returns:
        A ``200`` response when the database answered, otherwise ``503``
        with the same body shape so clients can always read the details.
    """
    try:
        with pool.connection() as conn:
            chunks_indexed = conn.execute("SELECT EXISTS (SELECT 1 FROM chunks)").fetchone()[0]
        database_ok = True
    except Exception:
        database_ok = False
        chunks_indexed = False

    embedding_model_loaded = getattr(req.app.state, "embedding_provider", None) is not None

    body = HealthResponse(
        status="ok" if database_ok else "database_unreachable",
        database=database_ok,
        chunks_indexed=chunks_indexed,
        embedding_model_loaded=embedding_model_loaded
    )

    status_code = 503 if not database_ok else 200

    return JSONResponse(status_code=status_code, content=body.model_dump())
