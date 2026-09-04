from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg import OperationalError

from app.api.routes_ask import router as ask_router
from app.api.routes_debug import router as debug_router
from app.api.routes_health import router as health_router
from app.composition import build_embedding_provider
from app.config.config import get_settings
from app.database.db import create_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the shared resources at startup and close them at shutdown.

    Loading the embedding model here costs several seconds once, instead of
    on every question. The pool is opened eagerly for the same reason.

    Args:
        app: The application whose state holds the shared resources.

    Yields:
        Control to the running application, with ``db_pool`` and
        ``embedding_provider`` available on ``app.state``.
    """
    settings = get_settings()
    app.state.db_pool = create_pool(settings)
    app.state.embedding_provider = build_embedding_provider(settings)
    yield
    app.state.db_pool.close()


app = FastAPI(title="RAG Assistant", lifespan=lifespan)
app.include_router(health_router)
app.include_router(debug_router)
app.include_router(ask_router)


@app.exception_handler(OperationalError)
async def database_unavailable_handler(request: Request, exc: OperationalError) -> JSONResponse:
    """Turn a database connection failure into a neutral ``503``.

    Catching it application-wide means no route has to guard its own
    queries, and the client never receives a driver error message carrying
    connection details.

    Args:
        request: The request that failed.
        exc: The psycopg error raised while reaching the database.

    Returns:
        A ``503`` response with a fixed message.
    """
    return JSONResponse(status_code=503, content={"detail": "Database not accessible."})
