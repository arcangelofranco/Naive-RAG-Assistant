from collections.abc import Generator
from contextlib import contextmanager

from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config.config import Settings

def _configure(conn):
    """Register the pgvector adapters on a newly opened connection.

    Args:
        conn: The connection the pool has just opened.
    """
    register_vector(conn)

def create_pool(settings: Settings) -> ConnectionPool:
    """Open a connection pool against the configured database.

    Args:
        settings: Application settings, read for ``database_url``.

    Returns:
        An open pool whose connections already understand vector types. The
        caller owns it and is responsible for closing it, which the API does
        when its lifespan ends.
    """
    return ConnectionPool(
        conninfo=settings.database_url,
        configure=_configure,
        open=True
    )

@contextmanager
def pool_for(settings: Settings) -> Generator[ConnectionPool, None, None]:
    """Provide a pool for the duration of a block, then close it.

    This is the short-lived counterpart to ``create_pool``, used by the
    ingestion entry point, where the process exits as soon as the work is
    done and nothing outlives the command.

    Args:
        settings: Application settings, read for ``database_url``.

    Yields:
        An open connection pool, closed on exit even if the block raises.
    """
    pool = create_pool(settings)
    try:
        yield pool
    finally:
        pool.close(timeout=1)
