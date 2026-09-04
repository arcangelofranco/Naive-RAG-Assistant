from pathlib import Path
import logging

from app.config.config import Settings
from app.ingestion.ingestor import IngestionReport

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "knowledge"
"""Path: The corpus directory, resolved relative to this file.

Deriving it from the module location keeps the command working both from the
repository root and from inside the container, where the code lives under a
different absolute path.
"""

def ingest_knowledge(settings: Settings, dir: Path | None = None) -> IngestionReport:
    """Index every Markdown document found under a directory.

    Files are ingested in sorted order, which makes a run reproducible and
    its log readable. The composition root is imported lazily, inside the
    function, so that importing this module stays cheap and free of side
    effects for callers that only want ``KNOWLEDGE_DIR``.

    Args:
        settings: Application settings, used to build the embedding provider
            and the database pool. The embedding model must be the same one
            the query path uses, which the composition root ensures.
        dir: Directory to scan recursively, or ``None`` for the bundled
            knowledge base.

    Returns:
        A report of the documents and chunks written.
    """
    from app.composition import build_ingestor
    from app.database.db import pool_for

    dir = dir or KNOWLEDGE_DIR
    paths = sorted(dir.rglob("*.md"))
    logger.info(f"{len(paths)} documents found in {dir}")
    with pool_for(settings) as pool:
        return build_ingestor(settings, pool).ingest(paths)


if __name__ == "__main__":
    from app.config.config import get_settings

    logging.basicConfig(level=logging.INFO)
    report = ingest_knowledge(get_settings())
    print(f"{report.document_count} documents, {report.chunk_count} chunks ingested.")
