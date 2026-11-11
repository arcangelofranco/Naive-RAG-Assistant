from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import logging

from app.domain.docs import embedding_text
from app.ingestion.chunker import chunk_doc
from app.ingestion.parser import parse_doc
from app.providers.embedding.token_counter import TokenCounter
from app.repositories.base import KnowledgeWriter

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class IngestedDoc:
    """What a single document contributed to the index.

    Attributes:
        document_id: Identifier of the ingested document.
        chunk_count: Number of chunks written for it.
    """

    document_id: str
    chunk_count: int

@dataclass(frozen=True)
class IngestionReport:
    """Summary of one ingestion run.

    Attributes:
        documents: One entry per ingested document, in processing order.
    """

    documents: list[IngestedDoc]

    @property
    def document_count(self) -> int:
        """int: Number of documents ingested in this run."""
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        """int: Total chunks written across every document."""
        return sum(d.chunk_count for d in self.documents)


class KnowledgeIngestor:
    """Turns Markdown files into indexed chunks with their vectors.

    It needs a ``TokenCounter`` rather than a plain embedding provider,
    because the chunker has to size windows with the very tokenizer that
    will encode them.
    """

    def __init__(
        self,
        embedding_provider: TokenCounter,
        repository: KnowledgeWriter
    ) -> None:
        """Wire the ingestor to its embedding backend and storage.

        Args:
            embedding_provider: Produces the vectors and counts the tokens.
            repository: Persists documents, chunks and vectors.
        """
        self._embedding_provider = embedding_provider
        self._repository = repository

    def ingest(self, paths: Iterable[Path]) -> IngestionReport:
        """Ingest a collection of documents, one at a time.

        Documents are processed independently, so a failure on one aborts
        the run while leaving the documents already written in place. Since
        ingestion is idempotent, rerunning the command after a fix is safe.

        Args:
            paths: Paths of the Markdown files to ingest.

        Returns:
            A report covering every document processed in this run.
        """
        return IngestionReport(
            [self._ingest_doc(path) for path in paths]
        )

    def _ingest_doc(self, path: Path) -> IngestedDoc:
        """Run the full pipeline for a single document.

        The text handed to the embedding model is rebuilt here with
        ``embedding_text``, so the vector carries the context header while
        the stored content does not. That asymmetry is deliberate: the
        header is added back when the prompt is built, and storing it would
        make it appear twice in every passage.

        Args:
            path: Path of the Markdown file to ingest.

        Returns:
            The record of what this document contributed to the index.

        Raises:
            ValueError: If the embedding provider returned a number of
                vectors that does not match the number of chunks, which
                would otherwise pair vectors with the wrong chunks.
        """
        doc = parse_doc(path)
        chunks = chunk_doc(
            doc,
            self._embedding_provider.count_tokens,
            self._embedding_provider.max_tokens
        )
        embeddings = self._embedding_provider.embed(
            [
                embedding_text(doc.title, c.section, c.content)
                for c in chunks
            ]
        )

        if len(embeddings) != len(chunks):
            raise ValueError(
                f"{doc.id}: {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        self._repository.save_document(doc, chunks, embeddings)

        logger.info(f"Ingested {doc.id}: {len(chunks)} chunks.")
        return IngestedDoc(document_id=doc.id, chunk_count=len(chunks))
