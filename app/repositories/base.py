from dataclasses import dataclass
from app.domain.docs import Chunk, ParsedDoc, RetrievedChunk
from typing import Protocol


class ChunkSearcher(Protocol):
    """The read side: nearest-neighbor lookup over the indexed chunks."""

    def search_similar(self, query_emb: list[float], top_k: int) -> list[RetrievedChunk]:
        """Find the chunks closest to a query vector.

        Implementations are expected to restrict the search to vectors
        produced by the configured embedding model, since vectors from
        different models occupy different spaces and cannot be compared.

        Args:
            query_emb: The embedded question.
            top_k: Maximum number of results to return.

        Returns:
            Up to ``top_k`` chunks ordered from most to least similar. The
            list is never filtered by relevance, so callers still have to
            apply the similarity threshold themselves.
        """
        pass

class KnowledgeWriter(Protocol):
    """The write side: persisting a document with its chunks and vectors."""

    def save_document(self, doc: ParsedDoc, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store a document, replacing whatever was indexed for it before.

        Implementations must be idempotent, so that re-ingesting an unchanged
        corpus leaves the same final state instead of duplicating rows.

        Args:
            doc: The parsed document, raw Markdown included.
            chunks: The chunks cut from it, in document order.
            embeddings: One vector per chunk, in the same order.
        """
        pass

@dataclass(frozen=True)
class ChunkToEmbed:
    """A stored chunk that needs a vector computed for it.

    Attributes:
        chunk_id: Database identifier of the chunk.
        title: Title of the source document, needed to rebuild the context
            header exactly as the original ingestion did.
        section: Heading of the section the chunk came from.
        content: The chunk text, without the header.
    """

    chunk_id: int
    title: str
    section: str
    content: str

class EmbeddingFiller(Protocol):
    """Backfills vectors for chunks that are already stored.

    It exists for the migration to a new embedding model, where chunk text
    stays put while a second vector table is populated in the background.
    That command does not exist yet, so this port currently has an
    implementation but no caller.
    """

    def chunks_to_embed(self) -> list[ChunkToEmbed]:
        """List the stored chunks together with their provenance.

        Returns:
            Every chunk in the corpus, ordered by identifier, with the
            fields needed to rebuild its embedding text.
        """
        pass

    def save_embeddings(self, model: str, vectors: dict[int, list[float]]) -> None:
        """Write or replace vectors for chunks that already exist.

        Args:
            model: Name of the model that produced the vectors, stored
                alongside them so that spaces stay separable.
            vectors: Mapping from chunk identifier to its vector.
        """
        pass
