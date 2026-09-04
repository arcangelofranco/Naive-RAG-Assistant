from typing import Protocol

class EmbeddingProvider(Protocol):
    """Turns texts into vectors that can be compared by cosine similarity."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into embedding vectors.

        Implementations are expected to return normalized vectors, so that
        cosine similarity coincides with the dot product, and to preserve
        the input order.

        Args:
            texts: The texts to encode. For chunks this is the text produced
                by ``embedding_text``, header included.

        Returns:
            One vector per input text, in the same order. Every vector has
            the dimensionality of the underlying model, which must match the
            ``vector`` column width in the database schema.
        """
        pass
