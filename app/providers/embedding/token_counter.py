from typing import Protocol
from app.providers.embedding.base import EmbeddingProvider

class TokenCounter(EmbeddingProvider, Protocol):
    """An embedding provider that also exposes its tokenizer limits."""

    def count_tokens(self, text: str) -> int:
        """Count the tokens the model would consume for a text.

        Args:
            text: The text to measure.

        Returns:
            The token count according to the model's own tokenizer, not an
            approximation derived from word or character counts.
        """
        pass

    @property
    def max_tokens(self) -> int:
        """int: Maximum tokens the model accepts in a single input.

        Anything longer is truncated by the model without raising, so the
        chunker treats this value as a hard ceiling to stay below.
        """
        pass
