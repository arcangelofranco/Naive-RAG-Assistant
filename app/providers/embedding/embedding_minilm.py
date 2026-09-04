from sentence_transformers import SentenceTransformer
from app.providers.embedding.token_counter import TokenCounter

class MiniLMEmbeddingProvider(TokenCounter):
    """Embeds text with a MiniLM sentence-transformers model.

    The default model produces 384-dimensional vectors and accepts at most
    256 tokens per input, which are the two figures the database schema and
    the chunker are built around. Swapping in a model with a different
    dimensionality requires a matching schema change and a full re-indexing
    of the corpus, because vectors from different models are not comparable.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the sentence-transformers model into memory.

        Loading takes several seconds, so callers are expected to build this
        once per process rather than per request. The API does so in its
        lifespan hook.

        Args:
            model_name: Name of the sentence-transformers model to load,
                resolved from the local cache when already present.
        """
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of texts into normalized embedding vectors.

        Normalization is requested explicitly, which makes cosine similarity
        equivalent to the dot product and lets the SQL layer compare vectors
        with pgvector's cosine operator.

        Args:
            texts: The texts to encode.

        Returns:
            One 384-dimensional vector per input text, in the same order,
            converted from the model's tensor output to plain Python lists.
        """
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32
        ).tolist()

    @property
    def max_tokens(self) -> int:
        """int: The model's maximum input length in tokens."""
        return self._model.max_seq_length

    def count_tokens(self, text: str) -> int:
        """Count the tokens the model would consume for a text.

        Args:
            text: The text to measure.

        Returns:
            The length of the tokenizer's ``input_ids``, which is the exact
            count the model receives rather than a word-based estimate.
        """
        return len(self._model.tokenizer(text)["input_ids"])
