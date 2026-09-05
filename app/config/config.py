from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Literal

class Settings(BaseSettings):
    """Runtime configuration for providers, storage and grounding.

    Unknown keys are ignored rather than rejected, so an ``.env`` file may
    carry variables meant for other tools.

    Attributes:
        llm_provider: Which LLM backend to build. ``google`` uses the Gemini
            SDK, ``openai`` uses the OpenAI SDK against any compatible
            endpoint. Any other value fails validation at startup, which is
            preferable to discovering it on the first request.
        llm_api_key: Provider credential. It is required by both backends,
            and the composition root raises a named error when it is empty.
        llm_model: Model identifier, for example ``gemini-3.5-flash``.
        llm_base_url: Endpoint root. Ignored by ``google`` and required by
            ``openai``.
        embedding_model: Name of the sentence-transformers model. It must
            produce 384-dimensional vectors to match the schema.
        embeddings_table: Table holding the vectors. Pointing it at a second
            table is what allows a new embedding space to be backfilled
            alongside the live one.
        database_url: PostgreSQL connection string. The default targets
            ``localhost``, which is correct when a script runs on the host;
            Compose overrides the host with ``db`` for the API container.
        similarity_threshold: Minimum cosine similarity for a chunk to enter
            the prompt. This value is a reasonable default rather than a
            calibrated one.
        top_k: Chunks retrieved per question, overridable per request.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: Literal["google", "openai"] = "google"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""

    embedding_model: str = ""
    embeddings_table: str = "chunk_embeddings"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/naive_rag_assistant"
    similarity_threshold: float = 0.5
    top_k: int = 5

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loading them once.

    The cache is what makes this usable as a FastAPI dependency without
    re-reading the environment on every request.

    Returns:
        The singleton settings instance.

    Raises:
        pydantic.ValidationError: If a value fails validation, for example
            an ``LLM_PROVIDER`` outside the supported pair.
    """
    return Settings()
