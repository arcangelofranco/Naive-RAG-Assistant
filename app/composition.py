from collections.abc import Callable
from psycopg_pool import ConnectionPool

from app.config.config import Settings
from app.domain.grounding import validate_citations
from app.domain.instructions import SYSTEM_PROMPT
from app.domain.prompt import count_passages
from app.ingestion.ingestor import KnowledgeIngestor
from app.providers.embedding.base import EmbeddingProvider
from app.providers.embedding.token_counter import TokenCounter
from app.providers.embedding.embedding_minilm import MiniLMEmbeddingProvider
from app.providers.llm.base import LLMProvider
from app.providers.llm.llm_cache import CachingLLMProvider
from app.providers.llm.llm_gemini import GeminiProvider
from app.providers.llm.llm_openai import OpenAIProvider
from app.providers.llm.llm_retry import RetryingLLMProvider
from app.repositories.postgres import PostgresRepository
from app.services.query_service import QueryService

MAX_OUTPUT_TOKENS = 800
"""int: Ceiling on generated answer length, and so on per-request cost."""

GENERATION_TEMPERATURE = 0.1
"""float: Sampling temperature, low enough for near deterministic output.

These two live here as constants rather than as environment variables
because they are not a deployment concern. They express a decision about how
the system should behave: this is constrained extraction and synthesis, not
creative writing.
"""

def build_embedding_provider(settings: Settings) -> TokenCounter:
    """Build the embedding provider, loading the model into memory.

    Args:
        settings: Application settings, read for ``embedding_model``.

    Returns:
        A provider that both embeds text and exposes its tokenizer, which
        the chunker needs in order to size windows correctly.
    """
    return MiniLMEmbeddingProvider(settings.embedding_model)

def _require(settings: Settings, field: str) -> str:
    """Read a setting that must not be empty, naming it if it is.

    Args:
        settings: Application settings.
        field: Attribute name to read.

    Returns:
        The setting value.

    Raises:
        ValueError: If the value is empty, with a message naming both the
            missing variable and the provider that requires it.
    """
    value = getattr(settings, field)
    if not value:
        raise ValueError(f"{field.upper()} not set: required with LLM_PROVIDER={settings.llm_provider}.")
    return value


def _build_gemini(settings: Settings) -> LLMProvider:
    """Build the Gemini adapter from configuration.

    Args:
        settings: Application settings. The API key is mandatory, while the
            base URL is ignored by this backend.

    Returns:
        A bare Gemini adapter, without the retry and cache layers.

    Raises:
        ValueError: If the API key is missing.
    """
    return GeminiProvider(
        api_key=_require(settings, "llm_api_key"),
        model=settings.llm_model,
        system_prompt=SYSTEM_PROMPT,
        temperature=GENERATION_TEMPERATURE,
        max_out_tokens=MAX_OUTPUT_TOKENS
    )

def _build_openai(settings: Settings) -> LLMProvider:
    """Build the OpenAI-compatible adapter from configuration.

    Args:
        settings: Application settings. Both the API key and the base URL
            are mandatory here, since there is no default endpoint for a
            compatible backend.

    Returns:
        A bare OpenAI-compatible adapter, without the retry and cache
        layers.

    Raises:
        ValueError: If the API key or the base URL is missing.
    """
    return OpenAIProvider(
        api_key=_require(settings, "llm_api_key"),
        model=settings.llm_model,
        system_prompt=SYSTEM_PROMPT,
        base_url=_require(settings, "llm_base_url"),
        temperature=GENERATION_TEMPERATURE,
        max_out_tokens=MAX_OUTPUT_TOKENS,
    )

_LLM_BUILDERS: dict[str, Callable[[Settings], LLMProvider]] = {
    "google": _build_gemini,
    "openai": _build_openai
}
"""dict: Registry mapping a provider name to its factory.

Adding a backend means registering a factory here. Nothing above this module
changes, because everything above depends on ``LLMProvider``.
"""

def _is_reusable_answer(prompt: str, answer: str) -> bool:
    """Decide whether an answer is worth keeping in the cache.

    This is the same rule the query service applies, reused rather than
    reimplemented. Duplicating it inside the cache would create two
    definitions of a valid answer that could drift apart, and the cache
    would eventually store answers the service intends to discard.

    Passing it in from here is also what keeps the providers package a leaf,
    since the cache receives a plain callable and imports nothing from the
    domain.

    Args:
        prompt: The prompt that was sent, from which the passage count is
            recovered.
        answer: The answer the model returned.

    Returns:
        ``True`` when the answer would survive citation validation, and is
        therefore safe to serve again from cache.
    """
    return validate_citations(answer.strip(), count_passages(prompt))


def build_llm_provider(settings: Settings) -> LLMProvider:
    """Build the full LLM stack: cache over retry over the adapter.

    The cache is outermost on purpose, so a hit short-circuits the retry
    layer as well as the network call.

    Args:
        settings: Application settings, read for the provider selection and
            its credentials.

    Returns:
        The decorated provider. Callers cannot tell how many layers are
        present, since all three implement the same port.

    Raises:
        ValueError: If the configured provider has no registered factory, or
            if a credential the chosen backend requires is missing. Failing
            here, at startup, beats surfacing the same mistake later as a
            confusing runtime error.
    """
    builder = _LLM_BUILDERS.get(settings.llm_provider)
    if builder is None:
        raise ValueError(f"LLM_PROVIDER:{settings.llm_provider} has no builders, available: {", ".join(sorted(_LLM_BUILDERS))}")

    return CachingLLMProvider(RetryingLLMProvider(builder(settings)), _is_reusable_answer)

def build_knowledge_repository(settings: Settings, pool: ConnectionPool) -> PostgresRepository:
    """Build the PostgreSQL adapter for the configured embedding space.

    Args:
        settings: Application settings, read for the embedding model name
            and the vector table.
        pool: An open connection pool with pgvector registered.

    Returns:
        The repository, which implements all three storage ports.
    """
    return PostgresRepository(
        pool, settings.embedding_model, settings.embeddings_table
    )

def build_ingestor(settings: Settings, pool: ConnectionPool) -> KnowledgeIngestor:
    """Build the indexing pipeline.

    It deliberately goes through the same embedding factory the query path
    uses, which is the guarantee that documents and questions land in one
    vector space.

    Args:
        settings: Application settings.
        pool: An open connection pool with pgvector registered.

    Returns:
        An ingestor ready to index a set of Markdown files.
    """
    return KnowledgeIngestor(
        embedding_provider=build_embedding_provider(settings),
        repository=build_knowledge_repository(settings, pool)
    )

def build_query_service(settings: Settings, pool: ConnectionPool, embedding_provider: EmbeddingProvider, llm_provider: LLMProvider | None) -> QueryService:
    """Build the query pipeline from already constructed collaborators.

    The embedding provider and the LLM stack are passed in rather than built
    here, because both are expensive and are held for the life of the
    process: the embedding model takes seconds to load, and rebuilding the
    LLM stack would discard the cache on every request.

    Args:
        settings: Application settings, read for ``top_k`` and the
            similarity threshold.
        pool: An open connection pool with pgvector registered.
        embedding_provider: The process-wide embedding provider.
        llm_provider: The decorated LLM stack, or ``None`` for a service
            that only retrieves. Retrieval needs no credentials, so leaving
            it out is what keeps that path available without an API key.

    Returns:
        A query service wired to its ports.
    """
    return QueryService(
        embedding_provider=embedding_provider,
        chunk_searcher=build_knowledge_repository(settings, pool),
        llm_provider=llm_provider,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold
    )
