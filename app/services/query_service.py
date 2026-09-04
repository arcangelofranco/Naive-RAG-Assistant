from dataclasses import dataclass

from app.domain.docs import RetrievedChunk
from app.domain.grounding import filter_by_threshold, validate_citations
from app.domain.instructions import REFUSAL
from app.domain.prompt import build_prompt
from app.providers.embedding.base import EmbeddingProvider
from app.providers.llm.base import LLMProvider
from app.repositories.base import ChunkSearcher

@dataclass
class AskResult:
    """The outcome of a question, whether answered or refused.

    Attributes:
        answer: The grounded answer with inline citations, or the exact
            refusal string when the question could not be answered.
        sources: The passages supporting the answer. It is empty on a
            refusal, so that a rejected answer never ships citations.
        grounded: Whether the answer is supported by retrieved context.
    """

    answer: str
    sources: list[RetrievedChunk]
    grounded: bool


class QueryService:
    """Orchestrates retrieval, prompting, generation and validation.

    It depends on three ports and on no concrete implementation, so the
    embedding backend, the vector store and the language model can each be
    swapped without touching this class.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        chunk_searcher: ChunkSearcher,
        llm_provider: LLMProvider,
        top_k: int,
        similarity_threshold: float
    ) -> None:
        """Wire the service to its ports and grounding parameters.

        Args:
            embedding_provider: Encodes the question into a query vector. It
                must be the same model that produced the indexed vectors,
                which the composition root guarantees.
            chunk_searcher: Runs the nearest-neighbor lookup.
            llm_provider: Generates the answer. In practice this is the
                cache wrapping the retry layer wrapping a concrete adapter.
            top_k: Default number of chunks to retrieve per question.
            similarity_threshold: Minimum similarity for a chunk to be
                allowed into the prompt.
        """
        self._embedding_provider = embedding_provider
        self._chunk_searcher = chunk_searcher
        self._llm_provider = llm_provider
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def retrive(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Embed a question and fetch its nearest chunks.

        Results come back unfiltered, which is what makes this method usable
        on its own by the debug endpoint: it exposes what retrieval actually
        returned, scores included, without spending an LLM call.

        Note:
            The method name is misspelled. It is kept as is because the
            debug route calls it under this name.

        Args:
            question: The user question.
            top_k: Number of chunks to retrieve, or ``None`` to use the
                configured default.

        Returns:
            The nearest chunks, ordered by decreasing similarity.
        """
        query_emb = self._embedding_provider.embed([question])[0]
        return self._chunk_searcher.search_similar(query_emb, top_k or self._top_k)

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        """Answer a question, or refuse when it cannot be grounded.

        Both anti-hallucination defenses are applied here. The threshold runs
        before generation, so an out-of-domain question never reaches the
        model and never gets the chance to be answered from memory. Citation
        validation runs after generation and discards the whole answer when
        it cites nothing, or cites a passage that was never in the prompt.

        Args:
            question: The user question.
            top_k: Number of chunks to retrieve for this call, or ``None``
                to use the configured default.

        Returns:
            An ``AskResult`` carrying the cited answer with its sources, or
            the refusal string with no sources at all.

        Raises:
            LLMRateLimitError: If the provider quota was exhausted after
                every retry.
            LLMUnavailableError: If the provider stayed unreachable.
            LLMInvalidRequestError: If the provider rejected the request.
        """
        chunks = self.retrive(question, top_k)
        grounded_chunks = filter_by_threshold(chunks, self._similarity_threshold)
        if not grounded_chunks:
            return AskResult(answer=REFUSAL, sources=[], grounded=False)

        prompt = build_prompt(question, grounded_chunks)
        answer = self._llm_provider.generate(prompt).strip()

        if not validate_citations(answer, len(grounded_chunks)):
            return AskResult(answer=REFUSAL, sources=[], grounded=False)

        return AskResult(answer=answer, sources=grounded_chunks, grounded=True)
