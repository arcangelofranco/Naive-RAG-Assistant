from collections.abc import Callable

from app.providers.llm.base import LLMProvider

IsCacheable = Callable[[str, str], bool]

class CachingLLMProvider:
    """Caches answers in memory, keyed by the normalized prompt.

    The cache lives for the life of the process and is never shared between
    workers, which is why the API memoizes the provider stack in application
    state: rebuilding it per request would hand every request an empty
    cache.
    """

    def __init__(self, wrapped: LLMProvider, is_cacheable: IsCacheable) -> None:
        """Wrap a provider with a validity-aware cache.

        The predicate is injected rather than imported, which keeps this
        package a leaf. The cache needs to know nothing about citations or
        passages, it only asks whether an answer may be reused.

        Args:
            wrapped: The provider to call on a cache miss.
            is_cacheable: Predicate receiving the prompt and the answer, and
                returning whether that answer is worth storing. In practice
                it is the same grounding rule the query service applies.
        """
        self._wrapped = wrapped
        self._cache: dict[str, str] = {}
        self._is_cacheable = is_cacheable

    def generate(self, prompt: str) -> str:
        """Return a cached answer, or generate and conditionally store one.

        Keys are normalized by collapsing whitespace and lowercasing, so
        prompts differing only in formatting share an entry.

        Args:
            prompt: The prompt text.

        Returns:
            The cached answer when the prompt was seen before, otherwise the
            freshly generated one, which is returned whether or not it was
            considered worth caching.
        """
        key = " ".join(prompt.split()).lower()
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        answer = self._wrapped.generate(prompt)
        # An answer the service would discard must not be reused: otherwise a single
        # output without valid citations pins that question to a refusal for the
        # entire life of the process.
        if self._is_cacheable(prompt, answer):
            self._cache[key] = answer
        return answer
