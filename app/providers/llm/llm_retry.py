import time
import logging

from app.providers.llm.base import LLMProvider
from app.providers.llm.llm_exceptions import LLMRateLimitError, LLMUnavailableError

logger = logging.getLogger(__name__)

class RetryingLLMProvider:
    """Retries transient provider failures with a linear backoff.

    It sits between the cache and the concrete adapter, which means a cache
    hit skips it entirely and only a real provider call can pay its cost.
    """

    def __init__(self, wrapped: LLMProvider, max_attempts: int = 3, delay: float = 1.0) -> None:
        """Wrap a provider with a retry policy.

        Args:
            wrapped: The provider to call, typically a concrete adapter.
            max_attempts: Total attempts, the first one included. The API
                layer surfaces the final failure as ``429`` or ``503``.
            delay: Base delay in seconds, multiplied by the attempt number
                to space out successive retries.
        """
        self._wrapped = wrapped
        self._max_attempts = max_attempts
        self._delay = delay

    def generate(self, prompt: str) -> str:
        """Call the wrapped provider, retrying transient failures.

        Args:
            prompt: The prompt text to forward unchanged.

        Returns:
            The answer from the first attempt that succeeds.

        Raises:
            LLMRateLimitError: If every attempt hit the provider quota. It
                reaches the API layer, which answers ``429`` and forwards
                the declared retry interval when there is one.
            LLMUnavailableError: If every attempt found the provider
                unreachable, or if the loop somehow ends without a result.
            LLMInvalidRequestError: Propagated immediately without retrying,
                since the request itself is the problem.
        """
        for attempt in range(1, self._max_attempts + 1):
            try:
                return self._wrapped.generate(prompt)
            except (LLMRateLimitError, LLMUnavailableError) as e:
                if attempt == self._max_attempts:
                    logger.warning(
                        f"{type(e).__name__}, attempts exhausted ({attempt}/{self._max_attempts})."
                    )
                    raise
                logger.warning(
                    f"{type(e).__name__}, attempt {attempt}/{self._max_attempts}."
                )
                time.sleep(_delay_for(e, attempt, self._delay))
        raise LLMUnavailableError()

def _delay_for(e: Exception, attempt: int, delay: float) -> float:
    """Decide how long to wait before the next attempt.

    A provider that declares its own retry interval wins over the local
    policy, since it knows when the quota actually resets. The attribute is
    read defensively rather than by type check, because only
    ``LLMRateLimitError`` carries one: ``LLMUnavailableError`` reaches this
    function too and always falls back to the linear backoff.

    Args:
        e: The transient error that caused the retry.
        attempt: Number of the attempt that just failed, starting at 1.
        delay: Base delay in seconds.

    Returns:
        The interval the provider declared, when there is one, otherwise
        the base delay multiplied by the attempt number.
    """
    declared = getattr(e, "retry_interval_sec", None)
    return declared if declared is not None else delay * attempt
