class LLMRateLimitError(Exception):
    """Raised when the provider rejected the request over quota.

    Attributes:
        retry_interval_sec: Seconds the provider asked the caller to wait,
            taken from its ``Retry-After`` header, or ``None`` when the
            provider declared no interval. The API layer forwards it to the
            client on the ``429`` response.
    """

    def __init__(self, retry_after_sec: float | None = None) -> None:
        """Record the provider-declared retry interval, when there is one.

        Args:
            retry_after_sec: Seconds to wait before retrying, or ``None``
                when the provider gave no such hint.
        """
        self.retry_interval_sec = retry_after_sec
        super().__init__("LLM rate limit exceeded.")

class LLMUnavailableError(Exception):
    """Raised on a timeout, a connection failure or a 5xx from the provider.

    It represents a transient condition, so the retry decorator will attempt
    the call again before letting it reach the API layer as a ``503``.
    """

    pass

class LLMInvalidRequestError(Exception):
    """Raised when the provider rejected the request as malformed.

    Typical causes are an invalid API key, an unknown model name, or a
    request the provider could not parse. It is deliberately not retried,
    since repeating it would only waste time.
    """

    pass
