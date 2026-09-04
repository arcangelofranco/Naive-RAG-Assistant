from typing import Protocol

class LLMProvider(Protocol):
    """Generates an answer from a fully assembled prompt."""

    def generate(self, prompt: str) -> str:
        """Send a prompt to the model and return its answer.

        The system prompt is not passed here. Adapters hold it from
        construction time, so every caller of this port gets the same
        grounding rules applied.

        Args:
            prompt: The prompt text produced by ``build_prompt``.

        Returns:
            The generated answer, as raw text.

        Raises:
            LLMRateLimitError: If the provider quota was exceeded.
            LLMUnavailableError: If the provider timed out, was unreachable,
                or answered with a server-side error.
            LLMInvalidRequestError: If the request itself was rejected, for
                example because of an invalid key or an unknown model name.
        """
        pass
