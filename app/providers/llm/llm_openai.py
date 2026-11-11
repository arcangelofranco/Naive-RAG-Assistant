import openai

from app.providers.llm.base import LLMProvider
from app.providers.llm.llm_exceptions import (
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMUnavailableError
)

class OpenAIProvider(LLMProvider):
    """Generates answers through an OpenAI-compatible chat endpoint.

    SDK-level retries are disabled, for the same reason as in the Gemini
    adapter: retry policy belongs to the decorator above, which reasons in
    terms of the project's own error taxonomy.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        base_url: str,
        temperature: float = 0.1,
        max_out_tokens: int = 800,
        timeout_sec: float = 30
    ) -> None:
        """Build the client and hold the generation parameters.

        Args:
            api_key: Credential for the target endpoint.
            model: Model identifier, for example ``qwen/qwen3.8-27b``.
            system_prompt: Instructions sent as the system message. An empty
                string suppresses the system message entirely.
            base_url: Endpoint root, for example
                ``https://api.groq.com/openai/v1``. It is required, since
                there is no meaningful default for a compatible backend.
            temperature: Sampling temperature, kept low for near
                deterministic output.
            max_out_tokens: Ceiling on the generated answer length.
            timeout_sec: Per-request timeout handed to the SDK.
        """
        self._model = model
        self._system_prompt = system_prompt
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_sec,
            max_retries=0
        )
        self._temperature = temperature
        self._max_out_tokens = max_out_tokens

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        """Build the message list for a chat completions call.

        Args:
            prompt: The prompt text to send as the user message.

        Returns:
            The system message followed by the user message, or the user
            message alone when no system prompt was configured. Some
            compatible backends reject an empty system message, so it is
            omitted rather than sent blank.
        """
        messages = []
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt
                }
            )
        messages.append({
            "role": "user",
            "content": prompt
        })
        return messages

    def generate(self, prompt: str) -> str:
        """Send a prompt to the endpoint and translate any SDK failure.

        Status codes are split at 500: server-side failures are transient
        and therefore worth retrying, while everything below is a problem
        with the request itself and will fail identically on a second try.

        Args:
            prompt: The prompt text to send.

        Returns:
            The generated answer text.

        Raises:
            LLMRateLimitError: If the endpoint answered with a rate limit,
                carrying the declared retry interval when present.
            LLMUnavailableError: On a timeout, a connection failure, a 5xx
                response, or an empty completion.
            LLMInvalidRequestError: On any other 4xx response, such as an
                invalid key or an unknown model name.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._messages(prompt),
                temperature=self._temperature,
                max_completion_tokens=self._max_out_tokens
            )
        except openai.RateLimitError as e:
            raise LLMRateLimitError(_retry_after_sec(e)) from e
        except openai.APITimeoutError as e:
            raise LLMUnavailableError(f"Timeout: {e}") from e
        except openai.APIConnectionError as e:
            raise LLMUnavailableError(f"Connection error: {e}") from e
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise LLMUnavailableError(str(e)) from e
            raise LLMInvalidRequestError(str(e)) from e

        content = response.choices[0].message.content
        if not content:
            raise LLMUnavailableError(
                f"{self._model} returned an empty response."
            )
        return content

def _retry_after_sec(e: openai.APIStatusError) -> float | None:
    """Read the ``retry-after`` header off a failed response.

    Args:
        e: The SDK error carrying the HTTP response.

    Returns:
        The declared wait in seconds, or ``None`` when the header is
        missing, unparseable, or the error carries no response at all.
    """
    try:
        val = e.response.headers.get("retry-after")
        return float(val) if val is not None else None
    except (AttributeError, ValueError, TypeError):
        return None
