import httpx
from google import genai
from google.genai import errors as genai_errors

from app.providers.llm.base import LLMProvider
from app.providers.llm.llm_exceptions import (
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMUnavailableError
)

_TransportErrors: tuple[type[Exception], ...] = (
    httpx.TimeoutException,
    httpx.ConnectError
)

class GeminiProvider(LLMProvider):
    """Generates answers through the Gemini API.

    SDK-level retries are disabled on purpose. Retrying is the job of the
    decorator that wraps this adapter, which can distinguish transient
    failures from permanent ones using the project's own error taxonomy.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.1,
        max_out_tokens: int = 800,
        timeout_sec: float = 30
    ) -> None:
        """Build the client and freeze the generation configuration.

        The system prompt is bound here rather than passed per call, so that
        every request goes out under the same grounding rules.

        Args:
            api_key: Credential for the Gemini API.
            model: Model identifier, for example ``gemini-3.5-flash``.
            system_prompt: Instructions sent as the system message.
            temperature: Sampling temperature. The default is low because
                the task is constrained extraction and synthesis rather
                than creative writing.
            max_out_tokens: Ceiling on the generated answer length, which
                also caps the cost of a single request.
            timeout_sec: Per-request timeout, converted to the
                milliseconds the SDK expects.
        """
        self._model = model
        self._system_prompt = system_prompt
        self._client = genai.Client(
            api_key=api_key
        )
        self._config = genai.types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_out_tokens,
            system_instruction=system_prompt,
            http_options=genai.types.HttpOptions(
                timeout=int(timeout_sec*1000),
                retry_options=genai.types.HttpRetryOptions(attempts=1)
            )
        )

    def generate(self, prompt: str) -> str:
        """Send a prompt to Gemini and translate any SDK failure.

        Args:
            prompt: The prompt text to send as user content.

        Returns:
            The generated answer text.

        Raises:
            LLMRateLimitError: If Gemini answered ``429``, carrying the
                declared retry interval when the response provided one.
            LLMInvalidRequestError: On any other client-side error, such as
                an invalid key or an unknown model name.
            LLMUnavailableError: On a timeout, a connection failure, or any
                other API error, which are all treated as transient.
        """
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=self._config
            )
        except genai_errors.ClientError as e:
            if e.code == 429:
                raise LLMRateLimitError(_retry_after_sec(e)) from e
            raise LLMInvalidRequestError(str(e)) from e
        except _TransportErrors as e:
            raise LLMUnavailableError(f"Timeout or Connection error: {e}") from e
        except genai_errors.APIError as e:
            raise LLMUnavailableError(str(e)) from e

        return response.text

def _retry_after_sec(e: genai_errors.APIError) -> float | None:
    """Read the ``Retry-After`` header off a failed Gemini response.

    The header is optional and its shape is not guaranteed, so every failure
    to read it is treated as an absent value rather than as an error worth
    propagating.

    Args:
        e: The SDK error carrying the HTTP response.

    Returns:
        The declared wait in seconds, or ``None`` when the header is
        missing, unparseable, or the error carries no response at all.
    """
    try:
        val = e.response.headers.get("Retry-After")
        return float(val) if val is not None else None
    except (AttributeError, ValueError, TypeError):
        return None
