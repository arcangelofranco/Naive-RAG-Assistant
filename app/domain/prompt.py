import re
from app.domain.docs import context_header
from typing import Protocol

_CITATION_RE = re.compile(r"\[(\d+)\]")
_PASSAGE_RE = re.compile(r"^\[(\d+)\] ", re.MULTILINE)

class ContextChunk(Protocol):
    """Structural type of anything that can be placed in the prompt.

    Declared as a ``Protocol`` so that ``build_prompt`` accepts any object
    exposing these three attributes, rather than a specific class. In
    practice this is ``RetrievedChunk``, but test doubles satisfy it too
    without importing anything from the domain.

    Attributes:
        title: Title of the source document.
        section: Heading of the section the chunk came from.
        content: The chunk text.
    """

    title: str
    section: str
    content: str

def build_prompt(question: str, chunks: list[ContextChunk]) -> str:
    """Assemble the context block and the question into a single prompt.

    Passages are numbered starting from 1, in the order they are given, and
    each one is prefixed with its provenance header. That number is what the
    model is instructed to cite, and what ``validate_citations`` later checks
    the answer against.

    Args:
        question: The user question, passed through unchanged.
        chunks: Retrieved passages to place in the context, already filtered
            by similarity. Order matters, because position within a long
            context can affect how well the model uses a passage.

    Returns:
        The full prompt text, with a ``CONTEXT`` block followed by a
        ``QUESTION`` block.
    """
    passages = "\n\n".join(
        f"[{i}] {context_header(c.title, c.section)}\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )

    return f"CONTEXT:\n{passages}\n\nQUESTION:\n{question}"

def count_passages(prompt: str) -> int:
    """Count the numbered passages a prompt contains.

    It recovers the passage count from the prompt text itself, which is what
    lets the cache validate an answer without being handed the chunk list.
    Only lines beginning with ``[n] `` are counted, so a bracketed number
    appearing inside a passage body is not mistaken for a new passage.

    Args:
        prompt: A prompt produced by ``build_prompt``.

    Returns:
        The number of passages placed in the context block.
    """
    return len(_PASSAGE_RE.findall(prompt))

def parse_citations(answer: str) -> set[int]:
    """Extract the passage numbers cited in a generated answer.

    Args:
        answer: The raw text returned by the language model.

    Returns:
        The set of distinct numbers appearing in ``[n]`` form. It is empty
        when the answer cites nothing, which callers treat as a failure
        rather than as a trivially valid result.
    """
    return {
        int(n) for n in _CITATION_RE.findall(answer)
    }
