from app.domain.prompt import parse_citations
from typing import Protocol, TypeVar

class _HasSimilarity(Protocol):
    """Structural type of anything carrying a similarity score.

    Attributes:
        similarity: Cosine similarity against the query vector.
    """

    similarity: float

T = TypeVar("T", bound=_HasSimilarity)

def filter_by_threshold(items: list[T], threshold: float) -> list[T]:
    """Drop the retrieval results that are not close enough to be relevant.

    This is the first defense, and it runs before the model is called at all.
    A nearest-neighbor search always returns its ``top_k`` results, even for a
    question the corpus cannot answer, so the threshold is what separates
    "this is the closest passage" from "this passage is relevant".

    Args:
        items: Retrieval results, each carrying a ``similarity`` attribute.
        threshold: Minimum similarity a result must reach to be kept.

    Returns:
        The results scoring at or above the threshold, in their original
        order. An empty list means the question is treated as out of domain
        and the language model is never invoked.
    """
    return [
        item for item in items if item.similarity >= threshold
    ]

def validate_citations(answer: str, n_passages: int) -> bool:
    """Check that an answer cites real passages, and cites at least one.

    This is the second defense, and it runs after generation. The rule has
    two parts, which rule out different failures: requiring at least one
    citation rejects anything unverifiable, such as an answer that cites
    nothing or a refusal the model phrased in its own words, while requiring
    every cited number to exist rejects fabricated references.

    The empty-set guard is not redundant. ``set().issubset(anything)``
    returns ``True`` in Python, so without it an answer carrying no citations
    at all would pass validation and reach the client marked as grounded.

    Args:
        answer: The generated answer text.
        n_passages: Number of passages placed in the prompt, which defines
            the valid citation range ``1..n_passages``.

    Returns:
        ``True`` when the answer cites at least one passage and every cited
        number exists in the prompt, ``False`` otherwise.
    """
    cited = parse_citations(answer)

    if not cited:
        return False

    return cited.issubset(
        set(
            range(1, n_passages + 1)
        )
    )
