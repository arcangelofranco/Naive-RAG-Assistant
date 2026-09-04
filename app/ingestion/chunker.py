import logging
from collections.abc import Callable

from app.domain.docs import Chunk, ParsedDoc, context_header, embedding_text

logger = logging.getLogger(__name__)

TargetTokens = 200
"""int: Maximum tokens a window may hold, header included."""

OverlapTokens = 40
"""int: Tokens carried from the end of a window into the start of the next.

Without an overlap, a statement straddling a window boundary would be split
across two chunks, and neither of them would carry the whole claim.
"""

CountTokens = Callable[[str], int]

def chunk_doc(
    doc: ParsedDoc,
    count_tokens: CountTokens,
    max_tokens: int
) -> list[Chunk]:
    """Split a parsed document into embeddable chunks.

    Sections are the unit of splitting, and a section short enough to fit
    the budget becomes a single chunk. Chunk indices run across the whole
    document rather than restarting per section, since they are the position
    stored in the database.

    Args:
        doc: The parsed document to split.
        count_tokens: Token counting function, which must come from the same
            model that will produce the vectors. Counting with one tokenizer
            and embedding with another would silently overflow the budget.
        max_tokens: The model's input limit, used as the safety ceiling that
            triggers a warning when a chunk exceeds it.

    Returns:
        The chunks in document order, each carrying its measured token count.

    Raises:
        ValueError: If the configured target does not leave room under the
            model limit, which would make every chunk overflow by design.
    """
    if TargetTokens >= max_tokens:
        raise ValueError(
            f"Target Tokens = {TargetTokens}"
        )

    chunks = []

    for section in doc.sections:
        header = context_header(doc.title, section.heading)
        for piece in _split_section(header, section.content, count_tokens):
            token_count = count_tokens(embedding_text(doc.title, section.heading, piece))
            if token_count > max_tokens:
                logger.warning(
                    f"Chunk truncated: {doc.id} - {section.heading} ({token_count} tokens)."
                )

            chunks.append(
                Chunk(
                    document_id=doc.id,
                    section=section.heading,
                    content=piece,
                    token_count=token_count,
                    chunk_index=len(chunks)
                )
            )

    return chunks

def _split_section(header: str, content: str, count_tokens: CountTokens) -> list[str]:
    """Pack a section's lines into overlapping windows under the budget.

    Lines are the atomic unit, so a single line longer than the budget is
    never broken up and produces an oversized chunk on its own. That is what
    the truncation warning in ``chunk_doc`` exists to surface.

    Args:
        header: The context header, whose token cost is subtracted from the
            budget up front because it is prepended at embedding time.
        content: The section body.
        count_tokens: Token counting function.

    Returns:
        The window texts, each one a run of lines joined by newlines, with
        consecutive windows sharing their overlap.
    """
    blocks = [line for line in content.split("\n") if line.strip()]
    budget = max(TargetTokens - count_tokens(header), 1)

    windows: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)
        if current and current_tokens + block_tokens > budget:
            windows.append(current)
            current, current_tokens = _tail_overlap(current, count_tokens)
        current.append(block)
        current_tokens += block_tokens

    if current:
        windows.append(current)

    return ["\n".join(w) for w in windows]

def _tail_overlap(blocks: list[str], count_tokens: CountTokens) -> tuple[list[str], int]:
    """Take the trailing lines of a window to seed the next one.

    Lines are collected from the end backwards until adding one more would
    exceed the overlap allowance. At least one line is always returned, even
    when it is on its own larger than the allowance, so that the overlap
    never comes back empty.

    Args:
        blocks: The lines of the window that just closed.
        count_tokens: Token counting function.

    Returns:
        A tuple of the carried-over lines, in their original order, and
        their combined token count.
    """
    tail: list[str] = []
    tail_tokens = 0
    for block in reversed(blocks):
        block_tokens = count_tokens(block)
        if tail and tail_tokens + block_tokens > OverlapTokens:
            break
        tail.insert(0, block)
        tail_tokens += block_tokens
    return tail, tail_tokens
