import re
from pathlib import Path
import frontmatter

from app.domain.docs import ParsedDoc, Section

_SectionRE = re.compile(r"^## (.+)$", re.MULTILINE)

def parse_doc(path: Path) -> ParsedDoc:
    """Read a knowledge base file and split it into sections.

    The raw file text is preserved on the returned document, because parsing
    is lossy: anything before the first ``##`` heading is dropped, and blank
    lines are discarded later by the chunker.

    Args:
        path: Path of the ``.md`` file to read, expected to carry ``id`` and
            ``title`` in its YAML frontmatter.

    Returns:
        The parsed document, with its sections in file order.

    Raises:
        KeyError: If the frontmatter has no ``id`` or no ``title``.
        UnicodeDecodeError: If the file is not valid UTF-8.
    """
    raw = path.read_text(encoding="utf-8")
    post = frontmatter.loads(raw)
    return ParsedDoc(
        id=post["id"],
        title=post["title"],
        category=post.get("category", ""),
        source_path=str(path),
        sections=_split_sections(post.content),
        raw_content=raw
    )


def _split_sections(body: str) -> list[Section]:
    """Cut a document body into sections on its ``##`` headings.

    Each section runs from the end of its heading to the start of the next
    one, or to the end of the body for the last section. Text preceding the
    first heading belongs to no section and is therefore never indexed.

    Args:
        body: The document body, frontmatter already removed.

    Returns:
        The sections in order. The list is empty when the body contains no
        ``##`` heading at all, which yields a document with no chunks.
    """
    matches = list(_SectionRE.finditer(body))
    sections = []

    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sections.append(Section(heading=match.group(1).strip(), content=body[start:end].strip()))

    return sections
