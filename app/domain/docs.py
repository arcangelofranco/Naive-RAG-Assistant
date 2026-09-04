from dataclasses import dataclass

@dataclass
class Section:
    """A single ``##`` section of a source document.

    Attributes:
        heading: The section heading, without the leading ``##`` marker.
        content: The section body, stripped of surrounding whitespace.
    """

    heading: str
    content: str

@dataclass
class ParsedDoc:
    """A knowledge base document after frontmatter parsing and splitting.

    Attributes:
        id: Stable document identifier taken from the frontmatter, for
            example ``design-patterns/adapter``. It is the primary key in
            the ``documents`` table.
        title: Human readable document title from the frontmatter.
        category: Document category from the frontmatter, for example
            ``design-pattern``. It is an empty string when absent.
        source_path: Path of the ``.md`` file this document was read from.
        sections: The document body split on ``##`` headings, in order.
        raw_content: The untouched source Markdown, frontmatter included.
            It is persisted as a backup, since chunking is lossy.
    """

    id: str
    title: str
    category: str
    source_path: str
    sections: list[Section]
    raw_content: str

@dataclass
class Chunk:
    """An indexable fragment of a document, sized for the embedding model.

    Attributes:
        document_id: Identifier of the document this chunk belongs to.
        section: Heading of the section the chunk was cut from.
        content: The chunk text, without the context header. The header is
            added back at embedding time and at prompt build time, so
            storing it here would duplicate it in every passage.
        token_count: Token count of the text that was actually embedded,
            context header included, measured with the embedding model's
            own tokenizer.
        chunk_index: Zero-based position of the chunk within its document.
    """

    document_id: str
    section: str
    content: str
    token_count: int
    chunk_index: int

@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by a similarity search, with its score.

    Attributes:
        document_id: Identifier of the source document.
        title: Title of the source document.
        section: Heading of the section the chunk came from.
        chunk_index: Zero-based position of the chunk within its document.
        content: The chunk text, as stored at ingestion time.
        similarity: Cosine similarity against the query vector, in the
            ``[-1, 1]`` range. It expresses semantic closeness and must not
            be read as a probability or as a percentage of relevance.
    """

    document_id: str
    title: str
    section: str
    chunk_index: int
    content: str
    similarity: float


def context_header(title: str, section: str) -> str:
    """Build the provenance label that identifies where a chunk came from.

    This is the single place where the label format is defined. The chunker,
    the ingestor and the prompt builder all call it, so that the text fed to
    the embedding model and the text shown to the language model follow the
    same convention.

    Args:
        title: Title of the source document.
        section: Heading of the section the chunk was cut from.

    Returns:
        The provenance label, for example ``Adapter - Structure``.
    """
    return f"{title} - {section}"

def embedding_text(title: str, section: str, content: str) -> str:
    """Build the text to embed for a chunk, header included.

    A short chunk read in isolation can be ambiguous, since a passage such as
    ``"Pros and cons"`` gives no clue about the concept it describes.
    Prefixing the provenance header makes the resulting vector carry that
    context as well.

    Args:
        title: Title of the source document.
        section: Heading of the section the chunk was cut from.
        content: The chunk text, without the header.

    Returns:
        The header and the content joined by a newline, which is what the
        embedding provider receives.
    """
    return f"{context_header(title, section)}\n{content}"
