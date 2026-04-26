"""Document chunking utilities.

The ingestion path keeps document structure first (pages / paragraph blocks),
then applies recursive chunking only when a block is too large. This avoids
splitting PDF tables and section lists into tiny fragments before retrieval.
"""

import re
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


def normalize_extracted_text(text: str) -> str:
    """Clean extractor artifacts while preserving meaningful document breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if not lines:
            continue
        paragraphs.append(" ".join(lines))

    return "\n\n".join(paragraphs).strip()


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """Split text into overlapping chunks.

    Args:
        text: Raw document text.
        chunk_size: Max characters per chunk (default from config).
        chunk_overlap: Overlap between chunks (default from config).

    Returns:
        List of chunk strings.
    """
    splitter = _build_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(normalize_extracted_text(text))


def chunk_document_blocks(
    blocks: List[Dict],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Dict]:
    """Chunk pre-extracted markdown blocks using structure-aware splitting.

    Args:
        blocks: Dicts with at least `text`.
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Chunk dicts with `text`, `page`, `block_type`, and sequence metadata.
    """
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    # Fallback splitter for sections that are still too large
    size_splitter = _build_splitter(chunk_size, chunk_overlap)
    
    # Semantic markdown splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    chunks: List[Dict] = []

    for block_index, block in enumerate(blocks):
        text = normalize_extracted_text(block.get("text", ""))
        if not text:
            continue

        # 1. Split semantically by markdown headers
        md_docs = markdown_splitter.split_text(text)
        
        # 2. For each semantic section, apply size constraints
        for doc in md_docs:
            split_texts = size_splitter.split_text(doc.page_content)
            
            for split_index, split_text in enumerate(split_texts):
                chunk_data = {
                    "text": split_text,
                    "page": block.get("page"),
                    "block_type": block.get("block_type", "markdown"),
                    "source_block_index": block_index,
                    "split_index": split_index,
                }
                # Inject semantic header metadata (e.g. {"Header 1": "Introduction"})
                chunk_data.update(doc.metadata)
                chunks.append(chunk_data)

    return chunks


def _build_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )
