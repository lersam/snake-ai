"""PDF ingestion: extract text, chunk and add to vector store."""
from __future__ import annotations

import logging

from typing import List

from .schemas import Config
from .embedder import get_default_embedder
from .vector_store import get_store_for_category, persist_store

logger = logging.getLogger(__name__)


def extract_text_from_pdf(path: str) -> str:
    try:
        import PyPDF2

        text_parts: List[str] = []
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except Exception as e:
        logger.error("Failed to extract PDF text from %s: %s", path, e)
        raise


def chunk_text(text: str, chunk_size_words: int, overlap_words: int) -> List[str]:
    words = text.split()
    chunks: List[str] = []
    i = 0
    n = len(words)
    while i < n:
        chunk = words[i : i + chunk_size_words]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_size_words - overlap_words)
    return chunks


def ingest_pdf_file(filepath: str, category: str, cfg: Config) -> int:
    """Extract text from PDF, chunk, embed and store. Returns number of chunks added."""
    if category not in __import__(".schemas", fromlist=["CATEGORY_LIST"]).CATEGORY_LIST:
        raise ValueError("Unknown category: %s" % (category,))
    text = extract_text_from_pdf(filepath)
    chunks = chunk_text(text, cfg.chunk_size_words, cfg.chunk_overlap_words)
    if not chunks:
        return 0
    embedder = get_default_embedder()
    embs = embedder.batch_embeddings(chunks)
    store = get_store_for_category(category, cfg.data_dir)
    store.add(chunks, embs)
    persist_store(category, store, cfg.data_dir)
    logger.info("Ingested %d chunks for %s", len(chunks), category)
    return len(chunks)
