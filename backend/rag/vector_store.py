"""
RAG Vector Store for ICU Diagnostic Risk Assistant.

Indexes medical guideline PDFs into ChromaDB for semantic retrieval,
enabling evidence-backed clinical decision support.
"""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

from backend.config import (
    GUIDELINES_DIR,
    CHROMA_DIR,
    RAG_TOP_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared embedding function – reused across build and query paths
# ---------------------------------------------------------------------------
_EMBEDDING_FN = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

_COLLECTION_NAME = "medical_guidelines"


# ---------------------------------------------------------------------------
# 1. PDF extraction
# ---------------------------------------------------------------------------
def extract_text_from_pdfs() -> list[dict]:
    """Read every PDF in GUIDELINES_DIR and return page-level text chunks.

    Returns a list of dicts: ``{text: str, source: str, page: int}``.
    Pages that fail to parse are silently skipped.
    """
    documents: list[dict] = []
    pdf_paths = sorted(Path(GUIDELINES_DIR).glob("*.pdf"))

    if not pdf_paths:
        logger.warning("No PDF files found in %s", GUIDELINES_DIR)
        return documents

    for pdf_path in pdf_paths:
        try:
            reader = PdfReader(str(pdf_path))
        except Exception:
            logger.warning("Could not open PDF: %s – skipping", pdf_path.name)
            continue

        for page_num, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                logger.warning(
                    "Failed to extract page %d from %s – skipping",
                    page_num + 1,
                    pdf_path.name,
                )
                continue

            text = text.strip()
            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_num + 1,  # 1-indexed for human readability
                }
            )

    logger.info(
        "Extracted %d pages from %d PDFs", len(documents), len(pdf_paths)
    )
    return documents


# ---------------------------------------------------------------------------
# 2. Chunking
# ---------------------------------------------------------------------------
def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split page-level text into overlapping character-level chunks.

    Each chunk retains its ``source``, ``page``, and a ``chunk_index``
    (sequential within the entire corpus).
    """
    chunks: list[dict] = []
    chunk_index = 0

    for doc in documents:
        text = doc["text"]
        # Normalise whitespace
        text = " ".join(text.split())

        if not text:
            continue

        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source": doc["source"],
                        "page": doc["page"],
                        "chunk_index": chunk_index,
                    }
                )
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    logger.info("Created %d chunks from %d pages", len(chunks), len(documents))
    return chunks


# ---------------------------------------------------------------------------
# 3. Build / load vector store
# ---------------------------------------------------------------------------
def build_vector_store(force_rebuild: bool = False) -> chromadb.Collection:
    """Build or load the ChromaDB collection of medical guideline chunks.

    If the persistent store already contains data and *force_rebuild* is
    ``False``, the existing collection is returned without re-indexing.
    """
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Try to load an existing collection
    if not force_rebuild:
        try:
            collection = client.get_collection(
                name=_COLLECTION_NAME,
                embedding_function=_EMBEDDING_FN,
            )
            if collection.count() > 0:
                logger.info(
                    "Loaded existing vector store with %d chunks",
                    collection.count(),
                )
                return collection
        except Exception:
            # Collection doesn't exist yet – will build below
            pass

    # --- Full (re-)build path ---
    logger.info("Building vector store from guideline PDFs …")

    # Delete stale collection if doing a rebuild
    try:
        client.delete_collection(name=_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_EMBEDDING_FN,
    )

    documents = extract_text_from_pdfs()
    if not documents:
        logger.warning("No documents extracted – vector store will be empty")
        return collection

    chunks = chunk_documents(documents)
    if not chunks:
        logger.warning("No chunks produced – vector store will be empty")
        return collection

    # ChromaDB has a per-call batch limit; add in slices of 5 000
    batch_size = 5000
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        collection.add(
            ids=[f"chunk_{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "page": c["page"],
                    "chunk_index": c["chunk_index"],
                }
                for c in batch
            ],
        )

    logger.info(
        "Vector store built – %d chunks indexed", collection.count()
    )
    return collection


# ---------------------------------------------------------------------------
# 4. Query
# ---------------------------------------------------------------------------
def query_guidelines(
    query: str, n_results: int = RAG_TOP_K
) -> list[dict]:
    """Search the vector store and return the most relevant guideline chunks.

    Each result is a dict with ``text``, ``source``, ``page``, and
    ``relevance_score`` (1 − ChromaDB distance, higher is better).
    """
    collection = build_vector_store()

    # Clamp n_results to the number of stored documents
    total = collection.count()
    if total == 0:
        logger.warning("Vector store is empty – no results to return")
        return []
    n_results = min(n_results, total)

    results = collection.query(query_texts=[query], n_results=n_results)

    output: list[dict] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(documents, metadatas, distances):
        output.append(
            {
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", 0),
                "relevance_score": round(1.0 - dist, 4),
            }
        )

    return output


# ---------------------------------------------------------------------------
# 5. Contextualised guideline retrieval for risk flags
# ---------------------------------------------------------------------------
def get_guideline_context(risk_flags: list[str]) -> str:
    """Retrieve and format guideline excerpts relevant to *risk_flags*.

    Duplicate chunks (same text) are removed.  The output is a
    ready-to-inject context string with source citations.
    """
    if not risk_flags:
        return ""

    seen_texts: set[str] = set()
    unique_results: list[dict] = []

    for flag in risk_flags:
        hits = query_guidelines(flag)
        for hit in hits:
            if hit["text"] not in seen_texts:
                seen_texts.add(hit["text"])
                unique_results.append(hit)

    if not unique_results:
        return "No relevant guideline excerpts found."

    # Sort by relevance (best first)
    unique_results.sort(key=lambda r: r["relevance_score"], reverse=True)

    lines: list[str] = []
    for r in unique_results:
        source_label = r["source"].replace(".pdf", "").replace("-", " ")
        lines.append(
            f'[Source: {source_label}, Page {r["page"]}]\n'
            f'"{r["text"]}"'
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 6. Convenience initialiser
# ---------------------------------------------------------------------------
def initialize_rag() -> chromadb.Collection:
    """Build or load the RAG vector store.  Call at application startup."""
    logger.info("Initialising RAG vector store …")
    collection = build_vector_store()
    logger.info(
        "RAG ready – %d chunks available for retrieval", collection.count()
    )
    return collection
