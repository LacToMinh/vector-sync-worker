"""
uploader.py — Manage the local ChromaDB vector store and Gemini embeddings for OptiBot.

Architecture (100% free):
  - Embeddings : Google Gemini "text-embedding-004"  (free, 1500 req/day)
  - Vector DB  : ChromaDB (runs in-process, persisted to ./chroma_db/)
  - Generation : Google Gemini "gemini-1.5-flash"    (free, 1500 req/day)

Each Markdown article is split into overlapping chunks so long articles are
handled correctly. Chunk metadata (article_url, title) is stored alongside each
vector so the chatbot can cite sources.

Chunking strategy:
  - Target chunk size : 600 words
  - Overlap           : 100 words  (~17 %)
  - Rationale         : Support articles are typically 200-1500 words. A 600-word
    window fits most articles in 1-2 chunks while the overlap preserves sentence
    context across boundaries. Small enough that retrieved chunks stay focused.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import chromadb
from google import genai
from google.genai import types as genai_types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "optibot-kb"
EMBED_MODEL = "gemini-embedding-001"   # confirmed available: supports embedContent
CHAT_MODEL = "gemini-2.5-flash"

CHUNK_SIZE_WORDS = 600
CHUNK_OVERLAP_WORDS = 100

# Gemini embedding free-tier rate limit: 15 RPM = 1 req per 4s minimum
# Use 5s to stay comfortably within limits (100 articles ≈ 8-10 min total)
EMBED_DELAY = 5.0  # seconds between embedding API calls


# ---------------------------------------------------------------------------
# Client setup
# ---------------------------------------------------------------------------

def _get_gemini_client() -> genai.Client:
    """Create and return a Gemini API client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def _get_chroma_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    Args:
        text:       Full article text (Markdown)
        chunk_size: Target chunk size in words
        overlap:    Number of words to repeat at chunk boundaries

    Returns:
        List of chunk strings
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]  # Short article — keep as single chunk

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_texts(texts: list[str], client: genai.Client) -> list[list[float]]:
    """
    Embed a list of text strings using Gemini embedding model.

    Handles free-tier rate limiting (15 RPM) with a delay between calls.
    Retries on 429 RESOURCE_EXHAUSTED with exponential backoff.
    """
    embeddings: list[list[float]] = []
    for i, text in enumerate(texts):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                result = client.models.embed_content(
                    model=EMBED_MODEL,
                    contents=text,
                    config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                )
                embeddings.append(result.embeddings[0].values)
                break  # success
            except Exception as exc:
                str_exc = str(exc)
                is_transient = any(err in str_exc for err in ["429", "503", "UNAVAILABLE", "Service Unavailable", "unavailable"])
                if is_transient and attempt < max_retries - 1:
                    wait = EMBED_DELAY * (2 ** attempt)  # exponential backoff
                    logger.warning(
                        "Transient error (%s). Waiting %.0fs before retry %d/%d…",
                        str_exc.split(".")[0], wait, attempt + 1, max_retries,
                    )
                    time.sleep(wait)
                else:
                    raise
        if i < len(texts) - 1:
            time.sleep(EMBED_DELAY)
    return embeddings


# ---------------------------------------------------------------------------
# Upload / delete
# ---------------------------------------------------------------------------

def upload_articles(articles: list[dict[str, Any]]) -> dict[str, int]:
    """
    Embed and upsert a batch of articles into the ChromaDB collection.

    Each article is split into overlapping chunks. Each chunk is stored with:
      - id          : "<article_id>_chunk_<n>"
      - document    : chunk text
      - metadata    : {article_id, article_url, title, chunk_index}

    Args:
        articles: List of dicts from scraper [{id, slug, path, content_hash, article_url}]

    Returns:
        Dict {article_id: chunk_count} for logging
    """
    collection = _get_chroma_collection()
    client = _get_gemini_client()
    stats: dict[str, int] = {}

    for article in articles:
        article_id = article["id"]
        file_path = Path(article["path"])

        if not file_path.exists():
            logger.warning("File not found, skipping: %s", file_path)
            continue

        full_text = file_path.read_text(encoding="utf-8")

        # Extract metadata from YAML frontmatter
        title = article.get("slug", article_id).replace("-", " ").title()
        article_url = article.get("article_url", "")

        # Quick title extraction from frontmatter
        title_match = re.search(r'^title:\s*"?([^"\n]+)"?', full_text, re.M)
        if title_match:
            title = title_match.group(1).strip()

        # Split into chunks
        chunks = _chunk_text(full_text, CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS)
        logger.info(
            "Article %s -> %d chunk(s) | %s", article_id, len(chunks), title[:50]
        )

        # Build IDs, documents, metadatas for this article
        chunk_ids = [f"{article_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "article_id": article_id,
                "article_url": article_url,
                "title": title,
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # Embed all chunks
        embeddings = _embed_texts(chunks, client)

        # Upsert into ChromaDB (idempotent — overwrites existing chunks)
        collection.upsert(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        stats[article_id] = len(chunks)

    total_chunks = sum(stats.values())
    logger.info(
        "Uploaded %d articles → %d total chunks indexed in ChromaDB",
        len(stats), total_chunks,
    )
    return stats


def delete_article_chunks(article_id: str) -> None:
    """
    Remove all chunks for a given article from ChromaDB.

    Called before re-uploading an updated article to avoid stale chunks.
    """
    _get_gemini_client()  # validate key early
    collection = _get_chroma_collection()

    # Query existing chunks for this article
    existing = collection.get(where={"article_id": article_id})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        logger.info(
            "Deleted %d stale chunks for article %s", len(existing["ids"]), article_id
        )


# ---------------------------------------------------------------------------
# High-level delta upload
# ---------------------------------------------------------------------------

def upload_delta(
    added: list[dict[str, Any]],
    updated: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Upload only new and changed articles.

    For updated articles: delete old chunks first, then upload fresh ones.

    Returns:
        Dict {article_id: chunk_count} for all articles processed
    """
    # Delete stale chunks for updated articles
    for article in updated:
        logger.info("Removing stale chunks for updated article: %s", article["id"])
        delete_article_chunks(article["id"])

    # Upload added + updated
    to_upload = added + updated
    if not to_upload:
        return {}

    return upload_articles(to_upload)


# ---------------------------------------------------------------------------
# Setup (idempotent — safe to call every run)
# ---------------------------------------------------------------------------

def setup() -> str:
    """
    Ensure ChromaDB collection exists and Gemini API key is valid.

    Returns:
        Collection name (for logging)
    """
    _get_gemini_client()  # validate key early
    collection = _get_chroma_collection()
    count = collection.count()
    logger.info(
        "ChromaDB collection '%s' ready — %d chunks currently indexed",
        COLLECTION_NAME, count,
    )
    return COLLECTION_NAME


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from dotenv import load_dotenv
    load_dotenv()
    name = setup()
    print(f"Collection: {name}")
    col = _get_chroma_collection()
    print(f"Chunks in store: {col.count()}")
