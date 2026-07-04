"""
test_query.py — Demo script: ask OptiBot a question using Gemini + ChromaDB RAG.

Flow:
  1. Embed the question with Gemini text-embedding-004
  2. Query ChromaDB for the top-K most relevant chunks
  3. Build a prompt with retrieved chunks + system instructions
  4. Generate an answer with Gemini 1.5 Flash
  5. Print the answer with "Article URL:" citations

Usage:
    python test_query.py
    python test_query.py "How do I set up a playlist?"

Take a screenshot of the output for the deliverable.
"""

import os
import sys
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = "optibot-kb"
EMBED_MODEL = "gemini-embedding-001"   # confirmed available
CHAT_MODEL = "gemini-2.5-flash"
TOP_K = 5  # Number of chunks to retrieve

SYSTEM_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.
• Tone: helpful, factual, concise.
• Only answer using the uploaded docs.
• Max 5 bullet points; else link to the doc.
• Cite up to 3 "Article URL:" lines per reply."""

DEFAULT_QUESTION = "How do I add a YouTube video?"


# ---------------------------------------------------------------------------
# RAG pipeline
# ---------------------------------------------------------------------------

def ask(question: str) -> None:
    """Retrieve relevant chunks and generate an answer with Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # Step 1: Embed the question
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}\n")
    print("🔍 Searching knowledge base...")

    embed_result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=question,
        config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    q_embedding = embed_result.embeddings[0].values

    # ------------------------------------------------------------------
    # Step 2: Query ChromaDB for top-K chunks
    # ------------------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("❌ Collection not found. Run `python main.py` first to populate the knowledge base.")
        sys.exit(1)

    chunk_count = collection.count()
    if chunk_count == 0:
        print("❌ Knowledge base is empty. Run `python main.py` first.")
        sys.exit(1)

    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(TOP_K, chunk_count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ------------------------------------------------------------------
    # Step 3: Build RAG prompt
    # ------------------------------------------------------------------
    context_parts: list[str] = []
    seen_urls: set[str] = set()
    cited_urls: list[str] = []

    for chunk, meta, dist in zip(chunks, metadatas, distances):
        url = meta.get("article_url", "")
        title = meta.get("title", "")
        relevance = 1 - dist  # cosine distance → similarity

        context_parts.append(
            f"--- Source: {title} ---\n"
            f"Article URL: {url}\n"
            f"Relevance: {relevance:.2f}\n\n"
            f"{chunk}\n"
        )

        if url and url not in seen_urls:
            seen_urls.add(url)
            cited_urls.append(url)

    context = "\n".join(context_parts)

    full_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== Context from Knowledge Base ===\n"
        f"{context}\n"
        f"=== End of Context ===\n\n"
        f"User question: {question}\n\n"
        f"Answer (cite Article URLs from the context above):"
    )

    # ------------------------------------------------------------------
    # Step 4: Generate answer with Gemini
    # ------------------------------------------------------------------
    print(f"🤖 Generating answer with Gemini 2.5 Flash...\n")

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=CHAT_MODEL,
                contents=full_prompt,
            )
            answer = response.text
            break
        except Exception as exc:
            str_exc = str(exc)
            is_transient = any(err in str_exc for err in ["429", "503", "UNAVAILABLE", "Service Unavailable", "unavailable"])
            if is_transient and attempt < max_retries - 1:
                wait = 3.0 * (2 ** attempt)
                print(f"⚠️  Google API busy/rate-limited. Retrying in {wait:.0f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

    # ------------------------------------------------------------------
    # Step 5: Print result
    # ------------------------------------------------------------------
    print("Answer:")
    print("-" * 60)
    print(answer)

    # Print any cited URLs not already in the answer body
    urls_missing = [u for u in cited_urls if u not in answer]
    if urls_missing:
        print("\nAdditional sources:")
        for url in urls_missing[:3]:
            print(f"  Article URL: {url}")

    print(f"\n{'='*60}")
    print(f"Retrieved {len(chunks)} chunks | {len(seen_urls)} unique source articles")
    print("Screenshot the above output for the deliverable ✅")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    ask(question)
