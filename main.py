"""
main.py — OptiBot knowledge-base sync job (Gemini + ChromaDB edition).

Orchestrates the full pipeline:
  1. Load previous run state (state.json)
  2. Scrape ≥ 30 articles from support.optisigns.com
  3. Diff against old state (added / updated / skipped) via SHA-256
  4. Embed and upsert only the delta into ChromaDB via Gemini API
  5. Save updated state

Exit codes:
  0 — success
  1 — error

Environment variables (set in .env or pass as -e to Docker):
  GEMINI_API_KEY   — required (free at https://aistudio.google.com/apikey)
  ZENDESK_DOMAIN   — optional, default: support.optisigns.com
  MIN_ARTICLES     — optional, default: 30
"""

import logging
import os
import sys
import time

from dotenv import load_dotenv

# Load .env file when running locally (no-op in Docker if file is absent)
load_dotenv()

import scraper
import state_manager
import uploader

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run() -> None:
    """Execute the full scrape → diff → embed & upload → persist cycle."""
    start = time.time()
    min_articles = int(os.getenv("MIN_ARTICLES", "30"))

    logger.info("=== OptiBot Sync Job starting ===")

    # ------------------------------------------------------------------
    # Step 1: Ensure ChromaDB collection exists
    # ------------------------------------------------------------------
    logger.info("Step 1/4 — Setting up ChromaDB vector store…")
    collection_name = uploader.setup()
    logger.info("Vector store: '%s' (ChromaDB, local)", collection_name)

    # ------------------------------------------------------------------
    # Step 2: Load old state
    # ------------------------------------------------------------------
    logger.info("Step 2/4 — Loading previous state…")
    old_state = state_manager.load_state()

    # ------------------------------------------------------------------
    # Step 3: Scrape articles
    # ------------------------------------------------------------------
    logger.info("Step 3/4 — Scraping articles (min: %d)…", min_articles)
    new_articles = scraper.scrape_and_save(min_articles=min_articles)

    # ------------------------------------------------------------------
    # Step 4: Diff
    # ------------------------------------------------------------------
    added, updated, skipped = state_manager.diff(old_state, new_articles)
    logger.info(
        "Delta: %d added | %d updated | %d skipped",
        len(added), len(updated), len(skipped),
    )

    # ------------------------------------------------------------------
    # Step 5: Upload delta (embed + upsert to ChromaDB)
    # ------------------------------------------------------------------
    logger.info("Step 4/4 — Embedding and indexing delta articles…")
    chunk_stats: dict[str, int] = {}

    if added or updated:
        chunk_stats = uploader.upload_delta(added, updated)
        total_chunks = sum(chunk_stats.values())
        logger.info(
            "Indexed %d articles → %d chunks total",
            len(chunk_stats), total_chunks,
        )
    else:
        logger.info("Nothing to upload — all articles unchanged.")

    # ------------------------------------------------------------------
    # Step 6: Persist updated state
    # ------------------------------------------------------------------
    new_state: dict = {}

    for article in added:
        # Use chunk count as a proxy for "vector_file_id" in state
        chunk_count = chunk_stats.get(article["id"], 0)
        state_manager.record_upload(new_state, article, f"chroma-chunks-{chunk_count}")

    for article in updated:
        chunk_count = chunk_stats.get(article["id"], 0)
        state_manager.record_upload(new_state, article, f"chroma-chunks-{chunk_count}")

    for article in skipped:
        state_manager.record_skipped(new_state, article, old_state)

    state_manager.save_state(new_state)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start
    total_chunks_uploaded = sum(chunk_stats.values())
    summary = (
        f"Scraped: {len(new_articles)} articles | "
        f"Added: {len(added)} | "
        f"Updated: {len(updated)} | "
        f"Skipped: {len(skipped)} | "
        f"Chunks indexed: {total_chunks_uploaded} | "
        f"Duration: {elapsed:.1f}s"
    )
    logger.info("=== %s ===", summary)
    print(f"\n✅  {summary}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        run()
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Sync job failed: %s", exc)
        sys.exit(1)
