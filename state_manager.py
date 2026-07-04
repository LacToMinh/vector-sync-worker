"""
state_manager.py — Persist article state between runs to enable delta uploads.

State file structure (state.json):
{
    "<article_id>": {
        "hash": "<sha256 of full markdown content>",
        "vector_file_id": "<OpenAI file ID in vector store>",
        "slug": "<filename slug>",
        "uploaded_at": "<ISO timestamp>"
    },
    ...
}
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_FILE = Path(os.getenv("STATE_FILE", "state.json"))


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def load_state() -> dict[str, dict[str, Any]]:
    """
    Load the current state from state.json.

    Returns an empty dict if the file does not exist yet (first run).
    """
    if not STATE_FILE.exists():
        logger.info("No state.json found — starting fresh (first run).")
        return {}

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info("Loaded state for %d articles from %s", len(state), STATE_FILE)
        return state
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read state.json (%s) — starting fresh.", exc)
        return {}


def save_state(state: dict[str, dict[str, Any]]) -> None:
    """
    Persist the updated state dict to state.json.

    Writes atomically using a temp file + rename to avoid corruption on crash.
    """
    tmp_path = STATE_FILE.with_suffix(".json.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        tmp_path.replace(STATE_FILE)
        logger.info("State saved: %d articles tracked in %s", len(state), STATE_FILE)
    except OSError as exc:
        logger.error("Failed to save state.json: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Delta detection
# ---------------------------------------------------------------------------

def diff(
    old_state: dict[str, dict[str, Any]],
    new_articles: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Compare the new scraped articles against the previously persisted state.

    Delta detection is based on SHA-256 of the full Markdown content
    (including YAML frontmatter). This catches both body changes and
    metadata updates (e.g. updated_at timestamp).

    Args:
        old_state:    Dict loaded from state.json  {article_id: {hash, ...}}
        new_articles: List of dicts from scraper   [{id, slug, path, content_hash, ...}]

    Returns:
        (added, updated, skipped) — three lists of article dicts
    """
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for article in new_articles:
        article_id = article["id"]
        new_hash = article["content_hash"]

        if article_id not in old_state:
            added.append(article)
        elif old_state[article_id]["hash"] != new_hash:
            # Carry over the old vector_file_id so we can delete it before re-uploading
            article["old_vector_file_id"] = old_state[article_id].get("vector_file_id")
            updated.append(article)
        else:
            skipped.append(article)

    return added, updated, skipped


# ---------------------------------------------------------------------------
# State update helpers
# ---------------------------------------------------------------------------

def record_upload(
    state: dict[str, dict[str, Any]],
    article: dict[str, Any],
    vector_file_id: str,
) -> None:
    """
    Update the state dict in-place to record a successful upload.

    Call this after successfully uploading an article to the vector store.
    """
    state[article["id"]] = {
        "hash": article["content_hash"],
        "vector_file_id": vector_file_id,
        "slug": article["slug"],
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }


def record_skipped(
    state: dict[str, dict[str, Any]],
    article: dict[str, Any],
    old_state: dict[str, dict[str, Any]],
) -> None:
    """
    Preserve the existing state entry for a skipped (unchanged) article.

    This keeps vector_file_id intact so future runs can still delete/update it.
    """
    state[article["id"]] = old_state[article["id"]]
