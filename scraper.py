"""
scraper.py — Fetch OptiSigns support articles via Zendesk Help Center API
and convert each article's HTML body to clean Markdown.

Zendesk Help Center API is public (no auth required for public help centers).
Docs: https://developer.zendesk.com/api-reference/help_center/help-center-api/articles/
"""

import hashlib
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ZENDESK_DOMAIN = os.getenv("ZENDESK_DOMAIN", "support.optisigns.com")
BASE_API_URL = f"https://{ZENDESK_DOMAIN}/api/v2/help_center"
ARTICLES_DIR = Path("articles")
REQUEST_DELAY = 0.3  # seconds between API requests (be polite)
PER_PAGE = 100  # max allowed by Zendesk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(title: str) -> str:
    """Convert an article title to a filesystem-safe slug."""
    slug = title.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80]  # cap length


def _clean_html(html_body: str) -> str:
    """
    Remove noisy elements (nav, script, style, footer, ads) from raw HTML
    before converting to Markdown.
    """
    soup = BeautifulSoup(html_body, "html.parser")

    # Remove noisy tags entirely
    for tag in soup(["nav", "script", "style", "footer", "aside", "form", "button"]):
        tag.decompose()

    # Remove elements that are typically ads or navigation breadcrumbs
    for cls in ["breadcrumb", "feedback", "vote", "share", "sidebar", "related"]:
        for el in soup.find_all(class_=re.compile(cls, re.I)):
            el.decompose()

    return str(soup)


def _html_to_markdown(html: str, article_url: str) -> str:
    """
    Convert HTML string to clean Markdown.
    Preserves: headings, code blocks, ordered/unordered lists, links.
    """
    clean = _clean_html(html)
    markdown = md(
        clean,
        heading_style="ATX",      # use # ## ### style
        bullets="-",              # consistent bullet character
        code_language="",         # don't guess code language
        strip=["img"],            # images don't add value in a text KB
        # links are preserved by default — no extra flag needed
    )
    # Collapse 3+ consecutive blank lines into 2
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def _sha256(text: str) -> str:
    """Return hex SHA-256 digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_frontmatter(article: dict[str, Any]) -> str:
    """Build YAML frontmatter header for a Markdown file."""
    title = article["title"].replace('"', '\\"')
    return (
        "---\n"
        f'title: "{title}"\n'
        f'article_url: {article["html_url"]}\n'
        f'article_id: {article["id"]}\n'
        f'updated_at: {article["updated_at"]}\n'
        "---\n\n"
    )


# ---------------------------------------------------------------------------
# Core scraping logic
# ---------------------------------------------------------------------------

def fetch_articles(min_articles: int = 30) -> list[dict[str, Any]]:
    """
    Fetch raw articles from the Zendesk Help Center API.

    Returns a list of raw article dicts from the API.
    Paginates automatically until `min_articles` are collected or all pages exhausted.
    """
    articles: list[dict[str, Any]] = []
    url: str | None = f"{BASE_API_URL}/articles.json?per_page={PER_PAGE}"

    print(f"Fetching articles from {ZENDESK_DOMAIN}...")

    while url and len(articles) < min_articles:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("articles", [])
        articles.extend(batch)
        url = data.get("next_page")  # None when there are no more pages

        print(f"  Fetched {len(articles)} articles so far (page done, next: {'yes' if url else 'no'})")
        time.sleep(REQUEST_DELAY)

    print(f"Total articles fetched: {len(articles)}")
    return articles


def scrape_and_save(min_articles: int = 30) -> list[dict[str, Any]]:
    """
    Main entry point for the scraper.

    1. Fetches raw articles from Zendesk API.
    2. Converts each to clean Markdown.
    3. Writes each to ./articles/<slug>.md with YAML frontmatter.
    4. Returns metadata list used by state_manager and uploader.

    Returns:
        List of dicts: [{id, slug, path, updated_at, content_hash}]
    """
    ARTICLES_DIR.mkdir(exist_ok=True)
    raw_articles = fetch_articles(min_articles)

    results: list[dict[str, Any]] = []
    seen_slugs: dict[str, int] = {}

    for article in tqdm(raw_articles, desc="Converting articles"):
        article_id = str(article["id"])
        title = article.get("title", "untitled")
        html_body = article.get("body") or ""
        article_url = article.get("html_url", "")
        updated_at = article.get("updated_at", "")

        # Skip articles with empty body
        if not html_body.strip():
            continue

        # Build a unique slug (handle duplicate titles)
        base_slug = _slugify(title)
        if not base_slug:
            base_slug = f"article-{article_id}"

        count = seen_slugs.get(base_slug, 0)
        slug = base_slug if count == 0 else f"{base_slug}-{count}"
        seen_slugs[base_slug] = count + 1

        # Convert HTML → Markdown
        markdown_body = _html_to_markdown(html_body, article_url)
        frontmatter = _build_frontmatter(article)
        full_content = frontmatter + markdown_body

        # Write to disk
        file_path = ARTICLES_DIR / f"{slug}.md"
        file_path.write_text(full_content, encoding="utf-8")

        content_hash = _sha256(full_content)

        results.append(
            {
                "id": article_id,
                "slug": slug,
                "path": str(file_path),
                "updated_at": updated_at,
                "content_hash": content_hash,
                "article_url": article_url,
            }
        )

    print(f"Saved {len(results)} articles to ./{ARTICLES_DIR}/")
    return results


# ---------------------------------------------------------------------------
# CLI entry point (for standalone testing)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    articles = scrape_and_save(min_articles=30)
    print(f"\nDone. {len(articles)} articles saved.")
    if articles:
        print(f"Sample: {articles[0]['path']} (hash: {articles[0]['content_hash'][:12]}...)")
