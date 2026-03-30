#!/usr/bin/env python3
"""Backfill cleaned text fields into the RSS dataset."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from text_cleaning import clean_text, clean_html_text, combine_article_text


def main() -> None:
    dataset_path = Path("dataset/rss_feeds_all.json")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    articles = json.loads(dataset_path.read_text(encoding="utf-8"))
    updated = 0

    for article in articles:
        raw_summary = article.get("summary", "")
        raw_content = article.get("content", [])

        cleaned_title = clean_text(article.get("title", "Untitled"))
        cleaned_source = clean_text(article.get("source", "Unknown"))
        cleaned_summary = clean_html_text(raw_summary)
        cleaned_content = combine_article_text(raw_summary, raw_content)

        if article.get("title") != cleaned_title:
            article["title"] = cleaned_title
            updated += 1
        if article.get("source") != cleaned_source:
            article["source"] = cleaned_source
            updated += 1

        if article.get("summary_text") != cleaned_summary:
            article["summary_text"] = cleaned_summary
            updated += 1
        if article.get("content_text") != cleaned_content:
            article["content_text"] = cleaned_content
            updated += 1

    dataset_path.write_text(
        json.dumps(articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Updated {len(articles)} articles, wrote {updated} field changes to {dataset_path}")


if __name__ == "__main__":
    main()
