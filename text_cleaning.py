"""Utilities for cleaning RSS/article text before indexing."""

from __future__ import annotations

import html
import re
from typing import Any


_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_BLOCK_TAG_RE = re.compile(r"</?(?:article|aside|blockquote|br|div|figcaption|figure|h[1-6]|hr|li|p|section|table|tbody|td|th|thead|tr|ul|ol)[^>]*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v\u00a0\u3000]+")
_NEWLINES_RE = re.compile(r"\n{3,}")


def extract_text_content(raw: Any) -> str:
    """Extract text-like content from strings, feedparser content lists, or dicts."""
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    if isinstance(raw, dict):
        if isinstance(raw.get("value"), str):
            return raw["value"]
        return "\n".join(
            extract_text_content(value)
            for value in raw.values()
            if isinstance(value, (str, list, dict))
        )

    if isinstance(raw, (list, tuple, set)):
        parts = [extract_text_content(item) for item in raw]
        return "\n\n".join(part for part in parts if part)

    return str(raw)


def clean_html_text(raw: Any) -> str:
    """Convert raw HTML-ish RSS content into readable plain text."""
    text = extract_text_content(raw)
    if not text:
        return ""

    text = html.unescape(text)
    text = _COMMENT_RE.sub(" ", text)
    text = _SCRIPT_STYLE_RE.sub(" ", text)
    text = _BLOCK_TAG_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_text(raw: Any) -> str:
    """Normalize a plain string field such as title or source."""
    text = clean_html_text(raw)
    return text.replace("\n", " ").strip()


def combine_article_text(summary: Any, content: Any) -> str:
    """Build a cleaned article body from RSS summary/content fields."""
    summary_text = clean_html_text(summary)
    content_text = clean_html_text(content)

    if summary_text and content_text:
        if summary_text in content_text:
            return content_text
        if content_text in summary_text:
            return summary_text
        return f"{summary_text}\n\n{content_text}".strip()

    return summary_text or content_text
