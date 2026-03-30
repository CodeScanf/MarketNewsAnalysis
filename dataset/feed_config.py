"""Utilities for loading RSS feed source configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_FEED_CONFIG_PATH = Path(__file__).with_name("rss_sources.json")


def load_feed_configs(
    config_path: str | Path | None = None,
    *,
    include_disabled: bool = False,
) -> list[dict[str, Any]]:
    """Load and normalize RSS feed definitions from JSON."""
    path = Path(config_path) if config_path else DEFAULT_FEED_CONFIG_PATH
    with path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    raw_feeds = raw_data.get("feeds", raw_data)
    if not isinstance(raw_feeds, list):
        raise ValueError(f"Feed config at {path} must contain a 'feeds' list")

    feeds: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    for index, item in enumerate(raw_feeds, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Feed entry #{index} in {path} must be an object")

        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        if not name or not url:
            raise ValueError(f"Feed entry #{index} in {path} must include non-empty 'name' and 'url'")
        if name in seen_names:
            raise ValueError(f"Duplicate feed name '{name}' found in {path}")
        seen_names.add(name)

        feed = {
            "name": name,
            "url": url,
            "category": str(item.get("category", "general")).strip() or "general",
            "region": str(item.get("region", "CN")).strip() or "CN",
            "language": str(item.get("language", "zh-CN")).strip() or "zh-CN",
            "source_type": str(item.get("source_type", "media")).strip() or "media",
            "enabled": bool(item.get("enabled", True)),
            "priority": int(item.get("priority", 0)),
            "tags": [str(tag).strip() for tag in item.get("tags", []) if str(tag).strip()],
        }
        feeds.append(feed)

    if not include_disabled:
        feeds = [feed for feed in feeds if feed["enabled"]]

    return feeds


def load_feed_map(config_path: str | Path | None = None) -> dict[str, str]:
    """Load enabled feed definitions as a name->url mapping."""
    return {feed["name"]: feed["url"] for feed in load_feed_configs(config_path)}
