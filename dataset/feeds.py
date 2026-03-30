import feedparser
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set
import re
import os
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from feed_config import DEFAULT_FEED_CONFIG_PATH, load_feed_configs
from text_cleaning import clean_text, clean_html_text, combine_article_text


class NewsRSSMonitor:
    """Monitor RSS feeds for new articles continuously"""

    def __init__(self, check_interval: int = 300, feed_config_path: str | None = None):
        """
        Initialize monitor
        check_interval: seconds between checks (default: 300 = 5 minutes)
        """
        self.check_interval = check_interval
        self.articals_loaded: Set[str] = set()
        self.cache_file = "./dataset/articals_loaded.json"
        self.feed_config_path = feed_config_path or str(DEFAULT_FEED_CONFIG_PATH)
        self.feed_configs = load_feed_configs(self.feed_config_path)
        self.feeds = {feed["name"]: feed["url"] for feed in self.feed_configs}

        # Load previously seen articles
        self.load_articals_loaded()

    def load_articals_loaded(self):
        """Load previously seen article IDs from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r") as f:
                    data = json.load(f)
                    self.articals_loaded = set(data.get("seen", []))
                print(f"Loaded {len(self.articals_loaded)} previously seen articles")
            except Exception as e:
                print(f"Error loading cache: {e}")

    def save_articals_loaded(self):
        """Save seen article IDs to cache"""
        try:
            with open(self.cache_file, "w") as f:
                json.dump({"seen": list(self.articals_loaded)}, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def fetch_feed(self, feed_config: Dict) -> List[Dict]:
        """Fetch and parse a single RSS feed, return only NEW articles from last 30 days"""
        feed_url = feed_config["url"]
        source = feed_config["name"]
        try:
            feed = feedparser.parse(feed_url)
            new_articles = []
            thirty_days_ago = datetime.now() - timedelta(days=30)

            for entry in feed.entries:
                article_id = entry.get("link", "")

                published_parsed = entry.get("published_parsed")
                if published_parsed:
                    article_date = datetime(*published_parsed[:6])
                    if article_date < thirty_days_ago:
                        continue

                if article_id and article_id not in self.articals_loaded:
                    raw_summary = entry.get("summary", "No summary")
                    raw_content = entry.get("content", [])
                    article = {
                        "id": article_id,
                        "title": clean_text(entry.get("title", "No title")),
                        "link": article_id,
                        "published": entry.get("published", "Unknown date"),
                        "updated": entry.get("updated", ""),
                        "summary": raw_summary,
                        "content": raw_content,
                        "summary_text": clean_html_text(raw_summary),
                        "content_text": combine_article_text(raw_summary, raw_content),
                        "author": clean_text(entry.get("author", "Unknown")),
                        "tags": entry.get("tags", []),
                        "category": entry.get("category", ""),
                        "source": clean_text(source),
                        "feed_url": feed_url,
                        "source_category": feed_config.get("category", "general"),
                        "source_region": feed_config.get("region", "CN"),
                        "source_language": feed_config.get("language", "zh-CN"),
                        "source_type": feed_config.get("source_type", "media"),
                        "source_priority": feed_config.get("priority", 0),
                        "source_tags": feed_config.get("tags", []),
                        "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    new_articles.append(article)
                    self.articals_loaded.add(article_id)

            return new_articles
        except Exception as e:
            print(f"Error fetching {source}: {str(e)}")
            return []

    def check_all_feeds(self, keywords: List[str] = None) -> List[Dict]:
        """Check all feeds for new articles"""
        all_new_articles = []

        for feed_config in self.feed_configs:
            new_articles = self.fetch_feed(feed_config)

            if keywords:
                filtered = []
                for article in new_articles:
                    title_lower = article["title"].lower()
                    summary_lower = article["summary"].lower()
                    if any(
                        kw.lower() in title_lower or kw.lower() in summary_lower
                        for kw in keywords
                    ):
                        filtered.append(article)
                all_new_articles.extend(filtered)
            else:
                all_new_articles.extend(new_articles)

        return all_new_articles

    def display_article(self, article: Dict):
        """Display a single article"""
        print("\n" + "=" * 80)
        print(f"NEW ARTICLE from {article['source']}")
        print("=" * 80)
        print(f"Title: {article['title']}")
        print(f"Published: {article['published']}")
        print(f"Link: {article['link']}")

        summary = re.sub("<[^<]+?>", "", article["summary"])
        print(f"Summary: {summary[:300]}...")
        print("=" * 80)

    def save_article_to_file(self, article: Dict, filename: str = "./dataset/rss_feeds_all.json"):
        """Save new article to a single JSON file"""
        try:
            articles = []
            if os.path.exists(filename):
                with open(filename, "r", encoding="utf-8") as f:
                    articles = json.load(f)

            articles.append(article)

            with open(filename, "w", encoding="utf-8") as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)

            txt_filename = "./dataset/published_dates.txt"
            with open(txt_filename, "a", encoding="utf-8") as f:
                f.write(f"{article['published']}\n")

            print(f"Saved to {filename} (Total: {len(articles)} articles)")
        except Exception as e:
            print(f"Error saving article: {e}")

    def monitor_continuous(self, keywords: List[str] = None, save_to_file: bool = True):
        """
        Continuously monitor feeds for new articles
        keywords: List of keywords to filter (None = all articles)
        save_to_file: Save new articles to JSON file
        """
        print("Starting continuous monitoring...")
        print(f"Checking every {self.check_interval} seconds")
        if keywords:
            print(f"Filtering for keywords: {', '.join(keywords)}")
        print("Press Ctrl+C to stop\n")

        try:
            iteration = 0
            while True:
                iteration += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n[{timestamp}] Check #{iteration}")

                new_articles = self.check_all_feeds(keywords)

                if new_articles:
                    print(f"Found {len(new_articles)} new article(s)!")
                    for article in new_articles:
                        self.display_article(article)
                        if save_to_file:
                            self.save_article_to_file(article)
                else:
                    print("No new articles found")

                self.save_articals_loaded()
                print(f"\nWaiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.save_articals_loaded()

    def check_once(self, keywords: List[str] = None, save_to_file: bool = True):
        """Check feeds once and return new articles"""
        print("Checking feeds for new articles...")
        new_articles = self.check_all_feeds(keywords)

        if new_articles:
            print(f"\nFound {len(new_articles)} new article(s)!\n")
            for article in new_articles:
                self.display_article(article)
                if save_to_file:
                    self.save_article_to_file(article)
        else:
            print("No new articles found")

        self.save_articals_loaded()
        return new_articles


def main():
    CHECK_INTERVAL = 300
    KEYWORDS = ["财经", "市场", "AI", "创业", "融资", "科技", "美股", "商业"]

    monitor = NewsRSSMonitor(check_interval=CHECK_INTERVAL)

    print("\nChoose mode:")
    print("1. Check once (run script manually each time)")
    print("2. Continuous monitoring (runs forever)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        monitor.check_once(keywords=KEYWORDS, save_to_file=True)
    else:
        monitor.monitor_continuous(keywords=KEYWORDS, save_to_file=True)


if __name__ == "__main__":
    main()
