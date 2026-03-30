import feedparser
from typing import List, Dict
import re
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).resolve().parent
if str(DATASET_DIR) not in sys.path:
    sys.path.insert(0, str(DATASET_DIR))

from feed_config import DEFAULT_FEED_CONFIG_PATH, load_feed_configs


class NewsRSSFetcher:
    """Fetch news articles from various RSS feeds"""

    def __init__(self, feed_config_path: str | None = None):
        self.feed_config_path = feed_config_path or str(DEFAULT_FEED_CONFIG_PATH)
        self.feed_configs = load_feed_configs(self.feed_config_path)
        self.feeds = {feed["name"]: feed["url"] for feed in self.feed_configs}

    def fetch_feed(self, feed_config: Dict) -> List[Dict]:
        """Fetch and parse a single RSS feed"""
        feed_url = feed_config["url"]
        try:
            feed = feedparser.parse(feed_url)
            articles = []

            for entry in feed.entries[:10]:
                article = {
                    "title": entry.get("title", "No title"),
                    "link": entry.get("link", ""),
                    "published": entry.get("published", "Unknown date"),
                    "summary": entry.get("summary", "No summary"),
                    "author": entry.get("author", "Unknown"),
                    "source": feed_config["name"],
                    "category": feed_config.get("category", "general"),
                    "region": feed_config.get("region", "CN"),
                    "language": feed_config.get("language", "zh-CN"),
                    "source_type": feed_config.get("source_type", "media"),
                    "source_tags": feed_config.get("tags", []),
                }
                articles.append(article)

            return articles
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {str(e)}")
            return []

    def fetch_all_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch all configured RSS feeds"""
        all_news = {}

        for feed_config in self.feed_configs:
            source = feed_config["name"]
            print(f"Fetching from {source}...")
            articles = self.fetch_feed(feed_config)
            if articles:
                all_news[source] = articles

        return all_news

    def search_keywords(self, articles: Dict[str, List[Dict]], keywords: List[str]) -> List[Dict]:
        """Search for specific keywords in article titles"""
        matching_articles = []

        for source, article_list in articles.items():
            for article in article_list:
                title_lower = article["title"].lower()
                if any(keyword.lower() in title_lower for keyword in keywords):
                    matching_articles.append(article)

        return matching_articles

    def display_articles(self, articles: List[Dict], max_articles: int = 20):
        """Display articles in a formatted way"""
        print("\n" + "=" * 80)
        print(f"Found {len(articles)} matching articles")
        print("=" * 80 + "\n")

        for i, article in enumerate(articles[:max_articles], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Published: {article['published']}")
            print(f"   Link: {article['link']}")
            if article.get("summary"):
                summary = re.sub("<[^<]+?>", "", article["summary"])
                print(f"   Summary: {summary[:200]}...")
            print()


def main():
    fetcher = NewsRSSFetcher()

    print("Fetching news from RSS feeds...")
    all_news = fetcher.fetch_all_feeds()

    print("\n### LATEST BUSINESS NEWS ###")
    for source, articles in all_news.items():
        print(f"\n--- {source} ---")
        for article in articles[:3]:
            print(f"- {article['title']}")

    print("\n\n### SEARCHING FOR SPECIFIC TOPICS ###")
    keywords = ["财经", "市场", "AI", "创业", "融资", "科技", "美股", "商业"]
    matching = fetcher.search_keywords(all_news, keywords)
    fetcher.display_articles(matching, max_articles=10)

    print("\n\n### CUSTOM KEYWORD SEARCH ###")
    custom_keywords = input("Enter keywords to search (comma-separated): ").split(",")
    custom_keywords = [k.strip() for k in custom_keywords if k.strip()]

    if custom_keywords:
        custom_results = fetcher.search_keywords(all_news, custom_keywords)
        fetcher.display_articles(custom_results)


if __name__ == "__main__":
    main()
