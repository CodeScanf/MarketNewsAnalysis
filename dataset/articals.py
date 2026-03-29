import feedparser
from typing import List, Dict
import re


class NewsRSSFetcher:
    """Fetch news articles from various RSS feeds"""

    def __init__(self):
        # Configured Chinese finance, business, and tech RSS feeds
        self.feeds = {
            "叶檀财经": "https://plink.anyfeeder.com/weixin/tancaijing",
            "华尔街见闻": "https://plink.anyfeeder.com/weixin/wallstreetcn",
            "财新网": "https://plink.anyfeeder.com/weixin/caix",
            "第一财经周刊": "https://plink.anyfeeder.com/weixin/CBNweekly",
            "经济观察网": "https://plink.anyfeeder.com/eeo",
            "财富中文网": "https://plink.anyfeeder.com/fortunechina",
            "路透中文": "https://plink.anyfeeder.com/reuters/cn",
            "雪球热门话题": "https://plink.anyfeeder.com/xueqiu/hot",
            "36氪": "https://plink.anyfeeder.com/36kr",
            "虎嗅": "https://plink.anyfeeder.com/weixin/huxiu",
            "钛媒体": "https://plink.anyfeeder.com/tmtpost",
            "界面新闻": "https://plink.anyfeeder.com/jiemian/business",
            "哈佛商业评论": "https://plink.anyfeeder.com/weixin/hbrchina",
            "吴晓波频道": "https://plink.anyfeeder.com/weixin/wuxiaobo",
            "德林社": "https://plink.anyfeeder.com/weixin/delin",
            "美股研究社": "https://plink.anyfeeder.com/weixin/meigu",
            "中国日报·财经": "https://plink.anyfeeder.com/chinadaily/business",
            "商业-财富子频道": "https://plink.anyfeeder.com/fortunechina/business",
            "喷嚏网·财经风云": "https://plink.anyfeeder.com/dapenti/cai",
        }

    def fetch_feed(self, feed_url: str) -> List[Dict]:
        """Fetch and parse a single RSS feed"""
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
                }
                articles.append(article)

            return articles
        except Exception as e:
            print(f"Error fetching feed {feed_url}: {str(e)}")
            return []

    def fetch_all_feeds(self) -> Dict[str, List[Dict]]:
        """Fetch all configured RSS feeds"""
        all_news = {}

        for source, feed_url in self.feeds.items():
            print(f"Fetching from {source}...")
            articles = self.fetch_feed(feed_url)
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
                    article["source"] = source
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
