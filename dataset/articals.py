import feedparser
import requests
from datetime import datetime
from typing import List, Dict
import re

class NewsRSSFetcher:
    """Fetch news articles from various RSS feeds"""
    
    def __init__(self):
        # Popular Indian news RSS feeds
        self.feeds = {
            'The Hindu Business': 'https://www.thehindu.com/business/feeder/default.rss',
            'Economic Times': 'https://economictimes.indiatimes.com/rssfeedstopstories.cms',
            'Business Standard': 'https://www.business-standard.com/rss/home_page_top_stories.rss',
            'Moneycontrol': 'https://www.moneycontrol.com/rss/latestnews.xml',
            'LiveMint': 'https://www.livemint.com/rss/companies',
            'NDTV Business': 'https://feeds.feedburner.com/ndtvprofit-latest',
            'Indian Express Business': 'https://indianexpress.com/section/business/feed/',
        }
    
    def fetch_feed(self, feed_url: str) -> List[Dict]:
        """Fetch and parse a single RSS feed"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:10]:  # Get top 10 articles
                article = {
                    'title': entry.get('title', 'No title'),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', 'Unknown date'),
                    'summary': entry.get('summary', 'No summary'),
                    'author': entry.get('author', 'Unknown')
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
    
    def search_keywords(self, articles: Dict[str, List[Dict]], 
                       keywords: List[str]) -> List[Dict]:
        """Search for specific keywords in article titles"""
        matching_articles = []
        
        for source, article_list in articles.items():
            for article in article_list:
                title_lower = article['title'].lower()
                if any(keyword.lower() in title_lower for keyword in keywords):
                    article['source'] = source
                    matching_articles.append(article)
        
        return matching_articles
    
    def display_articles(self, articles: List[Dict], max_articles: int = 20):
        """Display articles in a formatted way"""
        print("\n" + "="*80)
        print(f"Found {len(articles)} matching articles")
        print("="*80 + "\n")
        
        for i, article in enumerate(articles[:max_articles], 1):
            print(f"{i}. {article['title']}")
            print(f"   Source: {article.get('source', 'Unknown')}")
            print(f"   Published: {article['published']}")
            print(f"   Link: {article['link']}")
            if article.get('summary'):
                # Clean HTML tags from summary
                summary = re.sub('<[^<]+?>', '', article['summary'])
                print(f"   Summary: {summary[:200]}...")
            print()


def main():
    # Initialize fetcher
    fetcher = NewsRSSFetcher()
    
    # Fetch all news
    print("Fetching news from RSS feeds...")
    all_news = fetcher.fetch_all_feeds()
    
    # Example 1: Display all latest business news
    print("\n### LATEST BUSINESS NEWS ###")
    for source, articles in all_news.items():
        print(f"\n--- {source} ---")
        for article in articles[:3]:  # Top 3 from each source
            print(f"• {article['title']}")
    
    # Example 2: Search for specific topics
    print("\n\n### SEARCHING FOR SPECIFIC TOPICS ###")
    keywords = ['startup', 'IPO', 'share price', 'stock', 'mobility', 'app']
    matching = fetcher.search_keywords(all_news, keywords)
    fetcher.display_articles(matching, max_articles=10)
    
    # Example 3: Custom search
    print("\n\n### CUSTOM KEYWORD SEARCH ###")
    custom_keywords = input("Enter keywords to search (comma-separated): ").split(',')
    custom_keywords = [k.strip() for k in custom_keywords if k.strip()]
    
    if custom_keywords:
        custom_results = fetcher.search_keywords(all_news, custom_keywords)
        fetcher.display_articles(custom_results)


if __name__ == "__main__":
    # Required library: feedparser
    # Install with: pip install feedparser
    
    main()