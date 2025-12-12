import feedparser
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Set
import re
import os

class NewsRSSMonitor:
    """Monitor RSS feeds for new articles continuously"""
    
    def __init__(self, check_interval: int = 300):
        """
        Initialize monitor
        check_interval: seconds between checks (default: 300 = 5 minutes)
        """
        self.check_interval = check_interval
        self.articals_loaded: Set[str] = set()
        self.cache_file = './dataset/articals_loaded.json'
        
        # Popular Indian news RSS feeds
        self.feeds = {
            "The Hindu Business": "https://www.thehindu.com/business/feeder/default.rss",
            "Economic Times – Top Stories": "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
            "Business Standard – Top Stories": "https://www.business-standard.com/rss/home_page_top_stories.rss",
            "Moneycontrol – Latest News": "https://www.moneycontrol.com/rss/latestnews.xml",
            "LiveMint – Companies": "https://www.livemint.com/rss/companies",
            "NDTV Business": "https://feeds.feedburner.com/ndtvprofit-latest",
            "Indian Express – Business": "https://indianexpress.com/section/business/feed/",
            "Financial Express – Business": "https://www.financialexpress.com/feed/",
            "Financial Express – Markets": "https://www.financialexpress.com/market/feed/",
            "Trade Brains – Latest Articles": "https://tradebrains.in/feed/",
        }

        
        # Load previously seen articles
        self.load_articals_loaded()
    
    def load_articals_loaded(self):
        """Load previously seen article IDs from cache"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    self.articals_loaded = set(data.get('seen', []))
                print(f"Loaded {len(self.articals_loaded)} previously seen articles")
            except Exception as e:
                print(f"Error loading cache: {e}")
    
    def save_articals_loaded(self):
        """Save seen article IDs to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump({'seen': list(self.articals_loaded)}, f)
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def fetch_feed(self, feed_url: str, source: str) -> List[Dict]:
        """Fetch and parse a single RSS feed, return only NEW articles from last 30 days"""
        try:
            feed = feedparser.parse(feed_url)
            new_articles = []
            thirty_days_ago = datetime.now() - timedelta(days=30)
            
            for entry in feed.entries:
                # Use link as unique identifier
                article_id = entry.get('link', '')
                
                # Check article date
                published_parsed = entry.get('published_parsed')
                if published_parsed:
                    article_date = datetime(*published_parsed[:6])
                    if article_date < thirty_days_ago:
                        continue  # Skip articles older than 30 days
                
                # Check if this is a new article
                if article_id and article_id not in self.articals_loaded:
                    # Extract all available RSS fields
                    article = {
                        'id': article_id,
                        'title': entry.get('title', 'No title'),
                        'link': article_id,
                        'published': entry.get('published', 'Unknown date'),
                        # 'published_parsed': str(entry.get('published_parsed', '')),
                        'updated': entry.get('updated', ''),
                        # 'updated_parsed': str(entry.get('updated_parsed', '')),
                        'summary': entry.get('summary', 'No summary'),
                        # 'summary_detail': entry.get('summary_detail', {}),
                        'content': entry.get('content', []),
                        'author': entry.get('author', 'Unknown'),
                        # 'author_detail': entry.get('author_detail', {}),
                        # 'contributors': entry.get('contributors', []),
                        'tags': entry.get('tags', []),
                        'category': entry.get('category', ''),
                        # 'comments': entry.get('comments', ''),
                        # 'enclosures': entry.get('enclosures', []),
                        # 'media_content': entry.get('media_content', []),
                        # 'media_thumbnail': entry.get('media_thumbnail', []),
                        # 'rights': entry.get('rights', ''),
                        # 'publisher': entry.get('publisher', ''),
                        'source': source,
                        'feed_url': feed_url,
                        'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
        
        for source, feed_url in self.feeds.items():
            new_articles = self.fetch_feed(feed_url, source)
            
            # Filter by keywords if provided
            if keywords:
                filtered = []
                for article in new_articles:
                    title_lower = article['title'].lower()
                    summary_lower = article['summary'].lower()
                    if any(kw.lower() in title_lower or kw.lower() in summary_lower 
                           for kw in keywords):
                        filtered.append(article)
                all_new_articles.extend(filtered)
            else:
                all_new_articles.extend(new_articles)
        
        return all_new_articles
    
    def display_article(self, article: Dict):
        """Display a single article"""
        print("\n" + "="*80)
        print(f"🆕 NEW ARTICLE from {article['source']}")
        print("="*80)
        print(f"Title: {article['title']}")
        print(f"Published: {article['published']}")
        print(f"Link: {article['link']}")
        
        # Clean HTML tags from summary
        summary = re.sub('<[^<]+?>', '', article['summary'])
        print(f"Summary: {summary[:300]}...")
        print("="*80)
    
    def save_article_to_file(self, article: Dict, filename: str = './dataset/rss_feeds_all.json'):
        """Save new article to a single JSON file"""
        try:
            # Load existing articles
            articles = []
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    articles = json.load(f)
            
            # Add new article
            articles.append(article)
            
            # Save back with proper encoding
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(articles, f, indent=2, ensure_ascii=False)
            
            # Also save published dates to txt file
            txt_filename = './dataset/published_dates.txt'
            with open(txt_filename, 'a', encoding='utf-8') as f:
                f.write(f"{article['published']}\n")
            
            print(f"💾 Saved to {filename} (Total: {len(articles)} articles)")
        except Exception as e:
            print(f"Error saving article: {e}")
    
    def monitor_continuous(self, keywords: List[str] = None, 
                          save_to_file: bool = True):
        """
        Continuously monitor feeds for new articles
        keywords: List of keywords to filter (None = all articles)
        save_to_file: Save new articles to JSON file
        """
        print(f"🔍 Starting continuous monitoring...")
        print(f"⏱️  Checking every {self.check_interval} seconds")
        if keywords:
            print(f"🎯 Filtering for keywords: {', '.join(keywords)}")
        print("Press Ctrl+C to stop\n")
        
        try:
            iteration = 0
            while True:
                iteration += 1
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n[{timestamp}] Check #{iteration}")
                
                # Fetch new articles
                new_articles = self.check_all_feeds(keywords)
                
                if new_articles:
                    print(f"✅ Found {len(new_articles)} new article(s)!")
                    
                    for article in new_articles:
                        self.display_article(article)
                        
                        if save_to_file:
                            self.save_article_to_file(article)
                else:
                    print("No new articles found")
                
                # Save seen articles cache
                self.save_articals_loaded()
                
                # Wait before next check
                print(f"\n💤 Waiting {self.check_interval} seconds until next check...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            self.save_articals_loaded()
    
    def check_once(self, keywords: List[str] = None, save_to_file: bool = True):
        """Check feeds once and return new articles"""
        print("Checking feeds for new articles...")
        new_articles = self.check_all_feeds(keywords)
        
        if new_articles:
            print(f"\n✅ Found {len(new_articles)} new article(s)!\n")
            for article in new_articles:
                self.display_article(article)
                
                if save_to_file:
                    self.save_article_to_file(article)
        else:
            print("No new articles found")
        
        self.save_articals_loaded()
        return new_articles


def main():
    # Configuration
    CHECK_INTERVAL = 300  # 5 minutes (300 seconds)
    
    # Keywords to filter (leave empty list for all articles)
    KEYWORDS = ['startup', 'IPO', 'mobility', 'app', 'Namma Yatri', 
                'share price', 'stock', 'funding']
    
    # Create monitor
    monitor = NewsRSSMonitor(check_interval=CHECK_INTERVAL)
    
    print("\nChoose mode:")
    print("1. Check once (run script manually each time)")
    print("2. Continuous monitoring (runs forever)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Single check
        monitor.check_once(keywords=KEYWORDS, save_to_file=True)
    else:
        # Continuous monitoring
        monitor.monitor_continuous(keywords=KEYWORDS, save_to_file=True)


if __name__ == "__main__":
    # Required: pip install feedparser
    main()