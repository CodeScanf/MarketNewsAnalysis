# Dataset - Market News Data Collection

This directory contains the data collection infrastructure for the Market News Analysis project. It monitors and collects news articles from various Indian financial news RSS feeds.

## Overview

The data collection system continuously monitors RSS feeds from major Indian financial news sources and stores new articles for analysis. It implements deduplication to avoid processing the same article multiple times and filters articles by date relevance.

## Components

### Core Scripts

- **`feeds.py`**: Main RSS feed monitoring system
  - `NewsRSSMonitor` class - Continuously monitors RSS feeds at configurable intervals
  - Implements deduplication using article link hashing
  - Filters articles from the last 30 days
  - Default check interval: 300 seconds (5 minutes)

- **`articals.py`**: RSS feed fetcher utility
  - `NewsRSSFetcher` class - Fetches and parses RSS feeds
  - Extracts article metadata (title, link, summary, author, etc.)
  - Returns top 10 articles per feed

### Data Files

- **`rss_feeds_all.json`**: Complete collection of fetched articles
  - Contains full article metadata including:
    - Article ID (unique link)
    - Title, summary, and content
    - Publication and update timestamps
    - Author and source information
    - Tags and categories
    - Feed URL and fetch timestamp

- **`articals_loaded.json`**: Cache of previously seen articles
  - Stores article IDs to prevent duplicate processing
  - Enables incremental data collection across restarts

- **`seen_articles.json`**: Hash-based tracking of processed articles
  - Contains MD5 hashes of article identifiers
  - Used for efficient deduplication

- **`published_dates.txt`**: Log of article publication dates
  - Tracking file for temporal analysis

## RSS Feed Sources

The system monitors the following finance, business, and technology RSS sources:

1. **叶檀财经** - 财经评论与市场观察
2. **华尔街见闻** - 宏观、市场与商业资讯
3. **财新网** - 财经深度报道
4. **第一财经周刊** - 商业与消费报道
5. **经济观察网** - 财经与产业观察
6. **财富中文网** - 商业管理与公司新闻
7. **路透中文** - 国际财经新闻
8. **雪球热门话题** - 投资社区热点
9. **36氪** - 创投、科技和商业报道
10. **虎嗅** - 科技与商业评论
11. **钛媒体** - 科技商业资讯
12. **界面新闻** - 商业频道新闻
13. **哈佛商业评论** - 管理与商业洞察
14. **吴晓波频道** - 财经与商业内容
15. **德林社** - 投资与财经评论
16. **美股研究社** - 美股与科技公司资讯
17. **中国日报·财经** - 财经新闻
18. **商业-财富子频道** - 财富中文网商业子频道
19. **喷嚏网·财经风云** - 财经热点聚合

## Data Collection Process

### 1. Initialization
- Load previously seen article IDs from cache
- Configure RSS feed sources
- Set monitoring interval (default: 5 minutes)

### 2. Fetching
- Parse RSS feeds using `feedparser` library
- Extract article metadata (title, link, summary, author, tags, etc.)
- Apply date filter (last 30 days)

### 3. Deduplication
- Check article link against previously seen IDs
- Use MD5 hashing for efficient lookup
- Skip duplicate articles automatically

### 4. Storage
- Save new articles to `rss_feeds_all.json`
- Update `articals_loaded.json` with new article IDs
- Append to `seen_articles.json` for tracking

### 5. Continuous Monitoring
- Wait for configured interval
- Repeat fetch and process cycle
- Handle errors gracefully with retry logic

## Article Data Schema

Each article in `rss_feeds_all.json` contains:

```json
{
  "id": "unique_article_url",
  "title": "Article headline",
  "link": "Full article URL",
  "published": "Publication date (RFC 2822 format)",
  "updated": "Last update timestamp",
  "summary": "Article excerpt or description",
  "content": [],
  "author": "Author name or 'Unknown'",
  "tags": ["tag1", "tag2"],
  "category": "Article category",
  "source": "News source name",
  "feed_url": "RSS feed URL",
  "fetched_at": "Timestamp when article was collected"
}
```

## Usage

### Starting the RSS Monitor

```python
from dataset.feeds import NewsRSSMonitor

# Initialize with 5-minute check interval
monitor = NewsRSSMonitor(check_interval=300)

# Start monitoring
monitor.start_monitoring()
```

### Fetching Articles Once

```python
from dataset.articals import NewsRSSFetcher

fetcher = NewsRSSFetcher()
all_news = fetcher.fetch_all_feeds()
```

## Configuration

### Adjusting Check Interval
Modify the `check_interval` parameter when initializing `NewsRSSMonitor`:
```python
# Check every 10 minutes
monitor = NewsRSSMonitor(check_interval=600)
```

### Date Filter Window
Articles older than 30 days are automatically filtered. To modify this, update the `timedelta` in `feeds.py`:
```python
thirty_days_ago = datetime.now() - timedelta(days=30)  # Adjust days value
```

### Adding New RSS Feeds
Add new feeds to the `self.feeds` dictionary in either `feeds.py` or `articals.py`:
```python
self.feeds = {
    "Source Name": "https://example.com/rss/feed.xml",
    # ... existing feeds
}
```

## File Maintenance

- **`articals_loaded.json`**: Grows over time; consider periodic cleanup of old entries
- **`rss_feeds_all.json`**: Contains all collected articles; archive periodically if needed
- **`seen_articles.json`**: Hash cache; can be cleared to reset deduplication

## Error Handling

The system includes robust error handling:
- Failed feed fetches are logged and skipped
- Cache read/write errors are caught and reported
- Network timeouts are handled gracefully
- Invalid RSS feeds don't crash the monitor

## Dependencies

- `feedparser`: RSS feed parsing
- `requests`: HTTP requests (if needed)
- Standard library: `json`, `datetime`, `re`, `os`, `typing`

## Notes

- Article IDs are based on article URLs (links)
- Deduplication ensures each article is processed only once
- The system is designed for continuous operation
- All timestamps are in IST (India Standard Time, +0530)
