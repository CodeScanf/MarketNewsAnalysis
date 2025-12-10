# Quick Start Guide - Testing Persistence

## 1. Run the Automated Test

```bash
python3 test_persistence.py
```

**Expected Output:**
```
======================================================================
PERSISTENCE TEST
======================================================================

[TEST 1] First run - should ingest all articles
----------------------------------------------------------------------
📥 INGESTING 2 NEW ARTICLES
✅ Result: 2 new, 0 skipped

[TEST 2] Second run - should skip all articles
----------------------------------------------------------------------
📂 Loaded 2 existing stories from disk
📌 Skipping 2 already processed articles
✅ NO NEW ARTICLES TO PROCESS
✅ Result: 0 new, 2 skipped

[TEST 3] Third run - adding one new article
----------------------------------------------------------------------
📂 Loaded 2 existing stories from disk
📥 INGESTING 1 NEW ARTICLES
✅ Result: 1 new, 2 skipped

[TEST 4] Query test
----------------------------------------------------------------------
🔍 PROCESSING QUERY: "HDFC Bank"
✅ Query returned 1 stories

======================================================================
PERSISTENCE TEST COMPLETED
======================================================================

Summary:
  - First run: 2 articles ingested
  - Second run: 2 articles skipped (cached)
  - Third run: 1 new articles, 2 skipped
  - Query: Found 1 relevant stories

✅ Persistence is working correctly!
```

## 2. Run Interactive Demo (First Time)

```bash
python3 interactive_demo.py
```

**What Happens:**
- Processes all articles from `dataset/rss_feeds_all.json`
- Takes ~2 minutes (embedding generation)
- Saves cache to `dataset/` folder
- Ready for queries

**Output:**
```
🏦 FINANCIAL NEWS INTELLIGENCE SYSTEM - Interactive Demo
======================================================================

📂 Loading dataset: dataset/rss_feeds_all.json
   Found 400 articles

🔧 Initializing Intelligence System...

📥 Checking 400 articles for new content...
   Only new articles will be processed...

📥 INGESTING 400 NEW ARTICLES
======================================================================
[Agent processing...]

✅ INGESTION COMPLETE
   • New unique stories: 380
   • Duplicates in batch: 20
   • Total indexed: 380
   • Persisted to disk ✓

💬 INTERACTIVE QUERY MODE
   Type your query and press Enter
   Commands: 'stats', 'verbose on/off', 'clear cache', 'help', 'quit'
--------------------------------------------------------------

🔍 Query >
```

## 3. Run Interactive Demo (Second Time)

```bash
python3 interactive_demo.py
```

**What Happens:**
- Loads cache from disk (~2 seconds)
- Skips all 400 articles (already processed)
- Immediately ready for queries

**Output:**
```
🏦 FINANCIAL NEWS INTELLIGENCE SYSTEM - Interactive Demo
======================================================================

📂 Loading dataset: dataset/rss_feeds_all.json
   Found 400 articles

🔧 Initializing Intelligence System...
📂 Loaded 380 existing stories from disk

📥 Checking 400 articles for new content...
   Only new articles will be processed...

📌 Skipping 400 already processed articles

✅ NO NEW ARTICLES TO PROCESS
   Total stories in system: 380

💬 INTERACTIVE QUERY MODE [Ready in 2 seconds!]
--------------------------------------------------------------

🔍 Query >
```

## 4. Try Sample Queries

In the interactive mode, try:

```
🔍 Query > HDFC Bank news

⏳ Processing query...

🔍 QUERY RESULTS
======================================================================
   Query: "HDFC Bank news"
   Results: 3 stories found
   🎯 Detected Entities: HDFC Bank

──────────────────────────────────────────────────────────
💡 INTELLIGENT ANSWER
──────────────────────────────────────────────────────────
   HDFC Bank reported strong Q3 results with 15% profit growth...
──────────────────────────────────────────────────────────

📚 SOURCES (3 articles):
[1] HDFC Bank announces dividend
[2] HDFC Bank Q3 Results Beat Estimates
[3] Banking Sector Shows Strong Growth
```

More query examples:
```
> RBI policy
> IndiGo flight cancellations
> Banking sector update
> IPO surge
> Microsoft India investment
```

## 5. Check Statistics

```
🔍 Query > stats
📊 System Stats: 380 stories indexed
```

## 6. Clear Cache (if needed)

```
🔍 Query > clear cache
✅ Cache cleared. Restart demo to re-ingest all articles.

🔍 Query > quit
👋 Goodbye!
```

Then restart to re-process everything:
```bash
python3 interactive_demo.py
```

## 7. Check What Files Were Created

```bash
ls -lh dataset/
```

**Output:**
```
-rw-r--r--  seen_articles.json    (50 KB)
-rw-r--r--  vector_store.pkl      (5.2 MB)
-rw-r--r--  stories.pkl           (800 KB)
-rw-r--r--  rss_feeds_all.json    (2 MB)
```

## 8. Programmatic Usage

Create a test script:

```python
# my_test.py
from intanalysis import IntelligenceSystem

# Initialize
system = IntelligenceSystem(verbose=True)

# First ingestion
articles = [
    {
        "title": "Test Article 1",
        "content": "HDFC Bank reports growth",
        "url": "https://example.com/1"
    },
    {
        "title": "Test Article 2", 
        "content": "RBI policy update",
        "url": "https://example.com/2"
    }
]

result1 = system.ingest(articles)
print(f"First run: {result1['unique_count']} new, {result1['skipped_count']} skipped")

# Second ingestion (same articles)
result2 = system.ingest(articles)
print(f"Second run: {result2['unique_count']} new, {result2['skipped_count']} skipped")

# Query
response = system.query("HDFC Bank")
print(f"Found {len(response.stories)} stories")
print(f"Answer: {response.explanation}")
```

Run it:
```bash
python3 my_test.py
```

## 9. Performance Comparison

Time each run:

```bash
# First run (fresh cache)
time python3 interactive_demo.py
# → Real: 2m 15s

# Exit and run again
time python3 interactive_demo.py  
# → Real: 0m 3s  (98% faster!)
```

## 10. Troubleshooting

### Cache not working?
```bash
# Check if files exist
ls dataset/*.pkl

# Check permissions
ls -la dataset/

# Clear and retry
rm dataset/seen_articles.json dataset/*.pkl
python3 interactive_demo.py
```

### Want to force re-processing?
```python
system.ingest(articles, force=True)  # Bypasses cache
```

### Out of disk space?
```bash
# Check size
du -sh dataset/

# Clear cache
rm dataset/*.pkl dataset/seen_articles.json
```

## Summary of Commands

| Command | Purpose |
|---------|---------|
| `python3 test_persistence.py` | Automated test |
| `python3 interactive_demo.py` | Interactive mode |
| `stats` | Show system stats |
| `clear cache` | Reset all data |
| `help` | Show help |
| `quit` | Exit |

## Expected Behavior

✅ First run: Processes all articles (~2 minutes)
✅ Second run: Skips all articles (~2 seconds)
✅ With new articles: Only processes new ones
✅ Queries work immediately after loading cache
✅ No data loss between runs
✅ Disk usage: ~10MB per 1000 articles

🎉 **You're all set!**
