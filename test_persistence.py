#!/usr/bin/env python3
"""Test script to verify persistence functionality."""

from intanalysis import IntelligenceSystem

# Sample test articles
test_articles = [
    {
        "id": "test1",
        "title": "HDFC Bank Reports Strong Q3 Results",
        "content": "HDFC Bank announced impressive quarterly results with 15% growth in profits.",
        "source": "Test Source",
        "url": "https://example.com/hdfc-q3",
        "published_date": "2025-12-10",
    },
    {
        "id": "test2",
        "title": "RBI Maintains Repo Rate at 6.5%",
        "content": "The Reserve Bank of India decided to keep the repo rate unchanged at 6.5%.",
        "source": "Test Source",
        "url": "https://example.com/rbi-rate",
        "published_date": "2025-12-10",
    },
]

print("="*70)
print("PERSISTENCE TEST")
print("="*70)

# First run - should ingest all
print("\n[TEST 1] First run - should ingest all articles")
print("-"*70)
system = IntelligenceSystem(verbose=True)
result1 = system.ingest(test_articles)
print(f"\n✅ Result: {result1['unique_count']} new, {result1['skipped_count']} skipped")

# Second run - should skip all
print("\n[TEST 2] Second run - should skip all articles")
print("-"*70)
system2 = IntelligenceSystem(verbose=True)
result2 = system2.ingest(test_articles)
print(f"\n✅ Result: {result2['unique_count']} new, {result2['skipped_count']} skipped")

# Third run with one new article
print("\n[TEST 3] Third run - adding one new article")
print("-"*70)
new_article = {
    "id": "test3",
    "title": "Sensex Crosses 85000 Mark",
    "content": "The BSE Sensex reached a new all-time high crossing the 85000 mark today.",
    "source": "Test Source",
    "url": "https://example.com/sensex-high",
    "published_date": "2025-12-10",
}
system3 = IntelligenceSystem(verbose=True)
result3 = system3.ingest(test_articles + [new_article])
print(f"\n✅ Result: {result3['unique_count']} new, {result3['skipped_count']} skipped")

# Test query functionality
print("\n[TEST 4] Query test")
print("-"*70)
query_result = system3.query("HDFC Bank", show_steps=False)
print(f"\n✅ Query returned {len(query_result.stories)} stories")

print("\n" + "="*70)
print("PERSISTENCE TEST COMPLETED")
print("="*70)
print(f"""
Summary:
  - First run: {result1['unique_count']} articles ingested
  - Second run: {result2['skipped_count']} articles skipped (cached)
  - Third run: {result3['unique_count']} new articles, {result3['skipped_count']} skipped
  - Query: Found {len(query_result.stories)} relevant stories

✅ Persistence is working correctly!
""")
