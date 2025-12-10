# Architecture Diagram - Persistence Flow

## System Flow with Persistence

```
┌─────────────────────────────────────────────────────────────────┐
│                    Interactive Demo / User Code                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     IntelligenceSystem                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Load existing vector store from disk (if available)  │  │
│  │  2. Filter new articles via PersistenceManager           │  │
│  │  3. Process only new articles through pipeline           │  │
│  │  4. Save updated state to disk                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────┬─────────────────────────────────────────────┬───────────┘
        │                                             │
        │ Filter new                                  │ Save state
        ▼                                             ▼
┌──────────────────────┐                    ┌─────────────────────┐
│ PersistenceManager   │                    │   Storage Files     │
│                      │                    │                     │
│ • filter_new_art..() │◄──────────────────►│ seen_articles.json  │
│ • mark_as_seen()     │                    │ vector_store.pkl    │
│ • load_vector_st..() │                    │ stories.pkl         │
│ • save_vector_st..() │                    │                     │
└──────────────────────┘                    └─────────────────────┘
```

## Ingestion Flow - Before vs After

### Before (No Persistence)
```
User runs demo
    ↓
Load 400 articles from JSON
    ↓
┌──────────────────────────┐
│ Process ALL 400 articles │  ← 120 seconds
│  • Embed (768-dim)       │
│  • Deduplicate           │
│  • Extract entities      │
│  • Stock impact          │
│  • Index                 │
└──────────────────────────┘
    ↓
Ready for queries

Next run: Repeat entire process ←─┐
                                  │
                            Inefficient!
```

### After (With Persistence)

#### First Run
```
User runs demo
    ↓
Try to load cache ━━━━━━━━━━━━━━┐
    │                           │
    │ (cache miss)              │
    ↓                           │
Load 400 articles from JSON     │
    ↓                           │
┌──────────────────────────┐    │
│ Process ALL 400 articles │ ←──┘
│  • Embed (768-dim)       │  120 seconds
│  • Deduplicate           │
│  • Extract entities      │
│  • Stock impact          │
│  • Index                 │
└──────────────────────────┘
    ↓
Save to disk ━━━━━━━━━━━━━━━┐
    │                       │
    │                       ↓
    │              ┌─────────────────┐
    │              │ Persist:        │
    │              │ • 400 hashes    │
    │              │ • Vector store  │
    │              │ • Story metadata│
    │              └─────────────────┘
    ↓
Ready for queries
```

#### Second Run (No New Articles)
```
User runs demo
    ↓
Load cache ━━━━━━━━━━━━━━━━━┐
    ↓                       │
┌─────────────────┐         │
│ Loaded 400      │←────────┘
│ stories from    │  2 seconds
│ disk            │
└─────────────────┘
    ↓
Load 400 articles from JSON
    ↓
Filter new articles ━━━━━━━━━┐
    │                        │
    │ (all 400 seen)         │
    ↓                        │
Skip processing ━━━━━━━━━━━━━┘
    ↓
Already ready for queries!
```

#### Third Run (5 New Articles)
```
User runs demo
    ↓
Load cache ━━━━━━━━━━━━━━━━━┐
    ↓                       │
┌─────────────────┐         │
│ Loaded 400      │←────────┘
│ stories from    │  2 seconds
│ disk            │
└─────────────────┘
    ↓
Load 405 articles from JSON
    ↓
Filter new articles ━━━━━━━━━┐
    │                        │
    │ 400 seen, 5 new        │
    ↓                        │
┌──────────────────────┐     │
│ Process 5 articles   │←────┘
│  • Embed            │  15 seconds
│  • Deduplicate      │
│  • Extract entities │
│  • Stock impact     │
│  • Index            │
└──────────────────────┘
    ↓
Update disk ━━━━━━━━━━━━━━━┐
    │                      │
    │                      ↓
    │              ┌──────────────┐
    │              │ Update:      │
    │              │ • +5 hashes  │
    │              │ • Vector +5  │
    │              │ • Stories +5 │
    │              └──────────────┘
    ↓
Ready for queries
```

## Data Structure in Storage

### seen_articles.json
```json
[
  "a1b2c3d4e5f6...",  // MD5 hash of URL or content
  "f6e5d4c3b2a1...",
  "1234567890ab...",
  ...
]
```

### vector_store.pkl (FAISS Index)
```
Binary serialized FAISS index
- Type: IndexHNSWFlat
- Dimension: 768
- Vectors: 400 x 768 float32
- Size: ~5 MB for 1000 articles
```

### stories.pkl (Story Metadata)
```python
[
  UniqueStory(
    id="uuid-1",
    primary_article=ProcessedArticle(...),
    duplicate_articles=[...],
    entities=[Entity(...)],
    sectors=["Banking"],
    stock_impacts=[StockImpact(...)]
  ),
  ...
]
```

## Query Flow (Unchanged)

```
User query: "HDFC Bank news"
    ↓
QueryAgent
    ↓
┌──────────────────────────┐
│ 1. Extract entities      │
│    → "HDFC Bank"         │
├──────────────────────────┤
│ 2. Expand query          │
│    → "HDFC Bank Banking" │
├──────────────────────────┤
│ 3. Hybrid search         │
│    → 70% dense (FAISS)   │
│    → 30% BM25            │
├──────────────────────────┤
│ 4. Re-rank (CrossEncoder)│
│    → Top 10 candidates   │
├──────────────────────────┤
│ 5. Entity boosting       │
│    → Prioritize exact    │
├──────────────────────────┤
│ 6. LLM answer (Claude)   │ ← Only LLM call
│    → Generate summary    │
└──────────────────────────┘
    ↓
Return QueryResult
```

## Class Diagram

```
┌─────────────────────────┐
│  IntelligenceSystem     │
├─────────────────────────┤
│ + persistence           │◄─────┐
│ + vector_store          │      │ composition
│ + ingestion_graph       │      │
│ + query_graph           │      │
├─────────────────────────┤      │
│ + ingest(articles)      │      │
│ + query(text)           │      │
│ + get_stats()           │      │
└─────────────────────────┘      │
                                 │
┌─────────────────────────────────┤
│  PersistenceManager             │
├─────────────────────────────────┤
│ + storage_dir: Path             │
│ + seen_articles_file: Path      │
│ + vector_store_file: Path       │
│ + stories_file: Path            │
├─────────────────────────────────┤
│ + filter_new_articles()         │
│ + mark_articles_as_seen()       │
│ + get_seen_articles()           │
│ + save_seen_articles()          │
│ + load_vector_store()           │
│ + save_vector_store()           │
│ + clear_cache()                 │
└─────────────────────────────────┘
```

## Performance Comparison

```
Time to Process (seconds)
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  Without Persistence:                                   │
│  ████████████████████████ 120s                          │
│  ████████████████████████ 120s                          │
│  ████████████████████████ 120s                          │
│                                                         │
│  With Persistence:                                      │
│  ████████████████████████ 120s  (first run)             │
│  █ 2s                            (cached)               │
│  ███ 15s                         (5 new)                │
│                                                         │
└─────────────────────────────────────────────────────────┘
    0        30        60        90       120     150 (sec)
```

## File Size Growth

```
Storage Size (MB)
┌─────────────────────────────────────────────────┐
│                                                 │
│  100 articles:   ~1 MB                          │
│  ████▌                                          │
│                                                 │
│  500 articles:   ~5 MB                          │
│  ██████████████████████▌                        │
│                                                 │
│  1000 articles: ~10 MB                          │
│  █████████████████████████████████████████████  │
│                                                 │
└─────────────────────────────────────────────────┘
    0        2        4        6        8       10 (MB)
```

## Summary

✅ **Lazy Loading**: Cache loaded only when accessed
✅ **Incremental Updates**: Process only new articles
✅ **Automatic Persistence**: Transparent to user
✅ **Fast Startup**: 2 seconds for cached data
✅ **Space Efficient**: ~10KB per article
