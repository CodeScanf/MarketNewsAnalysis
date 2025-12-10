# IntAnalysis - Financial News Intelligence System

AI-powered multi-agent system for processing financial news using LangGraph.

## Features

- **Intelligent Deduplication**: Identifies duplicate articles with ≥95% accuracy using semantic embeddings
- **Entity Extraction**: Extracts companies, sectors, regulators with ≥90% precision
- **Stock Impact Mapping**: Maps news to impacted stocks with confidence scores
- **Context-Aware Queries**: Semantic search with entity expansion

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       LangGraph Workflow                          │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────┤
│ Ingestion│  Dedup   │  Entity  │  Stock   │ Storage  │  Query   │
│  Agent   │  Agent   │  Extract │  Impact  │  Agent   │  Agent   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

## Installation

```bash
cd IntAnalysis
pip install -e .
python -m spacy download en_core_web_sm
```

## Environment Variables

Create a `.env` file:

```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

### Interactive Demo (Recommended)

```bash
python3 interactive_demo.py
```

The demo automatically:
- ✅ Loads previously processed articles from disk
- ✅ Only ingests new articles (incremental updates)
- ✅ Persists vector store to avoid re-embedding
- ✅ Provides intelligent query interface

### Python API

```python
from intanalysis import IntelligenceSystem

# Initialize (loads from cache if available)
system = IntelligenceSystem()

# Ingest articles (only new ones are processed)
articles = [
    {"title": "HDFC Bank announces 15% dividend", "content": "...", "url": "..."},
    {"title": "RBI raises repo rate by 25bps", "content": "...", "url": "..."},
]
result = system.ingest(articles)
# Output: {'unique_count': 2, 'skipped_count': 0}

# Second run - articles are cached
result = system.ingest(articles)
# Output: {'unique_count': 0, 'skipped_count': 2}

# Query
response = system.query("HDFC Bank news")
print(response.explanation)  # AI-generated answer
for story in response.stories:
    print(story.primary_article.article.title)
```

### Persistence Features

The system automatically persists:
- **Processed article tracking** (`dataset/seen_articles.json`)
- **Vector store** (`dataset/vector_store.pkl`)
- **Story metadata** (`dataset/stories.pkl`)

This means:
- 🚀 **Instant startup** on subsequent runs
- 💰 **Cost savings** - no re-embedding of existing articles
- ⚡ **Incremental updates** - only process new articles

To clear cache and re-ingest:
```python
system.persistence.clear_cache()
```

## CLI

```bash
intanalysis ingest --file articles.json
intanalysis query "Banking sector news"
```

## License

MIT


 uvicorn api.main:app --reload --port 8000  
 npm install
npm run dev
