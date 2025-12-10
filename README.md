# IntAnalysis - AI-Powered Financial News Intelligence System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangGraph](https://img.shields.io/badge/Framework-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)

An AI-powered multi-agent system for processing financial news using LangGraph. The system identifies unique news stories from redundant coverage, extracts market entities, maps news to impacted stocks, and provides context-aware query responses for traders and investors.

## 🎯 Key Features

| Feature | Description | Target Accuracy |
|---------|-------------|-----------------|
| **Intelligent Deduplication** | Identifies duplicate articles using semantic embeddings | ≥95% |
| **Entity Extraction** | Extracts companies, sectors, regulators, people | ≥90% precision |
| **Stock Impact Mapping** | Maps news to impacted stocks with confidence scores | Direct: 100%, Sector: 60-80% |
| **Context-Aware Queries** | Semantic search with entity expansion and re-ranking | Top-5 relevance |
| **Hybrid Search** | Combines dense vectors (FAISS) + sparse (BM25) | Improved recall |
| **AI Explanations** | Natural language answers powered by Claude | - |

## 📐 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LangGraph Multi-Agent Pipeline                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│   │ Ingestion│──▶│  Dedup   │──▶│  Entity  │──▶│  Stock   │            │
│   │  Agent   │   │  Agent   │   │  Extract │   │  Impact  │            │
│   └──────────┘   └──────────┘   └──────────┘   └──────────┘            │
│        │                                              │                  │
│        │                                              ▼                  │
│        │         ┌──────────┐                  ┌──────────┐            │
│        │         │  Query   │◀─────────────────│ Storage  │            │
│        │         │  Agent   │                  │  Agent   │            │
│        │         └──────────┘                  └──────────┘            │
│        │              │                              │                  │
│        ▼              ▼                              ▼                  │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Vector Store (FAISS + BM25)                   │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Agent Responsibilities

| Agent | Function | Technology |
|-------|----------|------------|
| **Ingestion Agent** | Validates and normalizes raw articles | Pydantic models |
| **Deduplication Agent** | Clusters similar articles using embeddings | Sentence-Transformers, Union-Find |
| **Entity Extraction Agent** | Extracts entities using NER + rules + LLM | spaCy, rule-based, Claude fallback |
| **Stock Impact Agent** | Maps entities to stock symbols with confidence | Custom mapping + LLM |
| **Storage Agent** | Indexes articles in vector store | FAISS HNSW, BM25Okapi |
| **Query Agent** | Hybrid search + re-ranking + AI answer | CrossEncoder, Claude |

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- Anthropic API key (for Claude LLM features)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MarketNewsAnalysis.git
cd MarketNewsAnalysis

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode
pip install -e .

# Download spaCy model for NER
python -m spacy download en_core_web_sm
```

### Environment Setup

Create a `.env` file in the project root:

```bash
# Required for LLM features (AI explanations, enhanced entity extraction)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional: Customize model
# ANTHROPIC_MODEL=claude-3-5-haiku-latest
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Check the installation
python -c "from intanalysis import IntelligenceSystem; print('✅ Installation successful!')"
```

## 📖 Usage Examples

### Interactive Demo (Recommended for First-Time Users)

```bash
python interactive_demo.py
```

The interactive demo provides:
- ✅ Automatic article ingestion from RSS feeds
- ✅ Incremental updates (only new articles processed)
- ✅ Query interface with AI-powered answers
- ✅ Persistence across sessions

### Python API

#### Basic Usage

```python
from intanalysis import IntelligenceSystem

# Initialize the system
system = IntelligenceSystem(verbose=True)

# Ingest articles
articles = [
    {
        "title": "HDFC Bank announces 15% dividend, board approves stock buyback",
        "content": "HDFC Bank Limited declared a 15% dividend for its shareholders...",
        "source": "Economic Times",
        "url": "https://example.com/hdfc-news"
    },
    {
        "title": "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns",
        "content": "The Reserve Bank of India announced a rate hike today...",
        "source": "Business Standard",
        "url": "https://example.com/rbi-news"
    }
]

result = system.ingest(articles)
print(f"Unique stories: {result['unique_count']}")
print(f"Duplicates: {result['duplicate_count']}")
```

#### Querying the System

```python
# Query for company-specific news
response = system.query("HDFC Bank news")
print(f"Query: {response.query}")
print(f"Found: {len(response.stories)} relevant stories")
print(f"AI Answer: {response.explanation}")

for story in response.stories:
    article = story.primary_article.article
    print(f"\n📰 {article.title}")
    print(f"   Source: {article.source}")
    print(f"   Entities: {[e.name for e in story.primary_article.entities]}")
    print(f"   Stock Impacts: {[i.symbol for i in story.primary_article.stock_impacts]}")
```

#### Query Patterns

| Query Type | Example | Returns |
|------------|---------|---------|
| Company-specific | "HDFC Bank news" | Direct mentions + sector news |
| Sector-wide | "Banking sector update" | All banking-related articles |
| Regulator-specific | "RBI policy changes" | RBI-specific articles |
| Thematic | "Interest rate impact" | Semantically related articles |

#### Working with Entities

```python
# Access extracted entities
for story in response.stories:
    for entity in story.primary_article.entities:
        print(f"{entity.name} ({entity.type.value}): {entity.confidence:.0%}")
```

#### Stock Impact Analysis

```python
# Get stock impacts with confidence scores
for story in response.stories:
    for impact in story.primary_article.stock_impacts:
        print(f"{impact.symbol}: {impact.confidence:.0%} ({impact.impact_type.value})")
        print(f"  Reasoning: {impact.reasoning}")
```

### FastAPI REST API

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000
```

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest articles |
| `/query` | GET | Query the system |
| `/stats` | GET | Get system statistics |
| `/health` | GET | Health check |

#### Example API Calls

```bash
# Ingest articles
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '[{"title": "HDFC Bank news", "content": "..."}]'

# Query
curl "http://localhost:8000/query?q=HDFC%20Bank%20news"
```

### React Frontend

```bash
cd frontend
npm install
npm run dev
```

Access the web interface at `http://localhost:5173`

### Command Line Interface

```bash
# Ingest from file
intanalysis ingest --file articles.json

# Query from CLI
intanalysis query "Banking sector news"
```

## 🗂️ Project Structure

```
MarketNewsAnalysis/
├── intanalysis/              # Core package
│   ├── __init__.py          # Package exports
│   ├── agents.py            # 6 LangGraph agents
│   ├── core.py              # IntelligenceSystem main interface
│   ├── embeddings.py        # EmbeddingService, VectorStore, Reranker
│   ├── llm.py               # LLM service (Claude)
│   ├── mappings.py          # Stock/sector/regulator mappings
│   ├── models.py            # Pydantic data models
│   ├── persistence.py       # Disk persistence manager
│   ├── workflow.py          # LangGraph workflow definitions
│   └── cli.py               # Command line interface
├── api/                     # FastAPI REST API
│   └── main.py
├── frontend/                # React web interface
│   └── src/
├── dataset/                 # Data and persistence
│   ├── rss_feeds_all.json   # Sample articles
│   ├── seen_articles.json   # Processed article cache
│   ├── vector_store.pkl     # FAISS index
│   └── stories.pkl          # Story metadata
├── tests/                   # Test suite
│   ├── conftest.py          # Fixtures
│   ├── test_models.py       # Model tests
│   ├── test_agents.py       # Agent tests
│   ├── test_embeddings.py   # Embedding tests
│   ├── test_integration.py  # Integration tests
│   └── ...
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── API_SETUP.md         # API documentation
│   └── QUICKSTART.md        # Quick start guide
├── pyproject.toml           # Package configuration
└── README.md                # This file
```

## 💾 Persistence

The system automatically persists state to disk:

| File | Purpose | Format |
|------|---------|--------|
| `seen_articles.json` | Track processed articles | JSON array of MD5 hashes |
| `vector_store.pkl` | FAISS index + embeddings | Pickle (FAISS serialized) |
| `stories.pkl` | Story metadata | Pickle |

### Benefits

- 🚀 **Instant startup** on subsequent runs (~2s vs ~120s)
- 💰 **Cost savings** - no re-embedding of existing articles
- ⚡ **Incremental updates** - only process new articles

### Clear Cache

```python
system.persistence.clear_cache()
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=intanalysis --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v

# Run integration tests only
pytest tests/test_integration.py -v
```

## 📊 Performance Benchmarks

See [BENCHMARKS.md](docs/BENCHMARKS.md) for detailed performance metrics.

| Metric | Value |
|--------|-------|
| Deduplication Accuracy | ≥95% |
| Entity Extraction Precision | ≥90% |
| Query Latency (cached) | <500ms |
| Ingestion Speed | ~3 articles/sec |
| Storage per 1000 articles | ~10 MB |

## 🔧 Configuration

### Deduplication Threshold

```python
from intanalysis.agents import DeduplicationAgent

# Lower threshold = more aggressive deduplication
agent = DeduplicationAgent(threshold=0.60)  # Default: 0.60
```

### Hybrid Search Alpha

```python
# 0.7 = 70% dense (semantic) + 30% sparse (keyword)
results = vector_store.search(embedding, query_text, alpha=0.7)
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [spaCy NER](https://spacy.io/)

