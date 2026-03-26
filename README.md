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
- Node.js 18+ (for frontend)
- pip or conda
- DeepSeek API key (for LLM features)

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
API_KEY=your_deepseek_api_key_here
BASE_URL=https://api.deepseek.com
MODEL_ID=deepseek-chat

# Optional: Customize model or endpoint
# DEEPSEEK_MODEL=deepseek-chat
# DEEPSEEK_BASE_URL=https://api.deepseek.com
```

### Verify Installation

```bash
# Run the test suite
pytest tests/ -v

# Check the installation
python -c "from intanalysis import IntelligenceSystem; print('✅ Installation successful!')"
```

---

## 📰 Generating the Dataset

The system includes RSS feed scrapers to collect real financial news articles.

### One-Time Fetch

```bash
# Run the article fetcher (fetches from 10+ Indian financial news sources)
python dataset/articals.py
```

### Continuous Monitoring

```bash
# Start RSS feed monitor (checks every 5 minutes for new articles)
python dataset/feeds.py
```

This will:
- ✅ Fetch articles from 10+ Indian financial news RSS feeds
- ✅ Filter articles from the last 30 days
- ✅ Deduplicate and save to `dataset/rss_feeds_all.json`
- ✅ Track seen articles in `dataset/seen_articles.json`

### RSS Feed Sources

| Source | Feed |
|--------|------|
| Economic Times | Top Stories |
| Business Standard | Top Stories |
| Moneycontrol | Latest News |
| LiveMint | Companies |
| The Hindu Business | Business |
| NDTV Business | Latest |
| Indian Express | Business |
| Financial Express | Markets |
| Trade Brains | Latest |

### Custom Keywords Filter

```python
from dataset.feeds import NewsRSSMonitor

monitor = NewsRSSMonitor(check_interval=300)  # 5 minutes

# Filter for specific keywords
monitor.monitor_continuous(
    keywords=["HDFC", "RBI", "banking", "Sensex", "Nifty"],
    save_to_file=True
)
```

## 📖 Usage Examples

### 🖥️ Full Stack Demo (Recommended)

The easiest way to experience the system is with the **FastAPI backend + React frontend**:

#### Terminal 1: Start the Backend API

```bash
# Activate virtual environment
source .venv/bin/activate

# Start FastAPI server with uvicorn
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
✅ Intelligence System initialized
```

#### Terminal 2: Start the Frontend UI

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

You should see:
```
VITE v5.0.8  ready in 300 ms

➜  Local:   http://localhost:5173/
➜  Network: http://192.168.x.x:5173/
```

#### Open the Web UI

Navigate to **http://localhost:5173** in your browser.

<!-- Add your screenshot: docs/images/web-ui.png -->

**Features available in the UI:**

| Tab | Description |
|-----|-------------|
| 🔍 **Query** | Search financial news with natural language queries |
| 📥 **Ingest** | Paste JSON articles or load from RSS feeds |
| 🎮 **Demo** | One-click demo with sample articles |
| 📊 **Stats** | View system statistics (indexed stories, etc.) |

#### Example Workflow

1. **Load Data**: Click "Load from RSS" or paste articles in JSON format
2. **Ingest**: Click "Ingest Articles" to process them
3. **Query**: Type queries like:
   - `"HDFC Bank news"`
   - `"RBI policy changes"`
   - `"Banking sector update"`
   - `"What's happening with Infosys?"`
4. **View Results**: See AI-generated summaries with entity highlights

---

### 🐍 Interactive Demo (CLI)

```bash
python run.py
```

The interactive demo provides:
- ✅ Automatic article ingestion from RSS feeds
- ✅ Incremental updates (only new articles processed)
- ✅ Query interface with AI-powered answers
- ✅ Persistence across sessions

---

### 📚 Python API

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

### 🌐 FastAPI REST API

```bash
# Start the API server
uvicorn api.main:app --reload --port 8000

# With auto-reload for development
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

#### Interactive API Documentation

Once the server is running, access:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest articles into the system |
| `/query` | POST | Query the system with natural language |
| `/stats` | GET | Get system statistics |
| `/health` | GET | Health check |
| `/stories` | GET | List all indexed stories |
| `/chat/history` | GET | Get chat history |
| `/chat/clear` | POST | Clear chat history |

#### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Get system stats
curl http://localhost:8000/stats

# Ingest articles
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "articles": [
      {
        "title": "HDFC Bank announces 15% dividend",
        "content": "HDFC Bank Limited declared a 15% dividend for shareholders...",
        "source": "Economic Times",
        "url": "https://example.com/hdfc-news"
      }
    ]
  }'

# Query the system
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "HDFC Bank news"}'
```

#### Python Requests Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Ingest articles
articles = {
    "articles": [
        {
            "title": "RBI raises repo rate by 25bps",
            "content": "The Reserve Bank of India announced a rate hike...",
            "source": "Business Standard"
        }
    ]
}
response = requests.post(f"{BASE_URL}/ingest", json=articles)
print(response.json())

# Query
query = {"query": "RBI interest rate"}
response = requests.post(f"{BASE_URL}/query", json=query)
result = response.json()
print(f"Found {len(result['stories'])} stories")
print(f"AI Summary: {result['explanation']}")
```

---

### ⚛️ React Frontend

```bash
cd frontend
npm install
npm run dev
```

Access the web interface at **http://localhost:5173**

**Note:** The backend API must be running on port 8000 for the frontend to work.

---

### 💻 Command Line Interface

```bash
# Ingest from file
intanalysis ingest --file articles.json

# Query from CLI
intanalysis query "Banking sector news"
```

---

## 🏃 Quick Run Commands

| What | Command |
|------|---------|
| **Full Stack** | `uvicorn api.main:app --reload` + `cd frontend && npm run dev` |
| **Backend Only** | `uvicorn api.main:app --reload --port 8000` |
| **CLI Demo** | `python run.py` |
| **Generate Data** | `python dataset/feeds.py` |
| **Run Tests** | `pytest tests/ -v` |

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
│   └── main.py              # API endpoints
├── frontend/                # React web interface
│   ├── package.json         # Node.js dependencies
│   ├── vite.config.js       # Vite configuration
│   └── src/
│       ├── App.jsx          # Main React component
│       ├── api.js           # API client
│       └── index.css        # Styles
├── dataset/                 # Data generation & persistence
│   ├── articals.py          # One-time RSS fetcher
│   ├── feeds.py             # Continuous RSS monitor
│   ├── rss_feeds_all.json   # Collected articles
│   ├── seen_articles.json   # Processed article cache
│   ├── vector_store.pkl     # FAISS index
│   └── stories.pkl          # Story metadata
├── tests/                   # Test suite (149 tests)
│   ├── conftest.py          # Fixtures
│   ├── test_models.py       # Model tests
│   ├── test_agents.py       # Agent tests
│   ├── test_embeddings.py   # Embedding tests
│   ├── test_integration.py  # Integration tests
│   └── ...
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System architecture
│   ├── BENCHMARKS.md        # Performance metrics
│   ├── API_SETUP.md         # API documentation
│   └── QUICKSTART.md        # Quick start guide
├── run.py                   # Interactive CLI demo
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

The project includes a comprehensive test suite with **149 tests**.

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=intanalysis --cov-report=html

# Run specific test file
pytest tests/test_agents.py -v

# Run integration tests only
pytest tests/test_integration.py -v

# Run with markers
pytest tests/ -v -m "not slow"
```

### Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `models.py` | 28 | Data models, validation |
| `agents.py` | 24 | All 6 agents |
| `embeddings.py` | 16 | Vector store, reranker |
| `mappings.py` | 18 | Stock/sector mappings |
| `persistence.py` | 15 | Save/load operations |
| `workflow.py` | 21 | LangGraph flow |
| `integration` | 27 | End-to-end tests |

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

