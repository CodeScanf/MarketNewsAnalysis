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

```python
from intanalysis import IntelligenceSystem

# Initialize
system = IntelligenceSystem()

# Ingest articles
articles = [
    {"title": "HDFC Bank announces 15% dividend", "content": "..."},
    {"title": "RBI raises repo rate by 25bps", "content": "..."},
]
result = system.ingest(articles)

# Query
response = system.query("HDFC Bank news")
print(response.results)
```

## CLI

```bash
intanalysis ingest --file articles.json
intanalysis query "Banking sector news"
```

## License

MIT
