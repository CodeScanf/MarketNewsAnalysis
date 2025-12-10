# System Architecture

This document describes the system design, agent flow, and technical decisions for the IntAnalysis Financial News Intelligence System.

## Table of Contents

1. [Overview](#overview)
2. [System Design](#system-design)
3. [Agent Architecture](#agent-architecture)
4. [Data Flow](#data-flow)
5. [Technical Decisions](#technical-decisions)
6. [Persistence Layer](#persistence-layer)
7. [Performance Optimizations](#performance-optimizations)

---

## Overview

IntAnalysis is a multi-agent AI system built on LangGraph that processes financial news articles to:

- **Deduplicate** semantically similar articles
- **Extract** market entities (companies, sectors, regulators)
- **Map** news to impacted stocks with confidence scores
- **Enable** context-aware natural language queries

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              USER LAYER                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  CLI/Demo   │  │  REST API   │  │  React UI   │  │  Python SDK │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         CORE INTELLIGENCE LAYER                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    IntelligenceSystem (core.py)                    │  │
│  │  • Entry point for all operations                                  │  │
│  │  • Manages LangGraph workflows                                     │  │
│  │  • Handles persistence coordination                                │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
          │                                            │
          ▼                                            ▼
┌─────────────────────────────┐    ┌─────────────────────────────────────┐
│     INGESTION PIPELINE      │    │          QUERY PIPELINE              │
│  ┌───────────────────────┐  │    │  ┌─────────────────────────────────┐│
│  │   Ingestion Graph     │  │    │  │         Query Graph             ││
│  │  ┌─────┐ ┌─────┐      │  │    │  │  ┌─────────────────────────┐   ││
│  │  │Ingest│→│Dedup│→... │  │    │  │  │      Query Agent        │   ││
│  │  └─────┘ └─────┘      │  │    │  │  └─────────────────────────┘   ││
│  └───────────────────────┘  │    │  └─────────────────────────────────┘│
└─────────────────────────────┘    └─────────────────────────────────────┘
          │                                            │
          ▼                                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
│  │   VectorStore   │  │  Persistence    │  │    LLM Service          │ │
│  │   (FAISS+BM25)  │  │    Manager      │  │     (Claude)            │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## System Design

### Core Components

| Component | File | Responsibility |
|-----------|------|----------------|
| `IntelligenceSystem` | `core.py` | Main entry point, orchestrates pipelines |
| `Agents` | `agents.py` | Six specialized processing agents |
| `Workflow` | `workflow.py` | LangGraph graph definitions |
| `VectorStore` | `embeddings.py` | FAISS + BM25 hybrid storage |
| `EmbeddingService` | `embeddings.py` | Sentence-transformer embeddings |
| `Reranker` | `embeddings.py` | Cross-encoder re-ranking |
| `LLMService` | `llm.py` | Claude API wrapper |
| `PersistenceManager` | `persistence.py` | Disk storage management |
| `Models` | `models.py` | Pydantic data models |
| `Mappings` | `mappings.py` | Stock/sector/regulator lookups |

### Data Models

```python
# Core data models (models.py)

Article           # Raw news article with title, content, source, URL
Entity            # Extracted entity with name, type, confidence
StockImpact       # Stock impact with symbol, confidence, type
ProcessedArticle  # Article + entities + impacts + embedding
UniqueStory       # Primary article + duplicates cluster
QueryResult       # Query response with stories + explanation
```

### Entity Types

```python
class EntityType(Enum):
    COMPANY = "company"      # e.g., "HDFC Bank"
    SECTOR = "sector"        # e.g., "Banking"
    REGULATOR = "regulator"  # e.g., "RBI"
    PERSON = "person"        # e.g., "CEO Name"
```

### Impact Types

```python
class ImpactType(Enum):
    DIRECT = "direct"           # Company directly mentioned (100%)
    SECTOR = "sector"           # Sector-wide impact (60-80%)
    REGULATORY = "regulatory"   # Regulatory impact (variable)
```

---

## Agent Architecture

### Agent Design Pattern

All agents inherit from `BaseAgent` with a common interface:

```python
class BaseAgent(ABC):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")
    
    @abstractmethod
    def process(self, state: dict) -> dict:
        """Process state and return updated state."""
        pass
```

### The Six Agents

#### 1. Ingestion Agent
**Purpose:** Load and validate raw articles

```
Input:  state["raw_articles"] = [dict | Article, ...]
Output: state["articles"] = [Article, ...]
```

**Logic:**
- Converts dictionaries to Article objects
- Validates required fields (title, content)
- Generates unique article IDs via MD5 hash

---

#### 2. Deduplication Agent
**Purpose:** Cluster similar articles using semantic embeddings

```
Input:  state["articles"] = [Article, ...]
Output: state["unique_stories"] = [UniqueStory, ...]
        state["processed_articles"] = [ProcessedArticle, ...]
```

**Algorithm:**
1. Compute 768-dim embeddings for all articles
2. Build similarity matrix via cosine similarity
3. Apply Union-Find clustering with threshold (default: 0.60)
4. Create UniqueStory objects with primary + duplicates

```python
# Union-Find clustering
threshold = 0.60
for i, j in article_pairs:
    if cosine_similarity(emb[i], emb[j]) >= threshold:
        union(i, j)
```

**Technical Choice:** Union-Find was chosen over hierarchical clustering for O(n²) worst-case with near-linear practical performance.

---

#### 3. Entity Extraction Agent
**Purpose:** Extract structured entities using multi-method approach

```
Input:  state["unique_stories"] = [UniqueStory, ...]
Output: state["unique_stories"] with entities populated
```

**Multi-Method Pipeline:**
1. **Rule-based (Priority 1):** Match known companies, regulators from mapping tables
2. **spaCy NER (Priority 2):** Extract PERSON and ORG entities
3. **LLM Fallback (Priority 3):** Use Claude for complex/ambiguous cases

```python
# Confidence levels
RULE_BASED_COMPANY = 1.0    # Known company matched
RULE_BASED_REGULATOR = 1.0  # Known regulator matched
SPACY_PERSON = 0.85         # spaCy PERSON entity
SPACY_ORG = 0.70            # spaCy ORG entity
LLM_EXTRACTED = 0.80        # LLM-extracted entity
```

---

#### 4. Stock Impact Agent
**Purpose:** Map entities to impacted stocks with confidence scores

```
Input:  state["unique_stories"] with entities
Output: state["unique_stories"] with stock_impacts populated
```

**Impact Mapping Rules:**
| Entity Type | Impact Type | Confidence |
|-------------|-------------|------------|
| Company (direct mention) | DIRECT | 1.0 |
| Company (via sector) | SECTOR | 0.7 |
| Regulator | REGULATORY | 0.6 |

---

#### 5. Storage Agent
**Purpose:** Index articles in vector store for retrieval

```
Input:  state["unique_stories"] with embeddings
Output: state["vector_store"] updated
        state["storage_complete"] = True
```

**Vector Store Features:**
- FAISS HNSW index for approximate nearest neighbor
- BM25 sparse index for keyword matching
- Hybrid search combining both

---

#### 6. Query Agent
**Purpose:** Context-aware search with AI-powered answers

```
Input:  state["query"] = "HDFC Bank news"
        state["vector_store"] = VectorStore
Output: state["query_result"] = QueryResult
```

**Query Pipeline:**
```
Query: "HDFC Bank news"
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 1: Entity Extraction               │
│   → Company: "HDFC Bank Limited"        │
│   → Symbol: "HDFCBANK"                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 2: Query Expansion                 │
│   → "HDFC Bank news HDFC Bank Limited   │
│      Banking Financial Services"        │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 3: Hybrid Search                   │
│   → 70% Dense (FAISS cosine sim)        │
│   → 30% Sparse (BM25 keyword)           │
│   → Top 20 candidates                   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 4: Cross-Encoder Re-ranking        │
│   → ms-marco-MiniLM-L-6-v2              │
│   → Re-score query-document pairs       │
│   → Top 10 results                      │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 5: Entity Boosting                 │
│   → +0.3 for exact company match        │
│   → +0.2 for sector match               │
│   → +0.25 for regulator match           │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Step 6: LLM Answer Generation           │
│   → Claude generates natural language   │
│   → Filters to relevant sources only    │
│   → Returns explanation + stories       │
└─────────────────────────────────────────┘
```

---

## Data Flow

### Ingestion Flow

```
Raw Articles (JSON/Dict)
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                             │
│                                                                       │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐        │
│  │Ingestion│───▶│  Dedup   │───▶│  Entity  │───▶│  Stock   │        │
│  │  Agent  │    │  Agent   │    │ Extract  │    │  Impact  │        │
│  └─────────┘    └──────────┘    └──────────┘    └──────────┘        │
│       │              │               │               │               │
│       │              │               │               ▼               │
│       │              │               │        ┌──────────┐          │
│       │              │               │        │ Storage  │          │
│       │              │               │        │  Agent   │          │
│       │              │               │        └──────────┘          │
│       │              │               │               │               │
│       ▼              ▼               ▼               ▼               │
│  [Article]     [UniqueStory]   [+Entities]    [+Impacts]            │
│                 [+Embedding]    [+Sectors]     [Indexed]            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
Persisted to Disk
```

### Query Flow

```
User Query: "HDFC Bank news"
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          QUERY PIPELINE                               │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                       QUERY AGENT                              │  │
│  │                                                                │  │
│  │   [Entity Extraction] → [Query Expansion] → [Hybrid Search]   │  │
│  │           ↓                    ↓                  ↓           │  │
│  │   "HDFC Bank Limited"   + sectors/aliases    FAISS + BM25    │  │
│  │                                                                │  │
│  │   [Re-ranking] → [Entity Boosting] → [LLM Answer]             │  │
│  │        ↓               ↓                   ↓                   │  │
│  │   CrossEncoder    Score boost        Claude summary           │  │
│  │                                                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
    │
    ▼
QueryResult {
  query: "HDFC Bank news",
  stories: [UniqueStory, ...],
  matched_entities: [Entity, ...],
  explanation: "Based on your query..."
}
```

---

## Technical Decisions

### Why LangGraph?

| Requirement | LangGraph Feature |
|-------------|-------------------|
| Multi-agent orchestration | StateGraph with typed state |
| Conditional routing | `add_conditional_edges` |
| State persistence | Built-in checkpointing |
| Debugging | Visualization tools |
| Extensibility | Node-based composition |

### Why FAISS + BM25 Hybrid?

```
Dense Search (FAISS)     Sparse Search (BM25)      Hybrid (α=0.7)
─────────────────────    ────────────────────     ────────────────────
✓ Semantic similarity    ✓ Exact keyword match    ✓ Best of both
✓ "RBI" → "central bank" ✓ Rare terms preserved   ✓ 70% semantic
✗ Rare terms lost        ✗ No synonym matching    ✓ 30% keyword
```

### Why Cross-Encoder Re-ranking?

Bi-encoders (embedding search) trade accuracy for speed. Cross-encoders are more accurate but slower:

```
Stage 1: Bi-encoder retrieval (fast, top 20)
    → all-mpnet-base-v2 (768-dim embeddings)
    
Stage 2: Cross-encoder re-ranking (accurate, top 10)
    → ms-marco-MiniLM-L-6-v2 (query-document pairs)
```

### Why Union-Find for Deduplication?

| Algorithm | Time Complexity | Pros | Cons |
|-----------|-----------------|------|------|
| Agglomerative | O(n² log n) | Dendrogram | Slow |
| K-Means | O(nk) | Fast | Need to specify k |
| **Union-Find** | O(n² α(n)) ≈ O(n²) | Simple, transitive | Needs threshold |

Union-Find with threshold 0.60 provides:
- Transitive closure (if A~B and B~C, then A~C)
- Simple implementation
- Works well for duplicate detection

---

## Persistence Layer

### Architecture Diagram

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

### Storage Files

| File | Format | Content |
|------|--------|---------|
| `seen_articles.json` | JSON | MD5 hashes of processed articles |
| `vector_store.pkl` | Pickle | Serialized FAISS index |
| `stories.pkl` | Pickle | UniqueStory metadata list |

### Incremental Processing

```
First Run:
  Load 400 articles → Process ALL 400 → Save to disk (120s)

Second Run (No New):
  Load cache (2s) → Filter articles → 0 new → Skip processing

Third Run (5 New):
  Load cache (2s) → Filter → 5 new → Process 5 only (15s)
```

---

## Performance Optimizations

### Embedding Optimization

- **Model:** `all-mpnet-base-v2` (768-dim, normalized)
- **Batching:** Process articles in batches
- **Caching:** Singleton pattern for model loading

### FAISS Optimization

```python
# HNSW parameters
IndexHNSWFlat(dimension=768, M=32)
index.hnsw.efConstruction = 200  # Build-time accuracy
index.hnsw.efSearch = 100        # Query-time accuracy
```

### Lazy Loading

```python
@property
def vector_store(self) -> VectorStore:
    """Lazy load - only creates/loads when first accessed."""
    if self._vector_store is None:
        self._vector_store = self.persistence.load_vector_store()
    return self._vector_store
```

### Agent Singletons

```python
_agents = {}

def _get_agent(agent_class, **kwargs):
    """Singleton pattern for agents."""
    name = agent_class.__name__
    if name not in _agents:
        _agents[name] = agent_class(**kwargs)
    return _agents[name]
```

---

## Class Diagram

```
┌─────────────────────────┐       ┌─────────────────────────┐
│    IntelligenceSystem   │       │       BaseAgent         │
├─────────────────────────┤       ├─────────────────────────┤
│ + persistence           │       │ + verbose: bool         │
│ + vector_store          │       ├─────────────────────────┤
│ + ingestion_graph       │       │ + log(msg)              │
│ + query_graph           │       │ + process(state): dict  │
├─────────────────────────┤       └────────────┬────────────┘
│ + ingest(articles)      │                    │
│ + query(text)           │       ┌────────────┴────────────┐
│ + get_stats()           │       │                         │
└─────────────────────────┘  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
                             │Ingestion│  │  Dedup  │  │ Entity  │
                             │  Agent  │  │  Agent  │  │ Extract │
                             └─────────┘  └─────────┘  └─────────┘
                                                                    
┌─────────────────────────┐  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐
│  PersistenceManager     │  │  Stock  │  │ Storage │  │  Query  │
├─────────────────────────┤  │ Impact  │  │  Agent  │  │  Agent  │
│ + storage_dir           │  └─────────┘  └─────────┘  └─────────┘
│ + seen_articles_file    │
│ + vector_store_file     │  ┌─────────────────────────────────────┐
│ + stories_file          │  │            VectorStore              │
├─────────────────────────┤  ├─────────────────────────────────────┤
│ + filter_new_articles() │  │ + index: faiss.Index                │
│ + mark_as_seen()        │  │ + stories: List[UniqueStory]        │
│ + load_vector_store()   │  │ + _bm25: BM25Okapi                  │
│ + save_vector_store()   │  ├─────────────────────────────────────┤
│ + clear_cache()         │  │ + add(stories)                      │
└─────────────────────────┘  │ + search(embedding, text, k, alpha) │
                             │ + clear()                            │
                             └─────────────────────────────────────┘
```

---

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

---

## Summary

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| Framework | LangGraph | Multi-agent orchestration, state management |
| Embeddings | all-mpnet-base-v2 | Balance of quality and speed |
| Vector DB | FAISS HNSW | Scalable approximate search |
| Sparse Search | BM25Okapi | Keyword matching complement |
| Re-ranker | CrossEncoder | Accuracy boost for top results |
| LLM | Claude 3.5 | High-quality explanations |
| NER | spaCy + rules | Speed with customizability |
| Clustering | Union-Find | Simple transitive clustering |
| Persistence | Pickle + JSON | Simple, no external deps |

---

✅ **Lazy Loading**: Cache loaded only when accessed  
✅ **Incremental Updates**: Process only new articles  
✅ **Automatic Persistence**: Transparent to user  
✅ **Fast Startup**: 2 seconds for cached data  
✅ **Space Efficient**: ~10KB per article
