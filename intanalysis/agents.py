"""Six agents for the Financial News Intelligence System.

Implements:
1. News Ingestion Agent - Load and validate articles
2. Deduplication Agent - Cluster similar articles (≥95% accuracy)
3. Entity Extraction Agent - Extract entities (≥90% precision)
4. Stock Impact Agent - Map news to stocks
5. Storage Agent - Index for retrieval
6. Query Agent - Context-aware search
"""

from abc import ABC, abstractmethod
from typing import Optional
import uuid
import numpy as np
import spacy

from intanalysis.models import (
    Article, Entity, EntityType, StockImpact, ImpactType,
    ProcessedArticle, UniqueStory, QueryResult
)
from intanalysis.mappings import (
    COMPANY_TO_STOCK, SECTOR_TO_COMPANIES, REGULATORS,
    get_stock_symbol, get_companies_in_sector, get_sectors_for_company
)
from intanalysis.embeddings import EmbeddingService, VectorStore
from intanalysis.llm import LLMService


class BaseAgent(ABC):
    """Base agent with common interface."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.__class__.__name__}] {msg}")
    
    @abstractmethod
    def process(self, state: dict) -> dict:
        """Process state and return updated state."""
        pass


# =============================================================================
# Agent 1: News Ingestion
# =============================================================================

class IngestionAgent(BaseAgent):
    """Loads and validates news articles."""
    
    def process(self, state: dict) -> dict:
        raw_articles = state.get("raw_articles", [])
        
        articles = []
        for item in raw_articles:
            if isinstance(item, Article):
                articles.append(item)
            elif isinstance(item, dict):
                articles.append(Article(**item))
        
        self.log(f"Ingested {len(articles)} articles")
        state["articles"] = articles
        return state


# =============================================================================
# Agent 2: Deduplication (Target: ≥95% accuracy)
# =============================================================================

class DeduplicationAgent(BaseAgent):
    """Clusters similar articles using semantic embeddings."""
    
    def __init__(self, threshold: float = 0.60, verbose: bool = False):
        super().__init__(verbose)
        self.threshold = threshold
        self.embedder = EmbeddingService.get_instance()
    
    def process(self, state: dict) -> dict:
        articles: list[Article] = state.get("articles", [])
        if not articles:
            state["unique_stories"] = []
            return state
        
        # Compute embeddings
        texts = [a.full_text for a in articles]
        embeddings = self.embedder.embed_batch(texts)
        
        # Compute similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        # Cluster using Union-Find
        clusters = self._cluster_articles(len(articles), sim_matrix)
        
        # Build UniqueStory objects
        unique_stories = []
        processed_articles = []
        
        for cluster_id, indices in clusters.items():
            primary_idx = indices[0]
            
            primary = ProcessedArticle(
                article=articles[primary_idx],
                embedding=embeddings[primary_idx].tolist(),
                cluster_id=cluster_id,
                is_duplicate=False
            )
            processed_articles.append(primary)
            
            duplicates = []
            for idx in indices[1:]:
                dup = ProcessedArticle(
                    article=articles[idx],
                    embedding=embeddings[idx].tolist(),
                    cluster_id=cluster_id,
                    is_duplicate=True
                )
                duplicates.append(dup)
            
            unique_stories.append(UniqueStory(
                id=cluster_id,
                primary_article=primary,
                duplicate_articles=duplicates
            ))
        
        dup_count = len(articles) - len(unique_stories)
        self.log(f"Found {len(unique_stories)} unique stories, {dup_count} duplicates")
        
        state["unique_stories"] = unique_stories
        state["processed_articles"] = processed_articles
        return state
    
    def _cluster_articles(self, n: int, sim_matrix: np.ndarray) -> dict[str, list[int]]:
        """Cluster articles using Union-Find based on similarity."""
        parent = list(range(n))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # Union similar articles
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.threshold:
                    union(i, j)
        
        # Group by root
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)
        
        # Assign UUIDs to clusters
        return {str(uuid.uuid4()): indices for indices in clusters.values()}


# =============================================================================
# Agent 3: Entity Extraction (Target: ≥90% precision)
# =============================================================================

class EntityExtractionAgent(BaseAgent):
    """Extracts entities using spaCy + rule-based + LLM fallback."""
    
    def __init__(self, use_llm: bool = True, verbose: bool = False):
        super().__init__(verbose)
        self.use_llm = use_llm
        self._nlp = None
        self._llm = None
    
    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                self._nlp = spacy.blank("en")
        return self._nlp
    
    @property
    def llm(self) -> LLMService:
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMService.get_instance()
            except Exception:
                self._llm = None
        return self._llm
    
    def process(self, state: dict) -> dict:
        unique_stories: list[UniqueStory] = state.get("unique_stories", [])
        
        for story in unique_stories:
            text = story.primary_article.article.full_text
            entities = self._extract_entities(text)
            sectors = self._extract_sectors(entities)
            
            story.primary_article.entities = entities
            story.primary_article.sectors = sectors
            
            self.log(f"Extracted {len(entities)} entities from '{story.primary_article.article.title[:40]}...'")
        
        state["unique_stories"] = unique_stories
        return state
    
    def _extract_entities(self, text: str) -> list[Entity]:
        """Multi-method entity extraction."""
        entities = []
        text_lower = text.lower()
        
        # Rule-based: Known companies
        for key, (symbol, name, aliases) in COMPANY_TO_STOCK.items():
            if key in text_lower or any(a in text_lower for a in aliases):
                entities.append(Entity(name=name, type=EntityType.COMPANY, confidence=1.0))
        
        # Rule-based: Known regulators
        for key, info in REGULATORS.items():
            if key in text_lower or info["full_name"].lower() in text_lower:
                entities.append(Entity(name=info["full_name"], type=EntityType.REGULATOR, confidence=1.0))
        
        # spaCy NER for persons and orgs
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(Entity(name=ent.text, type=EntityType.PERSON, confidence=0.85))
            elif ent.label_ == "ORG" and not any(e.name.lower() == ent.text.lower() for e in entities):
                entities.append(Entity(name=ent.text, type=EntityType.COMPANY, confidence=0.7))
        
        # LLM fallback for complex cases
        if self.llm and len(entities) < 2:
            try:
                llm_entities = self.llm.extract_entities(text)
                for e in llm_entities:
                    if not any(existing.name.lower() == e["name"].lower() for existing in entities):
                        entities.append(Entity(
                            name=e["name"],
                            type=EntityType(e["type"]),
                            confidence=e.get("confidence", 0.8)
                        ))
            except Exception:
                pass
        
        return entities
    
    def _extract_sectors(self, entities: list[Entity]) -> list[str]:
        """Infer sectors from entities."""
        sectors = set()
        for entity in entities:
            if entity.type == EntityType.COMPANY:
                # Find company symbol
                for key, (symbol, name, _) in COMPANY_TO_STOCK.items():
                    if name.lower() == entity.name.lower():
                        sectors.update(get_sectors_for_company(symbol))
                        break
            elif entity.type == EntityType.REGULATOR:
                for key, info in REGULATORS.items():
                    if info["full_name"] == entity.name:
                        sectors.update(info.get("sectors", []))
        return list(sectors)


# =============================================================================
# Agent 4: Stock Impact Analysis
# =============================================================================

class StockImpactAgent(BaseAgent):
    """Maps news to impacted stocks with confidence scores."""
    
    def __init__(self, use_llm: bool = True, verbose: bool = False):
        super().__init__(verbose)
        self.use_llm = use_llm
        self._llm = None
    
    @property
    def llm(self) -> LLMService:
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMService.get_instance()
            except Exception:
                self._llm = None
        return self._llm
    
    def process(self, state: dict) -> dict:
        unique_stories: list[UniqueStory] = state.get("unique_stories", [])
        
        for story in unique_stories:
            impacts = self._analyze_impacts(story.primary_article)
            story.primary_article.stock_impacts = impacts
            self.log(f"Mapped {len(impacts)} stock impacts")
        
        state["unique_stories"] = unique_stories
        return state
    
    def _analyze_impacts(self, article: ProcessedArticle) -> list[StockImpact]:
        """Analyze stock impacts from entities."""
        impacts = []
        seen_symbols = set()
        
        for entity in article.entities:
            if entity.type == EntityType.COMPANY:
                # Direct company mention
                result = get_stock_symbol(entity.name)
                if result:
                    symbol, name, _ = result
                    if symbol not in seen_symbols:
                        impacts.append(StockImpact(
                            symbol=symbol,
                            company_name=name,
                            confidence=1.0,
                            impact_type=ImpactType.DIRECT,
                            reasoning="Company directly mentioned"
                        ))
                        seen_symbols.add(symbol)
            
            elif entity.type == EntityType.REGULATOR:
                # Regulatory impact on sector
                for key, info in REGULATORS.items():
                    if info["full_name"] == entity.name:
                        for sector in info.get("sectors", []):
                            for symbol in get_companies_in_sector(sector):
                                if symbol not in seen_symbols:
                                    impacts.append(StockImpact(
                                        symbol=symbol,
                                        company_name=symbol,
                                        confidence=0.6,
                                        impact_type=ImpactType.REGULATORY,
                                        reasoning=f"Regulatory impact via {entity.name}"
                                    ))
                                    seen_symbols.add(symbol)
        
        # Sector-wide impacts
        for sector in article.sectors:
            for symbol in get_companies_in_sector(sector):
                if symbol not in seen_symbols:
                    impacts.append(StockImpact(
                        symbol=symbol,
                        company_name=symbol,
                        confidence=0.7,
                        impact_type=ImpactType.SECTOR,
                        reasoning=f"Sector-wide impact ({sector})"
                    ))
                    seen_symbols.add(symbol)
        
        return impacts


# =============================================================================
# Agent 5: Storage & Indexing
# =============================================================================

class StorageAgent(BaseAgent):
    """Stores articles and builds vector index."""
    
    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self.vector_store: Optional[VectorStore] = None
    
    def process(self, state: dict) -> dict:
        unique_stories: list[UniqueStory] = state.get("unique_stories", [])
        
        # Initialize or get vector store from state
        if "vector_store" not in state:
            embedder = EmbeddingService.get_instance()
            state["vector_store"] = VectorStore(dimension=embedder.dimension)
        
        self.vector_store = state["vector_store"]
        self.vector_store.add(unique_stories)
        
        self.log(f"Indexed {len(unique_stories)} stories, total: {self.vector_store.index.ntotal}")
        state["storage_complete"] = True
        return state


# =============================================================================
# Agent 6: Query Processing
# =============================================================================

class QueryAgent(BaseAgent):
    """Context-aware query processing with hybrid search and re-ranking."""
    
    def __init__(self, use_llm: bool = True, use_reranker: bool = True, verbose: bool = False):
        super().__init__(verbose)
        self.use_llm = use_llm
        self.use_reranker = use_reranker
        self.embedder = EmbeddingService.get_instance()
        self._llm = None
        self._reranker = None
    
    @property
    def llm(self) -> LLMService:
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMService.get_instance()
            except Exception:
                self._llm = None
        return self._llm
    
    @property
    def reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is None and self.use_reranker:
            try:
                from intanalysis.embeddings import Reranker
                self._reranker = Reranker.get_instance()
            except Exception:
                self._reranker = None
        return self._reranker
    
    def process(self, state: dict) -> dict:
        query = state.get("query", "")
        if not query:
            return state
        
        vector_store: VectorStore = state.get("vector_store")
        if not vector_store:
            state["query_result"] = QueryResult(query=query, explanation="No articles indexed yet")
            return state
        
        # Step 1: Extract query entities
        self.log("Step 1: Extracting entities from query...")
        query_entities = self._extract_query_entities(query)
        if query_entities:
            self.log(f"   Found entities: {[e['name'] for e in query_entities]}")
        else:
            self.log("   No specific entities found, using semantic search")
        
        # Step 2: Expand query with related terms
        self.log("Step 2: Expanding query with related terms...")
        expanded_query = self._expand_query(query, query_entities)
        self.log(f"   Expanded: \"{expanded_query[:80]}...\"" if len(expanded_query) > 80 else f"   Expanded: \"{expanded_query}\"")
        
        # Step 3: Hybrid search (dense + BM25)
        self.log("Step 3: Running hybrid search (70% dense + 30% BM25)...")
        query_embedding = self.embedder.embed(expanded_query)
        results = vector_store.search(
            query_embedding, 
            query_text=expanded_query,
            k=20,  # Get more candidates for re-ranking
            alpha=0.7  # 70% dense, 30% BM25
        )
        self.log(f"   Retrieved {len(results)} candidates")
        
        # Step 4: Re-rank with cross-encoder
        if self.reranker and len(results) > 5:
            self.log("Step 4: Re-ranking with cross-encoder...")
            results = self.reranker.rerank(query, results, top_k=10)
            self.log(f"   Re-ranked to top {len(results)} results")
        else:
            self.log("Step 4: Skipping re-ranking (small result set)")
        
        # Step 5: Boost by entity match
        self.log("Step 5: Applying entity-based boosting...")
        filtered_results = self._filter_by_entities(results, query_entities)
        
        # Limit to top 5 results
        top_results = filtered_results[:5]
        
        # Step 6: Generate intelligent answer
        self.log("Step 6: Generating AI answer...")
        explanation = None
        if self.llm and top_results:
            try:
                # Include full content for better answers
                result_summaries = [
                    {
                        "title": s.primary_article.article.title,
                        "source": s.primary_article.article.source,
                        "content": (s.primary_article.article.content or "")[:500],
                        "entities": [e.name for e in s.primary_article.entities]
                    }
                    for s, _ in top_results
                ]
                explanation = self.llm.explain_query_results(query, result_summaries)
            except Exception as e:
                self.log(f"   Warning: Could not generate AI answer: {e}")

        state["query_result"] = QueryResult(
            query=query,
            stories=[s for s, _ in top_results],
            matched_entities=[Entity(name=e["name"], type=e["type"]) for e in query_entities],
            explanation=explanation
        )

        self.log(f"Query '{query}' returned {len(top_results)} results")
        return state
    
    def _extract_query_entities(self, query: str) -> list[dict]:
        """Extract entities from query string."""
        entities = []
        query_lower = query.lower()
        
        # Check companies
        result = get_stock_symbol(query_lower)
        if result:
            symbol, name, _ = result
            entities.append({"name": name, "type": EntityType.COMPANY, "symbol": symbol})
        
        # Check regulators
        for key, info in REGULATORS.items():
            if key in query_lower or info["full_name"].lower() in query_lower:
                entities.append({"name": info["full_name"], "type": EntityType.REGULATOR})
                break
        
        # Check sectors
        sector_keywords = {
            "banking": "Banking", "bank": "Banking", "aviation": "Aviation",
            "it": "IT", "tech": "IT", "auto": "Automobile", "automobile": "Automobile"
        }
        for keyword, sector in sector_keywords.items():
            if keyword in query_lower:
                entities.append({"name": sector, "type": EntityType.SECTOR})
                break
        
        return entities
    
    def _expand_query(self, query: str, entities: list[dict]) -> str:
        """Expand query with related terms."""
        terms = [query]
        
        for entity in entities:
            terms.append(entity["name"])
            
            if entity["type"] == EntityType.COMPANY and "symbol" in entity:
                # Add sector context
                sectors = get_sectors_for_company(entity["symbol"])
                terms.extend(sectors)
            
            elif entity["type"] == EntityType.SECTOR:
                # Add companies in sector
                companies = get_companies_in_sector(entity["name"])
                terms.extend(companies[:3])  # Top 3 companies
        
        return " ".join(terms)
    
    def _filter_by_entities(
        self, 
        results: list[tuple[UniqueStory, float]], 
        query_entities: list[dict]
    ) -> list[tuple[UniqueStory, float]]:
        """Filter and boost results matching query entities."""
        if not query_entities:
            return results
        
        scored = []
        for story, score in results:
            boost = 0
            article_entities = {e.name.lower() for e in story.primary_article.entities}
            article_sectors = set(story.primary_article.sectors)
            
            for qe in query_entities:
                if qe["type"] == EntityType.COMPANY and qe["name"].lower() in article_entities:
                    boost += 0.3
                elif qe["type"] == EntityType.SECTOR and qe["name"] in article_sectors:
                    boost += 0.2
                elif qe["type"] == EntityType.REGULATOR and qe["name"].lower() in article_entities:
                    boost += 0.25
            
            scored.append((story, score + boost))
        
        # Sort by boosted score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
