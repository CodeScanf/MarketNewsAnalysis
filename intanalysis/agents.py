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
from time import perf_counter
import re
import uuid
import numpy as np
import spacy

from intanalysis.models import (
    Article, Entity, EntityType, StockImpact, ImpactType,
    ProcessedArticle, QueryTiming, UniqueStory, QueryResult
)
from intanalysis.mappings import (
    COMPANY_TO_STOCK, SECTOR_TO_COMPANIES, REGULATORS,
    find_stock_symbols, get_stock_symbol, get_companies_in_sector, get_sectors_for_company
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
    
    DIGEST_TITLE_PATTERNS = (
        r"^\d+点\d*氪",
        r"早报",
        r"晚报",
        r"导览",
        r"汇总",
        r"合集",
        r"要闻",
    )

    def __init__(
        self,
        threshold: float = 0.72,
        title_threshold: float = 0.45,
        strong_threshold: float = 0.84,
        verbose: bool = False,
    ):
        super().__init__(verbose)
        self.threshold = threshold
        self.title_threshold = title_threshold
        self.strong_threshold = strong_threshold
        self.embedder = EmbeddingService.get_instance()
    
    def process(self, state: dict) -> dict:
        articles: list[Article] = state.get("articles", [])
        if not articles:
            state["unique_stories"] = []
            return state
        
        # Compute embeddings
        texts = [a.full_text for a in articles]
        embeddings = self.embedder.embed_batch(texts)
        title_embeddings = self.embedder.embed_batch([a.title for a in articles])
        
        # Compute similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)
        title_sim_matrix = np.dot(title_embeddings, title_embeddings.T)
        
        # Cluster using Union-Find
        clusters = self._cluster_articles(articles, sim_matrix, title_sim_matrix)
        
        # Build UniqueStory objects
        unique_stories = []
        processed_articles = []
        
        for cluster_id, indices in clusters.items():
            primary_idx = self._select_primary_index(indices, articles, sim_matrix)
            duplicate_indices = [idx for idx in indices if idx != primary_idx]
            
            primary = ProcessedArticle(
                article=articles[primary_idx],
                embedding=embeddings[primary_idx].tolist(),
                cluster_id=cluster_id,
                is_duplicate=False
            )
            processed_articles.append(primary)
            
            duplicates = []
            for idx in duplicate_indices:
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
    
    def _cluster_articles(
        self,
        articles: list[Article],
        sim_matrix: np.ndarray,
        title_sim_matrix: np.ndarray,
    ) -> dict[str, list[int]]:
        """Cluster articles using guarded Union-Find based on similarity."""
        n = len(articles)
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
                if self._should_cluster(
                    articles[i],
                    articles[j],
                    float(sim_matrix[i, j]),
                    float(title_sim_matrix[i, j]),
                ):
                    union(i, j)
        
        # Group by root
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)
        
        # Assign UUIDs to clusters
        return {str(uuid.uuid4()): indices for indices in clusters.values()}

    def _should_cluster(
        self,
        article_a: Article,
        article_b: Article,
        full_similarity: float,
        title_similarity: float,
    ) -> bool:
        """Decide whether two articles should be merged into one story."""
        if article_a.url and article_b.url and article_a.url == article_b.url:
            return True

        title_a = self._normalize_title(article_a.title)
        title_b = self._normalize_title(article_b.title)
        if title_a and title_a == title_b:
            return True

        if full_similarity < self.threshold:
            return False

        digest_a = self._is_digest_title(article_a.title)
        digest_b = self._is_digest_title(article_b.title)

        # Round-up posts often mention many companies and should not absorb standalone stories.
        if digest_a != digest_b:
            return False

        if title_similarity >= self.title_threshold:
            return True

        if full_similarity >= self.strong_threshold and self._lead_overlap(article_a.content, article_b.content):
            return True

        return False

    def _select_primary_index(
        self,
        indices: list[int],
        articles: list[Article],
        sim_matrix: np.ndarray,
    ) -> int:
        """Choose the best representative article for a cluster."""
        if len(indices) == 1:
            return indices[0]

        best_idx = indices[0]
        best_score = float("-inf")

        for idx in indices:
            similarity_score = sum(float(sim_matrix[idx, other]) for other in indices if other != idx)
            content_bonus = min(len(articles[idx].content), 4000) / 4000.0
            digest_penalty = 1.0 if self._is_digest_title(articles[idx].title) else 0.0
            score = similarity_score + (0.1 * content_bonus) - digest_penalty
            if score > best_score:
                best_idx = idx
                best_score = score

        return best_idx

    def _normalize_title(self, title: str) -> str:
        return re.sub(r"\s+", " ", (title or "").strip().lower())

    def _is_digest_title(self, title: str) -> bool:
        clean_title = title or ""
        return any(re.search(pattern, clean_title, re.IGNORECASE) for pattern in self.DIGEST_TITLE_PATTERNS)

    def _lead_overlap(self, content_a: str, content_b: str, limit: int = 180) -> bool:
        lead_a = re.sub(r"\s+", " ", (content_a or "")[:limit]).strip()
        lead_b = re.sub(r"\s+", " ", (content_b or "")[:limit]).strip()
        if not lead_a or not lead_b:
            return False
        return lead_a in lead_b or lead_b in lead_a


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
            aliases = info.get("aliases", [])
            if key in text_lower or info["full_name"].lower() in text_lower or any(alias.lower() in text_lower for alias in aliases):
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
        process_started = perf_counter()
        step_timings: dict[str, float] = {}

        query = state.get("query", "")
        if not query:
            return state
        
        vector_store: VectorStore = state.get("vector_store")
        if not vector_store:
            state["query_result"] = QueryResult(
                query=query,
                explanation="No articles indexed yet",
                timing=QueryTiming(
                    pipeline_ms=round((perf_counter() - process_started) * 1000, 1),
                    stages=step_timings,
                ),
            )
            return state
        
        # Step 1: Extract query entities
        self.log("Step 1: Extracting entities from query...")
        step_started = perf_counter()
        query_entities = self._extract_query_entities(query)
        step_timings["extract_entities_ms"] = round((perf_counter() - step_started) * 1000, 1)
        if query_entities:
            self.log(f"   Found entities: {[e['name'] for e in query_entities]}")
        else:
            self.log("   No specific entities found, using semantic search")
        
        # Step 2: Expand query with related terms
        self.log("Step 2: Expanding query with related terms...")
        step_started = perf_counter()
        expanded_query = self._expand_query(query, query_entities)
        step_timings["expand_query_ms"] = round((perf_counter() - step_started) * 1000, 1)
        self.log(f"   Expanded: \"{expanded_query[:80]}...\"" if len(expanded_query) > 80 else f"   Expanded: \"{expanded_query}\"")
        
        # Step 3: Hybrid search (dense + BM25)
        self.log("Step 3: Running hybrid search (70% dense + 30% BM25)...")
        step_started = perf_counter()
        query_embedding = self.embedder.embed(expanded_query)
        results = vector_store.search(
            query_embedding, 
            query_text=expanded_query,
            k=20,  # Get more candidates for re-ranking
            alpha=0.7  # 70% dense, 30% BM25
        )
        step_timings["search_ms"] = round((perf_counter() - step_started) * 1000, 1)
        self.log(f"   Retrieved {len(results)} candidates")
        
        # Step 4: Re-rank with cross-encoder
        if self.reranker and len(results) > 5:
            self.log("Step 4: Re-ranking with cross-encoder...")
            step_started = perf_counter()
            results = self.reranker.rerank(query, results, top_k=10)
            step_timings["rerank_ms"] = round((perf_counter() - step_started) * 1000, 1)
            self.log(f"   Re-ranked to top {len(results)} results")
        else:
            step_timings["rerank_ms"] = 0.0
            self.log("Step 4: Skipping re-ranking (small result set)")
        
        # Step 5: Boost by entity match
        self.log("Step 5: Applying entity-based boosting...")
        step_started = perf_counter()
        filtered_results = self._filter_by_entities(results, query_entities)
        step_timings["entity_boost_ms"] = round((perf_counter() - step_started) * 1000, 1)
        
        # Limit to top 10 results for LLM evaluation
        top_results = filtered_results[:10]
        
        # Step 6: Generate intelligent answer and filter relevant results
        self.log("Step 6: Generating AI answer and filtering relevant sources...")
        step_started = perf_counter()
        explanation = None
        final_results = []
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
                llm_response = self.llm.explain_query_results(query, result_summaries)
                
                # Handle structured response
                if isinstance(llm_response, dict):
                    explanation = llm_response.get("explanation", "")
                    relevant_indices = llm_response.get("relevant_indices", [])
                    
                    # Filter to only relevant results
                    final_results = [top_results[i] for i in relevant_indices if i < len(top_results)]
                    self.log(f"   LLM identified {len(final_results)} relevant sources out of {len(top_results)}")
                else:
                    # Fallback for string response
                    explanation = llm_response
                    final_results = top_results[:5]
            except Exception as e:
                self.log(f"   Warning: Could not generate AI answer: {e}")
                final_results = top_results[:5]
        else:
            final_results = top_results[:5]
        step_timings["answer_ms"] = round((perf_counter() - step_started) * 1000, 1)

        # Determine final entities based on whether we have relevant results
        final_entities = query_entities if final_results else []
        pipeline_ms = round((perf_counter() - process_started) * 1000, 1)

        state["query_result"] = QueryResult(
            query=query,
            stories=[s for s, _ in final_results],
            matched_entities=[Entity(name=e["name"], type=e["type"]) for e in final_entities],
            explanation=explanation,
            timing=QueryTiming(
                pipeline_ms=pipeline_ms,
                stages=step_timings,
            ),
        )

        self.log(f"Query '{query}' returned {len(final_results)} results")
        return state
    
    def _extract_query_entities(self, query: str) -> list[dict]:
        """Extract entities from query string."""
        entities = []
        query_lower = query.lower()
        
        # Check companies
        for symbol, name, _ in find_stock_symbols(query_lower):
            entities.append({"name": name, "type": EntityType.COMPANY, "symbol": symbol})
        
        # Check regulators
        for key, info in REGULATORS.items():
            aliases = info.get("aliases", [])
            if key in query_lower or info["full_name"].lower() in query_lower or any(alias.lower() in query_lower for alias in aliases):
                entities.append({"name": info["full_name"], "type": EntityType.REGULATOR})
                break
        
        # Check sectors
        sector_keywords = {
            "banking": "Banking",
            "bank": "Banking",
            "银行": "Chinese Banking",
            "aviation": "Aviation",
            "航空": "Aviation",
            "it": "IT",
            "tech": "IT",
            "科技": "Internet",
            "互联网": "Internet",
            "电商": "E-Commerce",
            "零售": "Consumer",
            "消费": "Consumer",
            "auto": "Automobile",
            "automobile": "Automobile",
            "汽车": "EV",
            "新能源车": "EV",
            "电动车": "EV",
            "家电": "Home Appliances"
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
            title_text = story.primary_article.article.title.lower()
            content_text = story.primary_article.article.content.lower()
            impact_symbols = {impact.symbol for impact in story.primary_article.stock_impacts}
            
            for qe in query_entities:
                if qe["type"] == EntityType.COMPANY:
                    query_terms = {qe["name"].lower()}
                    symbol = qe.get("symbol")
                    if symbol:
                        for _, (company_symbol, company_name, aliases) in COMPANY_TO_STOCK.items():
                            if company_symbol == symbol:
                                query_terms.add(company_name.lower())
                                query_terms.update(alias.lower() for alias in aliases)
                                break

                    if any(term in title_text for term in query_terms):
                        boost += 0.9
                    elif any(term in content_text for term in query_terms):
                        boost += 0.25
                    elif qe["name"].lower() in article_entities:
                        boost += 0.15

                    if symbol and symbol in impact_symbols:
                        boost += 0.15
                elif qe["type"] == EntityType.SECTOR and qe["name"] in article_sectors:
                    boost += 0.2
                elif qe["type"] == EntityType.REGULATOR and qe["name"].lower() in article_entities:
                    boost += 0.25
            
            scored.append((story, score + boost))
        
        # Sort by boosted score
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
