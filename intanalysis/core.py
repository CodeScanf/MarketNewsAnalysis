"""Core IntelligenceSystem - main interface for the package."""

from typing import Optional
from dotenv import load_dotenv

from intanalysis.models import Article, QueryResult, UniqueStory
from intanalysis.embeddings import VectorStore, EmbeddingService
from intanalysis.workflow import build_ingestion_graph, build_query_graph, PipelineState


class IntelligenceSystem:
    """
    Main interface for the Financial News Intelligence System.
    
    Example:
        system = IntelligenceSystem()
        
        # Ingest articles
        result = system.ingest([
            {"title": "HDFC Bank announces dividend", "content": "..."},
            {"title": "RBI raises rates", "content": "..."},
        ])
        
        # Query
        response = system.query("HDFC Bank news")
        for story in response.stories:
            print(story.primary_article.article.title)
    """
    
    def __init__(self, verbose: bool = True):
        """Initialize the intelligence system."""
        load_dotenv()
        
        self.verbose = verbose
        self._vector_store: Optional[VectorStore] = None
        self._ingestion_graph = None
        self._query_graph = None
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create vector store."""
        if self._vector_store is None:
            embedder = EmbeddingService.get_instance()
            self._vector_store = VectorStore(dimension=embedder.dimension)
        return self._vector_store
    
    @property
    def ingestion_graph(self):
        """Get compiled ingestion graph."""
        if self._ingestion_graph is None:
            self._ingestion_graph = build_ingestion_graph()
        return self._ingestion_graph
    
    @property
    def query_graph(self):
        """Get compiled query graph."""
        if self._query_graph is None:
            self._query_graph = build_query_graph()
        return self._query_graph
    
    def ingest(self, articles: list[dict | Article]) -> dict:
        """
        Ingest articles through the full pipeline.
        
        Args:
            articles: List of articles (dicts or Article objects)
            
        Returns:
            Processing result with unique stories and stats
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📥 INGESTING {len(articles)} ARTICLES")
            print(f"{'='*60}")
        
        initial_state: PipelineState = {
            "raw_articles": articles,
            "vector_store": self.vector_store,
            "errors": [],
        }
        
        result = self.ingestion_graph.invoke(initial_state)
        
        # Update our vector store reference
        self._vector_store = result.get("vector_store", self._vector_store)
        
        unique_stories = result.get("unique_stories", [])
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ INGESTION COMPLETE")
            print(f"   • Unique stories: {len(unique_stories)}")
            print(f"   • Duplicates removed: {len(articles) - len(unique_stories)}")
            print(f"   • Indexed: {self.vector_store.index.ntotal}")
            print(f"{'='*60}\n")
        
        return {
            "unique_stories": unique_stories,
            "total_articles": len(articles),
            "unique_count": len(unique_stories),
            "duplicate_count": len(articles) - len(unique_stories),
        }
    
    def query(self, query_text: str, show_steps: bool = True) -> QueryResult:
        """
        Query the system for relevant news.
        
        Args:
            query_text: Natural language query
            show_steps: Show intermediate processing steps
            
        Returns:
            QueryResult with matched stories and explanation
        """
        if self.verbose and show_steps:
            print(f"\n{'─'*50}")
            print(f"🔍 PROCESSING QUERY: \"{query_text}\"")
            print(f"{'─'*50}")
        
        initial_state: PipelineState = {
            "query": query_text,
            "vector_store": self.vector_store,
            "errors": [],
        }
        
        result = self.query_graph.invoke(initial_state)
        query_result = result.get("query_result", QueryResult(query=query_text))
        
        if self.verbose:
            print(f"\n📊 RESULTS: {len(query_result.stories)} stories found")
            if query_result.explanation:
                print(f"💡 {query_result.explanation}")
            print(f"{'='*60}\n")
        
        return query_result
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "indexed_stories": self.vector_store.index.ntotal,
            "total_stories": len(self.vector_store.stories),
        }
