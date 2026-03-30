"""Core IntelligenceSystem - main interface for the package."""

import sys
from typing import Optional
from time import perf_counter
from dotenv import load_dotenv

# Load .env before importing modules that may read HF-related env vars at import time.
load_dotenv()

from intanalysis.models import Article, QueryResult, QueryTiming, UniqueStory
from intanalysis.embeddings import VectorStore, EmbeddingService
from intanalysis.workflow import build_ingestion_graph, build_query_graph, PipelineState
from intanalysis.persistence import PersistenceManager


def _configure_console_encoding() -> None:
    """Prefer UTF-8 console output to avoid Windows GBK encoding errors."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_console_encoding()


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
    
    def __init__(self, verbose: bool = True, storage_dir: str = "dataset"):
        """Initialize the intelligence system.
        
        Args:
            verbose: Print processing information
            storage_dir: Directory for persistence files
        """
        load_dotenv()
        
        self.verbose = verbose
        self.persistence = PersistenceManager(storage_dir=storage_dir)
        self._vector_store: Optional[VectorStore] = None
        self._ingestion_graph = None
        self._query_graph = None
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create vector store, loading from disk if available."""
        if self._vector_store is None:
            embedder = EmbeddingService.get_instance()
            # Try loading from disk first
            self._vector_store = self.persistence.load_vector_store(dimension=embedder.dimension)
            if self._vector_store is None:
                self._vector_store = VectorStore(dimension=embedder.dimension)
            elif self.verbose:
                print(f"📂 Loaded {len(self._vector_store.stories)} existing stories from disk")
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
    
    def ingest(self, articles: list[dict | Article], force: bool = False) -> dict:
        """
        Ingest articles through the full pipeline (incremental).
        
        Args:
            articles: List of articles (dicts or Article objects)
            force: Force re-ingestion of all articles (ignore cache)
            
        Returns:
            Processing result with unique stories and stats
        """
        # Filter out already processed articles
        article_dicts = [
            art if isinstance(art, dict) else art.__dict__
            for art in articles
        ]
        
        if not force:
            new_articles, skipped = self.persistence.filter_new_articles(article_dicts)
            if self.verbose and skipped > 0:
                print(f"\n📌 Skipping {skipped} already processed articles")
        else:
            new_articles = article_dicts
            skipped = 0
        
        if not new_articles:
            if self.verbose:
                print(f"\n✅ No new articles to process")
            return {
                "unique_stories": [],
                "total_articles": len(articles),
                "unique_count": 0,
                "duplicate_count": 0,
                "skipped_count": skipped,
            }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📥 INGESTING {len(new_articles)} NEW ARTICLES")
            print(f"{'='*60}")
        
        initial_state: PipelineState = {
            "raw_articles": new_articles,
            "vector_store": self.vector_store,
            "errors": [],
        }
        
        result = self.ingestion_graph.invoke(initial_state)
        
        # Update our vector store reference
        self._vector_store = result.get("vector_store", self._vector_store)
        
        unique_stories = result.get("unique_stories", [])
        
        # Mark articles as seen and persist vector store
        self.persistence.mark_articles_as_seen(new_articles)
        self.persistence.save_vector_store(self.vector_store)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ INGESTION COMPLETE")
            print(f"   • New unique stories: {len(unique_stories)}")
            print(f"   • Duplicates in batch: {len(new_articles) - len(unique_stories)}")
            print(f"   • Total indexed: {self.vector_store.index.ntotal}")
            print(f"   • Persisted to disk ✓")
            print(f"{'='*60}\n")
        
        return {
            "unique_stories": unique_stories,
            "total_articles": len(articles),
            "unique_count": len(unique_stories),
            "duplicate_count": len(new_articles) - len(unique_stories),
            "skipped_count": skipped,
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
        
        query_started = perf_counter()
        result = self.query_graph.invoke(initial_state)
        query_result = result.get("query_result", QueryResult(query=query_text))
        if query_result.timing is None:
            query_result.timing = QueryTiming()
        query_result.timing.pipeline_ms = round((perf_counter() - query_started) * 1000, 1)
        
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
