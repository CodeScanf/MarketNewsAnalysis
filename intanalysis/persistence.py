"""Persistence layer for storing and loading processed articles and vector store."""

import json
import pickle
from pathlib import Path
from typing import Optional, Set
import hashlib

from intanalysis.models import UniqueStory
from intanalysis.embeddings import VectorStore


class PersistenceManager:
    """Manages persistent storage of articles and vector indices."""
    
    def __init__(self, storage_dir: str = "dataset"):
        """Initialize persistence manager.
        
        Args:
            storage_dir: Directory to store persistence files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.seen_articles_file = self.storage_dir / "seen_articles.json"
        self.vector_store_file = self.storage_dir / "vector_store.pkl"
        self.stories_file = self.storage_dir / "stories.pkl"
    
    def _article_hash(self, article: dict) -> str:
        """Generate unique hash for an article based on URL or content.
        
        Args:
            article: Article dictionary
            
        Returns:
            Hash string
        """
        # Prefer URL as unique identifier
        if article.get("url"):
            return hashlib.md5(article["url"].encode()).hexdigest()
        
        # Fallback to title + content hash
        content = f"{article.get('title', '')}{article.get('content', '')[:100]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_seen_articles(self) -> Set[str]:
        """Load set of seen article hashes.
        
        Returns:
            Set of article hashes
        """
        if not self.seen_articles_file.exists():
            return set()
        
        try:
            with open(self.seen_articles_file, 'r') as f:
                data = json.load(f)
                # Support both old format (URLs) and new format (hashes)
                if isinstance(data, dict) and "seen" in data:
                    # Old format - convert URLs to hashes
                    return {hashlib.md5(url.encode()).hexdigest() for url in data["seen"]}
                return set(data)
        except (json.JSONDecodeError, KeyError):
            return set()
    
    def save_seen_articles(self, article_hashes: Set[str]) -> None:
        """Save set of seen article hashes.
        
        Args:
            article_hashes: Set of article hashes to save
        """
        with open(self.seen_articles_file, 'w') as f:
            json.dump(list(article_hashes), f, indent=2)
    
    def filter_new_articles(self, articles: list[dict]) -> tuple[list[dict], int]:
        """Filter out articles that have already been processed.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Tuple of (new_articles, skipped_count)
        """
        seen = self.get_seen_articles()
        new_articles = []
        
        for article in articles:
            article_hash = self._article_hash(article)
            if article_hash not in seen:
                new_articles.append(article)
        
        skipped_count = len(articles) - len(new_articles)
        return new_articles, skipped_count
    
    def mark_articles_as_seen(self, articles: list[dict]) -> None:
        """Mark articles as processed.
        
        Args:
            articles: List of article dictionaries to mark as seen
        """
        seen = self.get_seen_articles()
        for article in articles:
            seen.add(self._article_hash(article))
        self.save_seen_articles(seen)
    
    def load_vector_store(self, dimension: int = 768) -> Optional[VectorStore]:
        """Load persisted vector store.
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            VectorStore instance or None if not found
        """
        if not self.vector_store_file.exists() or not self.stories_file.exists():
            return None
        
        try:
            # Load stories first
            with open(self.stories_file, 'rb') as f:
                stories = pickle.load(f)
            
            # Load vector store
            with open(self.vector_store_file, 'rb') as f:
                index_data = pickle.load(f)
            
            # Reconstruct vector store
            vector_store = VectorStore(dimension=dimension)
            vector_store.stories = stories
            vector_store._corpus_texts = [s.primary_article.article.full_text.lower() for s in stories]
            
            # Deserialize FAISS index
            import faiss
            vector_store.index = faiss.deserialize_index(index_data)
            
            # Rebuild BM25 index
            if vector_store._corpus_texts:
                from rank_bm25 import BM25Okapi
                tokenized = [text.split() for text in vector_store._corpus_texts]
                vector_store._bm25 = BM25Okapi(tokenized)
            
            return vector_store
        except Exception as e:
            print(f"⚠️  Warning: Could not load persisted data: {e}")
            return None
    
    def save_vector_store(self, vector_store: VectorStore) -> None:
        """Persist vector store to disk.
        
        Args:
            vector_store: VectorStore instance to save
        """
        try:
            # Save stories
            with open(self.stories_file, 'wb') as f:
                pickle.dump(vector_store.stories, f)
            
            # Serialize FAISS index
            import faiss
            index_data = faiss.serialize_index(vector_store.index)
            
            with open(self.vector_store_file, 'wb') as f:
                pickle.dump(index_data, f)
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save vector store: {e}")
    
    def clear_cache(self) -> None:
        """Clear all persisted data."""
        for file in [self.seen_articles_file, self.vector_store_file, self.stories_file]:
            if file.exists():
                file.unlink()
