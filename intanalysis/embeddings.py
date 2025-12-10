"""Embedding and vector store services with hybrid search."""

from typing import Optional, List, Tuple, Any
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

from intanalysis.models import ProcessedArticle, UniqueStory


class EmbeddingService:
    """Sentence-transformer embeddings with singleton pattern."""
    
    _instance: Optional["EmbeddingService"] = None
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Use all-mpnet-base-v2 (768-dim) for better quality."""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts."""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


class Reranker:
    """Cross-encoder re-ranker for improved precision."""
    
    _instance: Optional["Reranker"] = None
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name, max_length=512)
    
    @classmethod
    def get_instance(cls) -> "Reranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def rerank(
        self, 
        query: str, 
        results: List[Tuple[UniqueStory, float]], 
        top_k: int = 5
    ) -> List[Tuple[UniqueStory, float]]:
        """Re-rank results using cross-encoder."""
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, story.primary_article.article.full_text[:1000]) for story, _ in results]
        
        # Score with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)
        
        # Combine and sort
        scored = [(story, float(score)) for (story, _), score in zip(results, scores)]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[:top_k]


class VectorStore:
    """FAISS HNSW vector store with hybrid BM25 search."""
    
    def __init__(self, dimension: int = 768, use_hnsw: bool = True):
        self.dimension = dimension
        
        # Use HNSW for scalability, or FlatIP for small datasets
        if use_hnsw and dimension > 0:
            self.index = faiss.IndexHNSWFlat(dimension, 32)  # M=32 connections
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 100
        else:
            self.index = faiss.IndexFlatIP(dimension)
        
        self.stories: List[UniqueStory] = []
        self._bm25: Optional[BM25Okapi] = None
        self._corpus_texts: List[str] = []
    
    def add(self, stories: List[UniqueStory]) -> None:
        """Add stories with their embeddings and build BM25 index."""
        embeddings = []
        for story in stories:
            if story.primary_article.embedding:
                embeddings.append(story.primary_article.embedding)
                self.stories.append(story)
                self._corpus_texts.append(story.primary_article.article.full_text.lower())
        
        if embeddings:
            arr = np.array(embeddings, dtype=np.float32)
            self.index.add(arr)
            
            # Build BM25 index
            tokenized = [text.split() for text in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        query_text: str = "",
        k: int = 10,
        alpha: float = 0.7  # Weight for dense search (1-alpha for BM25)
    ) -> List[Tuple[UniqueStory, float]]:
        """Hybrid search combining dense vectors and BM25."""
        if self.index.ntotal == 0:
            return []
        
        k = min(k, self.index.ntotal)
        
        # Dense search
        query = query_embedding.reshape(1, -1).astype(np.float32)
        dense_scores, dense_indices = self.index.search(query, k * 2)
        
        # Normalize dense scores to [0, 1]
        dense_scores = dense_scores[0]
        if dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())
        
        # BM25 search
        bm25_scores = np.zeros(len(self.stories))
        if self._bm25 and query_text:
            bm25_raw = self._bm25.get_scores(query_text.lower().split())
            if bm25_raw.max() > 0:
                bm25_scores = bm25_raw / bm25_raw.max()  # Normalize
        
        # Fuse scores
        fused_scores = {}
        for i, idx in enumerate(dense_indices[0]):
            if idx >= 0 and idx < len(self.stories):
                dense_score = dense_scores[i]
                bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0
                fused_scores[idx] = alpha * dense_score + (1 - alpha) * bm25_score
        
        # Sort by fused score
        sorted_indices = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        
        results = []
        for idx in sorted_indices[:k]:
            results.append((self.stories[idx], fused_scores[idx]))
        
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        if hasattr(self.index, 'reset'):
            self.index.reset()
        else:
            self.index = faiss.IndexFlatIP(self.dimension)
        self.stories.clear()
        self._corpus_texts.clear()
        self._bm25 = None
        scores, indices = self.index.search(query, min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.stories):
                results.append((self.stories[idx], float(score)))
        return results
    
    def clear(self) -> None:
        """Clear the index."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.stories.clear()
