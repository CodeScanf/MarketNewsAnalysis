"""Embedding and vector store services with hybrid search."""

from typing import Optional, List, Tuple
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss

from intanalysis.models import ProcessedArticle, UniqueStory


_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9_.-]+")


def get_model_device() -> str:
    """Default to CPU unless the caller explicitly opts into another device."""
    return os.getenv("INTANALYSIS_DEVICE", "cpu").strip() or "cpu"


def tokenize_text(text: str) -> list[str]:
    """Tokenize mixed Chinese/Latin text for BM25-friendly lexical matching."""
    if not text:
        return []

    tokens: list[str] = []
    for chunk in _TOKEN_RE.findall(text.lower()):
        if _CJK_RE.fullmatch(chunk):
            if len(chunk) == 1:
                tokens.append(chunk)
                continue

            tokens.append(chunk)
            tokens.extend(chunk[i : i + 2] for i in range(len(chunk) - 1))
            if len(chunk) > 2:
                tokens.extend(chunk[i : i + 3] for i in range(len(chunk) - 2))
        else:
            tokens.append(chunk)

    return tokens


class EmbeddingService:
    """Sentence-transformer embeddings with singleton pattern."""
    
    _instance: Optional["EmbeddingService"] = None
    
    def __init__(self, model_name: str = "BAAI/bge-base-zh-v1.5", device: Optional[str] = None):
        """Use a Chinese-friendly embedding model for semantic retrieval."""
        self.device = device or get_model_device()
        self.model = SentenceTransformer(model_name, device=self.device)
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
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: Optional[str] = None):
        self.device = device or get_model_device()
        self.model = CrossEncoder(model_name, max_length=512, device=self.device)
    
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
            self.index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)  # M=32 connections
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
            tokenized = [tokenize_text(text) for text in self._corpus_texts]
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
            query_tokens = tokenize_text(query_text)
            bm25_raw = self._bm25.get_scores(query_tokens) if query_tokens else np.zeros(len(self.stories))
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
