"""Unit tests for intanalysis embeddings module."""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from intanalysis.models import ProcessedArticle, UniqueStory, Article


class TestEmbeddingService:
    """Tests for EmbeddingService class."""
    
    def test_singleton_pattern(self):
        """Test that EmbeddingService uses singleton pattern."""
        from intanalysis.embeddings import EmbeddingService
        
        instance1 = EmbeddingService.get_instance()
        instance2 = EmbeddingService.get_instance()
        
        assert instance1 is instance2
    
    def test_embed_single_text(self):
        """Test embedding a single text string."""
        from intanalysis.embeddings import EmbeddingService
        
        embedder = EmbeddingService.get_instance()
        embedding = embedder.embed("Test text for embedding")
        
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedder.dimension
        assert embedding.dtype in [np.float32, np.float64]
    
    def test_embed_batch(self):
        """Test embedding a batch of texts."""
        from intanalysis.embeddings import EmbeddingService
        
        embedder = EmbeddingService.get_instance()
        texts = ["Text one", "Text two", "Text three"]
        embeddings = embedder.embed_batch(texts)
        
        assert embeddings.shape == (3, embedder.dimension)
    
    def test_embeddings_normalized(self):
        """Test that embeddings are normalized (unit vectors)."""
        from intanalysis.embeddings import EmbeddingService
        
        embedder = EmbeddingService.get_instance()
        embedding = embedder.embed("Test normalization")
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Should be approximately 1
    
    def test_similar_texts_high_similarity(self):
        """Test that similar texts have high cosine similarity."""
        from intanalysis.embeddings import EmbeddingService
        
        embedder = EmbeddingService.get_instance()
        
        emb1 = embedder.embed("HDFC Bank announces dividend")
        emb2 = embedder.embed("HDFC Bank declares dividend payout")
        emb3 = embedder.embed("Completely unrelated topic about weather")
        
        # Cosine similarity (embeddings are normalized, so dot product = cosine sim)
        sim_12 = np.dot(emb1, emb2)
        sim_13 = np.dot(emb1, emb3)
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13


class TestVectorStore:
    """Tests for VectorStore class."""
    
    def test_vector_store_creation(self):
        """Test creating an empty vector store."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        
        assert store.dimension == 768
        assert store.index.ntotal == 0
        assert len(store.stories) == 0
    
    def test_add_stories(self, sample_processed_article):
        """Test adding stories to vector store."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        
        store.add([story])
        
        assert store.index.ntotal == 1
        assert len(store.stories) == 1
    
    def test_add_multiple_stories(self, sample_processed_article):
        """Test adding multiple stories."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        
        stories = []
        for i in range(5):
            article = Article(title=f"Article {i}", content=f"Content {i}")
            pa = ProcessedArticle(
                article=article,
                embedding=[float(i % 10) / 10] * 768
            )
            stories.append(UniqueStory(id=f"story-{i}", primary_article=pa))
        
        store.add(stories)
        
        assert store.index.ntotal == 5
        assert len(store.stories) == 5
    
    def test_search_returns_results(self, sample_processed_article):
        """Test that search returns relevant results."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        store.add([story])
        
        # Search with same embedding
        query_embedding = np.array(sample_processed_article.embedding, dtype=np.float32)
        results = store.search(query_embedding, k=5)
        
        assert len(results) == 1
        assert results[0][0].id == "test"
        assert results[0][1] >= 0  # Score should be non-negative
    
    def test_search_empty_store(self):
        """Test searching an empty store."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        query_embedding = np.random.randn(768).astype(np.float32)
        
        results = store.search(query_embedding, k=5)
        
        assert results == []
    
    def test_search_with_query_text(self, sample_processed_article):
        """Test hybrid search with query text for BM25."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        store.add([story])
        
        query_embedding = np.array(sample_processed_article.embedding, dtype=np.float32)
        results = store.search(
            query_embedding,
            query_text="HDFC Bank dividend",
            k=5,
            alpha=0.7
        )
        
        assert len(results) >= 1
    
    def test_clear_store(self, sample_processed_article):
        """Test clearing the vector store."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        store.add([story])
        
        assert store.index.ntotal == 1
        
        store.clear()
        
        assert store.index.ntotal == 0
        assert len(store.stories) == 0
    
    def test_stories_without_embeddings_skipped(self):
        """Test that stories without embeddings are skipped."""
        from intanalysis.embeddings import VectorStore
        
        store = VectorStore(dimension=768, use_hnsw=False)
        
        article = Article(title="No embedding", content="Content")
        pa = ProcessedArticle(article=article, embedding=None)  # No embedding
        story = UniqueStory(id="no-emb", primary_article=pa)
        
        store.add([story])
        
        assert store.index.ntotal == 0


class TestReranker:
    """Tests for Reranker class."""
    
    def test_reranker_singleton(self):
        """Test that Reranker uses singleton pattern."""
        from intanalysis.embeddings import Reranker
        
        instance1 = Reranker.get_instance()
        instance2 = Reranker.get_instance()
        
        assert instance1 is instance2
    
    def test_rerank_empty_results(self):
        """Test reranking empty results."""
        from intanalysis.embeddings import Reranker
        
        reranker = Reranker.get_instance()
        results = reranker.rerank("test query", [], top_k=5)
        
        assert results == []
    
    def test_rerank_returns_ordered_results(self, sample_processed_article):
        """Test that rerank returns ordered results."""
        from intanalysis.embeddings import Reranker
        
        reranker = Reranker.get_instance()
        
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        results = [(story, 0.5)]
        
        reranked = reranker.rerank("HDFC Bank news", results, top_k=5)
        
        assert len(reranked) >= 1
        assert isinstance(reranked[0][1], float)  # Has a score
    
    def test_rerank_limits_to_top_k(self, sample_processed_article):
        """Test that rerank limits to top_k results."""
        from intanalysis.embeddings import Reranker
        
        reranker = Reranker.get_instance()
        
        stories = []
        for i in range(10):
            article = Article(title=f"Article {i}", content=f"Content {i}")
            pa = ProcessedArticle(article=article, embedding=[0.1] * 768)
            stories.append((UniqueStory(id=f"story-{i}", primary_article=pa), 0.5))
        
        reranked = reranker.rerank("test query", stories, top_k=3)
        
        assert len(reranked) == 3
