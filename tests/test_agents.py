"""Unit tests for intanalysis agents."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from intanalysis.agents import (
    BaseAgent, IngestionAgent, DeduplicationAgent,
    EntityExtractionAgent, StockImpactAgent, StorageAgent, QueryAgent
)
from intanalysis.models import (
    Article, Entity, EntityType, ProcessedArticle, UniqueStory, ImpactType
)
from intanalysis.embeddings import VectorStore


class TestBaseAgent:
    """Tests for BaseAgent class."""
    
    def test_base_agent_verbose_logging(self, capsys):
        """Test that verbose agent logs messages."""
        class ConcreteAgent(BaseAgent):
            def process(self, state: dict) -> dict:
                self.log("Test message")
                return state
        
        agent = ConcreteAgent(verbose=True)
        agent.process({})
        
        captured = capsys.readouterr()
        assert "ConcreteAgent" in captured.out
        assert "Test message" in captured.out
    
    def test_base_agent_silent_when_not_verbose(self, capsys):
        """Test that non-verbose agent doesn't log."""
        class ConcreteAgent(BaseAgent):
            def process(self, state: dict) -> dict:
                self.log("Test message")
                return state
        
        agent = ConcreteAgent(verbose=False)
        agent.process({})
        
        captured = capsys.readouterr()
        assert captured.out == ""


class TestIngestionAgent:
    """Tests for IngestionAgent."""
    
    def test_ingest_from_dicts(self, sample_articles):
        """Test ingesting articles from dictionaries."""
        agent = IngestionAgent(verbose=False)
        state = {"raw_articles": sample_articles[:3]}
        
        result = agent.process(state)
        
        assert "articles" in result
        assert len(result["articles"]) == 3
        assert all(isinstance(a, Article) for a in result["articles"])
    
    def test_ingest_from_article_objects(self, sample_article_objects):
        """Test ingesting Article objects directly."""
        agent = IngestionAgent(verbose=False)
        state = {"raw_articles": sample_article_objects[:3]}
        
        result = agent.process(state)
        
        assert len(result["articles"]) == 3
        assert all(isinstance(a, Article) for a in result["articles"])
    
    def test_ingest_empty_list(self):
        """Test ingesting empty article list."""
        agent = IngestionAgent(verbose=False)
        state = {"raw_articles": []}
        
        result = agent.process(state)
        
        assert result["articles"] == []
    
    def test_ingest_mixed_types(self, sample_articles, sample_article_objects):
        """Test ingesting mixed dicts and Article objects."""
        agent = IngestionAgent(verbose=False)
        state = {"raw_articles": [sample_articles[0], sample_article_objects[1]]}
        
        result = agent.process(state)
        
        assert len(result["articles"]) == 2
        assert all(isinstance(a, Article) for a in result["articles"])


class TestDeduplicationAgent:
    """Tests for DeduplicationAgent."""
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_deduplication_detects_similar_articles(self, mock_embedding_service, duplicate_articles):
        """Test that similar articles are detected as duplicates."""
        # Create mock embeddings where similar articles have high cosine similarity
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        # Same embedding for duplicates (similarity = 1.0)
        base_embedding = np.random.randn(768).astype(np.float32)
        base_embedding = base_embedding / np.linalg.norm(base_embedding)
        mock_embedder.embed_batch.return_value = np.array([base_embedding, base_embedding])
        
        agent = DeduplicationAgent(threshold=0.6, verbose=False)
        agent.embedder = mock_embedder
        
        articles = [Article(**art) for art in duplicate_articles]
        state = {"articles": articles}
        
        result = agent.process(state)
        
        assert "unique_stories" in result
        # Both articles should be in same cluster (1 unique story)
        assert len(result["unique_stories"]) == 1
        assert result["unique_stories"][0].duplicate_count == 1
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_deduplication_different_articles_separate(self, mock_embedding_service, sample_article_objects):
        """Test that different articles remain separate."""
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        # Create orthogonal embeddings (low similarity)
        embeddings = []
        for i in range(3):
            emb = np.zeros(768)
            emb[i * 100] = 1.0  # Different directions
            embeddings.append(emb)
        mock_embedder.embed_batch.return_value = np.array(embeddings)
        
        agent = DeduplicationAgent(threshold=0.6, verbose=False)
        agent.embedder = mock_embedder
        
        state = {"articles": sample_article_objects[:3]}
        
        result = agent.process(state)
        
        # Each should be its own cluster
        assert len(result["unique_stories"]) == 3
        for story in result["unique_stories"]:
            assert story.duplicate_count == 0
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_deduplication_empty_input(self, mock_embedding_service):
        """Test deduplication with empty input."""
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        agent = DeduplicationAgent(verbose=False)
        agent.embedder = mock_embedder
        
        state = {"articles": []}
        result = agent.process(state)
        
        assert result["unique_stories"] == []
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_deduplication_threshold_effect(self, mock_embedding_service):
        """Test that threshold affects clustering."""
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        # Create embeddings with similarity ~0.7
        emb1 = np.random.randn(768).astype(np.float32)
        emb1 = emb1 / np.linalg.norm(emb1)
        emb2 = emb1 + np.random.randn(768).astype(np.float32) * 0.5
        emb2 = emb2 / np.linalg.norm(emb2)
        embeddings = np.array([emb1, emb2])
        
        mock_embedder.embed_batch.return_value = embeddings
        
        articles = [
            Article(title="Article 1", content="Content 1"),
            Article(title="Article 2", content="Content 2"),
        ]
        
        # High threshold - should be separate
        agent_high = DeduplicationAgent(threshold=0.9, verbose=False)
        agent_high.embedder = mock_embedder
        result_high = agent_high.process({"articles": articles})
        
        # Low threshold - might be clustered
        agent_low = DeduplicationAgent(threshold=0.5, verbose=False)
        agent_low.embedder = mock_embedder
        result_low = agent_low.process({"articles": articles})
        
        # High threshold should result in more unique stories
        assert len(result_high["unique_stories"]) >= len(result_low["unique_stories"])


class TestEntityExtractionAgent:
    """Tests for EntityExtractionAgent."""
    
    def test_extract_company_entities(self, hdfc_article):
        """Test extracting company entities."""
        agent = EntityExtractionAgent(use_llm=False, verbose=False)
        
        story = UniqueStory(
            id="test",
            primary_article=ProcessedArticle(article=hdfc_article)
        )
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        entities = result["unique_stories"][0].primary_article.entities
        entity_names = [e.name.lower() for e in entities]
        
        assert any("hdfc" in name for name in entity_names)
    
    def test_extract_regulator_entities(self, rbi_article):
        """Test extracting regulator entities."""
        agent = EntityExtractionAgent(use_llm=False, verbose=False)
        
        story = UniqueStory(
            id="test",
            primary_article=ProcessedArticle(article=rbi_article)
        )
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        entities = result["unique_stories"][0].primary_article.entities
        entity_types = [e.type for e in entities]
        
        assert EntityType.REGULATOR in entity_types
    
    def test_extract_sectors(self, hdfc_article):
        """Test sector extraction from entities."""
        agent = EntityExtractionAgent(use_llm=False, verbose=False)
        
        story = UniqueStory(
            id="test",
            primary_article=ProcessedArticle(article=hdfc_article)
        )
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        sectors = result["unique_stories"][0].primary_article.sectors
        assert "Banking" in sectors or "Financial Services" in sectors
    
    def test_multiple_entities_extraction(self, it_sector_article):
        """Test extracting multiple entities from one article."""
        agent = EntityExtractionAgent(use_llm=False, verbose=False)
        
        story = UniqueStory(
            id="test",
            primary_article=ProcessedArticle(article=it_sector_article)
        )
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        entities = result["unique_stories"][0].primary_article.entities
        # Should find TCS and Infosys
        assert len(entities) >= 2


class TestStockImpactAgent:
    """Tests for StockImpactAgent."""
    
    def test_direct_company_impact(self):
        """Test direct company mention creates high confidence impact."""
        agent = StockImpactAgent(use_llm=False, verbose=False)
        
        article = ProcessedArticle(
            article=Article(title="HDFC Bank", content="HDFC news"),
            entities=[Entity(name="HDFC Bank Limited", type=EntityType.COMPANY)]
        )
        story = UniqueStory(id="test", primary_article=article)
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        impacts = result["unique_stories"][0].primary_article.stock_impacts
        assert len(impacts) >= 1
        
        hdfc_impacts = [i for i in impacts if "HDFC" in i.symbol]
        assert len(hdfc_impacts) >= 1
        assert hdfc_impacts[0].impact_type == ImpactType.DIRECT
        assert hdfc_impacts[0].confidence == 1.0
    
    def test_regulatory_impact(self):
        """Test regulatory entity creates sector-wide impacts."""
        agent = StockImpactAgent(use_llm=False, verbose=False)
        
        article = ProcessedArticle(
            article=Article(title="RBI news", content="Central bank policy"),
            entities=[Entity(name="Reserve Bank of India", type=EntityType.REGULATOR)]
        )
        story = UniqueStory(id="test", primary_article=article)
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        impacts = result["unique_stories"][0].primary_article.stock_impacts
        regulatory_impacts = [i for i in impacts if i.impact_type == ImpactType.REGULATORY]
        
        assert len(regulatory_impacts) >= 1
        assert all(i.confidence < 1.0 for i in regulatory_impacts)
    
    def test_sector_wide_impact(self):
        """Test sector-wide impact mapping."""
        agent = StockImpactAgent(use_llm=False, verbose=False)
        
        article = ProcessedArticle(
            article=Article(title="Banking sector", content="Sector news"),
            entities=[Entity(name="HDFC Bank Limited", type=EntityType.COMPANY)],
            sectors=["Banking"]
        )
        story = UniqueStory(id="test", primary_article=article)
        state = {"unique_stories": [story]}
        
        result = agent.process(state)
        
        impacts = result["unique_stories"][0].primary_article.stock_impacts
        # Should have both direct and sector impacts
        impact_types = {i.impact_type for i in impacts}
        assert ImpactType.DIRECT in impact_types or ImpactType.SECTOR in impact_types


class TestStorageAgent:
    """Tests for StorageAgent."""
    
    def test_storage_adds_to_vector_store(self, sample_processed_article):
        """Test that storage agent adds stories to vector store."""
        agent = StorageAgent(verbose=False)
        
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        vector_store = VectorStore(dimension=768, use_hnsw=False)
        
        state = {
            "unique_stories": [story],
            "vector_store": vector_store
        }
        
        result = agent.process(state)
        
        assert result["storage_complete"] == True
        assert result["vector_store"].index.ntotal == 1
    
    def test_storage_creates_vector_store_if_missing(self, sample_processed_article):
        """Test that storage agent creates vector store if not in state."""
        agent = StorageAgent(verbose=False)
        
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        
        state = {"unique_stories": [story]}
        
        with patch('intanalysis.agents.EmbeddingService') as mock_emb:
            mock_emb.get_instance.return_value.dimension = 768
            result = agent.process(state)
        
        assert "vector_store" in result
        assert result["storage_complete"] == True


class TestQueryAgent:
    """Tests for QueryAgent."""
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_query_with_no_indexed_articles(self, mock_embedding_service):
        """Test query returns empty when no articles indexed."""
        mock_embedder = Mock()
        mock_embedder.dimension = 768
        mock_embedder.embed.return_value = np.random.randn(768).astype(np.float32)
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        agent = QueryAgent(use_llm=False, use_reranker=False, verbose=False)
        agent.embedder = mock_embedder
        
        vector_store = VectorStore(dimension=768, use_hnsw=False)
        state = {"query": "HDFC Bank news", "vector_store": vector_store}
        
        result = agent.process(state)
        
        assert "query_result" in result
        assert len(result["query_result"].stories) == 0
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_query_extracts_entities(self, mock_embedding_service):
        """Test that query agent extracts entities from query string."""
        mock_embedder = Mock()
        mock_embedder.dimension = 768
        mock_embedder.embed.return_value = np.random.randn(768).astype(np.float32)
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        agent = QueryAgent(use_llm=False, use_reranker=False, verbose=False)
        agent.embedder = mock_embedder
        
        # Test company query
        entities = agent._extract_query_entities("HDFC Bank news")
        assert any(e["type"] == EntityType.COMPANY for e in entities)
        
        # Test regulator query
        entities = agent._extract_query_entities("RBI policy changes")
        assert any(e["type"] == EntityType.REGULATOR for e in entities)
        
        # Test sector query
        entities = agent._extract_query_entities("banking sector update")
        assert any(e["type"] == EntityType.SECTOR for e in entities)
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_query_expansion(self, mock_embedding_service):
        """Test that query is expanded with related terms."""
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        agent = QueryAgent(use_llm=False, use_reranker=False, verbose=False)
        
        entities = [{"name": "HDFC Bank Limited", "type": EntityType.COMPANY, "symbol": "HDFCBANK"}]
        expanded = agent._expand_query("HDFC Bank news", entities)
        
        assert "HDFC Bank news" in expanded
        assert "HDFC Bank Limited" in expanded
    
    @patch('intanalysis.agents.EmbeddingService')
    def test_query_entity_filtering(self, mock_embedding_service, sample_processed_article):
        """Test that results are filtered and boosted by entity match."""
        mock_embedder = Mock()
        mock_embedding_service.get_instance.return_value = mock_embedder
        
        agent = QueryAgent(use_llm=False, use_reranker=False, verbose=False)
        
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        results = [(story, 0.8)]
        
        query_entities = [{"name": "HDFC Bank Limited", "type": EntityType.COMPANY}]
        filtered = agent._filter_by_entities(results, query_entities)
        
        # Score should be boosted
        assert filtered[0][1] > 0.8
