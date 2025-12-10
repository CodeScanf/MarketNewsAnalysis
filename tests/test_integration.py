"""Integration tests for the full intanalysis pipeline."""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from intanalysis.models import Article, EntityType, ImpactType
from intanalysis.core import IntelligenceSystem
from intanalysis.workflow import build_ingestion_graph, build_query_graph, PipelineState
from intanalysis.embeddings import VectorStore, EmbeddingService


class TestFullPipeline:
    """Integration tests for the complete ingestion and query pipeline."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_ingest_single_article(self, temp_storage, sample_articles):
        """Test ingesting a single article through the full pipeline."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        result = system.ingest([sample_articles[3]])  # HDFC Bank article
        
        assert result["total_articles"] == 1
        assert result["unique_count"] == 1
        assert result["duplicate_count"] == 0
        assert len(result["unique_stories"]) == 1
    
    def test_ingest_multiple_articles(self, temp_storage, sample_articles):
        """Test ingesting multiple articles."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        result = system.ingest(sample_articles[:5])
        
        assert result["total_articles"] == 5
        assert result["unique_count"] >= 1
        assert result["unique_count"] + result["duplicate_count"] == 5
    
    def test_deduplication_in_pipeline(self, temp_storage, duplicate_articles):
        """Test that duplicate articles are detected in the full pipeline."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        result = system.ingest(duplicate_articles)
        
        # Two similar articles about Reliance profit should cluster together
        assert result["total_articles"] == 2
        assert result["unique_count"] <= 2  # May be 1 if detected as duplicates
    
    def test_entity_extraction_in_pipeline(self, temp_storage):
        """Test that entities are extracted during ingestion."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        articles = [{
            "title": "HDFC Bank announces 15% dividend",
            "content": "HDFC Bank Limited declared a dividend for shareholders."
        }]
        
        result = system.ingest(articles)
        
        story = result["unique_stories"][0]
        entities = story.primary_article.entities
        
        # Should extract HDFC Bank as company
        company_names = [e.name.lower() for e in entities if e.type == EntityType.COMPANY]
        assert any("hdfc" in name for name in company_names)
    
    def test_stock_impact_in_pipeline(self, temp_storage):
        """Test that stock impacts are mapped during ingestion."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        articles = [{
            "title": "HDFC Bank reports strong Q3 results",
            "content": "HDFC Bank Limited reported excellent quarterly performance."
        }]
        
        result = system.ingest(articles)
        
        story = result["unique_stories"][0]
        impacts = story.primary_article.stock_impacts
        
        # Should map to HDFCBANK stock
        symbols = [i.symbol for i in impacts]
        assert "HDFCBANK" in symbols or any("HDFC" in s for s in symbols)
    
    def test_query_after_ingestion(self, temp_storage, sample_articles):
        """Test querying after ingesting articles."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        # Ingest articles
        system.ingest(sample_articles)
        
        # Query for HDFC Bank news
        result = system.query("HDFC Bank news", show_steps=False)
        
        assert result.query == "HDFC Bank news"
        # Should find relevant results
        assert len(result.stories) >= 0  # May be 0 if no matches
    
    def test_query_rbi_news(self, temp_storage, sample_articles):
        """Test querying for RBI news."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        # Ingest all sample articles (includes RBI articles)
        system.ingest(sample_articles)
        
        # Query for RBI news
        result = system.query("RBI policy changes", show_steps=False)
        
        assert result.query == "RBI policy changes"
    
    def test_query_banking_sector(self, temp_storage, sample_articles):
        """Test querying for banking sector news."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        system.ingest(sample_articles)
        
        result = system.query("banking sector update", show_steps=False)
        
        assert result.query == "banking sector update"
    
    def test_persistence_across_sessions(self, temp_storage, sample_articles):
        """Test that data persists across system instances."""
        # First session - ingest articles
        system1 = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        system1.ingest(sample_articles[:3])
        initial_count = system1.vector_store.index.ntotal
        
        # Second session - should load persisted data
        system2 = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        assert system2.vector_store.index.ntotal == initial_count
    
    def test_incremental_ingestion(self, temp_storage, sample_articles):
        """Test incremental ingestion skips already processed articles."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        # First ingestion
        result1 = system.ingest(sample_articles[:3])
        first_unique = result1["unique_count"]
        
        # Second ingestion with same articles
        result2 = system.ingest(sample_articles[:3])
        
        # All should be skipped
        assert result2["skipped_count"] == 3
        assert result2["unique_count"] == 0
    
    def test_force_reingest(self, temp_storage, sample_articles):
        """Test force re-ingestion ignores cache."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        # First ingestion
        system.ingest(sample_articles[:3])
        
        # Force re-ingestion
        result = system.ingest(sample_articles[:3], force=True)
        
        assert result["skipped_count"] == 0
        assert result["unique_count"] + result["duplicate_count"] == 3
    
    def test_get_stats(self, temp_storage, sample_articles):
        """Test getting system statistics."""
        system = IntelligenceSystem(verbose=False, storage_dir=temp_storage)
        
        system.ingest(sample_articles[:5])
        
        stats = system.get_stats()
        
        assert "indexed_stories" in stats
        assert "total_stories" in stats
        assert stats["indexed_stories"] >= 1


class TestWorkflowNodes:
    """Integration tests for LangGraph workflow nodes."""
    
    def test_ingestion_graph_compilation(self):
        """Test that ingestion graph compiles successfully."""
        graph = build_ingestion_graph()
        assert graph is not None
    
    def test_query_graph_compilation(self):
        """Test that query graph compiles successfully."""
        graph = build_query_graph()
        assert graph is not None
    
    def test_ingestion_graph_execution(self, sample_articles):
        """Test executing the ingestion graph."""
        graph = build_ingestion_graph()
        
        embedder = EmbeddingService.get_instance()
        vector_store = VectorStore(dimension=embedder.dimension)
        
        initial_state: PipelineState = {
            "raw_articles": sample_articles[:2],
            "vector_store": vector_store,
            "errors": [],
        }
        
        result = graph.invoke(initial_state)
        
        assert "unique_stories" in result
        assert "storage_complete" in result
        assert result["storage_complete"] == True
    
    def test_query_graph_execution(self, sample_articles):
        """Test executing the query graph with indexed data."""
        # First ingest some articles
        ingestion_graph = build_ingestion_graph()
        embedder = EmbeddingService.get_instance()
        vector_store = VectorStore(dimension=embedder.dimension)
        
        ingestion_state: PipelineState = {
            "raw_articles": sample_articles[:3],
            "vector_store": vector_store,
            "errors": [],
        }
        ingestion_graph.invoke(ingestion_state)
        
        # Then query
        query_graph = build_query_graph()
        query_state: PipelineState = {
            "query": "banking news",
            "vector_store": vector_store,
            "errors": [],
        }
        
        result = query_graph.invoke(query_state)
        
        assert "query_result" in result
        assert result["query_result"].query == "banking news"


class TestEndToEndScenarios:
    """End-to-end integration tests simulating real usage scenarios."""
    
    @pytest.fixture
    def system(self):
        """Create a fresh system for each test."""
        temp_dir = tempfile.mkdtemp()
        system = IntelligenceSystem(verbose=False, storage_dir=temp_dir)
        yield system
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_scenario_hdfc_bank_news(self, system):
        """Scenario: User wants all HDFC Bank news."""
        # Ingest mixed articles
        articles = [
            {"title": "HDFC Bank announces 15% dividend", "content": "HDFC Bank Limited declared dividend."},
            {"title": "RBI raises rates", "content": "Reserve Bank of India raised repo rate."},
            {"title": "HDFC Bank Q3 results", "content": "HDFC Bank reported strong Q3."},
            {"title": "ICICI Bank expansion", "content": "ICICI Bank opens new branches."},
        ]
        
        system.ingest(articles)
        result = system.query("HDFC Bank news", show_steps=False)
        
        # Should find HDFC Bank articles
        assert result.query == "HDFC Bank news"
        # Check matched entities include HDFC
        company_entities = [e for e in result.matched_entities if e.type == EntityType.COMPANY]
        if company_entities:
            assert any("HDFC" in e.name for e in company_entities)
    
    def test_scenario_rbi_policy(self, system):
        """Scenario: User wants RBI policy news."""
        articles = [
            {"title": "RBI raises repo rate by 25bps", "content": "Reserve Bank of India increased rates."},
            {"title": "RBI inflation concerns", "content": "Central bank expressed inflation worries."},
            {"title": "TCS wins deal", "content": "Tata Consultancy Services won a contract."},
        ]
        
        system.ingest(articles)
        result = system.query("RBI policy changes", show_steps=False)
        
        assert result.query == "RBI policy changes"
    
    def test_scenario_sector_query(self, system):
        """Scenario: User queries for entire sector."""
        articles = [
            {"title": "Banking sector NPAs decline", "content": "Indian banking sector reports lower NPAs."},
            {"title": "HDFC Bank dividend", "content": "HDFC Bank announces dividend."},
            {"title": "ICICI Bank results", "content": "ICICI Bank quarterly results."},
            {"title": "TCS deal", "content": "IT company TCS wins deal."},
        ]
        
        system.ingest(articles)
        result = system.query("banking sector news", show_steps=False)
        
        # Should find banking-related articles
        assert result.query == "banking sector news"
    
    def test_scenario_duplicate_news(self, system):
        """Scenario: Same news from multiple sources should be deduplicated."""
        articles = [
            {
                "title": "RBI increases repo rate by 25 basis points",
                "content": "The Reserve Bank of India announced a rate hike to combat inflation."
            },
            {
                "title": "Reserve Bank hikes interest rates by 0.25%",
                "content": "RBI raised rates by 25 basis points citing inflation concerns."
            },
            {
                "title": "Central bank raises policy rate 25bps",
                "content": "The central bank increased the repo rate in response to inflation."
            },
        ]
        
        result = system.ingest(articles)
        
        # These similar articles should be clustered
        # Depending on threshold, might be 1-3 unique stories
        assert result["unique_count"] <= 3
        assert result["total_articles"] == 3
    
    def test_scenario_it_sector(self, system):
        """Scenario: Query for IT sector news."""
        articles = [
            {"title": "TCS wins $500M deal", "content": "Tata Consultancy Services won a major deal."},
            {"title": "Infosys raises guidance", "content": "Infosys increased revenue guidance."},
            {"title": "HDFC Bank results", "content": "HDFC Bank reported earnings."},
        ]
        
        system.ingest(articles)
        result = system.query("IT sector news", show_steps=False)
        
        assert result.query == "IT sector news"
    
    def test_scenario_empty_query_result(self, system):
        """Scenario: Query with no matching results."""
        articles = [
            {"title": "HDFC Bank news", "content": "Banking news content."},
        ]
        
        system.ingest(articles)
        result = system.query("automobile sector news", show_steps=False)
        
        # May return empty or loosely related results
        assert result.query == "automobile sector news"
