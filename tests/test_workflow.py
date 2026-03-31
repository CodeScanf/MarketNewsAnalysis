"""Unit tests for intanalysis workflow module."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from intanalysis.workflow import (
    PipelineState, build_ingestion_graph, build_query_graph, build_full_graph, build_recommendation_graph,
    ingestion_node, deduplication_node, entity_extraction_node,
    stock_impact_node, storage_node, query_node,
    route_start, should_continue, _get_agent
)
from intanalysis.models import Article, ProcessedArticle, UniqueStory, QueryResult
from intanalysis.embeddings import VectorStore


class TestPipelineState:
    """Tests for PipelineState TypedDict."""
    
    def test_state_with_raw_articles(self):
        """Test state with raw articles."""
        state: PipelineState = {
            "raw_articles": [{"title": "Test", "content": "Content"}],
            "errors": [],
        }
        assert "raw_articles" in state
        assert len(state["raw_articles"]) == 1
    
    def test_state_with_query(self):
        """Test state with query."""
        state: PipelineState = {
            "query": "HDFC Bank news",
            "errors": [],
        }
        assert state["query"] == "HDFC Bank news"
    
    def test_state_with_vector_store(self):
        """Test state with vector store."""
        vs = VectorStore(dimension=768, use_hnsw=False)
        state: PipelineState = {
            "vector_store": vs,
            "errors": [],
        }
        assert state["vector_store"] is vs


class TestRoutingFunctions:
    """Tests for routing functions."""
    
    def test_route_start_query_path(self):
        """Test routing to query path."""
        vs = VectorStore(dimension=768, use_hnsw=False)
        state: PipelineState = {
            "query": "test query",
            "vector_store": vs,
        }
        
        result = route_start(state)
        assert result == "query_step"
    
    def test_route_start_ingestion_path(self):
        """Test routing to ingestion path."""
        state: PipelineState = {
            "raw_articles": [{"title": "Test", "content": "Content"}],
        }
        
        result = route_start(state)
        assert result == "ingestion"
    
    def test_route_start_ingestion_without_vector_store(self):
        """Test routing to ingestion when query but no vector store."""
        state: PipelineState = {
            "query": "test query",
            # No vector_store
        }
        
        result = route_start(state)
        assert result == "ingestion"
    
    def test_should_continue_with_articles(self):
        """Test should_continue returns deduplication when articles exist."""
        state: PipelineState = {
            "articles": [Article(title="Test", content="Content")],
        }
        
        result = should_continue(state)
        assert result == "deduplication"
    
    def test_should_continue_without_articles(self):
        """Test should_continue returns end when no articles."""
        state: PipelineState = {
            "articles": [],
        }
        
        result = should_continue(state)
        assert result == "__end__"


class TestAgentSingleton:
    """Tests for agent singleton pattern."""
    
    def test_get_agent_creates_instance(self):
        """Test that _get_agent creates agent instance."""
        from intanalysis.agents import IngestionAgent
        
        # Clear cache
        from intanalysis.workflow import _agents
        _agents.clear()
        
        agent = _get_agent(IngestionAgent, verbose=False)
        assert isinstance(agent, IngestionAgent)
    
    def test_get_agent_returns_same_instance(self):
        """Test that _get_agent returns same instance."""
        from intanalysis.agents import IngestionAgent
        
        # Clear cache
        from intanalysis.workflow import _agents
        _agents.clear()
        
        agent1 = _get_agent(IngestionAgent, verbose=False)
        agent2 = _get_agent(IngestionAgent, verbose=False)
        
        assert agent1 is agent2


class TestGraphBuilders:
    """Tests for graph builder functions."""
    
    def test_build_ingestion_graph_structure(self):
        """Test that ingestion graph has expected nodes."""
        graph = build_ingestion_graph()
        
        # Graph should compile successfully
        assert graph is not None
    
    def test_build_query_graph_structure(self):
        """Test that query graph has expected structure."""
        graph = build_query_graph()
        
        assert graph is not None
    
    def test_build_full_graph_structure(self):
        """Test that full graph has expected structure."""
        graph = build_full_graph()
        
        assert graph is not None

    def test_build_recommendation_graph_structure(self):
        """Test that recommendation graph compiles successfully."""
        graph = build_recommendation_graph()

        assert graph is not None


class TestWorkflowNodes:
    """Tests for individual workflow node functions."""
    
    def test_ingestion_node(self, sample_articles):
        """Test ingestion node processes articles."""
        state: PipelineState = {
            "raw_articles": sample_articles[:2],
        }
        
        result = ingestion_node(state)
        
        assert "articles" in result
        assert len(result["articles"]) == 2
    
    @patch('intanalysis.workflow._get_agent')
    def test_deduplication_node(self, mock_get_agent, sample_article_objects):
        """Test deduplication node processes articles."""
        mock_agent = Mock()
        mock_agent.process.return_value = {"unique_stories": []}
        mock_get_agent.return_value = mock_agent
        
        state = {"articles": sample_article_objects[:2]}
        
        result = deduplication_node(state)
        
        mock_agent.process.assert_called_once()
    
    @patch('intanalysis.workflow._get_agent')
    def test_entity_extraction_node(self, mock_get_agent, sample_unique_story):
        """Test entity extraction node."""
        mock_agent = Mock()
        mock_agent.process.return_value = {"unique_stories": [sample_unique_story]}
        mock_get_agent.return_value = mock_agent
        
        state = {"unique_stories": [sample_unique_story]}
        
        result = entity_extraction_node(state)
        
        mock_agent.process.assert_called_once()
    
    @patch('intanalysis.workflow._get_agent')
    def test_stock_impact_node(self, mock_get_agent, sample_unique_story):
        """Test stock impact node."""
        mock_agent = Mock()
        mock_agent.process.return_value = {"unique_stories": [sample_unique_story]}
        mock_get_agent.return_value = mock_agent
        
        state = {"unique_stories": [sample_unique_story]}
        
        result = stock_impact_node(state)
        
        mock_agent.process.assert_called_once()
    
    @patch('intanalysis.workflow._get_agent')
    def test_storage_node(self, mock_get_agent, sample_unique_story):
        """Test storage node."""
        mock_agent = Mock()
        mock_agent.process.return_value = {"storage_complete": True}
        mock_get_agent.return_value = mock_agent
        
        vs = VectorStore(dimension=768, use_hnsw=False)
        state = {
            "unique_stories": [sample_unique_story],
            "vector_store": vs,
        }
        
        result = storage_node(state)
        
        mock_agent.process.assert_called_once()
    
    @patch('intanalysis.workflow._get_agent')
    def test_query_node(self, mock_get_agent):
        """Test query node."""
        mock_agent = Mock()
        mock_agent.process.return_value = {
            "query_result": QueryResult(query="test")
        }
        mock_get_agent.return_value = mock_agent
        
        vs = VectorStore(dimension=768, use_hnsw=False)
        state = {
            "query": "test query",
            "vector_store": vs,
        }
        
        result = query_node(state)
        
        mock_agent.process.assert_called_once()


class TestGraphExecution:
    """Tests for graph execution."""
    
    def test_ingestion_graph_empty_input(self):
        """Test ingestion graph with empty input."""
        graph = build_ingestion_graph()
        
        vs = VectorStore(dimension=768, use_hnsw=False)
        state: PipelineState = {
            "raw_articles": [],
            "vector_store": vs,
            "errors": [],
        }
        
        result = graph.invoke(state)
        
        # Should complete without error
        assert "articles" in result
        assert result["articles"] == []
    
    def test_query_graph_empty_vector_store(self):
        """Test query graph with empty vector store."""
        graph = build_query_graph()
        
        vs = VectorStore(dimension=768, use_hnsw=False)
        state: PipelineState = {
            "query": "test query",
            "vector_store": vs,
            "errors": [],
        }
        
        result = graph.invoke(state)
        
        assert "query_result" in result
        assert result["query_result"].query == "test query"

    def test_recommendation_graph_latest_mode(self, sample_unique_story):
        """Test recommendation graph falls back to latest stories without chat history."""
        graph = build_recommendation_graph()

        vs = VectorStore(dimension=768, use_hnsw=False)
        sample_unique_story.primary_article.article.published_date = "2026-03-31T08:00:00+00:00"
        vs.stories = [sample_unique_story]
        state: PipelineState = {
            "user_id": 1,
            "chat_loader": lambda user_id, limit: [],
            "vector_store": vs,
            "errors": [],
        }

        result = graph.invoke(state)

        assert result["recommendation_mode"] == "latest"
        assert len(result["cards"]) == 1
        assert result["cards"][0]["title"] == sample_unique_story.primary_article.article.title

    def test_recommendation_graph_personalized_mode(self, sample_unique_story):
        """Test recommendation graph uses recent chat entities for personalized cards."""
        graph = build_recommendation_graph()

        vs = VectorStore(dimension=768, use_hnsw=False)
        sample_unique_story.primary_article.article.published_date = "2026-03-31T09:30:00+00:00"
        vs.stories = [sample_unique_story]
        state: PipelineState = {
            "user_id": 7,
            "chat_loader": lambda user_id, limit: [
                {
                    "query": "HDFC Bank latest news",
                    "matched_entities": [{"name": "HDFC Bank Limited", "type": "company"}],
                    "stories": [],
                    "created_at": "2026-03-31T09:00:00+00:00",
                }
            ],
            "vector_store": vs,
            "errors": [],
        }

        result = graph.invoke(state)

        assert result["recommendation_mode"] == "personalized"
        assert len(result["cards"]) == 1
        assert result["cards"][0]["recommendation_label"] == "重点关注公司"
        assert "HDFCBANK" in result["cards"][0]["stock_symbols"]
