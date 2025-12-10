"""LangGraph workflow for multi-agent orchestration."""

from typing import TypedDict, Optional, Literal
from langgraph.graph import StateGraph, END

from intanalysis.models import Article, UniqueStory, QueryResult
from intanalysis.embeddings import VectorStore
from intanalysis.agents import (
    IngestionAgent,
    DeduplicationAgent,
    EntityExtractionAgent,
    StockImpactAgent,
    StorageAgent,
    QueryAgent,
)


class PipelineState(TypedDict, total=False):
    """State flowing through the pipeline."""
    # Inputs
    raw_articles: list[dict | Article]
    query: Optional[str]
    
    # Processing
    articles: list[Article]
    unique_stories: list[UniqueStory]
    processed_articles: list
    
    # Storage
    vector_store: VectorStore
    storage_complete: bool
    
    # Output
    query_result: Optional[QueryResult]
    errors: list[str]


# Agent singletons
_agents = {}


def _get_agent(agent_class, **kwargs):
    """Get or create agent singleton."""
    name = agent_class.__name__
    if name not in _agents:
        _agents[name] = agent_class(**kwargs)
    return _agents[name]


# Node functions
def ingestion_node(state: PipelineState) -> PipelineState:
    """Run ingestion agent."""
    agent = _get_agent(IngestionAgent, verbose=True)
    return agent.process(dict(state))


def deduplication_node(state: PipelineState) -> PipelineState:
    """Run deduplication agent."""
    agent = _get_agent(DeduplicationAgent, verbose=True)
    return agent.process(dict(state))


def entity_extraction_node(state: PipelineState) -> PipelineState:
    """Run entity extraction agent."""
    agent = _get_agent(EntityExtractionAgent, verbose=True)
    return agent.process(dict(state))


def stock_impact_node(state: PipelineState) -> PipelineState:
    """Run stock impact analysis agent."""
    agent = _get_agent(StockImpactAgent, verbose=True)
    return agent.process(dict(state))


def storage_node(state: PipelineState) -> PipelineState:
    """Run storage agent."""
    agent = _get_agent(StorageAgent, verbose=True)
    return agent.process(dict(state))


def query_node(state: PipelineState) -> PipelineState:
    """Run query processing agent."""
    agent = _get_agent(QueryAgent, verbose=True)
    return agent.process(dict(state))


# Routing functions
def route_start(state: PipelineState) -> Literal["query", "ingestion"]:
    """Route based on whether this is a query or ingestion."""
    if state.get("query") and state.get("vector_store"):
        return "query"
    return "ingestion"


def should_continue(state: PipelineState) -> Literal["deduplication", "__end__"]:
    """Check if ingestion succeeded."""
    if state.get("articles"):
        return "deduplication"
    return "__end__"


def build_ingestion_graph() -> StateGraph:
    """Build the ingestion pipeline graph."""
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("deduplication", deduplication_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("stock_impact", stock_impact_node)
    graph.add_node("storage", storage_node)
    
    # Add edges (linear pipeline)
    graph.set_entry_point("ingestion")
    graph.add_conditional_edges("ingestion", should_continue, {"deduplication": "deduplication", "__end__": END})
    graph.add_edge("deduplication", "entity_extraction")
    graph.add_edge("entity_extraction", "stock_impact")
    graph.add_edge("stock_impact", "storage")
    graph.add_edge("storage", END)
    
    return graph.compile()


def build_query_graph() -> StateGraph:
    """Build the query processing graph."""
    graph = StateGraph(PipelineState)
    
    graph.add_node("query", query_node)
    graph.set_entry_point("query")
    graph.add_edge("query", END)
    
    return graph.compile()


def build_full_graph() -> StateGraph:
    """Build the complete graph with routing."""
    graph = StateGraph(PipelineState)
    
    # Add all nodes
    graph.add_node("ingestion", ingestion_node)
    graph.add_node("deduplication", deduplication_node)
    graph.add_node("entity_extraction", entity_extraction_node)
    graph.add_node("stock_impact", stock_impact_node)
    graph.add_node("storage", storage_node)
    graph.add_node("query", query_node)
    
    # Entry point routing
    graph.set_conditional_entry_point(route_start)
    
    # Ingestion pipeline
    graph.add_conditional_edges("ingestion", should_continue)
    graph.add_edge("deduplication", "entity_extraction")
    graph.add_edge("entity_extraction", "stock_impact")
    graph.add_edge("stock_impact", "storage")
    graph.add_edge("storage", END)
    
    # Query path
    graph.add_edge("query", END)
    
    return graph.compile()
