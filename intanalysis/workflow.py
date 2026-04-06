"""LangGraph workflow for multi-agent orchestration."""

from typing import Callable, Literal, Optional, TypedDict
from langgraph.graph import StateGraph, END

from intanalysis.models import Article, AttachmentContext, QueryResult, UniqueStory
from intanalysis.embeddings import VectorStore
from intanalysis.agents import (
    IngestionAgent,
    DeduplicationAgent,
    EntityExtractionAgent,
    StockImpactAgent,
    StorageAgent,
    QueryAgent,
)
from intanalysis.recommendations import (
    build_card,
    build_feed_summary,
    collect_personalized_candidates,
    extract_interest_entities,
    sort_latest_stories,
)


class PipelineState(TypedDict, total=False):
    """State flowing through the pipeline."""
    # Inputs
    raw_articles: list[dict | Article]
    query: Optional[str]
    attachment_context: Optional[AttachmentContext]
    
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

    # Recommendation inputs / outputs
    user_id: int
    storage_dir: str
    chat_loader: Callable[[int, int], list[dict]]
    chat_records: list[dict]
    interest_entities: list[dict]
    candidate_stories: list[UniqueStory]
    candidate_matches: dict[str, list[dict]]
    recommendation_mode: str
    cards: list[dict]
    feed_summary: str


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


def load_chat_context_node(state: PipelineState) -> PipelineState:
    """Load the current user's recent chat records."""
    user_id = state.get("user_id")
    chat_loader = state.get("chat_loader")
    if user_id is None or chat_loader is None:
        state["chat_records"] = []
        return state

    state["chat_records"] = chat_loader(user_id, 10) or []
    return state


def derive_interest_entities_node(state: PipelineState) -> PipelineState:
    """Extract recommendation entities from recent chats."""
    state["interest_entities"] = extract_interest_entities(state.get("chat_records", []), limit=10)
    return state


def route_recommendation_mode_node(state: PipelineState) -> PipelineState:
    """Choose personalized or latest recommendation mode."""
    state["recommendation_mode"] = "personalized" if state.get("interest_entities") else "latest"
    return state


def collect_candidates_node(state: PipelineState) -> PipelineState:
    """Collect candidate stories for recommendation cards."""
    candidate_limit = 50
    vector_store = state.get("vector_store")
    stories = list(vector_store.stories) if vector_store else []
    mode = state.get("recommendation_mode", "latest")

    candidate_stories: list[UniqueStory] = []
    candidate_matches: dict[str, list[dict]] = {}

    if mode == "personalized":
        candidates = collect_personalized_candidates(
            stories,
            state.get("interest_entities", []),
            limit=candidate_limit,
        )
        if candidates:
            candidate_stories = [story for story, _ in candidates]
            candidate_matches = {story.id: matches for story, matches in candidates}
        else:
            mode = "latest"

    if mode == "latest":
        candidate_stories = sort_latest_stories(stories, limit=candidate_limit)
        candidate_matches = {story.id: [] for story in candidate_stories}

    state["recommendation_mode"] = mode
    state["candidate_stories"] = candidate_stories
    state["candidate_matches"] = candidate_matches
    return state


def build_cards_node(state: PipelineState) -> PipelineState:
    """Build final recommendation cards and summary."""
    mode = state.get("recommendation_mode", "latest")
    candidate_matches = state.get("candidate_matches", {})
    cards = [
        build_card(story, mode, candidate_matches.get(story.id, []))
        for story in state.get("candidate_stories", [])
    ]
    state["cards"] = cards
    state["feed_summary"] = build_feed_summary(mode, cards, state.get("interest_entities", []))
    return state


# Routing functions
def route_start(state: PipelineState) -> Literal["query_step", "ingestion"]:
    """Route based on whether this is a query or ingestion."""
    if state.get("query") and state.get("vector_store"):
        return "query_step"
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

    # LangGraph disallows node names that collide with state keys like "query".
    graph.add_node("query_step", query_node)
    graph.set_entry_point("query_step")
    graph.add_edge("query_step", END)
    
    return graph.compile()


def build_recommendation_graph() -> StateGraph:
    """Build the recommendation workflow graph."""
    graph = StateGraph(PipelineState)
    graph.add_node("load_chat_context", load_chat_context_node)
    graph.add_node("derive_interest_entities", derive_interest_entities_node)
    graph.add_node("route_recommendation_mode", route_recommendation_mode_node)
    graph.add_node("collect_candidates", collect_candidates_node)
    graph.add_node("build_cards", build_cards_node)

    graph.set_entry_point("load_chat_context")
    graph.add_edge("load_chat_context", "derive_interest_entities")
    graph.add_edge("derive_interest_entities", "route_recommendation_mode")
    graph.add_edge("route_recommendation_mode", "collect_candidates")
    graph.add_edge("collect_candidates", "build_cards")
    graph.add_edge("build_cards", END)

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
    graph.add_node("query_step", query_node)
    
    # Entry point routing
    graph.set_conditional_entry_point(route_start)
    
    # Ingestion pipeline
    graph.add_conditional_edges("ingestion", should_continue)
    graph.add_edge("deduplication", "entity_extraction")
    graph.add_edge("entity_extraction", "stock_impact")
    graph.add_edge("stock_impact", "storage")
    graph.add_edge("storage", END)
    
    # Query path
    graph.add_edge("query_step", END)
    
    return graph.compile()
