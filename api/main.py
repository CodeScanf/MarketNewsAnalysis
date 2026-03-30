"""FastAPI application for Financial News Intelligence System."""

import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
from time import perf_counter

from intanalysis import IntelligenceSystem, Article
from intanalysis.chat_history import ChatHistoryManager
from intanalysis.models import QueryIntent, QueryTiming


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

# Initialize FastAPI app
app = FastAPI(
    title="Financial News Intelligence API",
    description="AI-Powered Financial News Intelligence System - Process news, detect duplicates, extract entities, and query intelligently.",
    version="1.0.0",
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Intelligence System
system = IntelligenceSystem(verbose=True)

# Initialize Chat History Manager
chat_history = ChatHistoryManager.get_instance()


# Request/Response Models
class ArticleInput(BaseModel):
    """Input model for a single article."""
    title: str
    content: str
    source: Optional[str] = None
    published_date: Optional[str] = None
    url: Optional[str] = None


class IngestRequest(BaseModel):
    """Request model for ingesting articles."""
    articles: List[ArticleInput]
    force: bool = False


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    total_articles: int
    unique_count: int
    duplicate_count: int
    skipped_count: int
    message: str
    unique_stories: List[dict] = []


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str


class EntityResponse(BaseModel):
    """Entity information."""
    name: str
    type: str
    confidence: float


class StockImpactResponse(BaseModel):
    """Stock impact information."""
    symbol: str
    company_name: str
    confidence: float
    impact_type: str
    reasoning: Optional[str] = None


class StoryResponse(BaseModel):
    """Story information."""
    id: str
    title: str
    content: str
    source: Optional[str]
    entities: List[EntityResponse]
    stock_impacts: List[StockImpactResponse]
    sectors: List[str]
    duplicate_count: int


class QueryResponse(BaseModel):
    """Response model for queries."""
    query: str
    intent: QueryIntent
    intent_source: str
    intent_reason: str
    stories: List[StoryResponse]
    matched_entities: List[EntityResponse]
    explanation: Optional[str]
    markdown_response: str
    timing: QueryTiming


class StatsResponse(BaseModel):
    """System statistics response."""
    indexed_stories: int
    total_stories: int


# Helper functions
def story_to_response(story) -> StoryResponse:
    """Convert UniqueStory to StoryResponse."""
    pa = story.primary_article
    return StoryResponse(
        id=story.id,
        title=pa.article.title,
        content=pa.article.content,
        source=pa.article.source,
        entities=[
            EntityResponse(name=e.name, type=e.type.value, confidence=e.confidence)
            for e in pa.entities
        ],
        stock_impacts=[
            StockImpactResponse(
                symbol=i.symbol,
                company_name=i.company_name,
                confidence=i.confidence,
                impact_type=i.impact_type.value,
                reasoning=i.reasoning
            )
            for i in pa.stock_impacts
        ],
        sectors=pa.sectors,
        duplicate_count=story.duplicate_count,
    )


def format_query_as_markdown(query: str, stories: list, explanation: str) -> str:
    """Format query results as markdown."""
    md = f"# Query Results: \"{query}\"\n\n"
    
    if explanation:
        md += f"## Summary\n{explanation}\n\n"
    
    md += f"## {len(stories)} source{'s' if len(stories) != 1 else ''} found.\n\n"
    
    for i, story in enumerate(stories, 1):
        md += f"### {i}. {story.title}\n\n"
        md += f"**Source:** {story.source or 'Unknown'}\n\n"
        md += f"{story.content}\n\n"
        
        if story.entities:
            md += "**Entities:** "
            md += ", ".join([f"`{e.name}` ({e.type})" for e in story.entities])
            md += "\n\n"
        
        if story.stock_impacts:
            md += "**Impacted Stocks:**\n"
            for impact in story.stock_impacts:
                md += f"- **{impact.symbol}** ({impact.company_name}): {impact.confidence:.0%} confidence ({impact.impact_type})\n"
            md += "\n"
        
        if story.sectors:
            md += f"**Sectors:** {', '.join(story.sectors)}\n\n"
        
        if story.duplicate_count > 0:
            md += f"⚠️ *This story has {story.duplicate_count} duplicate article(s)*\n\n"
        
        md += "---\n\n"
    
    return md


def format_general_as_markdown(query: str, explanation: str) -> str:
    """Format general-chat responses as markdown."""
    return f"# General Answer: \"{query}\"\n\n## Response\n{explanation or 'No answer generated.'}\n"


def format_update_as_markdown(query: str, stories: list, explanation: str) -> str:
    """Format news refresh responses as markdown."""
    md = f"# News Refresh: \"{query}\"\n\n"
    if explanation:
        md += f"## Summary\n{explanation}\n\n"
    md += f"## {len(stories)} new source{'s' if len(stories) != 1 else ''} indexed.\n\n"
    for i, story in enumerate(stories, 1):
        md += f"### {i}. {story.title}\n\n"
        md += f"**Source:** {story.source or 'Unknown'}\n\n"
        if story.content:
            md += f"{story.content[:400]}\n\n"
        md += "---\n\n"
    return md


def format_ingest_as_markdown(result: dict) -> str:
    """Format ingestion results as markdown."""
    md = "# Ingestion Results\n\n"
    md += f"## Statistics\n\n"
    md += f"- **Total articles processed:** {result['total_articles']}\n"
    md += f"- **Unique stories identified:** {result['unique_count']}\n"
    md += f"- **Duplicates detected:** {result['duplicate_count']}\n"
    md += f"- **Skipped (already processed):** {result['skipped_count']}\n\n"
    
    if result.get('unique_stories'):
        md += "## Unique Stories\n\n"
        for i, story in enumerate(result['unique_stories'], 1):
            pa = story.primary_article
            md += f"### {i}. {pa.article.title}\n\n"
            md += f"**Source:** {pa.article.source or 'Unknown'}\n\n"
            
            if pa.entities:
                md += "**Entities:** "
                md += ", ".join([f"`{e.name}` ({e.type.value})" for e in pa.entities])
                md += "\n\n"
            
            if pa.stock_impacts:
                md += "**Impacted Stocks:** "
                md += ", ".join([f"{i.symbol} ({i.confidence:.0%})" for i in pa.stock_impacts])
                md += "\n\n"
            
            if story.duplicate_count > 0:
                md += f"⚠️ *Has {story.duplicate_count} duplicate(s)*\n\n"
            
            md += "---\n\n"
    
    return md


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Financial News Intelligence API",
        "version": "1.0.0",
        "description": "AI-Powered Financial News Intelligence System",
        "endpoints": {
            "POST /ingest": "Ingest articles into the system",
            "POST /query": "Query the system for relevant news",
            "GET /stats": "Get system statistics",
            "GET /demo": "Run demo with sample data",
        }
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_articles(request: IngestRequest):
    """
    Ingest articles into the intelligence system.
    
    The system will:
    - Parse and validate articles
    - Detect and remove duplicates using semantic similarity
    - Extract entities (companies, sectors, regulators)
    - Map stock impacts with confidence levels
    - Store in vector database for querying
    """
    try:
        articles = [art.model_dump() for art in request.articles]
        result = system.ingest(articles, force=request.force)
        
        # Convert unique_stories to serializable format
        unique_stories_data = []
        for story in result.get("unique_stories", []):
            story_data = {
                "id": story.id,
                "title": story.primary_article.article.title,
                "source": story.primary_article.article.source,
                "entities": [{"name": e.name, "type": e.type.value} for e in story.primary_article.entities],
                "stock_impacts": [{"symbol": i.symbol, "confidence": i.confidence} for i in story.primary_article.stock_impacts],
                "duplicate_count": story.duplicate_count,
            }
            unique_stories_data.append(story_data)
        
        return IngestResponse(
            total_articles=result["total_articles"],
            unique_count=result["unique_count"],
            duplicate_count=result["duplicate_count"],
            skipped_count=result["skipped_count"],
            message=f"Successfully processed {result['total_articles']} articles. Found {result['unique_count']} unique stories.",
            unique_stories=unique_stories_data,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
async def query_system(request: QueryRequest):
    """
    Query the intelligence system for relevant news.
    
    The system handles various query patterns:
    - Company queries: "HDFC Bank news" → Direct mentions + sector news
    - Sector queries: "Banking sector update" → All related news
    - Regulator queries: "RBI policy changes" → Regulator-specific news
    - Thematic queries: "Interest rate impact" → Semantic matching
    """
    try:
        request_started = perf_counter()
        result = system.handle_user_query(request.query)
        
        stories = [story_to_response(story) for story in result.stories]
        entities = [
            EntityResponse(name=e.name, type=e.type.value, confidence=e.confidence)
            for e in result.matched_entities
        ]
        
        if result.intent == QueryIntent.GENERAL_CHAT:
            markdown = format_general_as_markdown(request.query, result.explanation)
        elif result.intent == QueryIntent.NEWS_UPDATE:
            markdown = format_update_as_markdown(request.query, stories, result.explanation)
        else:
            markdown = format_query_as_markdown(request.query, stories, result.explanation)
        result.timing.api_ms = round((perf_counter() - request_started) * 1000, 1)
        
        # Save chat to history
        try:
            chat_history.save_chat(
                query=request.query,
                intent=result.intent.value,
                intent_source=result.intent_source,
                explanation=result.explanation,
                stories=stories,
                matched_entities=entities,
                markdown_response=markdown,
                timing=result.timing.model_dump(),
            )
        except Exception:
            pass  # Don't fail the query if history save fails
        
        return QueryResponse(
            query=result.query,
            intent=result.intent,
            intent_source=result.intent_source,
            intent_reason=result.intent_reason,
            stories=stories,
            matched_entities=entities,
            explanation=result.explanation,
            markdown_response=markdown,
            timing=result.timing,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    try:
        stats = system.get_stats()
        return StatsResponse(
            indexed_stories=stats["indexed_stories"],
            total_stories=stats["total_stories"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo")
async def run_demo():
    """
    Run demo with sample data from the problem statement.
    
    Ingests sample articles and returns processing results in markdown format.
    """
    try:
        # Sample articles from problem statement
        articles = [
            {
                "title": "HDFC Bank announces 15% dividend, board approves stock buyback",
                "content": """HDFC Bank, India's largest private sector lender, announced a 15% dividend 
                for the fiscal year. The board also approved a stock buyback program worth Rs 2,500 crore. 
                The bank reported strong quarterly results with net profit growing 20% year-on-year.
                This move is expected to boost investor confidence in the banking sector.""",
                "source": "Economic Times"
            },
            {
                "title": "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns",
                "content": """The Reserve Bank of India (RBI) increased the repo rate by 25 basis points 
                to 6.75% in its monetary policy review. Governor Shaktikanta Das cited persistent 
                inflation concerns as the primary reason. This marks the sixth consecutive rate hike 
                by the central bank. Banking sector stocks saw mixed reactions to the announcement.""",
                "source": "MoneyControl"
            },
            {
                "title": "ICICI Bank opens 500 new branches across Tier-2 cities",
                "content": """ICICI Bank announced the opening of 500 new branches in Tier-2 and Tier-3 
                cities as part of its expansion strategy. The bank aims to increase its footprint 
                in underserved markets. This expansion will create over 3,000 new jobs and enhance 
                financial inclusion in these regions.""",
                "source": "Business Standard"
            },
            {
                "title": "Banking sector NPAs decline to 5-year low, credit growth at 16%",
                "content": """The Indian banking sector reported a significant decline in non-performing 
                assets (NPAs) to a 5-year low. Credit growth remained robust at 16% driven by retail 
                and MSME lending. Major banks including HDFC Bank, ICICI Bank, and SBI have shown 
                improved asset quality. Analysts expect the positive trend to continue.""",
                "source": "LiveMint"
            },
            {
                "title": "Reserve Bank hikes interest rates by 0.25% in surprise move",
                "content": """In a surprise announcement, the Reserve Bank of India increased interest 
                rates by 0.25 percentage points. The central bank governor cited inflationary pressures 
                as the key driver. The rate hike impacts all banks and is expected to increase EMIs 
                for home and auto loans.""",
                "source": "Financial Express"
            },
            {
                "title": "Central bank raises policy rate 25bps, signals hawkish stance",
                "content": """The RBI raised its policy repo rate by 25 basis points today, signaling 
                a hawkish monetary policy stance. The central bank indicated more rate hikes may follow 
                if inflation doesn't moderate. Bond markets reacted negatively to the announcement 
                while bank stocks showed resilience.""",
                "source": "Trade Brains"
            },
        ]
        
        result = system.ingest(articles)
        markdown = format_ingest_as_markdown(result)
        
        return {
            "message": "Demo completed successfully",
            "stats": {
                "total_articles": result["total_articles"],
                "unique_count": result["unique_count"],
                "duplicate_count": result["duplicate_count"],
                "skipped_count": result["skipped_count"],
            },
            "markdown_response": markdown,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "financial-news-intelligence"}


# Chat History Endpoints
@app.get("/chats")
async def get_chat_history(limit: int = 50):
    """
    Get recent chat history.
    
    Args:
        limit: Maximum number of chats to return (default 50)
    """
    try:
        chats = chat_history.get_recent_chats(limit=limit)
        return {"chats": chats, "count": len(chats)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats/{chat_id}")
async def get_chat(chat_id: int):
    """Get a specific chat by ID."""
    try:
        chat = chat_history.get_chat_by_id(chat_id)
        if not chat:
            raise HTTPException(status_code=404, detail="Chat not found")
        return chat
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chats/search/{search_query}")
async def search_chats(search_query: str, limit: int = 20):
    """Search chat history by query text."""
    try:
        chats = chat_history.search_chats(search_query, limit=limit)
        return {"chats": chats, "count": len(chats)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    """Delete a specific chat."""
    try:
        deleted = chat_history.delete_chat(chat_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Chat not found")
        return {"message": "Chat deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chats")
async def clear_chat_history():
    """Clear all chat history."""
    try:
        count = chat_history.clear_history()
        return {"message": f"Cleared {count} chats"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
