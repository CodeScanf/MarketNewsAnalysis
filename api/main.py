"""FastAPI application for Financial News Intelligence System."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import List, Optional

from fastapi import Cookie, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from intanalysis.app_services import (
    AppDatabase,
    AuthService,
    ChatHistoryManager,
    IntelligenceSystemResolver,
    KnowledgeContextResolver,
    build_authenticated_user,
)
from intanalysis.models import AuthenticatedUser, ConversationTurn, QueryIntent, QueryTiming


SESSION_COOKIE_NAME = "intanalysis_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30


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


@dataclass
class AppServices:
    """Runtime services attached to the FastAPI app."""

    dataset_root: Path
    app_db: AppDatabase
    auth_service: AuthService
    context_resolver: KnowledgeContextResolver
    system_resolver: IntelligenceSystemResolver
    chat_history: ChatHistoryManager


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
    unique_stories: List[dict] = Field(default_factory=list)


class ConversationTurnInput(BaseModel):
    """Recent conversation turn payload from the frontend."""

    role: str
    content: str
    intent: Optional[QueryIntent] = None
    matched_entities: List[str] = Field(default_factory=list)
    story_titles: List[str] = Field(default_factory=list)


class QueryRequest(BaseModel):
    """Request model for querying."""

    query: str
    history: List[ConversationTurnInput] = Field(default_factory=list)


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


class RegisterRequest(BaseModel):
    """Registration request payload."""

    username: str
    email: str
    password: str
    display_name: Optional[str] = None


class LoginRequest(BaseModel):
    """Login request payload."""

    identifier: str
    password: str


class NamespaceResponse(BaseModel):
    """Serialized namespace metadata."""

    id: int
    slug: str
    name: str
    scope_type: str
    owner_user_id: Optional[int] = None
    created_at: str


class MeResponse(BaseModel):
    """Authenticated user response."""

    id: int
    username: str
    email: str
    display_name: Optional[str]
    is_admin: bool
    status: str
    created_at: str
    last_login_at: Optional[str] = None
    default_private_namespace: Optional[NamespaceResponse] = None
    public_namespace: NamespaceResponse


class AuthResponse(MeResponse):
    """Auth response with a generic success message."""

    message: str


class ErrorDetailResponse(BaseModel):
    """Minimal error response model."""

    detail: str


def get_services(request: Request) -> AppServices:
    """Access runtime services from the app state."""
    return request.app.state.services


def serialize_namespace(namespace) -> Optional[NamespaceResponse]:
    """Convert namespace metadata to an API response."""
    if namespace is None:
        return None
    return NamespaceResponse(**namespace.model_dump())


def build_me_response(user: AuthenticatedUser) -> MeResponse:
    """Serialize an authenticated user."""
    return MeResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        display_name=user.display_name,
        is_admin=user.is_admin,
        status=user.status,
        created_at=user.created_at,
        last_login_at=user.last_login_at,
        default_private_namespace=serialize_namespace(user.default_private_namespace),
        public_namespace=serialize_namespace(user.public_namespace),
    )


def set_session_cookie(response: Response, token: str) -> None:
    """Set the auth cookie."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=SESSION_COOKIE_MAX_AGE,
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    """Clear the auth cookie."""
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )


def get_current_user(
    request: Request,
    session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
) -> AuthenticatedUser:
    """Require a valid session cookie and return the current user."""
    if not session_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    services = get_services(request)
    user = services.auth_service.get_user_for_session(session_token)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    context = services.context_resolver.get_context_for_user(user)
    return build_authenticated_user(user, context)


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
                reasoning=i.reasoning,
            )
            for i in pa.stock_impacts
        ],
        sectors=pa.sectors,
        duplicate_count=story.duplicate_count,
    )


def format_query_as_markdown(query: str, stories: list, explanation: Optional[str]) -> str:
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
                md += (
                    f"- **{impact.symbol}** ({impact.company_name}): "
                    f"{impact.confidence:.0%} confidence ({impact.impact_type})\n"
                )
            md += "\n"

        if story.sectors:
            md += f"**Sectors:** {', '.join(story.sectors)}\n\n"

        if story.duplicate_count > 0:
            md += f"*This story has {story.duplicate_count} duplicate article(s)*\n\n"

        md += "---\n\n"

    return md


def format_general_as_markdown(query: str, explanation: Optional[str]) -> str:
    """Format general-chat responses as markdown."""
    return f"# General Answer: \"{query}\"\n\n## Response\n{explanation or 'No answer generated.'}\n"


def format_update_as_markdown(query: str, stories: list, explanation: Optional[str]) -> str:
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
    md += "## Statistics\n\n"
    md += f"- **Total articles processed:** {result['total_articles']}\n"
    md += f"- **Unique stories identified:** {result['unique_count']}\n"
    md += f"- **Duplicates detected:** {result['duplicate_count']}\n"
    md += f"- **Skipped (already processed):** {result['skipped_count']}\n\n"

    if result.get("unique_stories"):
        md += "## Unique Stories\n\n"
        for i, story in enumerate(result["unique_stories"], 1):
            pa = story.primary_article
            md += f"### {i}. {pa.article.title}\n\n"
            md += f"**Source:** {pa.article.source or 'Unknown'}\n\n"
            if pa.entities:
                md += "**Entities:** "
                md += ", ".join([f"`{e.name}` ({e.type.value})" for e in pa.entities])
                md += "\n\n"
            if pa.stock_impacts:
                md += "**Impacted Stocks:** "
                md += ", ".join([f"{impact.symbol} ({impact.confidence:.0%})" for impact in pa.stock_impacts])
                md += "\n\n"
            if story.duplicate_count > 0:
                md += f"*Has {story.duplicate_count} duplicate(s)*\n\n"
            md += "---\n\n"

    return md


def create_app(dataset_root: str = "dataset", verbose: bool = True) -> FastAPI:
    """Create the FastAPI app with runtime services."""
    dataset_path = Path(dataset_root)
    dataset_path.mkdir(parents=True, exist_ok=True)
    app_db = AppDatabase(str(dataset_path / "app.db"))

    app = FastAPI(
        title="Financial News Intelligence API",
        description=(
            "AI-Powered Financial News Intelligence System - Process news, "
            "detect duplicates, extract entities, and query intelligently."
        ),
        version="1.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.services = AppServices(
        dataset_root=dataset_path,
        app_db=app_db,
        auth_service=AuthService(app_db),
        context_resolver=KnowledgeContextResolver(app_db, str(dataset_path)),
        system_resolver=IntelligenceSystemResolver(verbose=verbose),
        chat_history=ChatHistoryManager(app_db),
    )
    services = app.state.services

    @app.get("/")
    async def root():
        """Root endpoint with API info."""
        return {
            "name": "Financial News Intelligence API",
            "version": "1.1.0",
            "description": "AI-Powered Financial News Intelligence System",
            "endpoints": {
                "POST /auth/register": "Register a new user",
                "POST /auth/login": "Login with username or email",
                "POST /auth/logout": "Logout the current user",
                "GET /auth/me": "Get current session user",
                "POST /query": "Route and answer a user query",
                "POST /ingest": "Ingest articles into the public knowledge base",
                "GET /stats": "Get public knowledge base statistics",
                "GET /health": "Service health",
            },
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "financial-news-intelligence"}

    @app.post("/auth/register", response_model=AuthResponse)
    async def register(request: RegisterRequest, response: Response):
        """Register a user and create the default private namespace."""
        try:
            user = services.auth_service.register_user(
                username=request.username,
                email=request.email,
                password=request.password,
                display_name=request.display_name,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        token = services.auth_service.create_session(user.id)
        set_session_cookie(response, token)
        authenticated = build_authenticated_user(
            user,
            services.context_resolver.get_context_for_user(user),
        )
        return AuthResponse(
            **build_me_response(authenticated).model_dump(),
            message="Registered successfully",
        )

    @app.post("/auth/login", response_model=AuthResponse, responses={401: {"model": ErrorDetailResponse}})
    async def login(request: LoginRequest, response: Response):
        """Login via username or email and issue a cookie session."""
        user = services.auth_service.authenticate(request.identifier, request.password)
        if user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        token = services.auth_service.create_session(user.id)
        set_session_cookie(response, token)
        authenticated = build_authenticated_user(
            user,
            services.context_resolver.get_context_for_user(user),
        )
        return AuthResponse(
            **build_me_response(authenticated).model_dump(),
            message="Logged in successfully",
        )

    @app.post("/auth/logout")
    async def logout(
        response: Response,
        session_token: Optional[str] = Cookie(default=None, alias=SESSION_COOKIE_NAME),
    ):
        """Logout the current session."""
        if session_token:
            services.auth_service.delete_session(session_token)
        clear_session_cookie(response)
        return {"message": "Logged out successfully"}

    @app.get("/auth/me", response_model=MeResponse, responses={401: {"model": ErrorDetailResponse}})
    async def me(current_user: AuthenticatedUser = Depends(get_current_user)):
        """Return the current authenticated user."""
        return build_me_response(current_user)

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_articles(
        request: IngestRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
    ):
        """Ingest articles into the public knowledge base."""
        if not current_user.is_admin:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required")

        public_namespace = services.context_resolver.get_public_namespace()
        public_storage = services.context_resolver.get_storage_dir(public_namespace)
        system = services.system_resolver.get_system(
            storage_dir=public_storage,
            legacy_storage_dir=services.dataset_root,
        )

        try:
            articles = [art.model_dump() for art in request.articles]
            result = system.ingest(articles, force=request.force)
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
                message=(
                    f"Successfully processed {result['total_articles']} articles. "
                    f"Found {result['unique_count']} unique stories."
                ),
                unique_stories=unique_stories_data,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/query", response_model=QueryResponse)
    async def query_system(
        request: QueryRequest,
        current_user: AuthenticatedUser = Depends(get_current_user),
    ):
        """Route a query and answer it using the public knowledge base."""
        public_namespace = services.context_resolver.get_public_namespace()
        public_storage = services.context_resolver.get_storage_dir(public_namespace)
        system = services.system_resolver.get_system(
            storage_dir=public_storage,
            legacy_storage_dir=services.dataset_root,
        )

        try:
            request_started = perf_counter()
            history = [
                ConversationTurn(
                    role=turn.role,
                    content=turn.content,
                    intent=turn.intent,
                    matched_entities=turn.matched_entities,
                    story_titles=turn.story_titles,
                )
                for turn in request.history
            ]
            result = system.handle_user_query(request.query, history=history)

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

            services.chat_history.save_chat(
                user_id=current_user.id,
                query=request.query,
                intent=result.intent.value,
                intent_source=result.intent_source,
                explanation=result.explanation,
                stories=stories,
                matched_entities=entities,
                markdown_response=markdown,
                timing=result.timing.model_dump(),
            )

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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/stats", response_model=StatsResponse)
    async def get_stats(current_user: AuthenticatedUser = Depends(get_current_user)):
        """Get public knowledge base statistics."""
        public_namespace = services.context_resolver.get_public_namespace()
        public_storage = services.context_resolver.get_storage_dir(public_namespace)
        system = services.system_resolver.get_system(
            storage_dir=public_storage,
            legacy_storage_dir=services.dataset_root,
        )

        try:
            stats = system.get_stats()
            return StatsResponse(
                indexed_stories=stats["indexed_stories"],
                total_stories=stats["total_stories"],
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/demo")
    async def run_demo():
        """Run a demo against the public knowledge base."""
        public_namespace = services.context_resolver.get_public_namespace()
        public_storage = services.context_resolver.get_storage_dir(public_namespace)
        system = services.system_resolver.get_system(
            storage_dir=public_storage,
            legacy_storage_dir=services.dataset_root,
        )

        try:
            articles = [
                {
                    "title": "HDFC Bank announces 15% dividend, board approves stock buyback",
                    "content": (
                        "HDFC Bank, India's largest private sector lender, announced a 15% dividend "
                        "for the fiscal year. The board also approved a stock buyback program worth Rs 2,500 crore. "
                        "The bank reported strong quarterly results with net profit growing 20% year-on-year."
                    ),
                    "source": "Economic Times",
                },
                {
                    "title": "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns",
                    "content": (
                        "The Reserve Bank of India increased the repo rate by 25 basis points to 6.75% "
                        "in its monetary policy review. Governor Shaktikanta Das cited persistent inflation concerns."
                    ),
                    "source": "MoneyControl",
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
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/chats")
    async def get_chat_history(
        limit: int = 50,
        current_user: AuthenticatedUser = Depends(get_current_user),
    ):
        """Get current user's chat history."""
        try:
            chats = services.chat_history.get_recent_chats(current_user.id, limit=limit)
            return {"chats": chats, "count": len(chats)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/chats/{chat_id}")
    async def get_chat(chat_id: int, current_user: AuthenticatedUser = Depends(get_current_user)):
        """Get a specific chat owned by the current user."""
        try:
            chat = services.chat_history.get_chat_by_id(current_user.id, chat_id)
            if not chat:
                raise HTTPException(status_code=404, detail="Chat not found")
            return chat
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/chats/search/{search_query}")
    async def search_chats(
        search_query: str,
        limit: int = 20,
        current_user: AuthenticatedUser = Depends(get_current_user),
    ):
        """Search the current user's chat history."""
        try:
            chats = services.chat_history.search_chats(current_user.id, search_query, limit=limit)
            return {"chats": chats, "count": len(chats)}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/chats/{chat_id}")
    async def delete_chat(chat_id: int, current_user: AuthenticatedUser = Depends(get_current_user)):
        """Delete a current user's chat."""
        try:
            deleted = services.chat_history.delete_chat(current_user.id, chat_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Chat not found")
            return {"message": "Chat deleted successfully"}
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete("/chats")
    async def clear_chat_history(current_user: AuthenticatedUser = Depends(get_current_user)):
        """Clear all chat history for the current user."""
        try:
            count = services.chat_history.clear_history(current_user.id)
            return {"message": f"Cleared {count} chats"}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
