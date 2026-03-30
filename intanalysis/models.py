"""Data models for IntAnalysis."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field

from text_cleaning import clean_text, combine_article_text


class EntityType(str, Enum):
    """Entity types extracted from articles."""
    COMPANY = "company"
    SECTOR = "sector"
    REGULATOR = "regulator"
    PERSON = "person"


class ImpactType(str, Enum):
    """Stock impact types."""
    DIRECT = "direct"        # Company directly mentioned (100%)
    SECTOR = "sector"        # Sector-wide impact (60-80%)
    REGULATORY = "regulatory"  # Regulatory impact (variable)


class QueryIntent(str, Enum):
    """High-level user intent routing."""
    GENERAL_CHAT = "general_chat"
    NEWS_UPDATE = "news_update"
    FINANCIAL_QUERY = "financial_query"


class Article(BaseModel):
    """Raw news article."""
    id: str = Field(default_factory=lambda: "")
    title: str
    content: str
    source: Optional[str] = None
    published_date: Optional[str] = None  # Keep as string for flexibility
    url: Optional[str] = None

    def __init__(self, **data):
        data["title"] = clean_text(data.get("title", ""))
        data["content"] = combine_article_text(data.get("content", ""), "")
        if data.get("source") is not None:
            data["source"] = clean_text(data.get("source"))
        super().__init__(**data)
        if not self.id:
            import hashlib
            self.id = hashlib.md5(f"{self.title}{self.content[:100]}".encode()).hexdigest()[:12]

    @property
    def full_text(self) -> str:
        return f"{self.title}\n\n{self.content}"


class Entity(BaseModel):
    """Extracted entity."""
    name: str
    type: EntityType
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.name.lower(), self.type))


class StockImpact(BaseModel):
    """Stock impact mapping."""
    symbol: str
    company_name: str
    confidence: float
    impact_type: ImpactType
    reasoning: Optional[str] = None


class ProcessedArticle(BaseModel):
    """Article with extracted entities and embeddings."""
    article: Article
    entities: List[Entity] = Field(default_factory=list)
    stock_impacts: List[StockImpact] = Field(default_factory=list)
    sectors: List[str] = Field(default_factory=list)
    embedding: Optional[List[float]] = None
    cluster_id: Optional[str] = None
    is_duplicate: bool = False


class UniqueStory(BaseModel):
    """Deduplicated story cluster."""
    id: str
    primary_article: ProcessedArticle
    duplicate_articles: List[ProcessedArticle] = Field(default_factory=list)
    
    @property
    def duplicate_count(self) -> int:
        return len(self.duplicate_articles)


class QueryTiming(BaseModel):
    """Latency breakdown for a query."""
    pipeline_ms: float = 0.0
    api_ms: float = 0.0
    stages: dict[str, float] = Field(default_factory=dict)


class IntentDecision(BaseModel):
    """Result of intent classification."""
    intent: QueryIntent
    source: str = "rule"
    confidence: float = 0.0
    reason: str = ""


class QueryResult(BaseModel):
    """Query response."""
    query: str
    intent: QueryIntent = QueryIntent.FINANCIAL_QUERY
    intent_source: str = "pipeline"
    intent_reason: str = ""
    stories: List[UniqueStory] = Field(default_factory=list)
    matched_entities: List[Entity] = Field(default_factory=list)
    explanation: Optional[str] = None
    timing: QueryTiming = Field(default_factory=QueryTiming)


class User(BaseModel):
    """Application user."""
    id: int
    username: str
    email: str
    display_name: Optional[str] = None
    is_admin: bool = False
    status: str = "active"
    created_at: str
    last_login_at: Optional[str] = None


class Session(BaseModel):
    """Server-side session."""
    id: int
    user_id: int
    expires_at: str
    created_at: str
    last_seen_at: Optional[str] = None


class KnowledgeNamespace(BaseModel):
    """Knowledge namespace metadata."""
    id: int
    slug: str
    name: str
    scope_type: str
    owner_user_id: Optional[int] = None
    created_at: str


class AuthenticatedUser(User):
    """Authenticated user enriched with namespace context."""
    default_private_namespace: Optional[KnowledgeNamespace] = None
    public_namespace: Optional[KnowledgeNamespace] = None
