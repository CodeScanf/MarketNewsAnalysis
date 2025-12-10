"""Data models for IntAnalysis."""

from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


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


class Article(BaseModel):
    """Raw news article."""
    id: str = Field(default_factory=lambda: "")
    title: str
    content: str
    source: Optional[str] = None
    published_date: Optional[str] = None  # Keep as string for flexibility
    url: Optional[str] = None

    def __init__(self, **data):
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


class QueryResult(BaseModel):
    """Query response."""
    query: str
    stories: List[UniqueStory] = Field(default_factory=list)
    matched_entities: List[Entity] = Field(default_factory=list)
    explanation: Optional[str] = None
