"""Unit tests for intanalysis data models."""

import pytest
import hashlib
from intanalysis.models import (
    Article, Entity, EntityType, StockImpact, ImpactType,
    ProcessedArticle, UniqueStory, QueryResult
)


class TestEntityType:
    """Tests for EntityType enum."""
    
    def test_entity_type_values(self):
        """Test that all expected entity types exist."""
        assert EntityType.COMPANY.value == "company"
        assert EntityType.SECTOR.value == "sector"
        assert EntityType.REGULATOR.value == "regulator"
        assert EntityType.PERSON.value == "person"
    
    def test_entity_type_from_string(self):
        """Test creating EntityType from string."""
        assert EntityType("company") == EntityType.COMPANY
        assert EntityType("regulator") == EntityType.REGULATOR


class TestImpactType:
    """Tests for ImpactType enum."""
    
    def test_impact_type_values(self):
        """Test that all expected impact types exist."""
        assert ImpactType.DIRECT.value == "direct"
        assert ImpactType.SECTOR.value == "sector"
        assert ImpactType.REGULATORY.value == "regulatory"


class TestArticle:
    """Tests for Article model."""
    
    def test_article_creation(self):
        """Test basic article creation."""
        article = Article(
            title="Test Title",
            content="Test content here.",
            source="Test Source",
            url="https://example.com/article"
        )
        assert article.title == "Test Title"
        assert article.content == "Test content here."
        assert article.source == "Test Source"
        assert article.url == "https://example.com/article"
    
    def test_article_auto_id_generation(self):
        """Test that article ID is auto-generated if not provided."""
        article = Article(title="Test", content="Content")
        assert article.id is not None
        assert len(article.id) == 12  # MD5 hash truncated to 12 chars
    
    def test_article_id_consistency(self):
        """Test that same title/content produces same ID."""
        article1 = Article(title="Same Title", content="Same Content")
        article2 = Article(title="Same Title", content="Same Content")
        assert article1.id == article2.id
    
    def test_article_different_id_for_different_content(self):
        """Test that different content produces different ID."""
        article1 = Article(title="Title", content="Content A")
        article2 = Article(title="Title", content="Content B")
        assert article1.id != article2.id
    
    def test_article_full_text_property(self):
        """Test full_text property concatenates title and content."""
        article = Article(title="The Title", content="The content body")
        assert article.full_text == "The Title\n\nThe content body"
    
    def test_article_optional_fields(self):
        """Test that optional fields default to None."""
        article = Article(title="Title", content="Content")
        assert article.source is None
        assert article.published_date is None
        assert article.url is None
    
    def test_article_with_explicit_id(self):
        """Test article with explicitly provided ID."""
        article = Article(id="custom-id-123", title="Title", content="Content")
        assert article.id == "custom-id-123"


class TestEntity:
    """Tests for Entity model."""
    
    def test_entity_creation(self):
        """Test basic entity creation."""
        entity = Entity(name="HDFC Bank", type=EntityType.COMPANY, confidence=0.95)
        assert entity.name == "HDFC Bank"
        assert entity.type == EntityType.COMPANY
        assert entity.confidence == 0.95
    
    def test_entity_default_confidence(self):
        """Test that confidence defaults to 1.0."""
        entity = Entity(name="RBI", type=EntityType.REGULATOR)
        assert entity.confidence == 1.0
    
    def test_entity_hash_same_name_type(self):
        """Test that entities with same name and type have same hash."""
        entity1 = Entity(name="HDFC Bank", type=EntityType.COMPANY)
        entity2 = Entity(name="hdfc bank", type=EntityType.COMPANY)  # lowercase
        assert hash(entity1) == hash(entity2)
    
    def test_entity_hash_different_types(self):
        """Test that entities with different types have different hashes."""
        entity1 = Entity(name="Banking", type=EntityType.SECTOR)
        entity2 = Entity(name="Banking", type=EntityType.COMPANY)
        assert hash(entity1) != hash(entity2)
    
    def test_entity_in_set(self):
        """Test that entities with same hash can be deduplicated manually."""
        e1 = Entity(name="HDFC Bank", type=EntityType.COMPANY)
        e2 = Entity(name="hdfc bank", type=EntityType.COMPANY)  # Same (lowercase)
        e3 = Entity(name="ICICI Bank", type=EntityType.COMPANY)
        
        # Same hash for case-insensitive match
        assert hash(e1) == hash(e2)
        # Different hash for different entities
        assert hash(e1) != hash(e3)
        
        # Note: Pydantic models use object identity for __eq__, not hash
        # So sets won't auto-dedupe, but we can use hash for manual dedup
        seen_hashes = set()
        unique_entities = []
        for e in [e1, e2, e3]:
            h = hash(e)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique_entities.append(e)
        assert len(unique_entities) == 2


class TestStockImpact:
    """Tests for StockImpact model."""
    
    def test_stock_impact_creation(self):
        """Test basic stock impact creation."""
        impact = StockImpact(
            symbol="HDFCBANK",
            company_name="HDFC Bank Limited",
            confidence=1.0,
            impact_type=ImpactType.DIRECT,
            reasoning="Company directly mentioned"
        )
        assert impact.symbol == "HDFCBANK"
        assert impact.company_name == "HDFC Bank Limited"
        assert impact.confidence == 1.0
        assert impact.impact_type == ImpactType.DIRECT
        assert impact.reasoning == "Company directly mentioned"
    
    def test_stock_impact_optional_reasoning(self):
        """Test that reasoning is optional."""
        impact = StockImpact(
            symbol="TCS",
            company_name="Tata Consultancy Services",
            confidence=0.8,
            impact_type=ImpactType.SECTOR
        )
        assert impact.reasoning is None
    
    def test_stock_impact_different_types(self):
        """Test different impact types."""
        direct = StockImpact(
            symbol="A", company_name="A", confidence=1.0, impact_type=ImpactType.DIRECT
        )
        sector = StockImpact(
            symbol="B", company_name="B", confidence=0.7, impact_type=ImpactType.SECTOR
        )
        regulatory = StockImpact(
            symbol="C", company_name="C", confidence=0.6, impact_type=ImpactType.REGULATORY
        )
        
        assert direct.impact_type == ImpactType.DIRECT
        assert sector.impact_type == ImpactType.SECTOR
        assert regulatory.impact_type == ImpactType.REGULATORY


class TestProcessedArticle:
    """Tests for ProcessedArticle model."""
    
    def test_processed_article_creation(self, sample_processed_article):
        """Test processed article with all fields."""
        pa = sample_processed_article
        assert pa.article is not None
        assert len(pa.entities) == 2
        assert len(pa.sectors) == 2
        assert pa.embedding is not None
        assert pa.is_duplicate == False
    
    def test_processed_article_defaults(self):
        """Test processed article with default values."""
        article = Article(title="Test", content="Content")
        pa = ProcessedArticle(article=article)
        
        assert pa.entities == []
        assert pa.stock_impacts == []
        assert pa.sectors == []
        assert pa.embedding is None
        assert pa.cluster_id is None
        assert pa.is_duplicate == False
    
    def test_processed_article_with_stock_impacts(self):
        """Test processed article with stock impacts."""
        article = Article(title="Test", content="Content")
        impacts = [
            StockImpact(
                symbol="HDFCBANK",
                company_name="HDFC Bank",
                confidence=1.0,
                impact_type=ImpactType.DIRECT
            )
        ]
        pa = ProcessedArticle(article=article, stock_impacts=impacts)
        assert len(pa.stock_impacts) == 1
        assert pa.stock_impacts[0].symbol == "HDFCBANK"


class TestUniqueStory:
    """Tests for UniqueStory model."""
    
    def test_unique_story_creation(self, sample_unique_story):
        """Test unique story creation."""
        story = sample_unique_story
        assert story.id == "test-story-1"
        assert story.primary_article is not None
        assert story.duplicate_articles == []
    
    def test_duplicate_count_property(self, sample_processed_article):
        """Test duplicate_count property."""
        article2 = Article(title="Dup 1", content="Duplicate content 1")
        article3 = Article(title="Dup 2", content="Duplicate content 2")
        
        dup1 = ProcessedArticle(article=article2, is_duplicate=True)
        dup2 = ProcessedArticle(article=article3, is_duplicate=True)
        
        story = UniqueStory(
            id="story-with-dups",
            primary_article=sample_processed_article,
            duplicate_articles=[dup1, dup2]
        )
        
        assert story.duplicate_count == 2
    
    def test_unique_story_empty_duplicates(self, sample_processed_article):
        """Test unique story with no duplicates."""
        story = UniqueStory(
            id="no-dups",
            primary_article=sample_processed_article
        )
        assert story.duplicate_count == 0


class TestQueryResult:
    """Tests for QueryResult model."""
    
    def test_query_result_creation(self, sample_unique_story):
        """Test query result creation."""
        result = QueryResult(
            query="HDFC Bank news",
            stories=[sample_unique_story],
            matched_entities=[Entity(name="HDFC Bank", type=EntityType.COMPANY)],
            explanation="Found 1 story mentioning HDFC Bank"
        )
        
        assert result.query == "HDFC Bank news"
        assert len(result.stories) == 1
        assert len(result.matched_entities) == 1
        assert result.explanation is not None
    
    def test_query_result_defaults(self):
        """Test query result with default values."""
        result = QueryResult(query="test query")
        
        assert result.query == "test query"
        assert result.stories == []
        assert result.matched_entities == []
        assert result.explanation is None
    
    def test_query_result_multiple_stories(self, sample_processed_article):
        """Test query result with multiple stories."""
        stories = [
            UniqueStory(id=f"story-{i}", primary_article=sample_processed_article)
            for i in range(5)
        ]
        
        result = QueryResult(query="banking news", stories=stories)
        assert len(result.stories) == 5
