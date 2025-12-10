"""Pytest configuration and shared fixtures for IntAnalysis tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List
import numpy as np

from intanalysis.models import (
    Article, Entity, EntityType, StockImpact, ImpactType,
    ProcessedArticle, UniqueStory, QueryResult
)


# =============================================================================
# Sample Test Data
# =============================================================================

SAMPLE_ARTICLES = [
    {
        "title": "RBI increases repo rate by 25 basis points to combat inflation",
        "content": "The Reserve Bank of India announced a 25 basis point hike in the repo rate, bringing it to 6.75%. The central bank cited persistent inflation as the primary reason for this hawkish move.",
        "source": "Economic Times",
        "url": "https://example.com/rbi-rate-1"
    },
    {
        "title": "Reserve Bank hikes interest rates by 0.25% in surprise move",
        "content": "In a surprise decision, the RBI raised interest rates by 0.25%, signaling concerns about inflation. Markets reacted with mild volatility following the announcement.",
        "source": "Money Control",
        "url": "https://example.com/rbi-rate-2"
    },
    {
        "title": "Central bank raises policy rate 25bps, signals hawkish stance",
        "content": "The Reserve Bank of India increased the policy rate by 25 basis points today. Governor's commentary suggests more rate hikes may be on the horizon as inflation remains elevated.",
        "source": "Business Standard",
        "url": "https://example.com/rbi-rate-3"
    },
    {
        "title": "HDFC Bank announces 15% dividend, board approves stock buyback",
        "content": "HDFC Bank Limited declared a 15% dividend for its shareholders and approved a stock buyback program worth Rs 10,000 crores. The bank reported strong Q3 results with 18% profit growth.",
        "source": "LiveMint",
        "url": "https://example.com/hdfc-dividend"
    },
    {
        "title": "ICICI Bank opens 500 new branches across Tier-2 cities",
        "content": "ICICI Bank announced expansion plans with 500 new branches in Tier-2 and Tier-3 cities. The move aims to capture the growing banking demand in smaller towns and rural areas.",
        "source": "Financial Express",
        "url": "https://example.com/icici-expansion"
    },
    {
        "title": "Banking sector NPAs decline to 5-year low, credit growth at 16%",
        "content": "The Indian banking sector reported its lowest NPA levels in five years. Credit growth remained robust at 16%, driven by retail and corporate lending.",
        "source": "Economic Times",
        "url": "https://example.com/banking-npa"
    },
    {
        "title": "TCS wins $500 million deal with European bank",
        "content": "Tata Consultancy Services won a major digital transformation deal worth $500 million with a leading European bank. The 5-year contract will involve cloud migration and AI implementation.",
        "source": "Business Standard",
        "url": "https://example.com/tcs-deal"
    },
    {
        "title": "Infosys raises revenue guidance after strong Q3",
        "content": "Infosys Limited raised its FY24 revenue guidance to 1.5-2% after reporting strong Q3 results. The IT major saw growth in digital and cloud services.",
        "source": "Money Control",
        "url": "https://example.com/infosys-guidance"
    },
]

DUPLICATE_ARTICLES = [
    {
        "title": "Reliance Industries reports record quarterly profit",
        "content": "Reliance Industries Limited posted a record profit of Rs 18,000 crores in Q3. The oil-to-retail conglomerate saw growth across all business segments.",
        "source": "Economic Times",
        "url": "https://example.com/reliance-1"
    },
    {
        "title": "RIL posts all-time high quarterly profit of Rs 18000 cr",
        "content": "Reliance Industries announced its highest-ever quarterly profit. The company led by Mukesh Ambani reported Rs 18,000 crore net profit driven by refining margins.",
        "source": "Business Standard",
        "url": "https://example.com/reliance-2"
    },
]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_articles() -> List[dict]:
    """Return sample article dictionaries for testing."""
    return SAMPLE_ARTICLES.copy()


@pytest.fixture
def duplicate_articles() -> List[dict]:
    """Return duplicate article pairs for deduplication testing."""
    return DUPLICATE_ARTICLES.copy()


@pytest.fixture
def sample_article_objects() -> List[Article]:
    """Return sample Article objects."""
    return [Article(**art) for art in SAMPLE_ARTICLES]


@pytest.fixture
def sample_processed_article() -> ProcessedArticle:
    """Return a sample processed article with entities and embeddings."""
    article = Article(
        title="HDFC Bank announces 15% dividend",
        content="HDFC Bank Limited declared a 15% dividend for shareholders.",
        source="Test Source"
    )
    return ProcessedArticle(
        article=article,
        entities=[
            Entity(name="HDFC Bank Limited", type=EntityType.COMPANY, confidence=1.0),
            Entity(name="Banking", type=EntityType.SECTOR, confidence=0.8),
        ],
        sectors=["Banking", "Financial Services"],
        embedding=[0.1] * 768,  # Mock embedding
        is_duplicate=False
    )


@pytest.fixture
def sample_unique_story(sample_processed_article) -> UniqueStory:
    """Return a sample unique story."""
    return UniqueStory(
        id="test-story-1",
        primary_article=sample_processed_article,
        duplicate_articles=[]
    )


@pytest.fixture
def temp_storage_dir():
    """Create a temporary directory for storage tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_embeddings():
    """Return mock embedding function for testing without model loading."""
    def embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            np.random.seed(hash(text) % 2**32)
            emb = np.random.randn(768).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)
    return embed


@pytest.fixture
def hdfc_article() -> Article:
    """Return HDFC Bank article for entity extraction tests."""
    return Article(
        title="HDFC Bank announces 15% dividend, board approves stock buyback",
        content="HDFC Bank Limited declared a 15% dividend for its shareholders. The board also approved a stock buyback program. Banking sector remains strong.",
        source="Test Source"
    )


@pytest.fixture
def rbi_article() -> Article:
    """Return RBI article for regulator extraction tests."""
    return Article(
        title="RBI raises repo rate by 25bps",
        content="The Reserve Bank of India announced a rate hike today. The central bank Governor addressed inflation concerns in the monetary policy statement.",
        source="Test Source"
    )


@pytest.fixture
def it_sector_article() -> Article:
    """Return IT sector article for testing."""
    return Article(
        title="TCS wins major deal, Infosys follows",
        content="Tata Consultancy Services and Infosys reported strong deal wins. The IT sector outlook remains positive with digital transformation driving demand.",
        source="Test Source"
    )
