"""Unit tests for intanalysis persistence module."""

import pytest
import json
import tempfile
import shutil
from pathlib import Path

from intanalysis.persistence import PersistenceManager
from intanalysis.embeddings import VectorStore
from intanalysis.models import Article, ProcessedArticle, UniqueStory


class TestPersistenceManager:
    """Tests for PersistenceManager class."""
    
    @pytest.fixture
    def persistence(self, temp_storage_dir):
        """Create PersistenceManager with temp directory."""
        return PersistenceManager(storage_dir=temp_storage_dir)
    
    def test_initialization_creates_directory(self, temp_storage_dir):
        """Test that initialization creates storage directory."""
        pm = PersistenceManager(storage_dir=temp_storage_dir)
        assert Path(temp_storage_dir).exists()
    
    def test_article_hash_from_url(self, persistence):
        """Test that article hash is generated from URL when available."""
        article = {"url": "https://example.com/article", "title": "Test", "content": "Content"}
        hash1 = persistence._article_hash(article)
        
        # Same URL should produce same hash
        article2 = {"url": "https://example.com/article", "title": "Different", "content": "Different"}
        hash2 = persistence._article_hash(article2)
        
        assert hash1 == hash2
    
    def test_article_hash_from_content(self, persistence):
        """Test that article hash is generated from content when no URL."""
        article = {"title": "Test Title", "content": "Test content here"}
        hash1 = persistence._article_hash(article)
        
        # Different content should produce different hash
        article2 = {"title": "Test Title", "content": "Different content"}
        hash2 = persistence._article_hash(article2)
        
        assert hash1 != hash2
    
    def test_get_seen_articles_empty(self, persistence):
        """Test getting seen articles when file doesn't exist."""
        seen = persistence.get_seen_articles()
        assert seen == set()
    
    def test_save_and_get_seen_articles(self, persistence):
        """Test saving and retrieving seen articles."""
        hashes = {"hash1", "hash2", "hash3"}
        persistence.save_seen_articles(hashes)
        
        retrieved = persistence.get_seen_articles()
        assert retrieved == hashes
    
    def test_filter_new_articles(self, persistence, sample_articles):
        """Test filtering out already seen articles."""
        # Mark first 2 articles as seen
        persistence.mark_articles_as_seen(sample_articles[:2])
        
        # Filter all articles
        new_articles, skipped = persistence.filter_new_articles(sample_articles[:4])
        
        assert skipped == 2
        assert len(new_articles) == 2
    
    def test_mark_articles_as_seen(self, persistence, sample_articles):
        """Test marking articles as seen."""
        persistence.mark_articles_as_seen(sample_articles[:3])
        
        seen = persistence.get_seen_articles()
        assert len(seen) == 3
    
    def test_filter_all_new_articles(self, persistence, sample_articles):
        """Test filtering when no articles are seen."""
        new_articles, skipped = persistence.filter_new_articles(sample_articles[:3])
        
        assert skipped == 0
        assert len(new_articles) == 3
    
    def test_filter_all_seen_articles(self, persistence, sample_articles):
        """Test filtering when all articles are seen."""
        persistence.mark_articles_as_seen(sample_articles[:3])
        
        new_articles, skipped = persistence.filter_new_articles(sample_articles[:3])
        
        assert skipped == 3
        assert len(new_articles) == 0
    
    def test_save_and_load_vector_store(self, persistence, sample_processed_article):
        """Test saving and loading vector store."""
        # Create vector store with a story
        vector_store = VectorStore(dimension=768, use_hnsw=False)
        story = UniqueStory(id="test", primary_article=sample_processed_article)
        vector_store.add([story])
        
        # Save
        persistence.save_vector_store(vector_store)
        
        # Load
        loaded = persistence.load_vector_store(dimension=768)
        
        assert loaded is not None
        assert loaded.index.ntotal == 1
        assert len(loaded.stories) == 1
    
    def test_load_vector_store_missing_file(self, persistence):
        """Test loading vector store when file doesn't exist."""
        result = persistence.load_vector_store(dimension=768)
        assert result is None


class TestPersistenceEdgeCases:
    """Edge case tests for persistence."""
    
    def test_corrupted_seen_articles_file(self, temp_storage_dir):
        """Test handling of corrupted seen articles file."""
        pm = PersistenceManager(storage_dir=temp_storage_dir)
        
        # Write corrupted JSON
        with open(pm.seen_articles_file, 'w') as f:
            f.write("not valid json{}")
        
        # Should return empty set, not crash
        seen = pm.get_seen_articles()
        assert seen == set()
    
    def test_old_format_seen_articles(self, temp_storage_dir):
        """Test handling old format of seen articles (URLs)."""
        pm = PersistenceManager(storage_dir=temp_storage_dir)
        
        # Write old format
        old_format = {"seen": ["https://example.com/1", "https://example.com/2"]}
        with open(pm.seen_articles_file, 'w') as f:
            json.dump(old_format, f)
        
        # Should convert to hashes
        seen = pm.get_seen_articles()
        assert len(seen) == 2
        assert all(isinstance(h, str) for h in seen)
    
    def test_empty_article_list(self, temp_storage_dir):
        """Test filtering empty article list."""
        pm = PersistenceManager(storage_dir=temp_storage_dir)
        
        new_articles, skipped = pm.filter_new_articles([])
        
        assert new_articles == []
        assert skipped == 0
    
    def test_article_without_url_or_title(self, temp_storage_dir):
        """Test handling article with minimal data."""
        pm = PersistenceManager(storage_dir=temp_storage_dir)
        
        article = {"content": "Just content"}
        hash_val = pm._article_hash(article)
        
        assert isinstance(hash_val, str)
        assert len(hash_val) > 0

    def test_load_vector_store_from_legacy_public_storage(self, temp_storage_dir, sample_processed_article):
        """Test loading legacy global storage through a namespaced persistence manager."""
        legacy_store = VectorStore(dimension=768, use_hnsw=False)
        legacy_store.add([UniqueStory(id="legacy-story", primary_article=sample_processed_article)])

        legacy_pm = PersistenceManager(storage_dir=temp_storage_dir)
        legacy_pm.save_vector_store(legacy_store)
        legacy_pm.mark_articles_as_seen([sample_processed_article.article.model_dump()])

        public_storage = Path(temp_storage_dir) / "knowledge" / "public"
        public_pm = PersistenceManager(
            storage_dir=str(public_storage),
            legacy_storage_dir=temp_storage_dir,
        )

        loaded = public_pm.load_vector_store(dimension=768)
        seen = public_pm.get_seen_articles()

        assert loaded is not None
        assert loaded.index.ntotal == 1
        assert len(seen) == 1

    def test_save_vector_store_writes_to_namespaced_public_storage(self, temp_storage_dir, sample_processed_article):
        """Test saving after migration writes to the new public namespace path."""
        public_storage = Path(temp_storage_dir) / "knowledge" / "public"
        pm = PersistenceManager(
            storage_dir=str(public_storage),
            legacy_storage_dir=temp_storage_dir,
        )
        vector_store = VectorStore(dimension=768, use_hnsw=False)
        vector_store.add([UniqueStory(id="public-story", primary_article=sample_processed_article)])

        pm.save_vector_store(vector_store)
        pm.mark_articles_as_seen([sample_processed_article.article.model_dump()])

        assert (public_storage / "vector_store.pkl").exists()
        assert (public_storage / "stories.pkl").exists()
        assert (public_storage / "seen_articles.json").exists()
