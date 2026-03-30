"""Persistence layer for storing and loading processed articles and vector store."""

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional, Set

from intanalysis.embeddings import VectorStore, tokenize_text


class PersistenceManager:
    """Manages persistent storage of articles and vector indices."""

    def __init__(
        self,
        storage_dir: str = "dataset",
        legacy_storage_dir: Optional[str] = None,
    ):
        """Initialize persistence manager."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_storage_dir = Path(legacy_storage_dir) if legacy_storage_dir else None

        self.seen_articles_file = self.storage_dir / "seen_articles.json"
        self.vector_store_file = self.storage_dir / "vector_store.pkl"
        self.stories_file = self.storage_dir / "stories.pkl"

        if self.legacy_storage_dir is not None:
            self.legacy_seen_articles_file = self.legacy_storage_dir / "seen_articles.json"
            self.legacy_vector_store_file = self.legacy_storage_dir / "vector_store.pkl"
            self.legacy_stories_file = self.legacy_storage_dir / "stories.pkl"
        else:
            self.legacy_seen_articles_file = None
            self.legacy_vector_store_file = None
            self.legacy_stories_file = None

    def _article_hash(self, article: dict) -> str:
        """Generate a stable unique hash for an article."""
        if article.get("url"):
            return hashlib.md5(article["url"].encode()).hexdigest()
        content = f"{article.get('title', '')}{article.get('content', '')[:100]}"
        return hashlib.md5(content.encode()).hexdigest()

    @staticmethod
    def _load_seen_articles_from_file(path: Path) -> Set[str]:
        """Load seen article hashes from a concrete file path."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and "seen" in data:
                    return {hashlib.md5(url.encode()).hexdigest() for url in data["seen"]}
                return set(data)
        except (json.JSONDecodeError, KeyError):
            return set()

    def get_seen_articles(self) -> Set[str]:
        """Load the set of seen article hashes."""
        if self.seen_articles_file.exists():
            return self._load_seen_articles_from_file(self.seen_articles_file)
        if self.legacy_seen_articles_file and self.legacy_seen_articles_file.exists():
            return self._load_seen_articles_from_file(self.legacy_seen_articles_file)
        return set()

    def save_seen_articles(self, article_hashes: Set[str]) -> None:
        """Save the set of seen article hashes."""
        with open(self.seen_articles_file, "w") as f:
            json.dump(list(article_hashes), f, indent=2)

    def filter_new_articles(self, articles: list[dict]) -> tuple[list[dict], int]:
        """Filter out articles that have already been processed."""
        seen = self.get_seen_articles()
        new_articles = []

        for article in articles:
            article_hash = self._article_hash(article)
            if article_hash not in seen:
                new_articles.append(article)

        skipped_count = len(articles) - len(new_articles)
        return new_articles, skipped_count

    def mark_articles_as_seen(self, articles: list[dict]) -> None:
        """Mark a batch of articles as processed."""
        seen = self.get_seen_articles()
        for article in articles:
            seen.add(self._article_hash(article))
        self.save_seen_articles(seen)

    def load_vector_store(self, dimension: int = 768) -> Optional[VectorStore]:
        """Load a persisted vector store, optionally falling back to legacy storage."""
        stories_file = self.stories_file
        vector_store_file = self.vector_store_file

        if not stories_file.exists() or not vector_store_file.exists():
            if (
                self.legacy_stories_file
                and self.legacy_vector_store_file
                and self.legacy_stories_file.exists()
                and self.legacy_vector_store_file.exists()
            ):
                stories_file = self.legacy_stories_file
                vector_store_file = self.legacy_vector_store_file
            else:
                return None

        try:
            with open(stories_file, "rb") as f:
                stories = pickle.load(f)

            with open(vector_store_file, "rb") as f:
                index_data = pickle.load(f)

            vector_store = VectorStore(dimension=dimension)
            vector_store.stories = stories
            vector_store._corpus_texts = [s.primary_article.article.full_text.lower() for s in stories]

            import faiss

            vector_store.index = faiss.deserialize_index(index_data)

            if vector_store._corpus_texts:
                from rank_bm25 import BM25Okapi

                tokenized = [tokenize_text(text) for text in vector_store._corpus_texts]
                vector_store._bm25 = BM25Okapi(tokenized)

            return vector_store
        except Exception as e:
            print(f"Warning: Could not load persisted data: {e}")
            return None

    def save_vector_store(self, vector_store: VectorStore) -> None:
        """Persist a vector store to disk."""
        try:
            with open(self.stories_file, "wb") as f:
                pickle.dump(vector_store.stories, f)

            import faiss

            index_data = faiss.serialize_index(vector_store.index)

            with open(self.vector_store_file, "wb") as f:
                pickle.dump(index_data, f)
        except Exception as e:
            print(f"Warning: Could not save vector store: {e}")

    def clear_cache(self) -> None:
        """Clear persisted files in the active storage directory."""
        for file in [self.seen_articles_file, self.vector_store_file, self.stories_file]:
            if file.exists():
                file.unlink()
