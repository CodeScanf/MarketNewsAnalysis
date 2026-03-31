"""API tests for auth, session handling, chat isolation, and recommendations."""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi.testclient import TestClient

from api.main import create_app
from intanalysis.embeddings import VectorStore
from intanalysis.models import Article, Entity, EntityType, ProcessedArticle, UniqueStory


def register_user(
    client: TestClient,
    username: str,
    email: str,
    password: str = "secret123",
    display_name: Optional[str] = None,
):
    """Register a user through the API."""
    payload = {
        "username": username,
        "email": email,
        "password": password,
        "display_name": display_name or username.title(),
    }
    return client.post("/auth/register", json=payload)


def build_story(
    story_id: str,
    title: str,
    content: str,
    published_date: str,
    entity_name: str,
    entity_type: EntityType,
    sectors: Optional[list[str]] = None,
):
    """Build a minimal story for recommendation tests."""
    article = Article(
        title=title,
        content=content,
        source="Test Source",
        published_date=published_date,
    )
    processed = ProcessedArticle(
        article=article,
        entities=[Entity(name=entity_name, type=entity_type, confidence=1.0)],
        sectors=sectors or [],
        embedding=[0.1] * 768,
        is_duplicate=False,
    )
    return UniqueStory(id=story_id, primary_article=processed, duplicate_articles=[])


def seed_public_stories(app, dataset_root: Path, stories: list[UniqueStory]) -> None:
    """Seed the public vector store with in-memory stories."""
    services = app.state.services
    public_namespace = services.context_resolver.get_public_namespace()
    public_storage = services.context_resolver.get_storage_dir(public_namespace)
    system = services.system_resolver.get_system(
        storage_dir=public_storage,
        legacy_storage_dir=dataset_root,
    )
    vector_store = VectorStore(dimension=768, use_hnsw=False)
    vector_store.stories = stories
    system._vector_store = vector_store


class TestAuthApi:
    """Authentication and isolation tests."""

    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name)
        self.app = create_app(dataset_root=str(self.dataset_root), verbose=False)
        self.client = TestClient(self.app)

    def teardown_method(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_register_creates_default_private_namespace(self):
        response = register_user(self.client, "alice", "alice@example.com")
        assert response.status_code == 200
        data = response.json()
        assert data["username"] == "alice"
        assert data["is_admin"] is False
        assert data["public_namespace"]["slug"] == "public"
        assert data["default_private_namespace"]["slug"] == f"user-{data['id']}-default"

    def test_duplicate_username_registration_fails(self):
        register_user(self.client, "alice", "alice@example.com")
        response = register_user(self.client, "alice", "other@example.com")
        assert response.status_code == 400
        assert response.json()["detail"] == "username already exists"

    def test_duplicate_email_registration_fails(self):
        register_user(self.client, "alice", "alice@example.com")
        response = register_user(self.client, "bob", "alice@example.com")
        assert response.status_code == 400
        assert response.json()["detail"] == "email already exists"

    def test_login_with_username(self):
        register_user(self.client, "alice", "alice@example.com")
        self.client.post("/auth/logout")

        response = self.client.post(
            "/auth/login",
            json={"identifier": "alice", "password": "secret123"},
        )
        assert response.status_code == 200
        assert response.json()["username"] == "alice"

    def test_login_with_email(self):
        register_user(self.client, "alice", "alice@example.com")
        self.client.post("/auth/logout")

        response = self.client.post(
            "/auth/login",
            json={"identifier": "alice@example.com", "password": "secret123"},
        )
        assert response.status_code == 200
        assert response.json()["email"] == "alice@example.com"

    def test_login_with_wrong_password_fails(self):
        register_user(self.client, "alice", "alice@example.com")
        self.client.post("/auth/logout")

        response = self.client.post(
            "/auth/login",
            json={"identifier": "alice", "password": "wrong-password"},
        )
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid credentials"

    def test_protected_endpoints_require_authentication(self):
        query_response = self.client.post("/query", json={"query": "HDFC Bank news"})
        chats_response = self.client.get("/chats")
        recommendations_response = self.client.get("/recommendations")
        ingest_response = self.client.post("/ingest", json={"articles": []})
        stats_response = self.client.get("/stats")

        assert query_response.status_code == 401
        assert chats_response.status_code == 401
        assert recommendations_response.status_code == 401
        assert ingest_response.status_code == 401
        assert stats_response.status_code == 401

    def test_logout_invalidates_session(self):
        register_user(self.client, "alice", "alice@example.com")
        me_before = self.client.get("/auth/me")
        logout = self.client.post("/auth/logout")
        me_after = self.client.get("/auth/me")

        assert me_before.status_code == 200
        assert logout.status_code == 200
        assert me_after.status_code == 401

    def test_auth_me_returns_current_user(self):
        register_response = register_user(self.client, "alice", "alice@example.com")
        me_response = self.client.get("/auth/me")

        assert register_response.status_code == 200
        assert me_response.status_code == 200
        assert me_response.json()["username"] == "alice"

    def test_chat_history_is_isolated_per_user(self):
        user_a = register_user(self.client, "alice", "alice@example.com").json()
        services = self.app.state.services
        chat_a_id = services.chat_history.save_chat(
            user_id=user_a["id"],
            query="alice-only query",
            explanation="summary",
            stories=[],
            matched_entities=[],
            markdown_response="alice markdown",
            timing={},
        )
        self.client.post("/auth/logout")

        user_b = register_user(self.client, "bob", "bob@example.com").json()
        chat_b_id = services.chat_history.save_chat(
            user_id=user_b["id"],
            query="bob-only query",
            explanation="summary",
            stories=[],
            matched_entities=[],
            markdown_response="bob markdown",
            timing={},
        )

        chats_response = self.client.get("/chats")
        search_response = self.client.get("/chats/search/alice-only")
        foreign_chat_response = self.client.get(f"/chats/{chat_a_id}")
        delete_foreign_response = self.client.delete(f"/chats/{chat_a_id}")
        own_chat_response = self.client.get(f"/chats/{chat_b_id}")

        assert chats_response.status_code == 200
        assert chats_response.json()["count"] == 1
        assert chats_response.json()["chats"][0]["query"] == "bob-only query"
        assert search_response.status_code == 200
        assert search_response.json()["count"] == 0
        assert foreign_chat_response.status_code == 404
        assert delete_foreign_response.status_code == 404
        assert own_chat_response.status_code == 200

    def test_recommendations_use_recent_chat_entities(self):
        user = register_user(self.client, "alice", "alice@example.com").json()
        story = build_story(
            story_id="story-company",
            title="HDFC Bank announces new lending push",
            content="HDFC Bank expanded lending operations and signaled steady banking demand.",
            published_date="2026-03-31T09:00:00+00:00",
            entity_name="HDFC Bank Limited",
            entity_type=EntityType.COMPANY,
            sectors=["Banking"],
        )
        seed_public_stories(self.app, self.dataset_root, [story])
        self.app.state.services.chat_history.save_chat(
            user_id=user["id"],
            query="HDFC Bank latest update",
            explanation="summary",
            stories=[],
            matched_entities=[{"name": "HDFC Bank Limited", "type": "company"}],
            markdown_response="markdown",
            timing={},
        )

        response = self.client.get("/recommendations")

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "personalized"
        assert len(data["cards"]) == 1
        assert data["cards"][0]["title"] == "HDFC Bank announces new lending push"
        assert "HDFCBANK" in data["cards"][0]["stock_symbols"]

    def test_recommendations_fall_back_to_latest_stories_without_chat_history(self):
        register_user(self.client, "alice", "alice@example.com")
        latest_story = build_story(
            story_id="story-latest",
            title="Reserve Bank updates liquidity stance",
            content="The Reserve Bank of India adjusted liquidity operations and guided banks on funding costs.",
            published_date="2026-03-31T11:00:00+00:00",
            entity_name="Reserve Bank of India",
            entity_type=EntityType.REGULATOR,
            sectors=["Banking"],
        )
        older_story = build_story(
            story_id="story-older",
            title="Older banking recap",
            content="A prior summary of banking developments.",
            published_date="2026-03-29T08:00:00+00:00",
            entity_name="Banking",
            entity_type=EntityType.SECTOR,
            sectors=["Banking"],
        )
        seed_public_stories(self.app, self.dataset_root, [older_story, latest_story])

        response = self.client.get("/recommendations")

        assert response.status_code == 200
        data = response.json()
        assert data["mode"] == "latest"
        assert data["cards"][0]["title"] == "Reserve Bank updates liquidity stance"
        assert data["cards"][0]["recommendation_label"] == "最新资讯"

    def test_recommendations_repeat_same_snapshot_for_latest_mode(self):
        register_user(self.client, "alice", "alice@example.com")
        stories = [
            build_story(
                story_id=f"latest-{idx}",
                title=f"Latest story {idx}",
                content=f"Latest market recap {idx}",
                published_date=f"2026-03-{31 - min(idx, 9):02d}T{23 - (idx % 10):02d}:00:00+00:00",
                entity_name="Banking",
                entity_type=EntityType.SECTOR,
                sectors=["Banking"],
            )
            for idx in range(12)
        ]
        seed_public_stories(self.app, self.dataset_root, stories)

        first = self.client.get("/recommendations")
        second = self.client.get("/recommendations")

        assert first.status_code == 200
        assert second.status_code == 200
        first_data = first.json()
        second_data = second.json()
        assert first_data["mode"] == "latest"
        assert len(first_data["cards"]) == 10
        assert second_data == first_data

    def test_recommendations_repeat_same_snapshot_for_personalized_mode(self):
        user = register_user(self.client, "alice", "alice@example.com").json()
        stories = [
            build_story(
                story_id=f"personal-{idx}",
                title=f"HDFC Bank update {idx}",
                content=f"HDFC Bank expanded lending operations in update {idx}.",
                published_date=f"2026-03-{31 - min(idx, 9):02d}T{20 - (idx % 10):02d}:00:00+00:00",
                entity_name="HDFC Bank Limited",
                entity_type=EntityType.COMPANY,
                sectors=["Banking"],
            )
            for idx in range(12)
        ]
        seed_public_stories(self.app, self.dataset_root, stories)
        self.app.state.services.chat_history.save_chat(
            user_id=user["id"],
            query="HDFC Bank latest update",
            explanation="summary",
            stories=[],
            matched_entities=[{"name": "HDFC Bank Limited", "type": "company"}],
            markdown_response="markdown",
            timing={},
        )

        first = self.client.get("/recommendations")
        second = self.client.get("/recommendations")

        assert first.status_code == 200
        assert second.status_code == 200
        first_data = first.json()
        second_data = second.json()
        assert first_data["mode"] == "personalized"
        assert second_data["mode"] == "personalized"
        assert len(first_data["cards"]) == 10
        assert second_data == first_data
