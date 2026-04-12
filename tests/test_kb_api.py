"""API tests for the knowledge base endpoints."""

import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app
from intanalysis.models import (
    Article,
    AttachmentBlock,
    AttachmentContext,
    Entity,
    EntityType,
    ProcessedArticle,
    UniqueStory,
)


class _MockEmbedder:
    def __init__(self):
        self.dimension = 3

    def embed(self, text: str):
        lowered = text.lower()
        return np.array(
            [
                1.0 if "hdfc" in lowered or "回购" in lowered else 0.0,
                1.0 if "rbi" in lowered or "监管" in lowered else 0.0,
                float(len(lowered)) / 100.0,
            ],
            dtype=np.float32,
        )

    def embed_batch(self, texts):
        return np.array([self.embed(text) for text in texts], dtype=np.float32)


def register_user(client: TestClient, username: str, email: str):
    return client.post(
        "/auth/register",
        json={
            "username": username,
            "email": email,
            "password": "secret123",
            "display_name": username.title(),
        },
    )


def make_admin(app, user_id: int) -> None:
    with app.state.services.app_db.connection() as conn:
        conn.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))


def build_story(story_id: str, title: str, content: str, published_date: str) -> UniqueStory:
    article = Article(
        title=title,
        content=content,
        source="Test Source",
        published_date=published_date,
    )
    processed = ProcessedArticle(
        article=article,
        entities=[Entity(name="HDFC Bank Limited", type=EntityType.COMPANY, confidence=1.0)],
        sectors=["Banking"],
        embedding=[0.1] * 768,
    )
    return UniqueStory(id=story_id, primary_article=processed, duplicate_articles=[])


class _StubSystem:
    def __init__(self, stories):
        self._stories = stories
        self.vector_store = type("VectorStoreStub", (), {"stories": stories})()
        self.llm = None

    def ingest(self, articles, force=False):
        return {
            "unique_stories": self._stories,
            "total_articles": len(articles),
            "unique_count": len(self._stories),
            "duplicate_count": 0,
            "skipped_count": 0,
        }


class TestKnowledgeBaseApi:
    def setup_method(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = Path(self.temp_dir.name)
        self.app = create_app(dataset_root=str(self.dataset_root), verbose=False)
        self.client = TestClient(self.app)
        self.app.state.services.knowledge_base_service._embedder = _MockEmbedder()

    def teardown_method(self):
        self.client.close()
        self.temp_dir.cleanup()

    def test_kb_endpoints_require_auth(self):
        assert self.client.post("/kb/query", json={"query": "回购"}).status_code == 401
        assert self.client.get("/kb/documents").status_code == 401
        assert self.client.get("/kb/stats").status_code == 401

    def test_ingest_dual_writes_news_into_knowledge_base(self, monkeypatch):
        response = register_user(self.client, "admin", "admin@example.com")
        user_id = response.json()["id"]
        make_admin(self.app, user_id)
        story = build_story(
            "story-1",
            "HDFC Bank announces share buyback",
            "HDFC Bank plans a buyback and updates investors on capital allocation.",
            "2026-04-01T09:00:00+00:00",
        )
        stub_system = _StubSystem([story])
        monkeypatch.setattr(self.app.state.services.system_resolver, "get_system", lambda **kwargs: stub_system)

        ingest = self.client.post(
            "/ingest",
            json={
                "articles": [
                    {
                        "title": story.primary_article.article.title,
                        "content": story.primary_article.article.content,
                        "source": story.primary_article.article.source,
                        "published_date": story.primary_article.article.published_date,
                    }
                ]
            },
        )

        assert ingest.status_code == 200

        documents = self.client.get("/kb/documents")
        assert documents.status_code == 200
        payload = documents.json()
        assert payload["count"] == 1
        assert payload["documents"][0]["doc_type"] == "news_story"

        query = self.client.post("/kb/query", json={"query": "HDFC Bank 回购了什么？"})
        assert query.status_code == 200
        data = query.json()
        assert data["citations"]
        assert data["citations"][0]["title"] == "HDFC Bank announces share buyback"

    def test_kb_query_respects_empty_filters(self, monkeypatch):
        response = register_user(self.client, "admin", "admin@example.com")
        user_id = response.json()["id"]
        make_admin(self.app, user_id)
        story = build_story(
            "story-1",
            "HDFC Bank announces share buyback",
            "HDFC Bank plans a buyback and updates investors on capital allocation.",
            "2026-04-01T09:00:00+00:00",
        )
        stub_system = _StubSystem([story])
        monkeypatch.setattr(self.app.state.services.system_resolver, "get_system", lambda **kwargs: stub_system)

        ingest = self.client.post(
            "/ingest",
            json={
                "articles": [
                    {
                        "title": story.primary_article.article.title,
                        "content": story.primary_article.article.content,
                        "source": story.primary_article.article.source,
                        "published_date": story.primary_article.article.published_date,
                    }
                ]
            },
        )

        assert ingest.status_code == 200

        query = self.client.post(
            "/kb/query",
            json={
                "query": "HDFC Bank buyback",
                "sources": ["Missing Source"],
            },
        )

        assert query.status_code == 200
        data = query.json()
        assert data["citations"] == []
        assert data["related_documents"] == []
        assert data["answer"] == "No knowledge base documents match the current filters."

    def test_admin_can_upload_attachment_into_knowledge_base(self, monkeypatch):
        response = register_user(self.client, "admin", "admin@example.com")
        user_id = response.json()["id"]
        make_admin(self.app, user_id)

        monkeypatch.setattr(
            "intanalysis.knowledge_base.AttachmentParser.parse_file",
            lambda self, path, file_name=None, content_type=None: AttachmentContext(
                file_name=file_name or "notice.pdf",
                file_type="pdf",
                summary="公司拟回购股份。",
                query_text="公司拟回购股份。",
                page_count=1,
                blocks=[
                    AttachmentBlock(block_id="b1", page_no=1, block_type="paragraph", text="公司拟回购股份，用于员工激励。"),
                    AttachmentBlock(block_id="b2", page_no=1, block_type="paragraph", text="回购资金总额不超过 10 亿元。"),
                ],
                warnings=[],
            ),
        )

        upload = self.client.post(
            "/kb/documents/upload",
            files={"file": ("notice.pdf", b"%PDF-1.4 test", "application/pdf")},
        )

        assert upload.status_code == 200
        document = upload.json()
        assert document["doc_type"] == "attachment_pdf"

        detail = self.client.get(f"/kb/documents/{document['id']}")
        assert detail.status_code == 200
        assert len(detail.json()["chunks"]) == 2

        file_response = self.client.get(f"/kb/documents/{document['id']}/file")
        assert file_response.status_code == 200

    def test_non_admin_cannot_upload_or_rebuild(self):
        register_user(self.client, "alice", "alice@example.com")

        upload = self.client.post(
            "/kb/documents/upload",
            files={"file": ("notice.pdf", b"%PDF-1.4 test", "application/pdf")},
        )
        rebuild = self.client.post("/kb/rebuild-from-public-news")

        assert upload.status_code == 403
        assert rebuild.status_code == 403
