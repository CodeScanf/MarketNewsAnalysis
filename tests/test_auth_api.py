"""API tests for auth, session handling, and chat isolation."""

import tempfile
from pathlib import Path
from typing import Optional

from fastapi.testclient import TestClient

from api.main import create_app


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
        ingest_response = self.client.post("/ingest", json={"articles": []})
        stats_response = self.client.get("/stats")

        assert query_response.status_code == 401
        assert chats_response.status_code == 401
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
