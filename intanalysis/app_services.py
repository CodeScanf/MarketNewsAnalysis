"""Application services for auth, namespace metadata, and system resolution."""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

from intanalysis.core import IntelligenceSystem
from intanalysis.models import AuthenticatedUser, KnowledgeNamespace, User


def utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def utc_timestamp(value: Optional[datetime] = None) -> str:
    """Serialize a timestamp for SQLite storage."""
    return (value or utc_now()).replace(microsecond=0).isoformat()


@dataclass
class KnowledgeContext:
    """Namespace metadata used by the API layer."""

    public_namespace: KnowledgeNamespace
    default_private_namespace: Optional[KnowledgeNamespace] = None


class RecommendationService:
    """Run the recommendation workflow against chat history and a vector store."""

    FEED_LIMIT = 10
    APP_TZ = timezone(timedelta(hours=8))
    EMPTY_FEED_SUMMARY = "今天暂时没有新的推荐卡片了。"

    def __init__(self, db: "AppDatabase", chat_history: "ChatHistoryManager"):
        self.db = db
        self.chat_history = chat_history
        self._graph = None

    @property
    def graph(self):
        if self._graph is None:
            from intanalysis.workflow import build_recommendation_graph

            self._graph = build_recommendation_graph()
        return self._graph

    def get_recommendations(
        self,
        user_id: int,
        vector_store,
        storage_dir: str | Path,
    ) -> dict:
        """Build or return the daily recommendation snapshot for the current user."""
        served_date = self._served_date_key()
        snapshot = self.get_daily_snapshot(user_id, served_date)
        if snapshot is not None:
            return snapshot

        state = {
            "user_id": user_id,
            "storage_dir": str(Path(storage_dir)),
            "vector_store": vector_store,
            "chat_loader": self.chat_history.get_recent_chats,
            "errors": [],
        }
        result = self.graph.invoke(state)
        cards = result.get("cards", [])[: self.FEED_LIMIT]
        response = {
            "mode": result.get("recommendation_mode", "latest"),
            "feed_summary": result.get("feed_summary", "") if cards else self.EMPTY_FEED_SUMMARY,
            "cards": cards,
        }
        self.save_daily_snapshot(user_id=user_id, served_date=served_date, response=response)

        saved_snapshot = self.get_daily_snapshot(user_id, served_date)
        return saved_snapshot if saved_snapshot is not None else response

    @classmethod
    def _served_date_key(cls, now: Optional[datetime] = None) -> str:
        """Return the recommendation snapshot date key."""
        current = now.astimezone(cls.APP_TZ) if now is not None else datetime.now(cls.APP_TZ)
        return current.date().isoformat()

    def get_daily_snapshot(self, user_id: int, served_date: str) -> Optional[dict]:
        """Return the saved daily recommendation snapshot for a user."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT recommendation_mode, feed_summary, cards_json
                FROM recommendation_snapshots
                WHERE user_id = ? AND served_date = ?
                """,
                (user_id, served_date),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return {
            "mode": row["recommendation_mode"],
            "feed_summary": row["feed_summary"] or "",
            "cards": json.loads(row["cards_json"]) if row["cards_json"] else [],
        }

    def save_daily_snapshot(self, user_id: int, served_date: str, response: dict) -> None:
        """Persist the user's recommendation snapshot for the current day."""
        mode = response.get("mode", "latest")
        cards = response.get("cards", [])
        created_at = utc_timestamp()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO recommendation_snapshots (
                    user_id, served_date, recommendation_mode, feed_summary, cards_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    served_date,
                    mode,
                    response.get("feed_summary", ""),
                    json.dumps(cards, ensure_ascii=False),
                    created_at,
                ),
            )
            if cursor.rowcount <= 0 or not cards:
                return
            cursor.executemany(
                """
                INSERT OR IGNORE INTO recommendation_impressions (
                    user_id, story_id, recommendation_mode, served_date, created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (user_id, story_id, mode, served_date, created_at)
                    for story_id in [card.get("story_id") for card in cards]
                    if story_id
                ],
            )


class AppDatabase:
    """SQLite-backed application database."""

    def __init__(self, db_path: str = "dataset/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a SQLite connection with row access enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize application tables and indexes."""
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    display_name TEXT,
                    is_admin INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_login_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_token_hash TEXT NOT NULL UNIQUE,
                    user_id INTEGER NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_namespaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slug TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    scope_type TEXT NOT NULL CHECK (scope_type IN ('public', 'private')),
                    owner_user_id INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(owner_user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT 'financial_query',
                    intent_source TEXT NOT NULL DEFAULT 'pipeline',
                    explanation TEXT,
                    markdown_response TEXT,
                    stories_count INTEGER DEFAULT 0,
                    stories_json TEXT,
                    matched_entities_json TEXT,
                    attachments_json TEXT,
                    timing_json TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_impressions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    story_id TEXT NOT NULL,
                    recommendation_mode TEXT NOT NULL,
                    served_date TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, story_id, served_date)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recommendation_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    served_date TEXT NOT NULL,
                    recommendation_mode TEXT NOT NULL,
                    feed_summary TEXT NOT NULL,
                    cards_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    UNIQUE(user_id, served_date)
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id TEXT PRIMARY KEY,
                    doc_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT,
                    published_at TEXT,
                    language TEXT NOT NULL DEFAULT 'zh',
                    summary TEXT NOT NULL DEFAULT '',
                    url TEXT,
                    storage_path TEXT,
                    mime_type TEXT,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    entities_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_no INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    page_no INTEGER,
                    section_title TEXT,
                    anchor_label TEXT,
                    block_type TEXT NOT NULL DEFAULT 'paragraph',
                    FOREIGN KEY(document_id) REFERENCES knowledge_documents(id) ON DELETE CASCADE
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id
                ON sessions(user_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at
                ON sessions(expires_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_namespaces_owner
                ON knowledge_namespaces(owner_user_id)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chats_user_created_at
                ON chats(user_id, created_at DESC)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recommendation_impressions_user_date
                ON recommendation_impressions(user_id, served_date)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_recommendation_snapshots_user_date
                ON recommendation_snapshots(user_id, served_date)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_type_source
                ON knowledge_documents(doc_type, source)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_documents_published
                ON knowledge_documents(published_at)
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document
                ON knowledge_chunks(document_id, chunk_no)
                """
            )
            for statement in (
                "ALTER TABLE chats ADD COLUMN intent TEXT NOT NULL DEFAULT 'financial_query'",
                "ALTER TABLE chats ADD COLUMN intent_source TEXT NOT NULL DEFAULT 'pipeline'",
                "ALTER TABLE chats ADD COLUMN attachments_json TEXT",
            ):
                try:
                    cursor.execute(statement)
                except sqlite3.OperationalError:
                    pass
            self._ensure_public_namespace(cursor)

    def _ensure_public_namespace(self, cursor: sqlite3.Cursor) -> None:
        """Create the reserved public namespace."""
        cursor.execute(
            """
            INSERT OR IGNORE INTO knowledge_namespaces (slug, name, scope_type, owner_user_id, created_at)
            VALUES (?, ?, 'public', NULL, ?)
            """,
            ("public", "Public Knowledge Base", utc_timestamp()),
        )

    @staticmethod
    def row_to_user(row: sqlite3.Row) -> User:
        """Convert a row into a user model."""
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            display_name=row["display_name"],
            is_admin=bool(row["is_admin"]),
            status=row["status"],
            created_at=row["created_at"],
            last_login_at=row["last_login_at"],
        )

    @staticmethod
    def row_to_namespace(row: sqlite3.Row) -> KnowledgeNamespace:
        """Convert a row into a namespace model."""
        return KnowledgeNamespace(
            id=row["id"],
            slug=row["slug"],
            name=row["name"],
            scope_type=row["scope_type"],
            owner_user_id=row["owner_user_id"],
            created_at=row["created_at"],
        )


class AuthService:
    """User registration, authentication, and session lifecycle."""

    def __init__(self, db: AppDatabase, session_ttl_days: int = 30):
        self.db = db
        self.session_ttl = timedelta(days=session_ttl_days)

    @staticmethod
    def normalize_username(username: str) -> str:
        """Normalize a username for storage and lookup."""
        return username.strip().lower()

    @staticmethod
    def normalize_email(email: str) -> str:
        """Normalize an email for storage and lookup."""
        return email.strip().lower()

    @staticmethod
    def hash_password(password: str, salt_hex: str) -> str:
        """Hash a password with a stored salt."""
        salt = bytes.fromhex(salt_hex)
        return hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            390000,
        ).hex()

    @staticmethod
    def generate_salt() -> str:
        """Create a random password salt."""
        return secrets.token_hex(16)

    @staticmethod
    def hash_session_token(token: str) -> str:
        """Hash a raw session token for storage."""
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
        display_name: Optional[str] = None,
    ) -> User:
        """Create a user and the default private namespace."""
        normalized_username = self.normalize_username(username)
        normalized_email = self.normalize_email(email)
        if not normalized_username or not normalized_email or not password:
            raise ValueError("username, email, and password are required")

        salt = self.generate_salt()
        password_hash = self.hash_password(password, salt)
        created_at = utc_timestamp()

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT 1 FROM users WHERE username = ?",
                (normalized_username,),
            )
            if cursor.fetchone():
                raise ValueError("username already exists")
            cursor.execute(
                "SELECT 1 FROM users WHERE email = ?",
                (normalized_email,),
            )
            if cursor.fetchone():
                raise ValueError("email already exists")

            cursor.execute(
                """
                INSERT INTO users (
                    username, email, password_hash, password_salt,
                    display_name, is_admin, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, 0, 'active', ?)
                """,
                (
                    normalized_username,
                    normalized_email,
                    password_hash,
                    salt,
                    display_name.strip() if display_name else normalized_username,
                    created_at,
                ),
            )
            user_id = int(cursor.lastrowid)
            cursor.execute(
                """
                INSERT INTO knowledge_namespaces (
                    slug, name, scope_type, owner_user_id, created_at
                )
                VALUES (?, ?, 'private', ?, ?)
                """,
                (
                    f"user-{user_id}-default",
                    f"{normalized_username}'s Private Knowledge Base",
                    user_id,
                    created_at,
                ),
            )
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
        return self.db.row_to_user(row)

    def authenticate(self, identifier: str, password: str) -> Optional[User]:
        """Authenticate via username or email."""
        normalized_identifier = identifier.strip().lower()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM users
                WHERE username = ? OR email = ?
                """,
                (normalized_identifier, normalized_identifier),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            if row["status"] != "active":
                return None
            computed_hash = self.hash_password(password, row["password_salt"])
            if not secrets.compare_digest(computed_hash, row["password_hash"]):
                return None
            login_at = utc_timestamp()
            cursor.execute(
                "UPDATE users SET last_login_at = ? WHERE id = ?",
                (login_at, row["id"]),
            )
            cursor.execute("SELECT * FROM users WHERE id = ?", (row["id"],))
            return self.db.row_to_user(cursor.fetchone())

    def create_session(self, user_id: int) -> str:
        """Create a new session and return the raw token."""
        token = secrets.token_urlsafe(32)
        token_hash = self.hash_session_token(token)
        now = utc_now()
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_token_hash, user_id, expires_at, created_at, last_seen_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    token_hash,
                    user_id,
                    utc_timestamp(now + self.session_ttl),
                    utc_timestamp(now),
                    utc_timestamp(now),
                ),
            )
        return token

    def delete_session(self, token: str) -> None:
        """Delete a session token if present."""
        with self.db.connection() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE session_token_hash = ?",
                (self.hash_session_token(token),),
            )

    def get_user_for_session(self, token: str) -> Optional[User]:
        """Resolve the current user from a raw session token."""
        token_hash = self.hash_session_token(token)
        now = utc_timestamp()
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT sessions.id AS session_id, users.*
                FROM sessions
                JOIN users ON users.id = sessions.user_id
                WHERE sessions.session_token_hash = ?
                """,
                (token_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            cursor.execute(
                "SELECT expires_at FROM sessions WHERE id = ?",
                (row["session_id"],),
            )
            session_row = cursor.fetchone()
            if session_row is None or session_row["expires_at"] <= now:
                cursor.execute("DELETE FROM sessions WHERE id = ?", (row["session_id"],))
                return None
            cursor.execute(
                "UPDATE sessions SET last_seen_at = ? WHERE id = ?",
                (now, row["session_id"]),
            )
            return self.db.row_to_user(row)


class KnowledgeContextResolver:
    """Resolve namespace metadata and storage locations."""

    def __init__(self, db: AppDatabase, dataset_root: str = "dataset"):
        self.db = db
        self.dataset_root = Path(dataset_root)
        self.public_storage_dir = self.dataset_root / "knowledge" / "public"
        self.users_root_dir = self.dataset_root / "knowledge" / "users"

    def get_public_namespace(self) -> KnowledgeNamespace:
        """Fetch the reserved public namespace."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM knowledge_namespaces WHERE slug = ?",
                ("public",),
            )
            row = cursor.fetchone()
        if row is None:
            raise RuntimeError("public namespace is missing")
        return self.db.row_to_namespace(row)

    def get_default_private_namespace(self, user_id: int) -> Optional[KnowledgeNamespace]:
        """Fetch the default private namespace for a user."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM knowledge_namespaces
                WHERE owner_user_id = ? AND slug = ?
                """,
                (user_id, f"user-{user_id}-default"),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return self.db.row_to_namespace(row)

    def get_context_for_user(self, user: User) -> KnowledgeContext:
        """Return the public and private namespace context for a user."""
        return KnowledgeContext(
            public_namespace=self.get_public_namespace(),
            default_private_namespace=self.get_default_private_namespace(user.id),
        )

    def get_storage_dir(self, namespace: KnowledgeNamespace) -> Path:
        """Map a namespace to its storage directory."""
        if namespace.scope_type == "public":
            return self.public_storage_dir
        if namespace.owner_user_id is None:
            raise ValueError("private namespace must have an owner")
        return self.users_root_dir / str(namespace.owner_user_id) / "default"


class IntelligenceSystemResolver:
    """Cache IntelligenceSystem instances by storage directory."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._cache: dict[str, IntelligenceSystem] = {}

    def get_system(
        self,
        storage_dir: str | Path,
        legacy_storage_dir: Optional[str | Path] = None,
    ) -> IntelligenceSystem:
        """Return a cached system instance for a storage location."""
        key = str(Path(storage_dir))
        if key not in self._cache:
            self._cache[key] = IntelligenceSystem(
                verbose=self.verbose,
                storage_dir=key,
                legacy_storage_dir=str(Path(legacy_storage_dir)) if legacy_storage_dir else None,
            )
        return self._cache[key]


class ChatHistoryManager:
    """User-scoped chat history persistence backed by the application DB."""

    def __init__(self, db: AppDatabase):
        self.db = db

    def save_chat(
        self,
        user_id: int,
        query: str,
        explanation: Optional[str],
        stories: list,
        matched_entities: list,
        attachments: Optional[list[dict]] = None,
        markdown_response: Optional[str] = None,
        timing: Optional[dict] = None,
        intent: Optional[str] = None,
        intent_source: Optional[str] = None,
    ) -> int:
        """Persist a chat record for a user."""
        stories_json = json.dumps(
            [
                {
                    "id": s.get("id") if isinstance(s, dict) else getattr(s, "id", None),
                    "title": s.get("title") if isinstance(s, dict) else getattr(s, "title", None),
                    "source": s.get("source") if isinstance(s, dict) else getattr(s, "source", None),
                }
                for s in stories
            ]
        )
        entities_json = json.dumps(
            [
                {
                    "name": e.get("name") if isinstance(e, dict) else getattr(e, "name", None),
                    "type": e.get("type") if isinstance(e, dict) else getattr(e, "type", None),
                }
                for e in matched_entities
            ]
        )
        attachments_json = json.dumps(attachments or [], ensure_ascii=False)
        timing_json = json.dumps(timing) if timing else None

        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO chats (
                    user_id, query, intent, intent_source, explanation,
                    markdown_response, stories_count, stories_json,
                    matched_entities_json, attachments_json, timing_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    query,
                    intent or "financial_query",
                    intent_source or "pipeline",
                    explanation,
                    markdown_response,
                    len(stories),
                    stories_json,
                    entities_json,
                    attachments_json,
                    timing_json,
                    utc_timestamp(),
                ),
            )
            return int(cursor.lastrowid)

    def get_recent_chats(self, user_id: int, limit: int = 50) -> list[dict]:
        """Return the current user's recent chats."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, query, intent, intent_source, explanation, markdown_response,
                       stories_count, stories_json, matched_entities_json, attachments_json, timing_json, created_at
                FROM chats
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
        return [self._row_to_chat(row) for row in rows]

    def get_chat_by_id(self, user_id: int, chat_id: int) -> Optional[dict]:
        """Return a single chat owned by the current user."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, query, intent, intent_source, explanation, markdown_response,
                       stories_count, stories_json, matched_entities_json, attachments_json, timing_json, created_at
                FROM chats
                WHERE user_id = ? AND id = ?
                """,
                (user_id, chat_id),
            )
            row = cursor.fetchone()
        return self._row_to_chat(row) if row else None

    def search_chats(self, user_id: int, search_query: str, limit: int = 20) -> list[dict]:
        """Search the current user's chats."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, query, intent, intent_source, explanation, markdown_response,
                       stories_count, stories_json, matched_entities_json, attachments_json, timing_json, created_at
                FROM chats
                WHERE user_id = ? AND query LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, f"%{search_query}%", limit),
            )
            rows = cursor.fetchall()
        return [self._row_to_chat(row) for row in rows]

    def delete_chat(self, user_id: int, chat_id: int) -> bool:
        """Delete a chat owned by the current user."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM chats WHERE user_id = ? AND id = ?",
                (user_id, chat_id),
            )
            return cursor.rowcount > 0

    def clear_history(self, user_id: int) -> int:
        """Delete all chats for the current user."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chats WHERE user_id = ?", (user_id,))
            count = int(cursor.fetchone()[0])
            cursor.execute("DELETE FROM chats WHERE user_id = ?", (user_id,))
            return count

    @staticmethod
    def _row_to_chat(row: sqlite3.Row) -> dict:
        """Convert a chat row into an API response dict."""
        return {
            "id": row["id"],
            "query": row["query"],
            "intent": row["intent"] or "financial_query",
            "intent_source": row["intent_source"] or "pipeline",
            "explanation": row["explanation"],
            "markdown_response": row["markdown_response"],
            "stories_count": row["stories_count"],
            "stories": json.loads(row["stories_json"]) if row["stories_json"] else [],
            "matched_entities": json.loads(row["matched_entities_json"]) if row["matched_entities_json"] else [],
            "attachments": json.loads(row["attachments_json"]) if row["attachments_json"] else [],
            "timing": json.loads(row["timing_json"]) if row["timing_json"] else None,
            "created_at": row["created_at"],
        }


def build_authenticated_user(
    user: User,
    context: KnowledgeContext,
) -> AuthenticatedUser:
    """Attach namespace metadata to a user model."""
    return AuthenticatedUser(
        **user.model_dump(),
        default_private_namespace=context.default_private_namespace,
        public_namespace=context.public_namespace,
    )
