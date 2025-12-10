"""SQLite-based chat history persistence."""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from contextlib import contextmanager


class ChatHistoryManager:
    """Manages persistent storage of chat queries and responses using SQLite."""
    
    _instance = None
    
    def __init__(self, db_path: str = "dataset/chat_history.db"):
        """Initialize chat history manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()
    
    @classmethod
    def get_instance(cls, db_path: str = "dataset/chat_history.db") -> "ChatHistoryManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(db_path)
        return cls._instance
    
    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create chats table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    explanation TEXT,
                    markdown_response TEXT,
                    stories_count INTEGER DEFAULT 0,
                    stories_json TEXT,
                    matched_entities_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add markdown_response column if it doesn't exist (for existing DBs)
            try:
                cursor.execute("ALTER TABLE chats ADD COLUMN markdown_response TEXT")
            except Exception:
                pass  # Column already exists
            
            # Create index for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chats_created_at 
                ON chats(created_at DESC)
            """)
    
    def save_chat(
        self,
        query: str,
        explanation: Optional[str],
        stories: list,
        matched_entities: list,
        markdown_response: Optional[str] = None
    ) -> int:
        """Save a chat query and response.
        
        Args:
            query: User's query string
            explanation: AI-generated explanation
            stories: List of story responses
            matched_entities: List of matched entities
            markdown_response: Full markdown response to display
            
        Returns:
            ID of the saved chat
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Serialize stories and entities to JSON
            stories_json = json.dumps([
                {
                    "id": s.get("id") if isinstance(s, dict) else getattr(s, "id", None),
                    "title": s.get("title") if isinstance(s, dict) else getattr(s, "title", None),
                    "source": s.get("source") if isinstance(s, dict) else getattr(s, "source", None),
                }
                for s in stories
            ])
            
            entities_json = json.dumps([
                {
                    "name": e.get("name") if isinstance(e, dict) else getattr(e, "name", None),
                    "type": e.get("type") if isinstance(e, dict) else getattr(e, "type", None),
                }
                for e in matched_entities
            ])
            
            cursor.execute("""
                INSERT INTO chats (query, explanation, markdown_response, stories_count, stories_json, matched_entities_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query, explanation, markdown_response, len(stories), stories_json, entities_json))
            
            return cursor.lastrowid
    
    def get_recent_chats(self, limit: int = 50) -> list[dict]:
        """Get recent chat history.
        
        Args:
            limit: Maximum number of chats to return
            
        Returns:
            List of chat records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query, explanation, markdown_response, stories_count, stories_json, 
                       matched_entities_json, created_at
                FROM chats
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "query": row["query"],
                    "explanation": row["explanation"],
                    "markdown_response": row["markdown_response"],
                    "stories_count": row["stories_count"],
                    "stories": json.loads(row["stories_json"]) if row["stories_json"] else [],
                    "matched_entities": json.loads(row["matched_entities_json"]) if row["matched_entities_json"] else [],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
    
    def get_chat_by_id(self, chat_id: int) -> Optional[dict]:
        """Get a specific chat by ID.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Chat record or None
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query, explanation, markdown_response, stories_count, stories_json, 
                       matched_entities_json, created_at
                FROM chats
                WHERE id = ?
            """, (chat_id,))
            
            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "query": row["query"],
                    "explanation": row["explanation"],
                    "markdown_response": row["markdown_response"],
                    "stories_count": row["stories_count"],
                    "stories": json.loads(row["stories_json"]) if row["stories_json"] else [],
                    "matched_entities": json.loads(row["matched_entities_json"]) if row["matched_entities_json"] else [],
                    "created_at": row["created_at"],
                }
            return None
    
    def search_chats(self, search_query: str, limit: int = 20) -> list[dict]:
        """Search chat history by query text.
        
        Args:
            search_query: Text to search for
            limit: Maximum number of results
            
        Returns:
            List of matching chat records
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, query, explanation, markdown_response, stories_count, stories_json, 
                       matched_entities_json, created_at
                FROM chats
                WHERE query LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (f"%{search_query}%", limit))
            
            rows = cursor.fetchall()
            return [
                {
                    "id": row["id"],
                    "query": row["query"],
                    "explanation": row["explanation"],
                    "markdown_response": row["markdown_response"],
                    "stories_count": row["stories_count"],
                    "stories": json.loads(row["stories_json"]) if row["stories_json"] else [],
                    "matched_entities": json.loads(row["matched_entities_json"]) if row["matched_entities_json"] else [],
                    "created_at": row["created_at"],
                }
                for row in rows
            ]
    
    def delete_chat(self, chat_id: int) -> bool:
        """Delete a chat by ID.
        
        Args:
            chat_id: Chat ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            return cursor.rowcount > 0
    
    def clear_history(self) -> int:
        """Clear all chat history.
        
        Returns:
            Number of chats deleted
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM chats")
            count = cursor.fetchone()[0]
            cursor.execute("DELETE FROM chats")
            return count
