"""Knowledge base storage, ingestion, and query helpers."""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from intanalysis.attachments import AttachmentParser
from intanalysis.embeddings import EmbeddingService, Reranker, tokenize_text
from intanalysis.models import (
    AttachmentContext,
    Entity,
    KnowledgeChunk,
    KnowledgeCitation,
    KnowledgeDocument,
    KnowledgeDocType,
    KnowledgeQueryResult,
    QueryTiming,
    UniqueStory,
)
from intanalysis.recommendations import parse_timestamp
from text_cleaning import clean_html_text


_PARA_SPLIT_RE = re.compile(r"\n{2,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？.!?])\s+")
_RECENCY_KEYWORDS = ("最新", "今日", "今天", "最近", "近期", "本周", "newest", "latest", "today", "recent")


def utc_timestamp(value: Optional[datetime] = None) -> str:
    """Serialize a timezone-aware UTC timestamp."""
    current = value or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(text: str) -> str:
    """Normalize text for chunking and indexing."""
    cleaned = clean_html_text(text or "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def build_summary(text: str, limit: int = 300) -> str:
    """Build a compact summary from document text."""
    cleaned = normalize_text(text)
    if len(cleaned) <= limit:
        return cleaned
    snippet = cleaned[:limit].rsplit(" ", 1)[0].strip()
    if len(snippet) < max(120, limit // 2):
        snippet = cleaned[:limit].strip()
    return f"{snippet.rstrip('，。,.!;: ')}..."


def chunk_document_text(
    text: str,
    *,
    document_id: str,
    page_no: Optional[int] = None,
    block_type: str = "paragraph",
    anchor_prefix: str = "Section",
    max_chars: int = 900,
) -> list[KnowledgeChunk]:
    """Split document text into searchable chunks."""
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    paragraphs = [part.strip() for part in _PARA_SPLIT_RE.split(cleaned) if part.strip()]
    chunks: list[KnowledgeChunk] = []
    counter = 1
    for paragraph in paragraphs or [cleaned]:
        if len(paragraph) <= max_chars:
            chunks.append(
                KnowledgeChunk(
                    id=f"{document_id}-c{counter}",
                    document_id=document_id,
                    chunk_no=counter,
                    text=paragraph,
                    page_no=page_no,
                    section_title=None,
                    anchor_label=f"{anchor_prefix} {counter}",
                    block_type=block_type,
                )
            )
            counter += 1
            continue

        current = ""
        pieces: list[str] = []
        for sentence in _SENTENCE_SPLIT_RE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                pieces.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            pieces.append(current)

        for piece in pieces:
            chunks.append(
                KnowledgeChunk(
                    id=f"{document_id}-c{counter}",
                    document_id=document_id,
                    chunk_no=counter,
                    text=piece,
                    page_no=page_no,
                    section_title=None,
                    anchor_label=f"{anchor_prefix} {counter}",
                    block_type=block_type,
                )
            )
            counter += 1
    return chunks


def document_id_for_story(story: UniqueStory) -> str:
    """Build a stable knowledge document id for a news story."""
    return f"news-{story.id}"


def document_id_for_attachment(file_name: str, raw_bytes: bytes) -> str:
    """Build a deterministic-ish id for an uploaded attachment."""
    digest = hashlib.md5(raw_bytes).hexdigest()[:10]
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(file_name).stem).strip("-").lower() or "attachment"
    return f"att-{stem}-{digest}"


def story_to_knowledge(story: UniqueStory) -> tuple[KnowledgeDocument, list[KnowledgeChunk]]:
    """Map a deduplicated story into a knowledge document and chunks."""
    article = story.primary_article.article
    document_id = document_id_for_story(story)
    body = "\n\n".join(part for part in [article.title, article.content] if part)
    tags = list(dict.fromkeys([*story.primary_article.sectors, *[entity.name for entity in story.primary_article.entities[:6]]]))
    document = KnowledgeDocument(
        id=document_id,
        doc_type=KnowledgeDocType.NEWS_STORY,
        title=article.title,
        source=article.source,
        published_at=article.published_date,
        language="zh",
        summary=build_summary(article.content or article.title),
        url=article.url,
        storage_path=None,
        mime_type="text/news",
        tags=tags,
        entities=story.primary_article.entities,
        created_at=utc_timestamp(),
    )
    chunks = chunk_document_text(body, document_id=document_id, block_type="news", anchor_prefix="Section")
    return document, chunks


def attachment_to_knowledge(
    attachment_context: AttachmentContext,
    *,
    document_id: str,
    storage_path: str,
    mime_type: Optional[str],
    created_at: Optional[str] = None,
) -> tuple[KnowledgeDocument, list[KnowledgeChunk]]:
    """Map parsed attachment content into a knowledge document and chunks."""
    doc_type = (
        KnowledgeDocType.ATTACHMENT_PDF
        if attachment_context.file_type == "pdf"
        else KnowledgeDocType.ATTACHMENT_IMAGE
    )
    created_value = created_at or utc_timestamp()
    document = KnowledgeDocument(
        id=document_id,
        doc_type=doc_type,
        title=attachment_context.file_name,
        source="uploaded_attachment",
        published_at=created_value,
        language="zh",
        summary=attachment_context.summary,
        url=None,
        storage_path=storage_path,
        mime_type=mime_type,
        tags=[],
        entities=[],
        created_at=created_value,
    )
    chunks: list[KnowledgeChunk] = []
    for index, block in enumerate(attachment_context.blocks, start=1):
        anchor = f"Page {block.page_no}" if doc_type == KnowledgeDocType.ATTACHMENT_PDF else f"Image Block {index}"
        chunks.append(
            KnowledgeChunk(
                id=f"{document_id}-c{index}",
                document_id=document_id,
                chunk_no=index,
                text=block.text,
                page_no=block.page_no,
                section_title=None,
                anchor_label=anchor,
                block_type=block.block_type,
            )
        )
    return document, chunks


class KnowledgePersistenceManager:
    """Persist chunk index state to disk."""

    def __init__(self, storage_dir: str | Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "kb_index.faiss"
        self.chunks_file = self.storage_dir / "kb_chunks.pkl"
        self.meta_file = self.storage_dir / "kb_meta.json"

    def load_store(self, dimension: int) -> Optional["KnowledgeStore"]:
        """Load a persisted store if available."""
        if not self.index_file.exists() or not self.chunks_file.exists():
            return None
        try:
            with open(self.chunks_file, "rb") as fh:
                chunks = pickle.load(fh)
            index = faiss.read_index(str(self.index_file))
            store = KnowledgeStore(dimension=dimension)
            store.index = index
            store.chunks = chunks
            store._rebuild_auxiliary()
            return store
        except Exception:
            return None

    def save_store(self, store: "KnowledgeStore") -> None:
        """Persist a store to disk."""
        with open(self.chunks_file, "wb") as fh:
            pickle.dump(store.chunks, fh)
        faiss.write_index(store.index, str(self.index_file))
        with open(self.meta_file, "w", encoding="utf-8") as fh:
            json.dump(
                {"chunk_count": len(store.chunks), "updated_at": utc_timestamp()},
                fh,
                ensure_ascii=False,
                indent=2,
            )


class KnowledgeStore:
    """Chunk-level hybrid retrieval store for the knowledge base."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[KnowledgeChunk] = []
        self._chunk_positions_by_doc: dict[str, list[int]] = defaultdict(list)
        self._embeddings_matrix = np.zeros((0, dimension), dtype=np.float32)
        self._corpus_texts: list[str] = []
        self._bm25: Optional[BM25Okapi] = None

    def _rebuild_auxiliary(self) -> None:
        """Rebuild in-memory helpers from the stored chunk list."""
        self._chunk_positions_by_doc = defaultdict(list)
        embeddings: list[list[float]] = []
        self._corpus_texts = []
        for idx, chunk in enumerate(self.chunks):
            self._chunk_positions_by_doc[chunk.document_id].append(idx)
            self._corpus_texts.append(chunk.text.lower())
            embeddings.append(chunk.embedding or [0.0] * self.dimension)
        self._embeddings_matrix = (
            np.array(embeddings, dtype=np.float32)
            if embeddings
            else np.zeros((0, self.dimension), dtype=np.float32)
        )
        if self._corpus_texts:
            tokenized = [tokenize_text(text) for text in self._corpus_texts]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def add_chunks(self, chunks: Iterable[KnowledgeChunk]) -> None:
        """Append chunks to the store."""
        new_chunks = [chunk for chunk in chunks if chunk.embedding]
        if not new_chunks:
            return
        arr = np.array([chunk.embedding for chunk in new_chunks], dtype=np.float32)
        self.index.add(arr)
        self.chunks.extend(new_chunks)
        self._rebuild_auxiliary()

    def remove_document_ids(self, document_ids: set[str]) -> None:
        """Remove all chunks associated with a set of documents."""
        if not document_ids:
            return
        self.chunks = [chunk for chunk in self.chunks if chunk.document_id not in document_ids]
        self.index = faiss.IndexFlatIP(self.dimension)
        if self.chunks:
            arr = np.array([chunk.embedding for chunk in self.chunks if chunk.embedding], dtype=np.float32)
            if len(arr):
                self.index.add(arr)
        self._rebuild_auxiliary()

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        *,
        allowed_doc_ids: Optional[set[str]] = None,
        k: int = 20,
        alpha: float = 0.7,
    ) -> list[tuple[KnowledgeChunk, float]]:
        """Run hybrid retrieval over chunks."""
        if not self.chunks:
            return []

        if allowed_doc_ids:
            candidate_indices = [
                idx
                for doc_id in allowed_doc_ids
                for idx in self._chunk_positions_by_doc.get(doc_id, [])
            ]
        else:
            candidate_indices = list(range(len(self.chunks)))

        if not candidate_indices:
            return []

        candidate_matrix = self._embeddings_matrix[candidate_indices]
        dense_scores = np.dot(candidate_matrix, query_embedding).astype(float)
        if dense_scores.size and dense_scores.max() > dense_scores.min():
            dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min())

        bm25_scores = np.zeros(len(candidate_indices), dtype=float)
        if self._bm25 and query_text:
            query_tokens = tokenize_text(query_text)
            if query_tokens:
                raw_scores = self._bm25.get_scores(query_tokens)
                candidate_raw = raw_scores[candidate_indices]
                if candidate_raw.max() > 0:
                    bm25_scores = candidate_raw / candidate_raw.max()

        scored: list[tuple[KnowledgeChunk, float]] = []
        for pos, chunk_index in enumerate(candidate_indices):
            dense = float(dense_scores[pos]) if len(dense_scores) else 0.0
            lexical = float(bm25_scores[pos]) if len(bm25_scores) else 0.0
            scored.append((self.chunks[chunk_index], (alpha * dense) + ((1 - alpha) * lexical)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: min(k, len(scored))]


class KnowledgeReranker:
    """Chunk-level reranker reusing the existing cross-encoder model."""

    def __init__(self):
        self._reranker = None

    @property
    def reranker(self):
        if self._reranker is None:
            try:
                self._reranker = Reranker.get_instance()
            except Exception:
                self._reranker = None
        return self._reranker

    def rerank(
        self,
        query: str,
        results: list[tuple[KnowledgeChunk, float]],
        *,
        top_k: int = 8,
    ) -> list[tuple[KnowledgeChunk, float]]:
        """Rerank chunk results with the shared cross-encoder."""
        if not results or self.reranker is None:
            return results[:top_k]
        pairs = [(query, chunk.text[:1000]) for chunk, _ in results]
        try:
            scores = self.reranker.model.predict(pairs, show_progress_bar=False)
        except Exception:
            return results[:top_k]
        reranked = [(chunk, float(score)) for (chunk, _), score in zip(results, scores)]
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]


class KnowledgeBaseService:
    """Manage knowledge documents, chunk index state, and citation answers."""

    def __init__(self, db):
        self.db = db
        self._embedder = None
        self.reranker = KnowledgeReranker()
        self._stores: dict[str, KnowledgeStore] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = EmbeddingService.get_instance()
        return self._embedder

    def _persistence(self, storage_dir: str | Path) -> KnowledgePersistenceManager:
        kb_root = Path(storage_dir) / "kb"
        kb_root.mkdir(parents=True, exist_ok=True)
        return KnowledgePersistenceManager(kb_root)

    def get_store(self, storage_dir: str | Path) -> KnowledgeStore:
        """Return or create a cached store instance for a storage directory."""
        key = str(Path(storage_dir))
        if key not in self._stores:
            persistence = self._persistence(storage_dir)
            store = persistence.load_store(dimension=self.embedder.dimension)
            if store is None:
                store = KnowledgeStore(dimension=self.embedder.dimension)
            self._stores[key] = store
        return self._stores[key]

    def _persist_store(self, storage_dir: str | Path) -> None:
        self._persistence(storage_dir).save_store(self.get_store(storage_dir))

    def _insert_documents(self, documents: list[KnowledgeDocument], chunks: list[KnowledgeChunk]) -> None:
        with self.db.connection() as conn:
            cursor = conn.cursor()
            for document in documents:
                cursor.execute("DELETE FROM knowledge_documents WHERE id = ?", (document.id,))
                cursor.execute(
                    """
                    INSERT INTO knowledge_documents (
                        id, doc_type, title, source, published_at, language, summary,
                        url, storage_path, mime_type, tags_json, entities_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document.id,
                        document.doc_type.value,
                        document.title,
                        document.source,
                        document.published_at,
                        document.language,
                        document.summary,
                        document.url,
                        document.storage_path,
                        document.mime_type,
                        json.dumps(document.tags, ensure_ascii=False),
                        json.dumps([entity.model_dump() for entity in document.entities], ensure_ascii=False),
                        document.created_at,
                    ),
                )
            for chunk in chunks:
                cursor.execute("DELETE FROM knowledge_chunks WHERE id = ?", (chunk.id,))
                cursor.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        id, document_id, chunk_no, text, page_no, section_title, anchor_label, block_type
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_no,
                        chunk.text,
                        chunk.page_no,
                        chunk.section_title,
                        chunk.anchor_label,
                        chunk.block_type,
                    ),
                )

    def _delete_documents(self, document_ids: set[str]) -> None:
        if not document_ids:
            return
        placeholders = ",".join("?" for _ in document_ids)
        params = tuple(document_ids)
        with self.db.connection() as conn:
            conn.execute(f"DELETE FROM knowledge_chunks WHERE document_id IN ({placeholders})", params)
            conn.execute(f"DELETE FROM knowledge_documents WHERE id IN ({placeholders})", params)

    def ingest_news_stories(self, storage_dir: str | Path, stories: list[UniqueStory]) -> dict:
        """Ingest newly created stories into the public knowledge base."""
        if not stories:
            return {"documents_added": 0, "chunks_added": 0}

        documents: list[KnowledgeDocument] = []
        chunks: list[KnowledgeChunk] = []
        document_ids: set[str] = set()
        for story in stories:
            document, story_chunks = story_to_knowledge(story)
            documents.append(document)
            chunks.extend(story_chunks)
            document_ids.add(document.id)

        if chunks:
            embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()

        store = self.get_store(storage_dir)
        store.remove_document_ids(document_ids)
        self._delete_documents(document_ids)
        self._insert_documents(documents, chunks)
        store.add_chunks(chunks)
        self._persist_store(storage_dir)
        return {"documents_added": len(documents), "chunks_added": len(chunks)}

    def rebuild_news_from_stories(self, storage_dir: str | Path, stories: list[UniqueStory]) -> dict:
        """Replace all indexed news_story documents using the current public stories."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM knowledge_documents WHERE doc_type = ?",
                (KnowledgeDocType.NEWS_STORY.value,),
            )
            existing_ids = {row["id"] for row in cursor.fetchall()}

        store = self.get_store(storage_dir)
        store.remove_document_ids(existing_ids)
        self._delete_documents(existing_ids)

        result = self.ingest_news_stories(storage_dir, stories)
        result["replaced_documents"] = len(existing_ids)
        return result

    def upload_attachment(
        self,
        storage_dir: str | Path,
        *,
        file_name: str,
        content_type: Optional[str],
        raw_bytes: bytes,
    ) -> KnowledgeDocument:
        """Parse an uploaded attachment and store it as a knowledge document."""
        document_id = document_id_for_attachment(file_name, raw_bytes)
        suffix = Path(file_name).suffix.lower() or ".bin"
        file_root = Path(storage_dir) / "kb_files" / document_id
        file_root.mkdir(parents=True, exist_ok=True)
        stored_file = file_root / f"original{suffix}"
        stored_file.write_bytes(raw_bytes)

        attachment_context = AttachmentParser().parse_file(
            stored_file,
            file_name=file_name,
            content_type=content_type,
        )
        created_at = utc_timestamp()
        document, chunks = attachment_to_knowledge(
            attachment_context,
            document_id=document_id,
            storage_path=str(stored_file),
            mime_type=content_type,
            created_at=created_at,
        )
        if chunks:
            embeddings = self.embedder.embed_batch([chunk.text for chunk in chunks])
            for chunk, embedding in zip(chunks, embeddings):
                chunk.embedding = embedding.tolist()

        store = self.get_store(storage_dir)
        store.remove_document_ids({document.id})
        self._delete_documents({document.id})
        self._insert_documents([document], chunks)
        store.add_chunks(chunks)
        self._persist_store(storage_dir)
        return document

    @staticmethod
    def _row_to_document(row) -> KnowledgeDocument:
        raw_entities = json.loads(row["entities_json"]) if row["entities_json"] else []
        entities = [
            Entity.model_validate(entity)
            for entity in raw_entities
        ]
        return KnowledgeDocument(
            id=row["id"],
            doc_type=KnowledgeDocType(row["doc_type"]),
            title=row["title"],
            source=row["source"],
            published_at=row["published_at"],
            language=row["language"] or "zh",
            summary=row["summary"] or "",
            url=row["url"],
            storage_path=row["storage_path"],
            mime_type=row["mime_type"],
            tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
            entities=entities,
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_chunk(row) -> KnowledgeChunk:
        return KnowledgeChunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_no=row["chunk_no"],
            text=row["text"],
            page_no=row["page_no"],
            section_title=row["section_title"],
            anchor_label=row["anchor_label"],
            block_type=row["block_type"] or "paragraph",
        )

    def _fetch_documents_by_ids(self, document_ids: set[str]) -> dict[str, KnowledgeDocument]:
        if not document_ids:
            return {}
        placeholders = ",".join("?" for _ in document_ids)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT * FROM knowledge_documents WHERE id IN ({placeholders})",
                tuple(document_ids),
            )
            rows = cursor.fetchall()
        return {row["id"]: self._row_to_document(row) for row in rows}

    def query(
        self,
        storage_dir: str | Path,
        *,
        query: str,
        doc_types: Optional[list[str]] = None,
        sources: Optional[list[str]] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        llm=None,
    ) -> KnowledgeQueryResult:
        """Run a chunk-based knowledge query with citations."""
        process_started = datetime.now(timezone.utc)
        timings: dict[str, float] = {}
        has_filters = bool(doc_types or sources or date_from or date_to)
        with self.db.connection() as conn:
            cursor = conn.cursor()
            clauses = []
            params: list[object] = []
            if doc_types:
                placeholders = ",".join("?" for _ in doc_types)
                clauses.append(f"doc_type IN ({placeholders})")
                params.extend(doc_types)
            if sources:
                placeholders = ",".join("?" for _ in sources)
                clauses.append(f"source IN ({placeholders})")
                params.extend(sources)
            if date_from:
                clauses.append("(published_at IS NOT NULL AND published_at >= ?)")
                params.append(date_from)
            if date_to:
                clauses.append("(published_at IS NOT NULL AND published_at <= ?)")
                params.append(date_to)

            sql = "SELECT * FROM knowledge_documents"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            cursor.execute(sql, tuple(params))
            document_rows = cursor.fetchall()

        allowed_documents = {row["id"]: self._row_to_document(row) for row in document_rows}
        allowed_doc_ids = set(allowed_documents.keys()) if has_filters else None

        if has_filters and not allowed_documents:
            total_ms = round((datetime.now(timezone.utc) - process_started).total_seconds() * 1000, 1)
            return KnowledgeQueryResult(
                query=query,
                answer="No knowledge base documents match the current filters.",
                citations=[],
                related_documents=[],
                timing=QueryTiming(pipeline_ms=total_ms, stages=timings),
            )

        store = self.get_store(storage_dir)
        search_started = datetime.now(timezone.utc)
        query_embedding = self.embedder.embed(query)
        scored_chunks = store.search(query_embedding, query, allowed_doc_ids=allowed_doc_ids, k=20, alpha=0.7)
        timings["search_ms"] = round((datetime.now(timezone.utc) - search_started).total_seconds() * 1000, 1)

        if not allowed_documents:
            allowed_documents = self._fetch_documents_by_ids({chunk.document_id for chunk, _ in scored_chunks})

        if any(keyword in query.lower() for keyword in _RECENCY_KEYWORDS):
            adjusted: list[tuple[KnowledgeChunk, float]] = []
            now = datetime.now(timezone.utc)
            for chunk, score in scored_chunks:
                document = allowed_documents.get(chunk.document_id)
                adjusted_score = score
                if document and document.doc_type == KnowledgeDocType.NEWS_STORY and document.published_at:
                    published = parse_timestamp(document.published_at)
                    if published is not None:
                        age_days = max(0.0, (now - published).total_seconds() / 86400.0)
                        adjusted_score += max(0.0, 1.5 - min(age_days, 7.0) * 0.2)
                adjusted.append((chunk, adjusted_score))
            adjusted.sort(key=lambda item: item[1], reverse=True)
            scored_chunks = adjusted

        rerank_started = datetime.now(timezone.utc)
        reranked_chunks = self.reranker.rerank(query, scored_chunks, top_k=8)
        timings["rerank_ms"] = round((datetime.now(timezone.utc) - rerank_started).total_seconds() * 1000, 1)

        answer_started = datetime.now(timezone.utc)
        top_chunks = reranked_chunks[:4]
        chunk_summaries = []
        for index, (chunk, score) in enumerate(reranked_chunks[:8]):
            document = allowed_documents.get(chunk.document_id)
            if document is None:
                continue
            chunk_summaries.append(
                {
                    "index": index,
                    "chunk_id": chunk.id,
                    "title": document.title,
                    "doc_type": document.doc_type.value,
                    "source": document.source,
                    "published_at": document.published_at,
                    "anchor_label": chunk.anchor_label,
                    "page_no": chunk.page_no,
                    "content": chunk.text[:900],
                    "score": round(score, 4),
                }
            )

        relevant_indices = list(range(len(top_chunks)))
        answer = ""
        if llm and chunk_summaries:
            try:
                response = llm.explain_knowledge_results(query, chunk_summaries)
                answer = str(response.get("explanation", "")).strip()
                relevant_indices = [
                    idx for idx in response.get("relevant_chunk_indices", [])
                    if 0 <= idx < len(chunk_summaries)
                ] or relevant_indices
            except Exception:
                answer = ""

        if not answer and chunk_summaries:
            answer = chunk_summaries[0]["content"][:220]
        if not answer:
            answer = "当前知识库中没有足够证据回答该问题。"
        timings["answer_ms"] = round((datetime.now(timezone.utc) - answer_started).total_seconds() * 1000, 1)

        selected_chunks = [
            reranked_chunks[index]
            for index in relevant_indices[:4]
            if 0 <= index < len(reranked_chunks)
        ] or top_chunks

        citations: list[KnowledgeCitation] = []
        seen_documents: list[str] = []
        related_documents: list[KnowledgeDocument] = []
        for chunk, _ in selected_chunks[:4]:
            document = allowed_documents.get(chunk.document_id)
            if document is None:
                continue
            citations.append(
                KnowledgeCitation(
                    document_id=document.id,
                    chunk_id=chunk.id,
                    title=document.title,
                    doc_type=document.doc_type,
                    source=document.source,
                    published_at=document.published_at,
                    snippet=chunk.text[:320],
                    anchor_label=chunk.anchor_label,
                    page_no=chunk.page_no,
                    storage_path=document.storage_path,
                )
            )
            if document.id not in seen_documents:
                seen_documents.append(document.id)
                related_documents.append(document)

        if not citations:
            answer = "当前知识库中没有足够证据回答该问题。"

        total_ms = round((datetime.now(timezone.utc) - process_started).total_seconds() * 1000, 1)
        return KnowledgeQueryResult(
            query=query,
            answer=answer,
            citations=citations,
            related_documents=related_documents[:4],
            timing=QueryTiming(pipeline_ms=total_ms, stages=timings),
        )

    def list_documents(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        doc_type: Optional[str] = None,
        source: Optional[str] = None,
    ) -> tuple[list[KnowledgeDocument], int]:
        """Return paginated knowledge documents."""
        clauses = []
        params: list[object] = []
        if doc_type:
            clauses.append("doc_type = ?")
            params.append(doc_type)
        if source:
            clauses.append("source = ?")
            params.append(source)
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM knowledge_documents {where_sql}", tuple(params))
            total = int(cursor.fetchone()[0])
            cursor.execute(
                f"""
                SELECT * FROM knowledge_documents
                {where_sql}
                ORDER BY COALESCE(published_at, created_at) DESC, created_at DESC
                LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            )
            rows = cursor.fetchall()
        return [self._row_to_document(row) for row in rows], total

    def get_document(self, document_id: str) -> tuple[Optional[KnowledgeDocument], list[KnowledgeChunk]]:
        """Fetch a document and its stored chunks."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM knowledge_documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
            if row is None:
                return None, []
            cursor.execute(
                """
                SELECT * FROM knowledge_chunks
                WHERE document_id = ?
                ORDER BY chunk_no ASC
                """,
                (document_id,),
            )
            chunk_rows = cursor.fetchall()
        return self._row_to_document(row), [self._row_to_chunk(chunk_row) for chunk_row in chunk_rows]

    def get_document_file_path(self, document_id: str) -> Optional[Path]:
        """Return the stored file path for an attachment document."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT storage_path FROM knowledge_documents WHERE id = ?", (document_id,))
            row = cursor.fetchone()
        if row is None or not row["storage_path"]:
            return None
        return Path(row["storage_path"])

    def get_stats(self) -> dict:
        """Return high-level knowledge base statistics."""
        with self.db.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM knowledge_documents")
            document_count = int(cursor.fetchone()[0])
            cursor.execute("SELECT COUNT(*) FROM knowledge_chunks")
            chunk_count = int(cursor.fetchone()[0])
            cursor.execute(
                "SELECT doc_type, COUNT(*) AS count FROM knowledge_documents GROUP BY doc_type"
            )
            type_counts = {row["doc_type"]: row["count"] for row in cursor.fetchall()}
            cursor.execute(
                "SELECT COALESCE(source, 'unknown') AS source, COUNT(*) AS count FROM knowledge_documents GROUP BY COALESCE(source, 'unknown')"
            )
            source_counts = {row["source"]: row["count"] for row in cursor.fetchall()}
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "doc_type_counts": type_counts,
            "source_counts": source_counts,
        }
