"""Core IntelligenceSystem - main interface for the package."""

import json
import re
import sys
from typing import Optional
from time import perf_counter
from pathlib import Path
from dotenv import load_dotenv

# Load .env before importing modules that may read HF-related env vars at import time.
load_dotenv()

from intanalysis.models import (
    Article,
    ConversationTurn,
    QueryIntent,
    QueryResult,
    QueryTiming,
    ShortTermContext,
    UniqueStory,
)
from intanalysis.embeddings import VectorStore, EmbeddingService
from intanalysis.workflow import build_ingestion_graph, build_query_graph, PipelineState
from intanalysis.persistence import PersistenceManager
from intanalysis.intent import IntentClassifier
from intanalysis.llm import LLMService
from intanalysis.mappings import REGULATORS, find_stock_symbols
from text_cleaning import clean_text, combine_article_text


def _configure_console_encoding() -> None:
    """Prefer UTF-8 console output to avoid Windows GBK encoding errors."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_console_encoding()


class IntelligenceSystem:
    """
    Main interface for the Financial News Intelligence System.
    
    Example:
        system = IntelligenceSystem()
        
        # Ingest articles
        result = system.ingest([
            {"title": "HDFC Bank announces dividend", "content": "..."},
            {"title": "RBI raises rates", "content": "..."},
        ])
        
        # Query
        response = system.query("HDFC Bank news")
        for story in response.stories:
            print(story.primary_article.article.title)
    """

    CONTEXT_FOLLOWUP_PATTERNS = (
        re.compile(r"^(那|那它|那他|那她|那这|那这个|那家公司|那只|那家)"),
        re.compile(r"^(再|继续|接着|顺便|还有|另外|然后)"),
        re.compile(r"(它|他|她|这家|那家|这个|那个|该公司|该股|这只|那只)"),
        re.compile(r"(比较|对比|相比|谁更|哪个好|哪家更)"),
    )
    
    def __init__(
        self,
        verbose: bool = True,
        storage_dir: str = "dataset",
        legacy_storage_dir: Optional[str] = None,
    ):
        """Initialize the intelligence system.
        
        Args:
            verbose: Print processing information
            storage_dir: Directory for persistence files
            legacy_storage_dir: Optional fallback directory used for read compatibility
        """
        load_dotenv()
        
        self.verbose = verbose
        self.persistence = PersistenceManager(
            storage_dir=storage_dir,
            legacy_storage_dir=legacy_storage_dir,
        )
        self._vector_store: Optional[VectorStore] = None
        self._ingestion_graph = None
        self._query_graph = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._llm: Optional[LLMService] = None
    
    @property
    def vector_store(self) -> VectorStore:
        """Get or create vector store, loading from disk if available."""
        if self._vector_store is None:
            embedder = EmbeddingService.get_instance()
            # Try loading from disk first
            self._vector_store = self.persistence.load_vector_store(dimension=embedder.dimension)
            if self._vector_store is None:
                self._vector_store = VectorStore(dimension=embedder.dimension)
            elif self.verbose:
                print(f"📂 Loaded {len(self._vector_store.stories)} existing stories from disk")
        return self._vector_store
    
    @property
    def ingestion_graph(self):
        """Get compiled ingestion graph."""
        if self._ingestion_graph is None:
            self._ingestion_graph = build_ingestion_graph()
        return self._ingestion_graph
    
    @property
    def query_graph(self):
        """Get compiled query graph."""
        if self._query_graph is None:
            self._query_graph = build_query_graph()
        return self._query_graph

    @property
    def intent_classifier(self) -> IntentClassifier:
        """Get or create the intent classifier."""
        if self._intent_classifier is None:
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    @property
    def llm(self) -> Optional[LLMService]:
        """Get optional LLM service."""
        if self._llm is None:
            try:
                self._llm = LLMService.get_instance()
            except Exception:
                self._llm = None
        return self._llm
    
    def ingest(self, articles: list[dict | Article], force: bool = False) -> dict:
        """
        Ingest articles through the full pipeline (incremental).
        
        Args:
            articles: List of articles (dicts or Article objects)
            force: Force re-ingestion of all articles (ignore cache)
            
        Returns:
            Processing result with unique stories and stats
        """
        # Filter out already processed articles
        article_dicts = [
            art if isinstance(art, dict) else art.__dict__
            for art in articles
        ]
        
        if not force:
            new_articles, skipped = self.persistence.filter_new_articles(article_dicts)
            if self.verbose and skipped > 0:
                print(f"\n📌 Skipping {skipped} already processed articles")
        else:
            new_articles = article_dicts
            skipped = 0
        
        if not new_articles:
            if self.verbose:
                print(f"\n✅ No new articles to process")
            return {
                "unique_stories": [],
                "total_articles": len(articles),
                "unique_count": 0,
                "duplicate_count": 0,
                "skipped_count": skipped,
            }
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📥 INGESTING {len(new_articles)} NEW ARTICLES")
            print(f"{'='*60}")
        
        initial_state: PipelineState = {
            "raw_articles": new_articles,
            "vector_store": self.vector_store,
            "errors": [],
        }
        
        result = self.ingestion_graph.invoke(initial_state)
        
        # Update our vector store reference
        self._vector_store = result.get("vector_store", self._vector_store)
        
        unique_stories = result.get("unique_stories", [])
        
        # Mark articles as seen and persist vector store
        self.persistence.mark_articles_as_seen(new_articles)
        self.persistence.save_vector_store(self.vector_store)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ INGESTION COMPLETE")
            print(f"   • New unique stories: {len(unique_stories)}")
            print(f"   • Duplicates in batch: {len(new_articles) - len(unique_stories)}")
            print(f"   • Total indexed: {self.vector_store.index.ntotal}")
            print(f"   • Persisted to disk ✓")
            print(f"{'='*60}\n")
        
        return {
            "unique_stories": unique_stories,
            "total_articles": len(articles),
            "unique_count": len(unique_stories),
            "duplicate_count": len(new_articles) - len(unique_stories),
            "skipped_count": skipped,
        }
    
    def query(self, query_text: str, show_steps: bool = True) -> QueryResult:
        """
        Query the system for relevant news.
        
        Args:
            query_text: Natural language query
            show_steps: Show intermediate processing steps
            
        Returns:
            QueryResult with matched stories and explanation
        """
        if self.verbose and show_steps:
            print(f"\n{'─'*50}")
            print(f"🔍 PROCESSING QUERY: \"{query_text}\"")
            print(f"{'─'*50}")
        
        initial_state: PipelineState = {
            "query": query_text,
            "vector_store": self.vector_store,
            "errors": [],
        }
        
        query_started = perf_counter()
        result = self.query_graph.invoke(initial_state)
        query_result = result.get("query_result", QueryResult(query=query_text))
        if query_result.timing is None:
            query_result.timing = QueryTiming()
        query_result.timing.pipeline_ms = round((perf_counter() - query_started) * 1000, 1)
        
        if self.verbose:
            print(f"\n📊 RESULTS: {len(query_result.stories)} stories found")
            if query_result.explanation:
                print(f"💡 {query_result.explanation}")
            print(f"{'='*60}\n")
        
        return query_result

    def handle_user_query(
        self,
        query_text: str,
        history: list[dict | ConversationTurn] | None = None,
        show_steps: bool = True,
    ) -> QueryResult:
        """Classify intent and dispatch to the appropriate handling path."""
        total_started = perf_counter()
        context_started = perf_counter()
        short_context = self._build_short_term_context(history or [])
        resolved_query = self._resolve_query_with_context(query_text, short_context)
        context_ms = round((perf_counter() - context_started) * 1000, 1)

        classify_started = perf_counter()
        decision = self.intent_classifier.classify(resolved_query)
        classify_ms = round((perf_counter() - classify_started) * 1000, 1)

        if self.verbose and show_steps:
            if short_context.applied and short_context.recent_entities:
                print(
                    f"\n🧠 Context: recent entities={short_context.recent_entities[:3]} "
                    f"resolved_query=\"{resolved_query}\""
                )
            print(f"\n🧭 Intent: {decision.intent.value} ({decision.source}, {decision.confidence:.0%})")
            if decision.reason:
                print(f"   Reason: {decision.reason}")

        if decision.intent == QueryIntent.NEWS_UPDATE:
            result = self._handle_news_refresh(query_text)
        elif decision.intent == QueryIntent.GENERAL_CHAT:
            result = self._handle_general_chat(query_text)
        else:
            result = self.query(resolved_query, show_steps=show_steps)
            result.query = query_text

        if result.timing is None:
            result.timing = QueryTiming()
        result.intent = decision.intent
        result.intent_source = decision.source
        result.intent_reason = decision.reason
        result.timing.stages = {
            "context_build_ms": context_ms,
            "intent_classify_ms": classify_ms,
            **result.timing.stages,
        }
        result.timing.pipeline_ms = round((perf_counter() - total_started) * 1000, 1)
        return result

    def _build_short_term_context(self, history: list[dict | ConversationTurn]) -> ShortTermContext:
        """Build structured short-term context from recent conversation turns."""
        if not history:
            return ShortTermContext()

        normalized: list[ConversationTurn] = []
        for item in history[-6:]:
            try:
                turn = item if isinstance(item, ConversationTurn) else ConversationTurn.model_validate(item)
                normalized.append(turn)
            except Exception:
                continue

        recent_entities: list[str] = []
        recent_story_titles: list[str] = []
        last_intent: QueryIntent | None = None
        last_user_query: str | None = None

        def append_unique(target: list[str], values: list[str], limit: int = 4) -> None:
            for value in values:
                cleaned = clean_text(value or "")
                if cleaned and cleaned not in target:
                    target.append(cleaned)
                if len(target) >= limit:
                    break

        for turn in reversed(normalized):
            if turn.role == "assistant":
                if last_intent is None and turn.intent is not None:
                    last_intent = turn.intent
                append_unique(recent_entities, turn.matched_entities, limit=4)
                append_unique(recent_story_titles, turn.story_titles, limit=3)
            elif turn.role == "user":
                if last_user_query is None and turn.content.strip():
                    last_user_query = turn.content.strip()
                append_unique(recent_entities, self._extract_context_entities(turn.content), limit=4)

        return ShortTermContext(
            recent_entities=recent_entities,
            recent_story_titles=recent_story_titles,
            last_intent=last_intent,
            last_user_query=last_user_query,
            turn_count=len(normalized),
        )

    def _extract_context_entities(self, text: str) -> list[str]:
        """Extract lightweight context entities from recent user text."""
        if not text:
            return []

        lowered = text.lower()
        entities = [name for _, name, _ in find_stock_symbols(lowered)]

        for key, info in REGULATORS.items():
            aliases = info.get("aliases", [])
            if key in lowered or info["full_name"].lower() in lowered or any(alias.lower() in lowered for alias in aliases):
                entities.append(info["full_name"])

        # Keep order while deduplicating.
        seen = set()
        ordered = []
        for name in entities:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def _should_apply_context(self, query_text: str, context: ShortTermContext) -> bool:
        """Decide whether the current query should inherit short-term context."""
        if not context.recent_entities:
            return False

        text = (query_text or "").strip()
        if not text:
            return False

        lowered = text.lower()
        explicit_entities = self._extract_context_entities(text)

        if any(pattern.search(text) for pattern in self.CONTEXT_FOLLOWUP_PATTERNS):
            return True

        if len(text) <= 10 and not explicit_entities:
            return True

        if text.endswith(("呢", "吗", "怎么样", "如何")) and not explicit_entities:
            return True

        if any(keyword in lowered for keyword in ("比较", "对比", "相比", "谁更", "哪个好")) and context.recent_entities:
            return True

        return False

    def _resolve_query_with_context(self, query_text: str, context: ShortTermContext) -> str:
        """Augment ambiguous follow-up queries with recent context."""
        if not self._should_apply_context(query_text, context):
            context.applied = False
            context.resolved_query = query_text
            return query_text

        parts = [query_text]
        if context.last_user_query:
            parts.append(f"上轮问题：{context.last_user_query}")
        if context.recent_entities:
            parts.append(f"相关实体：{' '.join(context.recent_entities[:3])}")
        if context.recent_story_titles:
            parts.append(f"相关主题：{'；'.join(context.recent_story_titles[:2])}")

        resolved_query = " ".join(parts)
        context.applied = True
        context.resolved_query = resolved_query
        return resolved_query

    def _handle_general_chat(self, query_text: str) -> QueryResult:
        """Handle non-financial general chat directly with the LLM."""
        started = perf_counter()
        stages: dict[str, float] = {}

        llm_started = perf_counter()
        answer = None
        if self.llm is not None:
            try:
                answer = self.llm.answer_general_query(query_text)
            except Exception:
                answer = None
        stages["general_llm_ms"] = round((perf_counter() - llm_started) * 1000, 1)

        if not answer:
            answer = "当前没有可用的大模型服务来处理通识问答，请稍后再试，或补充更明确的金融新闻问题。"

        return QueryResult(
            query=query_text,
            intent=QueryIntent.GENERAL_CHAT,
            intent_source="direct",
            intent_reason="Handled as a general conversation query.",
            explanation=answer,
            timing=QueryTiming(
                pipeline_ms=round((perf_counter() - started) * 1000, 1),
                stages=stages,
            ),
        )

    def _handle_news_refresh(self, query_text: str) -> QueryResult:
        """Refresh RSS feeds and sync any dataset articles that are not yet indexed."""
        from dataset.feeds import NewsRSSMonitor

        started = perf_counter()
        stages: dict[str, float] = {}

        refresh_started = perf_counter()
        monitor = NewsRSSMonitor(check_interval=300)
        fetched_articles = monitor.check_all_feeds()
        monitor.save_articals_loaded()
        stages["refresh_feeds_ms"] = round((perf_counter() - refresh_started) * 1000, 1)

        persist_started = perf_counter()
        if fetched_articles:
            self._append_fetched_articles(fetched_articles)
        stages["refresh_persist_ms"] = round((perf_counter() - persist_started) * 1000, 1)

        scan_started = perf_counter()
        dataset_articles = self._load_rss_dataset_articles()
        pending_articles, _ = self.persistence.filter_new_articles(dataset_articles)
        stages["refresh_dataset_scan_ms"] = round((perf_counter() - scan_started) * 1000, 1)

        ingest_started = perf_counter()
        ingest_result = {
            "unique_stories": [],
            "total_articles": 0,
            "unique_count": 0,
            "duplicate_count": 0,
            "skipped_count": 0,
        }
        if pending_articles:
            ingest_result = self.ingest(pending_articles, force=True)
        stages["refresh_ingest_ms"] = round((perf_counter() - ingest_started) * 1000, 1)

        pending_count = len(pending_articles)
        total_dataset = len(dataset_articles)
        fetched_count = len(fetched_articles)

        if fetched_count or pending_count:
            explanation = (
                f"已检查 {len(monitor.feed_configs)} 个 RSS 源，拉取到 {fetched_count} 篇新文章。"
                f" 当前 RSS 数据集共有 {total_dataset} 篇文章，其中 {pending_count} 篇此前尚未入库，"
                f"本次新增 {ingest_result['unique_count']} 条主 story，识别出 {ingest_result['duplicate_count']} 篇重复内容。"
            )
        else:
            explanation = (
                f"已检查 {len(monitor.feed_configs)} 个 RSS 源，当前没有发现新的文章。"
                f" 本地 RSS 数据集共 {total_dataset} 篇，已全部完成入库同步。"
            )

        return QueryResult(
            query=query_text,
            intent=QueryIntent.NEWS_UPDATE,
            intent_source="direct",
            intent_reason="Handled as a news refresh request.",
            stories=ingest_result.get("unique_stories", []),
            explanation=explanation,
            timing=QueryTiming(
                pipeline_ms=round((perf_counter() - started) * 1000, 1),
                stages=stages,
            ),
        )

    def _append_fetched_articles(self, new_articles: list[dict]) -> None:
        """Append fetched raw feed entries to the RSS dataset file."""
        dataset_file = Path(self.persistence.storage_dir) / "rss_feeds_all.json"
        existing: list[dict] = []
        if dataset_file.exists():
            with dataset_file.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.extend(new_articles)
        with dataset_file.open("w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def _load_rss_dataset_articles(self) -> list[dict]:
        """Load the persisted RSS dataset and convert it into ingestion-ready articles."""
        dataset_file = Path(self.persistence.storage_dir) / "rss_feeds_all.json"
        if not dataset_file.exists():
            return []

        with dataset_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return self._convert_feed_entries_to_articles(data)

    def _convert_feed_entries_to_articles(self, items: list[dict]) -> list[dict]:
        """Convert raw feed entries to ingestion-ready article dicts."""
        articles = []
        for item in items:
            articles.append({
                "id": item.get("id", "")[:50],
                "title": clean_text(item.get("title", "Untitled")),
                "content": item.get("content_text") or combine_article_text(item.get("summary", ""), item.get("content")),
                "source": clean_text(item.get("source", "Unknown")),
                "url": item.get("link", ""),
                "published_date": item.get("published"),
            })
        return articles
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "indexed_stories": self.vector_store.index.ntotal,
            "total_stories": len(self.vector_store.stories),
        }
