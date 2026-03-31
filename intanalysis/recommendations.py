"""Helpers for recommendation workflow and card generation."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from typing import Iterable, Optional

from intanalysis.mappings import (
    COMPANY_TO_STOCK,
    REGULATORS,
    SECTOR_TO_COMPANIES,
    find_stock_symbols,
    get_companies_in_sector,
    get_stock_symbol,
)
from intanalysis.models import EntityType, UniqueStory


SUMMARY_MAX_CHARS = 180
SUMMARY_MIN_CHARS = 120
SECTOR_KEYWORDS: dict[str, str] = {
    "banking": "Banking",
    "bank": "Banking",
    "银行": "Chinese Banking",
    "aviation": "Aviation",
    "航空": "Aviation",
    "it": "IT",
    "tech": "IT",
    "科技": "Internet",
    "互联网": "Internet",
    "电商": "E-Commerce",
    "零售": "Consumer",
    "消费": "Consumer",
    "auto": "Automobile",
    "automobile": "Automobile",
    "汽车": "EV",
    "新能源车": "EV",
    "电动车": "EV",
    "家电": "Home Appliances",
}


def normalize_text(value: Optional[str]) -> str:
    """Normalize text for loose matching."""
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse a publication timestamp into timezone-aware UTC."""
    if not value:
        return None

    text = value.strip()
    if not text:
        return None

    candidates = [
        text,
        text.replace("Z", "+00:00"),
    ]
    for candidate in candidates:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(text, fmt)
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    try:
        parsed = parsedate_to_datetime(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def dedupe_entities(entities: Iterable[dict], limit: int = 10) -> list[dict]:
    """Keep the first occurrence of each entity name/type pair."""
    seen: set[tuple[str, str]] = set()
    ordered: list[dict] = []
    for entity in entities:
        name = (entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "").strip()
        if not name or not entity_type:
            continue
        key = (name.lower(), entity_type.lower())
        if key in seen:
            continue
        seen.add(key)
        ordered.append({"name": name, "type": entity_type})
        if len(ordered) >= limit:
            break
    return ordered


def extract_query_entities(query: str) -> list[dict]:
    """Extract lightweight company, regulator, and sector entities from a query."""
    lowered = normalize_text(query)
    if not lowered:
        return []

    entities: list[dict] = []
    for _, company_name, _ in find_stock_symbols(lowered):
        entities.append({"name": company_name, "type": EntityType.COMPANY.value})

    for key, info in REGULATORS.items():
        aliases = info.get("aliases", [])
        full_name = info["full_name"]
        if key in lowered or full_name.lower() in lowered or any(alias.lower() in lowered for alias in aliases):
            entities.append({"name": full_name, "type": EntityType.REGULATOR.value})

    for keyword, sector_name in SECTOR_KEYWORDS.items():
        if keyword in lowered:
            entities.append({"name": sector_name, "type": EntityType.SECTOR.value})

    return dedupe_entities(entities)


def extract_interest_entities(chat_records: list[dict], limit: int = 10) -> list[dict]:
    """Build a recent-interest entity list from chat history."""
    collected: list[dict] = []
    for chat in chat_records:
        matched_entities = chat.get("matched_entities") or []
        if matched_entities:
            for entity in matched_entities:
                if isinstance(entity, dict):
                    collected.append(
                        {
                            "name": entity.get("name"),
                            "type": entity.get("type"),
                        }
                    )
                elif isinstance(entity, str):
                    collected.extend(extract_query_entities(entity))
        else:
            collected.extend(extract_query_entities(chat.get("query", "")))
        if len(collected) >= limit * 2:
            break

    return dedupe_entities(collected, limit=limit)


def get_story_timestamp(story: UniqueStory) -> Optional[datetime]:
    """Return the story timestamp if available."""
    return parse_timestamp(story.primary_article.article.published_date)


def sort_latest_stories(stories: list[UniqueStory], limit: int = 10) -> list[UniqueStory]:
    """Return the latest stories, preferring valid publication dates and newer insertions."""
    ranked = sorted(
        enumerate(stories),
        key=lambda item: (
            get_story_timestamp(item[1]) is not None,
            get_story_timestamp(item[1]) or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        ),
        reverse=True,
    )
    return [story for _, story in ranked[:limit]]


def _company_terms(company_name: str) -> set[str]:
    lowered = normalize_text(company_name)
    terms = {lowered}
    for key, (_, full_name, aliases) in COMPANY_TO_STOCK.items():
        if lowered in {normalize_text(key), normalize_text(full_name)}:
            terms.add(normalize_text(key))
            terms.add(normalize_text(full_name))
            terms.update(normalize_text(alias) for alias in aliases if alias)
            break
    return {term for term in terms if term}


def _regulator_terms(regulator_name: str) -> set[str]:
    lowered = normalize_text(regulator_name)
    terms = {lowered}
    for key, info in REGULATORS.items():
        full_name = normalize_text(info["full_name"])
        aliases = {normalize_text(alias) for alias in info.get("aliases", []) if alias}
        if lowered == normalize_text(key) or lowered == full_name or lowered in aliases:
            terms.add(normalize_text(key))
            terms.add(full_name)
            terms.update(aliases)
            break
    return {term for term in terms if term}


def _entity_weight(entity_type: str) -> float:
    if entity_type == EntityType.COMPANY.value:
        return 3.0
    if entity_type == EntityType.REGULATOR.value:
        return 2.2
    return 1.6


def score_story_for_entities(story: UniqueStory, interest_entities: list[dict]) -> tuple[float, list[dict]]:
    """Score a story against interest entities and return matched entity metadata."""
    title_text = normalize_text(story.primary_article.article.title)
    content_text = normalize_text(story.primary_article.article.content)
    entity_names = {normalize_text(entity.name) for entity in story.primary_article.entities}
    story_sectors = {normalize_text(sector) for sector in story.primary_article.sectors}

    score = 0.0
    matched_entities: list[dict] = []

    for entity in interest_entities:
        entity_name = (entity.get("name") or "").strip()
        entity_type = str(entity.get("type") or "").strip().lower()
        if not entity_name or not entity_type:
            continue

        weight = _entity_weight(entity_type)
        matched = False

        if entity_type == EntityType.COMPANY.value:
            for term in _company_terms(entity_name):
                if term in title_text:
                    score += weight * 3.0
                    matched = True
                    break
            if not matched:
                for term in _company_terms(entity_name):
                    if term in content_text:
                        score += weight * 1.5
                        matched = True
                        break
            if not matched and normalize_text(entity_name) in entity_names:
                score += weight
                matched = True
        elif entity_type == EntityType.REGULATOR.value:
            for term in _regulator_terms(entity_name):
                if term in title_text:
                    score += weight * 2.5
                    matched = True
                    break
            if not matched:
                for term in _regulator_terms(entity_name):
                    if term in content_text:
                        score += weight * 1.25
                        matched = True
                        break
            if not matched and normalize_text(entity_name) in entity_names:
                score += weight
                matched = True
        else:
            sector_name = normalize_text(entity_name)
            if sector_name in story_sectors:
                score += weight * 2.0
                matched = True
            elif sector_name in title_text:
                score += weight * 1.5
                matched = True
            elif sector_name in content_text:
                score += weight
                matched = True

        if matched:
            matched_entities.append({"name": entity_name, "type": entity_type})

    if matched_entities and story.primary_article.stock_impacts:
        score += 0.75

    published_at = get_story_timestamp(story)
    if published_at is not None:
        age_days = max(0.0, (datetime.now(timezone.utc) - published_at).total_seconds() / 86400.0)
        score += max(0.0, 2.5 - min(age_days, 10.0) * 0.2)

    return score, matched_entities


def collect_personalized_candidates(
    stories: list[UniqueStory],
    interest_entities: list[dict],
    limit: int = 10,
) -> list[tuple[UniqueStory, list[dict]]]:
    """Collect the top personalized stories for the interest entity set."""
    scored: list[tuple[float, datetime, int, UniqueStory, list[dict]]] = []
    for index, story in enumerate(stories):
        score, matched_entities = score_story_for_entities(story, interest_entities)
        if score <= 0 or not matched_entities:
            continue
        scored.append(
            (
                score,
                get_story_timestamp(story) or datetime.min.replace(tzinfo=timezone.utc),
                index,
                story,
                matched_entities,
            )
        )

    scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return [(story, matched_entities) for _, _, _, story, matched_entities in scored[:limit]]


def summarize_story(story: UniqueStory) -> str:
    """Generate a compact summary for a recommendation card."""
    content = re.sub(r"\s+", " ", (story.primary_article.article.content or "")).strip()
    if not content:
        return story.primary_article.article.title
    if len(content) <= SUMMARY_MAX_CHARS:
        return content

    snippet = content[:SUMMARY_MAX_CHARS].rsplit(" ", 1)[0].strip()
    if len(snippet) < SUMMARY_MIN_CHARS:
        snippet = content[:SUMMARY_MAX_CHARS].strip()
    return f"{snippet.rstrip('，。,.!;: ')}..."


def infer_stock_symbols(story: UniqueStory) -> list[str]:
    """Infer up to three stock symbols for a story."""
    symbols: list[str] = []

    for impact in story.primary_article.stock_impacts:
        if impact.symbol and impact.symbol not in symbols:
            symbols.append(impact.symbol)
        if len(symbols) >= 3:
            return symbols

    for entity in story.primary_article.entities:
        if entity.type == EntityType.COMPANY:
            result = get_stock_symbol(entity.name)
            if result and result[0] not in symbols:
                symbols.append(result[0])
        elif entity.type == EntityType.REGULATOR:
            for info in REGULATORS.values():
                if info["full_name"] == entity.name:
                    for sector_name in info.get("sectors", []):
                        for symbol in get_companies_in_sector(sector_name):
                            if symbol not in symbols:
                                symbols.append(symbol)
                            if len(symbols) >= 3:
                                return symbols
                    break
        if len(symbols) >= 3:
            return symbols

    for sector_name in story.primary_article.sectors:
        for symbol in SECTOR_TO_COMPANIES.get(sector_name, []):
            if symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= 3:
                return symbols

    return symbols[:3]


def build_card(story: UniqueStory, mode: str, matched_entities: list[dict]) -> dict:
    """Build a single recommendation card."""
    stock_symbols = infer_stock_symbols(story)
    matched_names = [entity["name"] for entity in matched_entities]
    if not matched_names:
        matched_names = [entity.name for entity in story.primary_article.entities[:3]]
    if not matched_names and story.primary_article.sectors:
        matched_names = story.primary_article.sectors[:3]

    if mode == "personalized":
        entity_type = matched_entities[0]["type"] if matched_entities else ""
        primary_name = matched_names[0] if matched_names else story.primary_article.article.title
        if entity_type == EntityType.COMPANY.value:
            recommendation_label = "重点关注公司"
            recommendation_reason = f"这条新闻与你最近关注的公司 {primary_name} 直接相关，建议继续跟踪后续公告和市场反应。"
        elif entity_type == EntityType.REGULATOR.value:
            recommendation_label = "关注监管动态"
            recommendation_reason = f"这条新闻涉及你最近关注的监管机构 {primary_name}，建议关注对相关板块和个股的后续影响。"
        else:
            recommendation_label = "观察板块联动"
            recommendation_reason = f"这条新闻与你最近关注的行业 {primary_name} 相关，建议观察板块是否出现持续联动。"
    else:
        recommendation_label = "最新资讯"
        recommendation_reason = "最新资讯，建议浏览。"

    article = story.primary_article.article
    return {
        "story_id": story.id,
        "title": article.title,
        "source": article.source,
        "published_date": article.published_date,
        "summary": summarize_story(story),
        "matched_entities": matched_names,
        "stock_symbols": stock_symbols,
        "recommendation_label": recommendation_label,
        "recommendation_reason": recommendation_reason,
    }


def build_feed_summary(mode: str, cards: list[dict], interest_entities: list[dict]) -> str:
    """Build the section-level summary text."""
    if mode == "personalized":
        if not cards:
            return "根据你最近的对话暂未找到强相关资讯。"
        entity_names = "、".join(entity["name"] for entity in interest_entities[:3])
        return f"根据你最近 10 条对话整理出 {len(cards)} 条相关资讯，重点围绕 {entity_names}。"
    if cards:
        return "最新 10 篇资讯摘要"
    return "当前知识库里还没有可推荐的资讯。"
