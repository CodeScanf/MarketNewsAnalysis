"""Intent classification for routing user queries."""

from __future__ import annotations

import re

from intanalysis.llm import LLMService
from intanalysis.mappings import REGULATORS, find_stock_symbols
from intanalysis.models import IntentDecision, QueryIntent


class IntentClassifier:
    """Rule-first intent classifier with optional LLM fallback."""

    UPDATE_PATTERNS = (
        re.compile(r"(更新|刷新|同步|拉取|抓取|采集|重(新)?获取).{0,8}(新闻|资讯|文章|rss|feed|数据源|内容|快讯)"),
        re.compile(r"(新闻|资讯|文章|rss|feed|数据源|内容|快讯).{0,8}(更新|刷新|同步|拉取|抓取|采集)"),
    )
    GENERAL_PATTERNS = (
        re.compile(r"^(什么是|啥是|解释一下|介绍一下|帮我解释|请解释|科普一下)"),
        re.compile(r"(你是谁|你能做什么|怎么用|什么意思|如何理解)"),
    )
    UPDATE_KEYWORDS = {"更新", "刷新", "同步", "拉取", "抓取", "采集", "补抓", "重跑", "重建索引", "更新新闻", "刷新新闻"}
    NEWS_KEYWORDS = {"新闻", "资讯", "文章", "快讯", "研报", "报道", "消息", "动态"}
    FINANCE_KEYWORDS = {
        "股票", "股价", "港股", "美股", "a股", "财报", "业绩", "估值", "市盈率", "市值", "公司",
        "公告", "行业", "板块", "市场", "金融", "财经", "宏观", "经济", "利率", "汇率", "央行",
        "证监会", "交易所", "做多", "做空", "回购", "分红", "ipo", "earnings", "stock", "shares",
        "market", "finance", "financial", "ticker",
    }
    SECTOR_KEYWORDS = {
        "banking", "bank", "银行", "aviation", "航空", "it", "tech", "科技", "互联网",
        "电商", "零售", "消费", "auto", "automobile", "汽车", "新能源车", "电动车", "家电",
    }
    TICKER_RE = re.compile(r"\b\d{4,6}\.(?:hk|sh|sz)\b", re.IGNORECASE)

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None

    @property
    def llm(self) -> LLMService | None:
        if self._llm is None and self.use_llm:
            try:
                self._llm = LLMService.get_instance()
            except Exception:
                self._llm = None
        return self._llm

    def classify(self, query: str) -> IntentDecision:
        """Classify a user query into one of the supported intents."""
        rule_decision = self._classify_by_rules(query)
        if rule_decision.confidence >= 0.85 or self.llm is None:
            return rule_decision

        try:
            llm_decision = self.llm.classify_intent(query)
            if llm_decision.confidence >= rule_decision.confidence:
                return llm_decision
        except Exception:
            pass

        return rule_decision

    def _classify_by_rules(self, query: str) -> IntentDecision:
        text = (query or "").strip()
        if not text:
            return IntentDecision(
                intent=QueryIntent.GENERAL_CHAT,
                source="rule",
                confidence=1.0,
                reason="Empty query defaults to general chat.",
            )

        lowered = text.lower()

        for pattern in self.UPDATE_PATTERNS:
            if pattern.search(text):
                return IntentDecision(
                    intent=QueryIntent.NEWS_UPDATE,
                    source="rule",
                    confidence=0.98,
                    reason="Matched explicit news refresh pattern.",
                )

        if any(keyword in text for keyword in self.UPDATE_KEYWORDS):
            return IntentDecision(
                intent=QueryIntent.NEWS_UPDATE,
                source="rule",
                confidence=0.9,
                reason="Matched explicit refresh keyword.",
            )

        if any(pattern.search(text) for pattern in self.GENERAL_PATTERNS):
            finance_hits = sum(keyword in lowered for keyword in self.FINANCE_KEYWORDS)
            has_news_signal = any(keyword in text for keyword in self.NEWS_KEYWORDS)
            if finance_hits <= 1 and not has_news_signal:
                return IntentDecision(
                    intent=QueryIntent.GENERAL_CHAT,
                    source="rule",
                    confidence=0.9,
                    reason="Matched concept-explanation or casual-help pattern without strong news intent.",
                )

        if find_stock_symbols(lowered) or self.TICKER_RE.search(lowered):
            return IntentDecision(
                intent=QueryIntent.FINANCIAL_QUERY,
                source="rule",
                confidence=0.97,
                reason="Matched known company or ticker alias.",
            )

        for key, info in REGULATORS.items():
            aliases = info.get("aliases", [])
            if key in lowered or info["full_name"].lower() in lowered or any(alias.lower() in lowered for alias in aliases):
                return IntentDecision(
                    intent=QueryIntent.FINANCIAL_QUERY,
                    source="rule",
                    confidence=0.97,
                    reason="Matched regulator keyword.",
                )

        finance_hits = sum(keyword in lowered for keyword in self.FINANCE_KEYWORDS)
        sector_hits = sum(keyword in lowered for keyword in self.SECTOR_KEYWORDS)

        if finance_hits >= 2 or sector_hits >= 1:
            return IntentDecision(
                intent=QueryIntent.FINANCIAL_QUERY,
                source="rule",
                confidence=0.9,
                reason="Matched multiple finance or sector keywords.",
            )

        if finance_hits == 1:
            return IntentDecision(
                intent=QueryIntent.FINANCIAL_QUERY,
                source="rule",
                confidence=0.7,
                reason="Matched a single finance keyword.",
            )

        return IntentDecision(
            intent=QueryIntent.GENERAL_CHAT,
            source="rule",
            confidence=0.45,
            reason="No strong finance or refresh signals found.",
        )
