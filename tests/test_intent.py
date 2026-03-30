"""Tests for high-level intent routing."""

import pytest

from intanalysis.intent import IntentClassifier
from intanalysis.models import QueryIntent


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("你好", QueryIntent.GENERAL_CHAT),
        ("帮我解释一下什么是通货膨胀", QueryIntent.GENERAL_CHAT),
        ("更新一下新闻", QueryIntent.NEWS_UPDATE),
        ("把未入库文章同步到知识库", QueryIntent.NEWS_UPDATE),
        ("泡泡玛特的股票能买吗", QueryIntent.FINANCIAL_QUERY),
        ("最近港股消费板块怎么样", QueryIntent.FINANCIAL_QUERY),
    ],
)
def test_rule_intent_classifier(query, expected):
    """Rule-first classifier should handle core routing cases deterministically."""
    classifier = IntentClassifier(use_llm=False)
    decision = classifier.classify(query)
    assert decision.intent == expected
    assert decision.source == "rule"
