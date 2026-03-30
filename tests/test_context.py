"""Tests for lightweight short-term conversation context."""

from intanalysis import IntelligenceSystem


def test_build_short_term_context_collects_recent_entities():
    """Recent assistant entities should be carried into short-term context."""
    system = IntelligenceSystem(verbose=False)
    context = system._build_short_term_context(
        [
            {"role": "user", "content": "泡泡玛特的股票能买吗？"},
            {
                "role": "assistant",
                "content": "略",
                "intent": "financial_query",
                "matched_entities": ["泡泡玛特"],
                "story_titles": ["泡泡玛特主动“降温”，加剧港股寒冬？"],
            },
        ]
    )

    assert context.last_user_query == "泡泡玛特的股票能买吗？"
    assert context.last_intent.value == "financial_query"
    assert "泡泡玛特" in context.recent_entities


def test_followup_query_is_resolved_with_context():
    """Ambiguous follow-up questions should inherit recent entities."""
    system = IntelligenceSystem(verbose=False)
    context = system._build_short_term_context(
        [
            {"role": "user", "content": "泡泡玛特的股票能买吗？"},
            {
                "role": "assistant",
                "content": "略",
                "intent": "financial_query",
                "matched_entities": ["泡泡玛特"],
                "story_titles": ["泡泡玛特主动“降温”，加剧港股寒冬？"],
            },
        ]
    )

    resolved = system._resolve_query_with_context("那它呢？", context)

    assert context.applied is True
    assert "泡泡玛特" in resolved
    assert "上轮问题" in resolved
