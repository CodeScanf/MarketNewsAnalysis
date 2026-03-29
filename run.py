#!/usr/bin/env python3
"""Interactive demo for IntAnalysis with real RSS feed data."""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from intanalysis import IntelligenceSystem
from intanalysis.models import Article, UniqueStory, QueryResult


def load_rss_feeds(file_path: str, limit: Optional[int] = None) -> list[dict]:
    """Load RSS feed articles from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    articles = []
    for item in data:
        # Combine summary and content for full text
        content = item.get("summary", "")
        if item.get("content"):
            content += "\n" + str(item["content"])
        
        articles.append({
            "id": item.get("id", "")[:50],
            "title": item.get("title", "Untitled"),
            "content": content,
            "source": item.get("source", "Unknown"),
            "url": item.get("link", ""),
            "published_date": item.get("published"),
        })
    
    if limit:
        articles = articles[:limit]
    
    return articles


def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print("🏦 FINANCIAL NEWS INTELLIGENCE SYSTEM - Interactive Demo")
    print("   Powered by LangGraph Multi-Agent Architecture")
    print("="*70)


def print_story_details(story: UniqueStory, index: int, verbose: bool = True):
    """Print detailed story information."""
    article = story.primary_article.article
    entities = story.primary_article.entities
    impacts = story.primary_article.stock_impacts
    sectors = story.primary_article.sectors
    
    print(f"\n{'─'*60}")
    print(f"📰 [{index}] {article.title}")
    print(f"{'─'*60}")
    print(f"   📌 Source: {article.source}")
    
    if verbose:
        if article.content:
            summary = article.content[:200] + "..." if len(article.content) > 200 else article.content
            print(f"   📝 Summary: {summary}")
        
        if entities:
            companies = [e.name for e in entities if e.type.value == "company"]
            regulators = [e.name for e in entities if e.type.value == "regulator"]
            
            if companies:
                print(f"   🏢 Companies: {', '.join(companies)}")
            if regulators:
                print(f"   🏛️  Regulators: {', '.join(regulators)}")
        
        if sectors:
            print(f"   📊 Sectors: {', '.join(sectors)}")
        
        if impacts:
            print(f"   📈 Stock Impacts:")
            for imp in impacts[:5]:
                print(f"      • {imp.symbol}: {imp.confidence:.0%} ({imp.impact_type.value})")
        
        if story.duplicate_count > 0:
            print(f"   ⚠️  {story.duplicate_count} duplicate(s) consolidated")


def print_query_result(result: QueryResult, verbose: bool = True):
    """Print query result with details."""
    print(f"\n{'='*60}")
    print(f"🔍 QUERY RESULTS")
    print(f"{'='*60}")
    print(f"   Query: \"{result.query}\"")
    print(f"   Results: {len(result.stories)} stories found")
    
    if result.matched_entities:
        print(f"   🎯 Detected Entities: {', '.join(e.name for e in result.matched_entities)}")
    
    # Show AI Answer prominently
    if result.explanation:
        print(f"\n{'─'*60}")
        print(f"💡 INTELLIGENT ANSWER")
        print(f"{'─'*60}")
        print(f"   {result.explanation}")
        print(f"{'─'*60}")
    
    print(f"\n📚 SOURCES ({len(result.stories)} articles):")
    for i, story in enumerate(result.stories, 1):
        print_story_details(story, i, verbose=verbose)


def interactive_loop(system: IntelligenceSystem):
    """Run interactive query loop."""
    print("\n" + "─"*60)
    print("💬 INTERACTIVE QUERY MODE")
    print("   Type your query and press Enter")
    print("   Commands: 'stats', 'verbose on/off', 'clear cache', 'help', 'quit'")
    print("─"*60)
    
    verbose = True
    
    while True:
        try:
            query = input("\n🔍 Query > ").strip()
            
            if not query:
                continue
            
            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'stats':
                stats = system.get_stats()
                print(f"📊 System Stats: {stats['indexed_stories']} stories indexed")
                continue
            
            if query.lower() == 'verbose on':
                verbose = True
                print("✅ Verbose mode enabled")
                continue
            
            if query.lower() == 'verbose off':
                verbose = False
                print("✅ Verbose mode disabled")
                continue
            
            if query.lower() == 'clear cache':
                system.persistence.clear_cache()
                print("✅ Cache cleared. Restart demo to re-ingest all articles.")
                continue
            
            if query.lower() == 'help':
                print("""
📚 HELP - Query Examples:
   • "HDFC Bank news" - Company-specific search
   • "Banking sector update" - Sector-wide search
   • "RBI policy" - Regulator-specific search
   • "Stock market rally" - Topic search
   • "IndiGo DGCA" - Multi-entity search
   • "IPO surge" - Event-based search
   
🔧 Commands:
   • stats - Show system statistics
   • verbose on/off - Toggle detailed output
   • clear cache - Clear all persisted data
   • help - Show this help
   • quit - Exit the demo
""")
                continue
            
            # Process query
            print(f"\n⏳ Processing query...")
            result = system.query(query)
            print_query_result(result, verbose=verbose)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    print_header()
    
    # Default dataset path
    dataset_path = Path(__file__).parent / "dataset" / "rss_feeds_all.json"
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please provide the path to rss_feeds_all.json")
        sys.exit(1)
    
    # Load articles
    print(f"\n📂 Loading dataset: {dataset_path}")
    articles = load_rss_feeds(str(dataset_path))
    print(f"   Found {len(articles)} articles")
    
    # Initialize system
    print("\n🔧 Initializing Intelligence System...")
    system = IntelligenceSystem(verbose=True)
    
    # Ingest articles (only new ones)
    print(f"\n📥 Checking {len(articles)} articles for new content...")
    print("   Only new articles will be processed...")
    
    result = system.ingest(articles)
    
    print(f"\n{'─'*60}")
    print("📊 INGESTION SUMMARY")
    print(f"{'─'*60}")
    print(f"   Total articles checked: {result['total_articles']}")
    print(f"   Already processed: {result['skipped_count']}")
    print(f"   New unique stories: {result['unique_count']}")
    print(f"   Duplicates in new batch: {result['duplicate_count']}")
    if result['skipped_count'] > 0:
        print(f"   \u2705 Incremental ingestion: Skipped {result['skipped_count']} previously processed articles")
    
    # Show sample stories if any new ones were processed
    if result['unique_stories']:
        print(f"\n{'─'*60}")
        print("📰 NEWLY INDEXED STORIES (first 5)")
        print(f"{'─'*60}")
        
        for i, story in enumerate(result['unique_stories'][:5], 1):
            print_story_details(story, i, verbose=False)
    else:
        print(f"\n{'─'*60}")
        print("📰 NO NEW STORIES TO DISPLAY")
        print(f"   All articles have been processed previously")
        print(f"   Total stories in system: {system.get_stats()['indexed_stories']}")
        print(f"{'─'*60}")
    
    # Enter interactive mode
    interactive_loop(system)


if __name__ == "__main__":
    main()


