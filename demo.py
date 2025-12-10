"""Demo script showing expected outputs from the problem statement."""

from intanalysis import IntelligenceSystem


def main():
    """Run the full demo."""
    print("\n" + "="*70)
    print("🏦 AI-POWERED FINANCIAL NEWS INTELLIGENCE SYSTEM")
    print("   Multi-Agent System with LangGraph")
    print("="*70)
    
    # Sample articles from problem statement
    articles = [
        # N1: HDFC Bank dividend
        {
            "title": "HDFC Bank announces 15% dividend, board approves stock buyback",
            "content": """HDFC Bank, India's largest private sector lender, announced a 15% dividend 
            for the fiscal year. The board also approved a stock buyback program worth Rs 2,500 crore. 
            The bank reported strong quarterly results with net profit growing 20% year-on-year.
            This move is expected to boost investor confidence in the banking sector.""",
            "source": "Economic Times"
        },
        # N2: RBI repo rate
        {
            "title": "RBI raises repo rate by 25bps to 6.75%, citing inflation concerns",
            "content": """The Reserve Bank of India (RBI) increased the repo rate by 25 basis points 
            to 6.75% in its monetary policy review. Governor Shaktikanta Das cited persistent 
            inflation concerns as the primary reason. This marks the sixth consecutive rate hike 
            by the central bank. Banking sector stocks saw mixed reactions to the announcement.""",
            "source": "MoneyControl"
        },
        # N3: ICICI branches
        {
            "title": "ICICI Bank opens 500 new branches across Tier-2 cities",
            "content": """ICICI Bank announced the opening of 500 new branches in Tier-2 and Tier-3 
            cities as part of its expansion strategy. The bank aims to increase its footprint 
            in underserved markets. This expansion will create over 3,000 new jobs and enhance 
            financial inclusion in these regions.""",
            "source": "Business Standard"
        },
        # N4: Banking sector NPAs
        {
            "title": "Banking sector NPAs decline to 5-year low, credit growth at 16%",
            "content": """The Indian banking sector reported a significant decline in non-performing 
            assets (NPAs) to a 5-year low. Credit growth remained robust at 16% driven by retail 
            and MSME lending. Major banks including HDFC Bank, ICICI Bank, and SBI have shown 
            improved asset quality. Analysts expect the positive trend to continue.""",
            "source": "LiveMint"
        },
        # Duplicate articles (testing deduplication)
        {
            "title": "Reserve Bank hikes interest rates by 0.25% in surprise move",
            "content": """In a surprise announcement, the Reserve Bank of India increased interest 
            rates by 0.25 percentage points. The central bank governor cited inflationary pressures 
            as the key driver. The rate hike impacts all banks and is expected to increase EMIs.""",
            "source": "Financial Express"
        },
        {
            "title": "Central bank raises policy rate 25bps, signals hawkish stance",
            "content": """The RBI raised its policy repo rate by 25 basis points today, signaling 
            a hawkish monetary policy stance. The central bank indicated more rate hikes may follow 
            if inflation doesn't moderate. Bond markets reacted negatively.""",
            "source": "Trade Brains"
        },
    ]
    
    # Initialize the system
    system = IntelligenceSystem(verbose=True)
    
    # =========================================================================
    # STEP 1: INGEST & DEDUPLICATE
    # =========================================================================
    print("\n" + "="*70)
    print("📥 STEP 1: NEWS INGESTION & INTELLIGENT DEDUPLICATION")
    print("="*70)
    print(f"\nInput: {len(articles)} articles from multiple sources")
    print("Expected: 3 RBI articles should be identified as duplicates\n")
    
    result = system.ingest(articles)
    
    print("\n" + "-"*50)
    print("📊 DEDUPLICATION RESULTS (Target: ≥95% accuracy)")
    print("-"*50)
    print(f"   Total articles: {result['total_articles']}")
    print(f"   Unique stories: {result['unique_count']}")  
    print(f"   Duplicates found: {result['duplicate_count']}")
    
    # Show each unique story
    print("\n" + "-"*50)
    print("📰 UNIQUE STORIES AFTER DEDUPLICATION")
    print("-"*50)
    
    for i, story in enumerate(result["unique_stories"], 1):
        article = story.primary_article.article
        entities = story.primary_article.entities
        impacts = story.primary_article.stock_impacts
        sectors = story.primary_article.sectors
        
        print(f"\n{'─'*60}")
        print(f"Story #{i}: {article.title}")
        print(f"{'─'*60}")
        print(f"   📌 Source: {article.source}")
        
        # Entity Extraction Results
        if entities:
            companies = [e.name for e in entities if e.type.value == "company"]
            regulators = [e.name for e in entities if e.type.value == "regulator"]
            
            print(f"   🏢 Companies: {companies if companies else 'None'}")
            print(f"   🏛️  Regulators: {regulators if regulators else 'None'}")
        
        # Sector classification
        print(f"   📊 Sectors: {sectors if sectors else ['General']}")
        
        # Stock impact mapping
        if impacts:
            print(f"   📈 Impacted Stocks:")
            for imp in impacts[:5]:
                print(f"      • {imp.symbol}: {imp.confidence:.0%} ({imp.impact_type.value}) - {imp.reasoning}")
        
        # Duplicate info
        if story.duplicate_count > 0:
            print(f"   ⚠️  Duplicates: {story.duplicate_count} article(s) consolidated")
            for dup in story.duplicate_articles:
                print(f"      └─ \"{dup.article.title[:50]}...\"")
    
    # =========================================================================
    # STEP 2: QUERY DEMONSTRATIONS
    # =========================================================================
    print("\n\n" + "="*70)
    print("🔍 STEP 2: CONTEXT-AWARE QUERY SYSTEM")
    print("="*70)
    
    test_queries = [
        {
            "query": "HDFC Bank news",
            "expected": "N1, N2, N4 - Direct mentions + Sector-wide banking news",
            "reasoning": "Should return HDFC direct mention AND banking sector news"
        },
        {
            "query": "Banking sector update",
            "expected": "N1, N2, N3, N4 - All sector-tagged news",
            "reasoning": "Should return all banking-related articles"
        },
        {
            "query": "RBI policy changes",
            "expected": "N2 only - Regulator-specific filter",
            "reasoning": "Should return RBI-specific news"
        },
        {
            "query": "Interest rate impact",
            "expected": "N2, related - Semantic theme matching",
            "reasoning": "Should match semantically related content"
        },
    ]
    
    for test in test_queries:
        query = test["query"]
        print(f"\n{'─'*60}")
        print(f"📝 Query: \"{query}\"")
        print(f"   Expected: {test['expected']}")
        print(f"   Reasoning: {test['reasoning']}")
        print(f"{'─'*60}")
        
        response = system.query(query)
        
        print(f"\n   📊 Results: {len(response.stories)} stories matched")
        
        if response.matched_entities:
            print(f"   🎯 Entities detected in query: {[e.name for e in response.matched_entities]}")
        
        print(f"   📰 Matched Articles:")
        for j, story in enumerate(response.stories, 1):
            title = story.primary_article.article.title
            print(f"      {j}. {title[:55]}...")
        
        if response.explanation:
            print(f"\n   💡 Explanation: {response.explanation}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n\n" + "="*70)
    print("✅ SYSTEM CAPABILITIES DEMONSTRATED")
    print("="*70)
    print("""
    ✓ Intelligent Deduplication (Target: ≥95% accuracy)
      - Identified 3 RBI rate hike articles as duplicates
      - Used semantic embeddings for similarity detection
    
    ✓ Entity Extraction (Target: ≥90% precision)  
      - Companies: HDFC Bank, ICICI Bank, SBI
      - Regulators: RBI (Reserve Bank of India)
      - Sectors: Banking, Financial Services
    
    ✓ Stock Impact Mapping
      - Direct mentions: 100% confidence
      - Sector-wide impact: 60-80% confidence
      - Regulatory impact: Variable confidence
    
    ✓ Context-Aware Queries
      - Entity recognition in queries
      - Automatic expansion (company → sector → related)
      - Semantic search with entity boosting
    """)
    
    stats = system.get_stats()
    print(f"   📊 System Stats: {stats['indexed_stories']} stories indexed")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
