"""LLM service using Claude/Anthropic API."""

import os
import json
from typing import Optional
from pathlib import Path

from anthropic import Anthropic
from pydantic import BaseModel

# Load .env file from project root or parent directories
def _load_env():
    """Load .env file from current or parent directories."""
    try:
        from dotenv import load_dotenv
        # Try multiple locations
        for path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            env_file = path / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                return
    except ImportError:
        # dotenv not installed, try manual loading
        for path in [Path.cwd(), Path.cwd().parent, Path(__file__).parent.parent]:
            env_file = path / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ.setdefault(key.strip(), value.strip().strip('"\''))
                return

_load_env()


class LLMService:
    """Claude API wrapper with singleton pattern."""
    
    _instance: Optional["LLMService"] = None
    
    def __init__(self, api_key: Optional[str] = None):
        # Check multiple possible env var names
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY required")
        self.client = Anthropic(api_key=self.api_key)
        self.model = "claude-3-haiku-20240307"
    
    @classmethod
    def get_instance(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Generate text completion."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return response.content[0].text
    
    def extract_json(self, prompt: str, system: str = "") -> dict:
        """Generate and parse JSON response."""
        full_system = f"{system}\nRespond ONLY with valid JSON, no markdown or explanation."
        response = self.generate(prompt, full_system)
        
        # Clean markdown code blocks if present
        if "```" in response:
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        return json.loads(response.strip())
    
    def extract_entities(self, text: str) -> list[dict]:
        """Extract financial entities using Claude."""
        prompt = f"""Extract financial entities from this news article.

Article:
{text}

Extract:
1. Companies (with stock symbols if known)
2. Sectors (Banking, IT, Aviation, etc.)
3. Regulators (RBI, SEBI, DGCA, etc.)
4. Key persons mentioned

Return JSON array:
[{{"name": "entity name", "type": "company|sector|regulator|person", "confidence": 0.0-1.0}}]"""

        return self.extract_json(prompt, "You are a financial entity extraction expert.")
    
    def analyze_stock_impact(self, article_text: str, entities: list[dict]) -> list[dict]:
        """Analyze stock impact using Claude."""
        prompt = f"""Analyze stock market impact of this news.

Article:
{article_text}

Entities found: {json.dumps(entities)}

For each impacted stock, provide:
- symbol: NSE stock symbol
- company_name: Full company name
- confidence: 0.0-1.0 (1.0 for direct mention, 0.6-0.8 for sector impact)
- impact_type: "direct" | "sector" | "regulatory"
- reasoning: Brief explanation

Return JSON array of impacted stocks."""

        return self.extract_json(prompt, "You are a financial market analyst.")
    
    def explain_query_results(self, query: str, results: list[dict]) -> str:
        """Generate intelligent answer to query using article content."""
        prompt = f"""You are a financial news analyst. Answer the user's question directly using the information from these news articles.

User Question: {query}

Relevant News Articles:
{json.dumps(results[:5], indent=2)}

Instructions:
1. Directly answer the question using facts from the articles
2. Cite the source when mentioning specific information
3. If the articles contain the answer, provide it clearly
4. Keep the response concise but informative (3-5 sentences)
5. If the articles don't fully answer the question, say what they do tell us

Answer:"""

        return self.generate(prompt, "You are an intelligent financial news assistant that provides direct, factual answers based on news sources.", max_tokens=500)
