"""LLM service using DeepSeek's OpenAI-compatible API."""

import os
import json
import re
from typing import Optional, Any
from pathlib import Path

from openai import OpenAI

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
    """DeepSeek API wrapper with singleton pattern."""
    
    _instance: Optional["LLMService"] = None
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # Support both the user's requested variable names and DeepSeek-specific names.
        self.api_key = api_key or os.getenv("API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY or DEEPSEEK_API_KEY required")
        self.base_url = (
            base_url
            or os.getenv("BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.model = (
            model
            or os.getenv("MODEL_ID")
            or os.getenv("DEEPSEEK_MODEL")
            or "deepseek-chat"
        )
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
    
    @classmethod
    def get_instance(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024) -> str:
        """Generate text completion."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        return ""
    
    def extract_json(self, prompt: str, system: str = "") -> Any:
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
        """Extract financial entities using DeepSeek."""
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
        """Analyze stock impact using DeepSeek."""
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
    
    def explain_query_results(self, query: str, results: list[dict]) -> dict:
        """Generate intelligent answer to query using article content.
        
        Returns a dict with:
        - explanation: str - The answer text
        - relevant_indices: list[int] - Indices of articles that are relevant to the query
        """
        # Number articles for reference
        numbered_results = []
        for i, r in enumerate(results[:10]):
            numbered_results.append({
                "index": i,
                "title": r.get("title", ""),
                "source": r.get("source", ""),
                "content": r.get("content", "")[:500],
                "entities": r.get("entities", [])
            })
        
        prompt = f"""You are a financial news analyst. Analyze the user's query and the provided news articles.

User Question: {query}

News Articles:
{json.dumps(numbered_results, indent=2)}

Instructions:
1. First determine if the question is related to financial news/markets
2. If NOT related to financial news (like greetings, general chat, off-topic), return empty relevant_indices
3. If it IS a financial query, identify ONLY the articles that are DIRECTLY relevant to answering the question
4. An article is relevant ONLY if it specifically discusses the topic in the query (e.g., for "RBI policy", only articles about RBI/Reserve Bank policies)
5. Do NOT include articles that are only tangentially related or just happen to mention banking/finance in general

You MUST respond with ONLY a valid JSON object in this exact format (no other text before or after):
{{"explanation": "Your 2-4 sentence answer using facts from relevant articles only. If no relevant articles, say so.", "relevant_indices": [0, 2]}}"""

        response = self.generate(prompt, "You are a JSON-only response assistant. Output valid JSON only, no markdown, no explanation outside JSON.", max_tokens=600)
        
        # Parse the response
        try:
            # Clean up response - remove markdown code blocks if present
            clean_response = response.strip()
            if clean_response.startswith("```"):
                # Remove markdown code block
                clean_response = re.sub(r'^```(?:json)?\s*', '', clean_response)
                clean_response = re.sub(r'\s*```$', '', clean_response)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*"explanation"[^{}]*"relevant_indices"[^{}]*\[[\d,\s]*\][^{}]*\}', clean_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                # Validate the structure
                if "explanation" in parsed and "relevant_indices" in parsed:
                    return parsed
            
            # Try direct parse
            parsed = json.loads(clean_response)
            if "explanation" in parsed and "relevant_indices" in parsed:
                return parsed
                
        except (json.JSONDecodeError, AttributeError) as e:
            pass
        
        # Fallback: return all results if parsing fails
        return {
            "explanation": response,
            "relevant_indices": list(range(len(results[:10])))
        }
