import os

import requests
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

_ROUTER_SYSTEM_PROMPT = """
You are the assistant for a stock fundamental analysis service. Your only job is to understand what stock the user wants analyzed and trigger the analysis once confirmed.

TOOLS:
- search_ticker: call this any time the user mentions a company name or ticker symbol.
- confirm_analysis: call this ONLY when you have a verified ticker from search_ticker AND the user clearly wants analysis.

WORKFLOW:
1. User mentions a stock → call search_ticker with whatever they typed.
2a. If what the user typed closely matches the result name (e.g. "AAPL" → "Apple Inc.", "apple" → "Apple Inc.") → call confirm_analysis immediately without asking.
2b. If the user's input differs from the found name (e.g. "tata steels" → found "Tata Steel Ltd") → reply with EXACTLY this pattern: "Did you mean Tata Steel Ltd (TATASTEEL.NS)? Reply yes to run a fundamental analysis." Use the actual name and ticker from search_ticker.
2c. If search_ticker returns nothing → reply: "I couldn't find a publicly listed stock matching that name. Please try a different company name or ticker symbol."
3. User says yes/sure/go ahead → look at your previous search_ticker result in the conversation, then call confirm_analysis with that ticker and name.

RULES:
- Never invent or guess tickers. Only use tickers returned by search_ticker.
- Never use markdown formatting. No asterisks, bold, italics, bullet points, or headers. Plain text only.
- Keep replies to 1–2 sentences maximum.
- Do not run analysis on non-equity securities (ETFs, futures, forex).
- If the user asks something unrelated to stocks, briefly redirect in one sentence.
"""


def _yahoo_search(query: str) -> dict | None:
    try:
        resp = requests.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={"q": query, "quotesCount": 5, "newsCount": 0},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        resp.raise_for_status()
        for quote in resp.json().get("quotes", []):
            if quote.get("quoteType") == "EQUITY":
                return {
                    "ticker": quote["symbol"],
                    "name": quote.get("shortname") or quote.get("longname") or query,
                }
    except Exception:
        pass
    return None


def make_router_agent(history: list[dict]) -> tuple[Agent, dict]:
    """
    Create a fresh RouterAgent per request.
    Returns (agent, analysis_slot).
    After calling agent(user_message), check analysis_slot for {"ticker", "name"}
    to know if analysis was triggered.
    """
    analysis_slot: dict = {}

    @tool
    def search_ticker(query: str) -> str:
        """Search Yahoo Finance for a publicly listed stock by company name or ticker symbol.
        Returns the best matching equity ticker and full company name, or a not-found message.

        Args:
            query: The company name or ticker symbol to search for.
        """
        result = _yahoo_search(query)
        if result:
            return f"Found: {result['name']} (Ticker: {result['ticker']})"
        return f"No publicly listed equity found matching '{query}'. Ask the user to try a different name or ticker."

    @tool
    def confirm_analysis(ticker: str, name: str) -> str:
        """Signal that the user has confirmed they want fundamental analysis on this stock.
        Call only when you have a verified ticker and the user's intent is clear.

        Args:
            ticker: The exact ticker symbol returned by search_ticker (e.g. TATASTEEL.NS).
            name: The company name returned by search_ticker (e.g. Tata Steel Ltd).
        """
        analysis_slot["ticker"] = ticker
        analysis_slot["name"] = name
        return "Analysis confirmed. The pipeline will now run."

    agent = Agent(
        name="RouterAgent",
        model=BedrockModel(
            model_id=os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        ),
        system_prompt=_ROUTER_SYSTEM_PROMPT,
        tools=[search_ticker, confirm_analysis],
        messages=list(history),
        callback_handler=None,
    )

    return agent, analysis_slot
