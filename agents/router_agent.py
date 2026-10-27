import os

import requests
from langchain_aws import ChatBedrockConverse
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.base import BaseCheckpointSaver

_ROUTER_SYSTEM_PROMPT = """
You are the assistant for a stock fundamental analysis service. Your only job is to understand what stock the user wants analyzed and trigger the analysis once confirmed.

TOOLS:
- search_ticker: call this any time the user mentions a company name or ticker symbol.
- ask_confirmation: call this when search_ticker found a match but the user's input differs from the result name. This sends a confirmation question to the user.
- confirm_analysis: call this ONLY when you have a verified ticker and the user clearly wants analysis.

WORKFLOW:
1. User mentions a stock → call search_ticker with whatever they typed.
2a. If the user's input closely matches the found name (e.g. "AAPL" → "Apple Inc.", "apple" → "Apple Inc.") → call confirm_analysis immediately.
2b. If the user's input differs from the found name (e.g. "tata steels" → found "Tata Steel Ltd") → call ask_confirmation(ticker, name). Do nothing else.
2c. If search_ticker returns nothing → reply in one sentence: "I couldn't find a publicly listed stock matching that name. Please try a different company name or ticker symbol."
3. If the user replies "yes" or another affirmative to a previous ask_confirmation → call confirm_analysis with the ticker and name from that earlier confirmation. Do not search again.

RULES:
- Never invent or guess tickers. Only use tickers returned by search_ticker.
- Never use markdown formatting. No asterisks, bold, italics, bullet points, or headers. Plain text only.
- Keep replies to 1 sentence maximum.
- Do not run analysis on non-equity securities (ETFs, futures, forex).
- If the user asks something unrelated to stocks, redirect in one sentence.
"""

_REGION = os.getenv("AWS_REGION", "us-east-1")
_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")

_graph = None


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
def ask_confirmation(ticker: str, name: str) -> str:
    """Ask the user to confirm this is the stock they want analyzed.
    Call when the user's input differs from the found stock name.

    Args:
        ticker: The exact ticker symbol returned by search_ticker (e.g. TATASTEEL.NS).
        name: The company name returned by search_ticker (e.g. Tata Steel Limited).
    """
    return f"Confirmation requested for {name} ({ticker})."


@tool
def confirm_analysis(ticker: str, name: str) -> str:
    """Signal that the user has confirmed they want fundamental analysis on this stock.
    Call only when you have a verified ticker and the user's intent is clear.

    Args:
        ticker: The exact ticker symbol returned by search_ticker (e.g. TATASTEEL.NS).
        name: The company name returned by search_ticker (e.g. Tata Steel Ltd).
    """
    return f"Analysis confirmed for {name} ({ticker}). The pipeline will now run."


_graph = None


def init_router(checkpointer: BaseCheckpointSaver) -> None:
    """Call once at app startup. Checkpointer is created and managed by the caller."""
    global _graph
    llm = ChatBedrockConverse(model_id=_MODEL_ID, region_name=_REGION)
    _graph = create_react_agent(
        model=llm,
        tools=[search_ticker, ask_confirmation, confirm_analysis],
        prompt=_ROUTER_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )


def run_router(session_id: str, message: str) -> dict:
    """
    Runs the router agent for a given session turn.
    Redis checkpointer restores full conversation history automatically.

    Returns {"text": str, "analysis_slot": dict|None, "pending_slot": dict|None}.
    """
    graph = _graph
    config = {
        "configurable": {
            "thread_id": session_id,
        }
    }
    inputs = {"messages": [{"role": "user", "content": message}]}

    analysis_slot = None
    pending_slot = None
    final_text = ""

    for chunk in graph.stream(inputs, stream_mode="updates", config=config):
        if "agent" in chunk:
            for msg in chunk["agent"]["messages"]:
                for tc in getattr(msg, "tool_calls", []):
                    if tc["name"] == "confirm_analysis":
                        analysis_slot = tc["args"]
                    elif tc["name"] == "ask_confirmation":
                        pending_slot = tc["args"]
                content = getattr(msg, "content", "")
                if isinstance(content, str) and content:
                    final_text = content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                            final_text = block["text"]

    return {"text": final_text, "analysis_slot": analysis_slot, "pending_slot": pending_slot}
