import os

import requests
from langchain_aws import ChatBedrockConverse
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.base import BaseCheckpointSaver

from tools.financial_tools import query_income_statement, query_balance_sheet, query_cash_flow, query_key_ratios
from tools.news_tools import query_news_sentiment

_ROUTER_SYSTEM_PROMPT = """
You are the assistant for a stock fundamental analysis service. You have two modes:

MODE A — ROUTING: Help the user find and confirm a stock for analysis.
MODE B — Q&A: Answer follow-up questions about a completed analysis report that appears in the conversation history.

TOOLS (only use in Mode A):
- search_ticker: call this any time the user mentions a company name or ticker symbol.
- ask_confirmation: call this when search_ticker found a match but the user's input differs from the result name.
- confirm_analysis: call this ONLY when you have a verified ticker and the user clearly wants analysis.

MODE A WORKFLOW:
1. User mentions a stock → call search_ticker with whatever they typed.
2a. If the user's input closely matches the found name or ticker (e.g. "AAPL" → "Apple Inc.", "sail" → "SAIL") → call confirm_analysis immediately. Do not ask for confirmation.
2b. If the user's input clearly differs from the found name (e.g. user typed a full company name but a different company was returned) → call ask_confirmation(ticker, name) with EXACTLY the ticker and name returned by search_ticker. Do nothing else.
2c. If search_ticker returns nothing → reply: "I couldn't find a publicly listed stock matching that name. Please try a different company name or ticker symbol."
3. If the user replies "yes" or any affirmative to a previous ask_confirmation → call confirm_analysis using the exact ticker and name from that ask_confirmation call. Do not call search_ticker again.
4. If the user rejects a confirmation or provides a different/corrected company name (e.g. "no, I meant X", "I meant steel authority of india") → call search_ticker with exactly what the user provided. Treat it as a fresh search from step 1.

MODE B RULES:
- If a completed analysis report is present in the conversation history and the user asks a question about it → answer from the report first.
- For specific data questions (e.g. "what was the exact P/E?", "show me the revenue numbers", "what did the news say?") → use the query_* tools to fetch the cached raw data for that ticker, then answer precisely.
- Available Q&A tools: query_key_ratios, query_income_statement, query_balance_sheet, query_cash_flow, query_news_sentiment.
- Keep answers concise and grounded in the data only.

GENERAL RULES:
- STRICT: Never mention any company name, ticker symbol, or alternative that was not returned by a search_ticker call. All company information must come exclusively from search_ticker results.
- STRICT: Never present multiple options or alternatives from your own knowledge. search_ticker returns one result — work only with that result.
- Never use markdown formatting in your conversational replies. Plain text only.
- Do not run analysis on non-equity securities (ETFs, futures, forex).
- If the user asks something unrelated to stocks or the report, redirect in one sentence.
"""

_REGION = os.getenv("AWS_REGION", "us-east-1")
_MODEL_ID = os.getenv("BEDROCK_HAIKU_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001")

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
        tools=[
            search_ticker, ask_confirmation, confirm_analysis,
            query_key_ratios, query_income_statement, query_balance_sheet,
            query_cash_flow, query_news_sentiment,
        ],
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


def inject_report(session_id: str, ticker: str, name: str, report: str) -> None:
    """Add the completed analysis report into the LangGraph checkpoint so follow-up questions can reference it."""
    from langchain_core.messages import AIMessage
    config = {"configurable": {"thread_id": session_id}}
    _graph.update_state(
        config,
        {"messages": [AIMessage(content=f"[Analysis complete for {name} ({ticker})]\n\n{report}")]},
    )
