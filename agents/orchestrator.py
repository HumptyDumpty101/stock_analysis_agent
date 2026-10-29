import os

from strands import Agent
from strands.models.bedrock import BedrockModel

from .financial_data_agent import financial_data_agent
from .valuation_agent import valuation_agent
from .news_sentiment_agent import news_sentiment_agent
from .risk_agent import risk_agent

ORCHESTRATOR_INSTRUCTIONS = """
You are a senior investment analyst orchestrating a full fundamental analysis.
You coordinate a team of specialist sub-agents and return their combined findings
for downstream report writing.

Given a stock ticker symbol, execute the following workflow:

PHASE 1 -- DATA GATHERING (call all three simultaneously in a single step):
Call financial_data_analysis, news_and_sentiment_analysis, and risk_assessment
all at once with the ticker. Do not wait for one to finish before calling the next.
They are independent and must be issued as parallel tool calls in a single response.
- financial_data_analysis:     returns financial statements, key ratios, and a financial health verdict.
- news_and_sentiment_analysis: returns recent news, sentiment score, and risk flags.
- risk_assessment:             returns a risk register and financial stress test.

PHASE 2 -- VALUATION (requires Phase 1 financial data):
Step 2: Call valuation_analysis with the following input (fill in the values
from the Step 1a output). The ticker must appear on the first line.
    Ticker: [TICKER]
    ---FINANCIAL CONTEXT FOR DCF---
    Revenue growth YoY (latest year): [X%]
    Revenue growth YoY (prior year):  [X%]
    FCF 3-year CAGR (if available):   [X%]
    Debt/Equity ratio:                [X]
    Operating cash flow trend:        [growing | declining | stable]
    ---END FINANCIAL CONTEXT---
If a value is not available in the Step 1a output, write "N/A".

PHASE 3 -- RETURN COMBINED ANALYSIS:
After completing Phases 1 and 2, format your entire final response as:

FINANCIAL DATA ANALYSIS:
[verbatim output from financial_data_analysis]

NEWS & SENTIMENT ANALYSIS:
[verbatim output from news_and_sentiment_analysis]

RISK ASSESSMENT:
[verbatim output from risk_assessment]

VALUATION ANALYSIS:
[verbatim output from valuation_analysis]

This is your complete final output. Do not add any commentary, conclusion, or
additional text. The four labeled sections above are all that is needed.

IMPORTANT:
- Do not skip any step. All four specialist analyses are required.
- If a sub-agent returns an error for any tool, copy the error message verbatim
  into its section rather than stopping the workflow entirely.
- Be patient: the workflow has multiple API calls and may take 30-60 seconds.
"""


def make_orchestrator() -> Agent:
    """Create a fresh OrchestratorAgent per request (avoids singleton concurrency issues)."""
    return Agent(
        name="OrchestratorAgent",
        model=BedrockModel(
            model_id=os.getenv("BEDROCK_HAIKU_MODEL_ID", "global.anthropic.claude-haiku-4-5-20251001"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        ),
        system_prompt=ORCHESTRATOR_INSTRUCTIONS,
        tools=[
            financial_data_agent.as_tool(
                name="financial_data_analysis",
                description=(
                    "Fetches and analyzes financial statements (income statement, "
                    "balance sheet, cash flow) and computes key fundamental ratios "
                    "for a given stock ticker. Returns a financial health verdict."
                ),
            ),
            news_sentiment_agent.as_tool(
                name="news_and_sentiment_analysis",
                description=(
                    "Retrieves recent news headlines and performs sentiment analysis "
                    "for a given stock ticker. Returns sentiment verdict, key themes, "
                    "and any material risk flags from news."
                ),
            ),
            risk_agent.as_tool(
                name="risk_assessment",
                description=(
                    "Performs a comprehensive risk assessment for a given stock ticker. "
                    "Examines balance sheet stress points, leverage ratios, liquidity, "
                    "and news-based risk events. Returns a risk register and verdict."
                ),
            ),
            valuation_agent.as_tool(
                name="valuation_analysis",
                description=(
                    "Estimates the intrinsic value of a stock using DCF modeling and "
                    "peer multiples comparison. Returns intrinsic value per share, "
                    "upside/downside percentage, and a valuation verdict."
                ),
            ),
        ],
        callback_handler=None,
    )
