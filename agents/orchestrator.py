import os

from strands import Agent
from strands.models.bedrock import BedrockModel

from .financial_data_agent import financial_data_agent
from .valuation_agent import valuation_agent
from .news_sentiment_agent import news_sentiment_agent
from .risk_agent import risk_agent
from .report_writer_agent import report_writer_agent

ORCHESTRATOR_INSTRUCTIONS = """
You are a senior investment analyst orchestrating a full fundamental analysis.
You coordinate a team of specialist sub-agents and synthesize their findings
into a final investment brief.

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

PHASE 3 -- SYNTHESIS:
Step 3: Call write_investment_report with ALL findings from Steps 1a, 1b, 1c, and 2.
Structure your input as:
    FINANCIAL DATA ANALYSIS:    [output from step 1a]
    NEWS & SENTIMENT ANALYSIS:  [output from step 1b]
    RISK ASSESSMENT:            [output from step 1c]
    VALUATION ANALYSIS:         [output from step 2]

IMPORTANT:
- Do not skip any step. All four specialist analyses are required.
- If a sub-agent returns an error for any tool, note it in the final report
  rather than stopping the workflow entirely.
- Your final output must be the ReportWriterAgent's output copied verbatim,
  character for character. Do not add commentary, remove sections, reformat,
  or summarize any part of it.
- Be patient: the workflow has multiple API calls and may take 30-60 seconds.
"""

orchestrator = Agent(
    name="OrchestratorAgent",
    model=BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
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
        report_writer_agent.as_tool(
            name="write_investment_report",
            description=(
                "Synthesizes all analysis (financial data, valuation, sentiment, risk) "
                "into a final, professional investment brief. Takes the combined "
                "findings from all specialist agents and produces the final report."
            ),
        ),
    ],
)
