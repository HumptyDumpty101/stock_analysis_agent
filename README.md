# Stock Fundamental Analysis Agent

An AI-powered stock analysis backend that performs full fundamental analysis on any publicly listed equity. Built with AWS Bedrock, LangGraph, and FastAPI.

## What it does

Type any company name or ticker — the agent finds the right stock, confirms ambiguous matches, then runs a multi-phase analysis and returns a professional investment brief.

**Analysis pipeline:**
1. Financial data — income statement, balance sheet, cash flow, key ratios
2. News & sentiment — recent headlines, sentiment score, risk flags
3. Risk assessment — leverage, liquidity, stress test
4. Valuation — DCF model + peer multiples, intrinsic value vs current price
5. Investment brief — synthesised report from all four analyses

## Architecture

```
Browser → Next.js API → FastAPI Agent Server
                              │
                    ┌─────────┴──────────┐
                    │   RouterAgent      │  LangGraph + Redis checkpointer
                    │   (session memory) │  — disambiguates company names
                    └─────────┬──────────┘
                              │ confirmed ticker
                    ┌─────────┴──────────┐
                    │  OrchestratorAgent │  AWS Bedrock (Claude Sonnet)
                    └─────────┬──────────┘
                              │ parallel tool calls
              ┌───────────────┼───────────────┐
        Financial Data    News/Sentiment    Risk Assessment
              └───────────────┼───────────────┘
                          Valuation
                        Report Writer
```

**Streaming:** Results are streamed phase-by-phase over SSE so the frontend can show live progress.

**Session memory:** LangGraph checkpointer (backed by Redis) saves full conversation state per session — tool calls included — so multi-turn disambiguation ("did you mean X?") works correctly across requests.

## Stack

- **AWS Bedrock** — Claude Sonnet 4.6 via Bedrock Converse API
- **Strands Agents** — orchestrator and specialist sub-agents
- **LangGraph + LangChain** — router agent with Redis-backed session memory
- **FastAPI** — SSE streaming server
- **Redis (Upstash)** — LangGraph checkpointer backend

## Live demo

[rohithks.com/finance](https://rohithks.com/finance)
