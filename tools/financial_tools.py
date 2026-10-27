import json

import yfinance as yf
from strands import tool
from langchain.tools import tool as lc_tool

from .utils import _safe_float, _millions, _extract_row, make_cached

_cached = make_cached(ttl=86_400)


@tool
def get_income_statement(ticker: str) -> str:
    """Get the last 4 years of income statement for a stock, plus TTM headline numbers.

    Returns revenue, gross profit, operating income, EBITDA, net income, R&D in USD millions,
    plus trailing/forward EPS and margin percentages from the latest available data.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
    """
    def fetch():
        try:
            stock = yf.Ticker(ticker.upper())
            df = stock.income_stmt
            if df is None or df.empty:
                return json.dumps({"error": f"No income statement for {ticker}"})

            result = {"annual_data": {}}
            for col in df.columns[:4]:
                year = str(col.year)
                result["annual_data"][year] = {
                    "Revenue (M)":          _extract_row(df, "Total Revenue", col=col),
                    "Gross Profit (M)":     _extract_row(df, "Gross Profit", col=col),
                    "Operating Income (M)": _extract_row(df, "Operating Income", "EBIT", col=col),
                    "EBITDA (M)":           _extract_row(df, "EBITDA", col=col),
                    "Net Income (M)":       _extract_row(df, "Net Income", "Net Income Common Stockholders", col=col),
                    "R&D (M)":              _extract_row(df, "Research And Development", col=col),
                }

            info = stock.info
            result["eps_ttm"]             = _safe_float(info.get("trailingEps"))
            result["eps_forward"]         = _safe_float(info.get("forwardEps"))
            result["revenue_ttm_M"]       = _millions(info.get("totalRevenue"))
            result["gross_margin_pct"]    = round(info.get("grossMargins") * 100, 2) if info.get("grossMargins") else None
            result["operating_margin_pct"]= round(info.get("operatingMargins") * 100, 2) if info.get("operatingMargins") else None
            result["net_margin_pct"]      = round(info.get("profitMargins") * 100, 2) if info.get("profitMargins") else None

            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "ticker": ticker})

    return _cached(f"income:{ticker.upper()}", fetch)


@tool
def get_balance_sheet(ticker: str) -> str:
    """Get the most recent balance sheet for a stock.

    Returns total assets, liabilities, equity, total debt, and cash in USD millions.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
    """
    def fetch():
        df = yf.Ticker(ticker.upper()).balance_sheet
        if df is None or df.empty:
            return json.dumps({"error": f"No balance sheet for {ticker}"})

        col = df.columns[0]  # most recent fiscal year
        result = {
            "Total Assets (M)":        _extract_row(df, "Total Assets", col=col),
            "Total Liabilities (M)":   _extract_row(df, "Total Liabilities Net Minority Interest", "Total Liabilities", col=col),
            "Stockholders Equity (M)": _extract_row(df, "Stockholders Equity", "Common Stock Equity", col=col),
            "Total Debt (M)":          _extract_row(df, "Total Debt", col=col),
            "Cash (M)":                _extract_row(df, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments", col=col),
        }
        return json.dumps(result)

    return _cached(f"balance:{ticker.upper()}", fetch)


@tool
def get_cash_flow_statement(ticker: str) -> str:
    """Get the last 3 years of cash flow data for a stock.

    Returns operating cash flow, capital expenditure, and free cash flow in USD millions.
    FCF = Operating CF + Capex (yfinance reports capex as negative).

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
    """
    def fetch():
        df = yf.Ticker(ticker.upper()).cash_flow
        if df is None or df.empty:
            return json.dumps({"error": f"No cash flow data for {ticker}"})

        result = {}
        for col in df.columns[:4]:
            year = str(col.year)
            ocf   = _extract_row(df, "Operating Cash Flow", col=col)
            capex = _extract_row(df, "Capital Expenditure", col=col)
            fcf   = round(ocf + capex, 1) if ocf is not None and capex is not None else None
            result[year] = {
                "Operating Cash Flow (M)": ocf,
                "Capital Expenditure (M)": capex,
                "Free Cash Flow (M)":      fcf,
            }
        return json.dumps(result)

    return _cached(f"cashflow:{ticker.upper()}", fetch)


@tool
def compute_key_ratios(ticker: str) -> str:
    """Compute key financial ratios for a stock.

    Returns margins, ROE, debt/equity, P/E, EPS, market cap, price, sector.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
    """
    def fetch():
        try:
            stock = yf.Ticker(ticker.upper())
            info  = stock.info

            def pct(key):
                f = _safe_float(info.get(key))
                return round(f * 100, 2) if f is not None else None

            def num(key):
                f = _safe_float(info.get(key))
                return round(f, 2) if f is not None else None

            # Real revenue YoY: (newest − previous) ÷ previous × 100
            # income_stmt columns are newest-first, so col 0 = this year, col 1 = last year
            revenue_yoy = None
            try:
                fin = stock.income_stmt
                if fin is not None and not fin.empty and len(fin.columns) >= 2:
                    rev_this = _extract_row(fin, "Total Revenue", col=fin.columns[0])
                    rev_last = _extract_row(fin, "Total Revenue", col=fin.columns[1])
                    if rev_this and rev_last and rev_last != 0:
                        revenue_yoy = round((rev_this - rev_last) / rev_last * 100, 2)
            except Exception:
                pass

            # FCF yield: free cash flow ÷ market cap × 100
            market_cap = info.get("marketCap")
            fcf        = info.get("freeCashflow")
            fcf_yield  = None
            if market_cap and fcf and float(market_cap) > 0:
                fcf_yield = round((float(fcf) / float(market_cap)) * 100, 2)

            # Debt/Equity unit fix: Yahoo returns 102.63 meaning 102.63% (= 1.03×)
            debt_to_eq = _safe_float(info.get("debtToEquity"))
            if debt_to_eq is not None:
                debt_to_eq = round(debt_to_eq / 100, 2)

            result = {
                "ticker":         ticker.upper(),
                "company_name":   info.get("longName", ticker.upper()),
                "sector":         info.get("sector", "N/A"),
                "industry":       info.get("industry", "N/A"),
                "current_price":  _safe_float(info.get("currentPrice") or info.get("regularMarketPrice")),
                "market_cap_M":   _millions(market_cap),
                "52_week_high":   _safe_float(info.get("fiftyTwoWeekHigh")),
                "52_week_low":    _safe_float(info.get("fiftyTwoWeekLow")),
                "beta":           num("beta"),
                "valuation": {
                    "pe_trailing": num("trailingPE"),
                    "pe_forward":  num("forwardPE"),
                    "pb_ratio":    num("priceToBook"),
                    "ps_ratio":    num("priceToSalesTrailing12Months"),
                    "ev_ebitda":   num("enterpriseToEbitda"),
                    "ev_revenue":  num("enterpriseToRevenue"),
                },
                "profitability": {
                    "roe_pct":          pct("returnOnEquity"),
                    "roa_pct":          pct("returnOnAssets"),
                    "gross_margin_pct": pct("grossMargins"),
                    "op_margin_pct":    pct("operatingMargins"),
                    "net_margin_pct":   pct("profitMargins"),
                },
                "leverage": {
                    "current_ratio":  num("currentRatio"),
                    "quick_ratio":    num("quickRatio"),
                    "debt_to_equity": debt_to_eq,  # already in × (divided by 100)
                },
                "growth": {
                    "revenue_yoy_pct":     revenue_yoy,
                    "revenue_growth_3y":   pct("revenueGrowth"),
                    "earnings_growth_yoy": pct("earningsGrowth"),
                    "earnings_quarterly":  pct("earningsQuarterlyGrowth"),
                },
                "cash_flow": {
                    "fcf_yield_pct":    fcf_yield,
                    "free_cash_flow_M": _millions(fcf),
                    "operating_cf_M":   _millions(info.get("operatingCashflow")),
                },
                "income": {
                    "eps_ttm":     num("trailingEps"),
                    "eps_forward": num("forwardEps"),
                },
                "dividend": {
                    "yield_pct":    pct("dividendYield"),
                    "payout_ratio": pct("payoutRatio"),
                    "annual_div":   _safe_float(info.get("dividendRate")),
                },
            }
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "ticker": ticker})

    return _cached(f"ratios:{ticker.upper()}", fetch)


# ── LangChain Q&A wrappers (cache-only, used by RouterAgent) ─────────────────

@lc_tool
def query_income_statement(ticker: str) -> str:
    """Get cached income statement for a previously analyzed stock. Returns revenue, profit, and margin data."""
    return _cached(f"income:{ticker.upper()}", lambda: "", cache_only=True)


@lc_tool
def query_balance_sheet(ticker: str) -> str:
    """Get cached balance sheet for a previously analyzed stock. Returns assets, liabilities, equity, and debt."""
    return _cached(f"balance:{ticker.upper()}", lambda: "", cache_only=True)


@lc_tool
def query_cash_flow(ticker: str) -> str:
    """Get cached cash flow data for a previously analyzed stock. Returns operating cash flow, capex, and FCF."""
    return _cached(f"cashflow:{ticker.upper()}", lambda: "", cache_only=True)


@lc_tool
def query_key_ratios(ticker: str) -> str:
    """Get cached key financial ratios for a previously analyzed stock. Returns P/E, ROE, margins, growth rates, and valuation multiples."""
    return _cached(f"ratios:{ticker.upper()}", lambda: "", cache_only=True)
