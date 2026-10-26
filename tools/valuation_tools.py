import json
import os

import yfinance as yf
import requests
from strands import tool

from .utils import _safe_float, _millions, make_cached

_cached = make_cached(ttl=86_400)


def _load_dcf_inputs(ticker: str):
    """Pull FCF, net debt, shares outstanding, and current price from yfinance."""
    stock = yf.Ticker(ticker)
    info  = stock.info

    fcf    = _safe_float(info.get("freeCashflow"))
    debt   = _safe_float(info.get("totalDebt", 0) or 0)
    cash   = _safe_float(info.get("totalCash", 0) or 0)
    shares = _safe_float(info.get("sharesOutstanding"))
    price  = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    if not all([fcf, shares, price]):
        raise ValueError(f"Missing required DCF inputs for {ticker}: fcf={fcf}, shares={shares}, price={price}")

    net_debt = (debt or 0) - (cash or 0)
    return fcf, net_debt, shares, price


def _dcf_verdict(upside: float) -> str:
    if upside > 30:   return "SIGNIFICANTLY UNDERVALUED"
    if upside > 15:   return "SLIGHTLY UNDERVALUED"
    if upside >= -15: return "FAIRLY VALUED"
    if upside >= -30: return "SLIGHTLY OVERVALUED"
    return "SIGNIFICANTLY OVERVALUED"


@tool
def run_dcf_model(ticker: str, growth_rate: float, discount_rate: float,
                  terminal_growth_rate: float) -> str:
    """Estimate intrinsic value per share with a 5-year DCF and compare it to the current price.

    Uses free cash flow projected forward 5 years, then a Gordon Growth terminal value.
    Returns intrinsic value, current price, upside/downside %, and a valuation verdict.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
        growth_rate: Expected annual FCF growth rate as a decimal (e.g. 0.08 for 8%)
        discount_rate: WACC / required return as a decimal (e.g. 0.10 for 10%). Must exceed terminal_growth_rate.
        terminal_growth_rate: Perpetual growth rate after year 5 (e.g. 0.03). Keep below 3% — never above discount_rate.
    """
    try:
        fcf, net_debt, shares, price = _load_dcf_inputs(ticker.upper())

        if discount_rate <= terminal_growth_rate:
            return json.dumps({"error": "discount_rate must exceed terminal_growth_rate to avoid division by zero"})

        # Step 1 — project FCF for 5 years
        fcfs = [fcf * (1 + growth_rate) ** t for t in range(1, 6)]

        # Step 2 — discount each year's FCF back to today
        pv_fcfs = [f / (1 + discount_rate) ** t for t, f in enumerate(fcfs, start=1)]

        # Step 3 — terminal value (Gordon growth), discounted back from year 5
        terminal_value = fcfs[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
        pv_terminal    = terminal_value / (1 + discount_rate) ** 5

        # Step 4 — enterprise value → equity value → per share
        enterprise_value = sum(pv_fcfs) + pv_terminal
        equity_value     = enterprise_value - net_debt
        intrinsic_value  = equity_value / shares

        upside = (intrinsic_value / price - 1) * 100

        return json.dumps({
            "intrinsic_value": round(intrinsic_value, 2),
            "current_price":   price,
            "upside_pct":      round(upside, 1),
            "verdict":         _dcf_verdict(upside),
            "inputs": {
                "fcf_M":               _millions(fcf),
                "net_debt_M":          _millions(net_debt),
                "shares_M":            _millions(shares),
                "growth_rate":         growth_rate,
                "discount_rate":       discount_rate,
                "terminal_growth_rate": terminal_growth_rate,
            },
        })
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_peer_multiples(ticker: str) -> str:
    """Compare a stock's valuation multiples against its sector peers.

    Fetches P/E, P/B, EV/EBITDA, P/S, forward P/E, and ROE for the company and
    a set of sector peers. Returns averages, premiums/discounts, and a valuation verdict.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
    """
    def fetch():
        try:
            ticker_upper = ticker.upper()
            peers        = _get_peers(ticker_upper)

            def get_multiples(t: str) -> dict:
                try:
                    i = yf.Ticker(t).info
                    return {
                        "ticker":      t,
                        "name":        i.get("shortName", t),
                        "pe":          _safe_float(i.get("trailingPE")),
                        "forward_pe":  _safe_float(i.get("forwardPE")),
                        "pb":          _safe_float(i.get("priceToBook")),
                        "ps":          _safe_float(i.get("priceToSalesTrailing12Months")),
                        "ev_ebitda":   _safe_float(i.get("enterpriseToEbitda")),
                        "roe_pct":     round(i.get("returnOnEquity", 0) * 100, 2) if i.get("returnOnEquity") else None,
                    }
                except Exception:
                    return {"ticker": t, "pe": None, "forward_pe": None, "pb": None,
                            "ps": None, "ev_ebitda": None, "roe_pct": None}

            subject   = get_multiples(ticker_upper)
            peer_data = [get_multiples(p) for p in peers[:6]]

            def avg(key: str):
                vals = [p[key] for p in peer_data if p.get(key) is not None]
                return round(sum(vals) / len(vals), 2) if vals else None

            peer_avg = {
                "pe":         avg("pe"),
                "forward_pe": avg("forward_pe"),
                "pb":         avg("pb"),
                "ps":         avg("ps"),
                "ev_ebitda":  avg("ev_ebitda"),
                "roe_pct":    avg("roe_pct"),
            }

            def premium(subject_val, peer_val) -> str | None:
                if subject_val is None or peer_val is None or peer_val == 0:
                    return None
                pct = round(((subject_val - peer_val) / peer_val) * 100, 1)
                return f"{'+' if pct >= 0 else ''}{pct}%"

            verdict, weighted_avg = _peer_verdict(subject, peer_avg)

            result = {
                "ticker":       ticker_upper,
                "subject":      subject,
                "peers":        peer_data,
                "peer_avg":     peer_avg,
                "vs_peers": {
                    "pe":        premium(subject.get("pe"),        peer_avg.get("pe")),
                    "forward_pe":premium(subject.get("forward_pe"),peer_avg.get("forward_pe")),
                    "pb":        premium(subject.get("pb"),        peer_avg.get("pb")),
                    "ps":        premium(subject.get("ps"),        peer_avg.get("ps")),
                    "ev_ebitda": premium(subject.get("ev_ebitda"), peer_avg.get("ev_ebitda")),
                },
                "weighted_premium_pct": weighted_avg,
                "verdict": verdict,
            }
            return json.dumps(result, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "ticker": ticker})

    return _cached(f"peers:{ticker.upper()}", fetch)


_WEIGHTS = {"forward_pe": 0.30, "ev_ebitda": 0.25, "pe": 0.20, "ps": 0.15, "pb": 0.10}


def _peer_verdict(subject: dict, peer_avg: dict) -> tuple[str, float | None]:
    """Weighted-average premium vs peers → valuation verdict."""
    total, weight_used = 0.0, 0.0
    for key, w in _WEIGHTS.items():
        s, p = subject.get(key), peer_avg.get(key)
        if s is None or not p:      # missing on either side → skip
            continue
        total       += w * (s / p - 1) * 100   # premium of this multiple × its weight
        weight_used += w

    if weight_used == 0:
        return "N/A", None

    weighted_avg = round(total / weight_used, 1)    # re-scale so skipped weights don't count

    if weighted_avg > 30:    verdict = "SIGNIFICANTLY OVERVALUED"
    elif weighted_avg > 15:  verdict = "SLIGHTLY OVERVALUED"
    elif weighted_avg >= -15: verdict = "FAIRLY VALUED"
    elif weighted_avg >= -30: verdict = "SLIGHTLY UNDERVALUED"
    else:                     verdict = "SIGNIFICANTLY UNDERVALUED"

    return verdict, weighted_avg


def _get_peers(ticker: str) -> list[str]:
    """Try FMP peer endpoint first; fall back to hardcoded sector peers."""
    fmp_key = os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    if fmp_key:
        try:
            url  = f"https://financialmodelingprep.com/api/v4/stock_peers?symbol={ticker}&apikey={fmp_key}"
            resp = requests.get(url, timeout=10)
            data = resp.json()
            if data and isinstance(data, list) and data[0].get("peersList"):
                return [p for p in data[0]["peersList"] if p != ticker]
        except Exception:
            pass

    # Fallback: sector peers for common tickers
    _SECTOR_PEERS: dict[str, list[str]] = {
        "AAPL":  ["MSFT", "GOOGL", "META", "AMZN", "NVDA"],
        "MSFT":  ["AAPL", "GOOGL", "AMZN", "CRM", "ORCL"],
        "GOOGL": ["META", "MSFT", "AMZN", "SNAP", "PINS"],
        "NVDA":  ["AMD", "INTC", "QCOM", "TSM", "AVGO"],
        "TSLA":  ["GM", "F", "RIVN", "NIO", "BMW.DE"],
        "AMZN":  ["WMT", "TGT", "SHOP", "BABA", "JD"],
        "META":  ["GOOGL", "SNAP", "PINS", "TWTR", "TTD"],
    }
    if ticker in _SECTOR_PEERS:
        return _SECTOR_PEERS[ticker]

    # Last resort: fetch sector from yfinance and pick generic peers
    try:
        sector = yf.Ticker(ticker).info.get("sector", "")
        _GENERIC: dict[str, list[str]] = {
            "Technology":          ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
            "Financial Services":  ["JPM", "BAC", "WFC", "GS", "MS"],
            "Healthcare":          ["JNJ", "PFE", "UNH", "ABBV", "MRK"],
            "Consumer Cyclical":   ["AMZN", "TSLA", "HD", "NKE", "MCD"],
            "Energy":              ["XOM", "CVX", "COP", "SLB", "EOG"],
            "Industrials":         ["HON", "UPS", "CAT", "DE", "LMT"],
        }
        return [p for p in _GENERIC.get(sector, ["SPY"]) if p != ticker]
    except Exception:
        return ["SPY"]
