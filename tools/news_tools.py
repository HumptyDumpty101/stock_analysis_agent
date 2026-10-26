import json
import re
import os

import boto3
import requests
from strands import tool

from .utils import make_cached

_cached = make_cached(ttl=3_600)  # news expires in 1 h

_BEDROCK_MODEL  = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6")
_BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")


def _score_with_llm(texts: list[str]) -> list[tuple[str, float]]:
    """Batch-score headlines with Claude on Bedrock. One converse call for all headlines."""
    client   = boto3.client("bedrock-runtime", region_name=_BEDROCK_REGION)
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))

    resp = client.converse(
        modelId=_BEDROCK_MODEL,
        system=[{"text": (
            "You are a financial news sentiment classifier. "
            "For each numbered headline return exactly this JSON: "
            "{\"results\": [{\"label\": \"BULLISH\"|\"NEUTRAL\"|\"BEARISH\", \"score\": 0.0-1.0}, ...]} "
            "— one entry per headline, same order, no extra keys."
        )}],
        messages=[{"role": "user", "content": [{"text": numbered}]}],
        inferenceConfig={"temperature": 0, "maxTokens": 512},
    )

    text = resp["output"]["message"]["content"][0]["text"]
    # extract JSON even if Claude wraps it in markdown
    start = text.find("{")
    end   = text.rfind("}") + 1
    data  = json.loads(text[start:end])

    out = []
    for item in data.get("results", []):
        label = item.get("label", "NEUTRAL").upper()
        score = round(float(item.get("score", 0.5)), 3)
        out.append((label, score))
    return out


# ── Keyword fallback ──────────────────────────────────────────────────────────
_BULLISH_SIGNALS = [
    "beat", "beats", "record", "surge", "upgrade", "growth", "strong",
    "raises", "buyback", "outperform", "jumps", "rally", "gains", "boosts",
]
_BEARISH_SIGNALS = [
    "miss", "misses", "fall", "falls", "drop", "decline", "cut", "downgrade",
    "lawsuit", "probe", "investigation", "antitrust", "recall", "fine", "weak", "loss",
]


def _score_headline_keyword(text: str) -> tuple[str, float]:
    words = re.findall(r"[a-z]+", text.lower())
    bull  = sum(w in _BULLISH_SIGNALS for w in words)
    bear  = sum(w in _BEARISH_SIGNALS for w in words)
    if bull + bear == 0:
        return "NEUTRAL", 0.0
    net = (bull - bear) / (bull + bear)
    if net > 0.15:  return "BULLISH",  round(net, 2)
    if net < -0.15: return "BEARISH",  round(net, 2)
    return "NEUTRAL", round(net, 2)


def _score_headlines(texts: list[str]) -> list[tuple[str, float]]:
    """Try LLM scorer first; fall back to keyword scorer."""
    try:
        return _score_with_llm(texts)
    except Exception:
        return [_score_headline_keyword(t) for t in texts]


@tool
def search_recent_news(ticker: str, max_results: int = 10) -> str:
    """Fetch recent news articles and headlines for a given stock ticker.

    Uses Tavily search to retrieve the latest news. Returns titles, publishers,
    publication dates, article links, and a FinBERT sentiment score per article.
    Results are cached for 1 hour.

    Args:
        ticker: Stock ticker symbol e.g. AAPL, RELIANCE.NS
        max_results: Maximum number of articles to return (1-20). Default 10.
    """
    max_results = max(1, min(int(max_results), 20))     # clamp to 1..20

    def fetch():
        try:
            api_key = os.getenv("TAVILY_API_KEY", "")
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":      api_key,
                    "query":        f"{ticker.upper()} stock news earnings analyst",
                    "max_results":  max_results,
                    "search_depth": "basic",
                    "topic":        "news",
                },
                timeout=15,
            )
            resp.raise_for_status()
            raw = resp.json().get("results", [])

            # Build stubs, then batch-score with FinBERT
            stubs = []
            for item in raw:
                title          = item.get("title") or "No title"
                summary        = item.get("content") or ""
                first_sentence = summary.split(".")[0].strip() if summary else ""
                stubs.append({
                    "title":          title,
                    "publisher":      item.get("source") or "Unknown",
                    "published_at":   item.get("published_date") or "",
                    "url":            item.get("url") or "",
                    "first_sentence": first_sentence,
                    "scoring_text":   f"{title}. {first_sentence}",
                })

            scoring_texts = [s["scoring_text"] for s in stubs]
            scores        = _score_headlines(scoring_texts) if scoring_texts else []

            articles = []
            for stub, (label, score) in zip(stubs, scores):
                articles.append({
                    "title":           stub["title"],
                    "publisher":       stub["publisher"],
                    "published_at":    stub["published_at"],
                    "sentiment":       label,
                    "sentiment_score": score,
                    "url":             stub["url"],
                    "first_sentence":  stub["first_sentence"],
                })

            # Count labels for a quick summary
            counts = {"BULLISH": 0, "NEUTRAL": 0, "BEARISH": 0}
            for a in articles:
                counts[a["sentiment"]] = counts.get(a["sentiment"], 0) + 1

            return json.dumps({
                "ticker":        ticker.upper(),
                "article_count": len(articles),
                "sentiment_counts": counts,
                "articles":     articles,
            }, default=str)
        except Exception as e:
            return json.dumps({"error": str(e), "ticker": ticker})

    return _cached(f"news:{ticker.upper()}:{max_results}", fetch)


_HIGH_RISK_KEYWORDS = [
    "fraud", "investigation", "antitrust", "lawsuit",
    "recall", "scandal", "penalty", "fine",
]


@tool
def analyze_sentiment(news_json: str) -> str:
    """Perform a structured sentiment analysis on a set of news articles.

    Takes the JSON output from search_recent_news and produces a detailed
    sentiment report: an aggregate sentiment score, key bullish and bearish
    themes extracted from headlines, and an overall market narrative summary.

    Args:
        news_json: JSON string output from search_recent_news containing
                   a list of articles with titles and sentiment scores.
    """
    try:
        data     = json.loads(news_json)
        articles = data.get("articles", [])
        ticker   = data.get("ticker", "UNKNOWN")

        if not articles:
            return json.dumps({"error": "No articles to analyze", "ticker": ticker})

        # Convert LLM confidence scores to signed values: BEARISH → negative
        def signed(a):
            s = float(a.get("sentiment_score", 0.0))
            if a.get("sentiment") == "BEARISH":
                return -abs(s)
            if a.get("sentiment") == "BULLISH":
                return abs(s)
            return 0.0

        scores    = [signed(a) for a in articles]
        avg_score = round(sum(scores) / len(scores), 3) if scores else 0.0

        # Keyword theme extraction from all titles combined
        all_titles    = " ".join(a.get("title", "") for a in articles).lower()
        bullish_themes = [w for w in _BULLISH_SIGNALS if w in all_titles][:5]
        bearish_themes = [w for w in _BEARISH_SIGNALS if w in all_titles][:5]

        # Most extreme headlines
        sorted_articles = sorted(articles, key=signed, reverse=True)
        top_bullish     = [a for a in sorted_articles if a.get("sentiment") == "BULLISH"][:3]
        top_bearish     = [a for a in reversed(sorted_articles) if a.get("sentiment") == "BEARISH"][:3]

        # High-risk headline flags
        risk_flags = [
            a["title"] for a in articles
            if any(k in a.get("title", "").lower() for k in _HIGH_RISK_KEYWORDS)
        ]

        # Overall verdict
        if avg_score > 0.40:    overall = "STRONGLY BULLISH"
        elif avg_score > 0.15:  overall = "BULLISH"
        elif avg_score < -0.40: overall = "STRONGLY BEARISH"
        elif avg_score < -0.15: overall = "BEARISH"
        else:                   overall = "NEUTRAL"

        result = {
            "ticker":               ticker,
            "overall_sentiment":    overall,
            "aggregate_score":      avg_score,
            "interpretation":       f"Score of {avg_score:.3f} on a scale of -1 (very bearish) to +1 (very bullish)",
            "article_count":        len(articles),
            "sentiment_breakdown":  data.get("sentiment_counts", {}),
            "key_bullish_themes":   bullish_themes,
            "key_bearish_themes":   bearish_themes,
            "top_bullish_headlines": [h["title"] for h in top_bullish],
            "top_bearish_headlines": [h["title"] for h in top_bearish],
            "risk_flags":           risk_flags,
            "risk_flag_count":      len(risk_flags),
            "all_articles": [
                {
                    "title":          a["title"],
                    "publisher":      a["publisher"],
                    "published_at":   a["published_at"],
                    "sentiment":      a["sentiment"],
                    "url":            a["url"],
                    "first_sentence": a.get("first_sentence", ""),
                }
                for a in articles
            ],
        }
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
