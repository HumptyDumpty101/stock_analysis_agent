import io
import contextlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from guardrail import validate_and_resolve
from agents.orchestrator import orchestrator

app = FastAPI()


class AnalyzeRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    check = validate_and_resolve(req.query)
    if not check["ok"]:
        raise HTTPException(status_code=422, detail=check["reply"])

    ticker = check["ticker"]
    prompt = (
        f"Perform a complete fundamental analysis for the stock ticker: {ticker}\n"
        f"Follow the full workflow: gather financial data, news sentiment, and risk "
        f"assessment, then run valuation, and finally produce the investment brief."
    )

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = orchestrator(prompt)

    report = str(result)
    return {
        "ticker": ticker,
        "name": check["name"],
        "report": report,
    }
