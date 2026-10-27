import asyncio
import io
import json
import contextlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from guardrail import validate_and_resolve
from agents.orchestrator import orchestrator

app = FastAPI()


class AnalyzeRequest(BaseModel):
    query: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# Phase labels emitted at these elapsed-second thresholds after phase 1
PHASE_SCHEDULE = [
    (35, 2, "Running valuation analysis…"),
    (55, 3, "Writing investment report…"),
]


def _run_orchestrator(prompt: str) -> str:
    """Run orchestrator in a thread with its own event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = orchestrator(prompt)
        return str(result)
    finally:
        loop.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    check = validate_and_resolve(req.query)
    if not check["ok"]:
        raise HTTPException(status_code=422, detail=check["reply"])

    ticker = check["ticker"]
    name = check["name"]
    prompt = (
        f"Perform a complete fundamental analysis for the stock ticker: {ticker}\n"
        f"Follow the full workflow: gather financial data, news sentiment, and risk "
        f"assessment, then run valuation, and finally produce the investment brief."
    )

    async def event_stream():
        loop = asyncio.get_event_loop()
        task = loop.run_in_executor(None, _run_orchestrator, prompt)

        yield sse({"event": "phase", "phase": 1, "label": "Gathering market data…"})

        start = loop.time()
        next_phase = 0

        while not task.done():
            await asyncio.sleep(3)
            elapsed = loop.time() - start

            while next_phase < len(PHASE_SCHEDULE) and elapsed >= PHASE_SCHEDULE[next_phase][0]:
                _, phase_num, label = PHASE_SCHEDULE[next_phase]
                yield sse({"event": "phase", "phase": phase_num, "label": label})
                next_phase += 1

            # Keepalive comment to prevent proxy timeouts
            yield ": keepalive\n\n"

        try:
            report = await task
            yield sse({"event": "result", "ticker": ticker, "name": name, "report": report})
        except Exception:
            yield sse({"event": "error", "message": "Analysis failed."})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
