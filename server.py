import asyncio
import io
import json
import contextlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from agents.router_agent import run_router
from agents.orchestrator import make_orchestrator
from errors import AppError

app = FastAPI()


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.json)


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=AppError("GEN001").json)


class ChatRequest(BaseModel):
    message: str
    session_id: str


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


PHASE_SCHEDULE = [
    (35, 2, "Running valuation analysis…"),
    (55, 3, "Writing investment report…"),
]


def _run_orchestrator(prompt: str) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    buf = io.StringIO()
    try:
        orchestrator = make_orchestrator()
        with contextlib.redirect_stdout(buf):
            result = orchestrator(prompt)
        return str(result)
    finally:
        loop.close()


async def _stream_analysis(ticker: str, name: str):
    """Async generator that runs the orchestrator and yields SSE strings."""
    loop = asyncio.get_event_loop()
    prompt = (
        f"Perform a complete fundamental analysis for the stock ticker: {ticker}\n"
        f"Follow the full workflow: gather financial data, news sentiment, and risk "
        f"assessment, then run valuation, and finally produce the investment brief."
    )
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
        yield ": keepalive\n\n"

    try:
        report = await task
        yield sse({"event": "result", "ticker": ticker, "name": name, "report": report})
    except Exception:
        yield sse({"event": "error", **AppError("AGNT003").json})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: run_router(req.session_id, req.message)
            )
        except Exception:
            yield sse({"event": "error", **AppError("GEN001").json})
            return

        if result["analysis_slot"]:
            slot = result["analysis_slot"]
            async for chunk in _stream_analysis(slot["ticker"], slot["name"]):
                yield chunk
        elif result["pending_slot"]:
            slot = result["pending_slot"]
            yield sse({
                "event": "message",
                "text": f"Did you mean {slot['name']} ({slot['ticker']})? Reply yes to run a fundamental analysis.",
            })
        else:
            yield sse({"event": "message", "text": result["text"]})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
