import asyncio
import contextlib
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from langgraph.checkpoint.postgres import PostgresSaver

from agents.router_agent import init_router, run_router, inject_report
from agents.orchestrator import make_orchestrator
from agents.report_writer_agent import make_report_writer_agent
from errors import AppError


@asynccontextmanager
async def lifespan(_: FastAPI):
    with PostgresSaver.from_conn_string(os.getenv("DATABASE_URL")) as checkpointer:
        checkpointer.setup()
        init_router(checkpointer)
        yield


app = FastAPI(lifespan=lifespan)


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


# Time-based phase labels while orchestrator (phases 1-2) runs.
# Phase 3 (report writing) is triggered explicitly after orchestrator finishes.
PHASE_SCHEDULE = [
    (35, 2, "Running valuation analysis…"),
]


def _run_orchestrator(prompt: str) -> str:
    """Run orchestrator phases 1-4 in a thread. Returns labeled combined analysis text."""
    thread_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(thread_loop)
    buf = io.StringIO()
    try:
        orchestrator = make_orchestrator()
        with contextlib.redirect_stdout(buf):
            result = orchestrator(prompt)
        return str(result)
    finally:
        thread_loop.close()


def _run_report_writer_streaming(
    combined_context: str,
    token_queue: asyncio.Queue,
    main_loop: asyncio.AbstractEventLoop,
) -> str:
    """Run the report writer in a thread, pushing each token to token_queue on the main loop."""
    thread_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(thread_loop)

    def callback(**kwargs):
        data = kwargs.get("data", "")
        if data:
            main_loop.call_soon_threadsafe(token_queue.put_nowait, data)

    try:
        agent = make_report_writer_agent(callback_handler=callback)
        result = agent(combined_context)
        return str(result)
    finally:
        main_loop.call_soon_threadsafe(token_queue.put_nowait, None)  # sentinel
        thread_loop.close()


async def _stream_analysis(ticker: str, name: str, session_id: str):
    """Async generator — runs orchestrator then streams report writer tokens as SSE."""
    loop = asyncio.get_event_loop()
    prompt = (
        f"Perform a complete fundamental analysis for the stock ticker: {ticker}\n"
        f"Follow the full workflow: gather financial data, news sentiment, and risk "
        f"assessment in parallel, then run valuation analysis."
    )

    # ── Phases 1 & 2: data gathering + valuation ─────────────────────────────
    yield sse({"event": "phase", "phase": 1, "label": "Gathering market data…"})

    orch_task = loop.run_in_executor(None, _run_orchestrator, prompt)

    start = loop.time()
    next_phase = 0
    while not orch_task.done():
        await asyncio.sleep(3)
        elapsed = loop.time() - start
        while next_phase < len(PHASE_SCHEDULE) and elapsed >= PHASE_SCHEDULE[next_phase][0]:
            _, phase_num, label = PHASE_SCHEDULE[next_phase]
            yield sse({"event": "phase", "phase": phase_num, "label": label})
            next_phase += 1
        yield ": keepalive\n\n"

    try:
        combined_context = await orch_task
    except Exception:
        yield sse({"event": "error", **AppError("AGNT003").json})
        return

    # ── Phase 3: streaming report writer ─────────────────────────────────────
    yield sse({"event": "phase", "phase": 3, "label": "Writing investment report…"})
    yield sse({"event": "writing_start", "ticker": ticker, "name": name})

    token_queue: asyncio.Queue = asyncio.Queue()
    report_task = loop.run_in_executor(
        None, _run_report_writer_streaming, combined_context, token_queue, loop
    )

    full_report = ""
    while True:
        token = await token_queue.get()
        if token is None:  # sentinel — report writer finished
            break
        full_report += token
        yield sse({"event": "token", "text": token})

    try:
        await report_task
        inject_report(session_id=session_id, ticker=ticker, name=name, report=full_report)
        yield sse({"event": "result", "ticker": ticker, "name": name, "report": full_report})
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
            async for chunk in _stream_analysis(slot["ticker"], slot["name"], req.session_id):
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
