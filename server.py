import asyncio
import io
import json
import contextlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agents.router_agent import make_router_agent
from agents.orchestrator import make_orchestrator
from session import load_history, save_history

app = FastAPI()


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
    """Run a fresh orchestrator instance in a thread with its own event loop."""
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    history = await load_history(req.session_id)
    router, analysis_slot = make_router_agent(history)

    async def event_stream():
        loop = asyncio.get_event_loop()

        # Run the router agent (fast: intent + optional search_ticker call)
        router_result = await loop.run_in_executor(None, lambda: router(req.message))

        # Persist updated conversation (router's full message list includes tool turns)
        updated_history = list(router.messages)

        if analysis_slot:
            # Router triggered an analysis — save history then stream phases
            await save_history(req.session_id, updated_history)

            ticker = analysis_slot["ticker"]
            name = analysis_slot["name"]
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
                yield sse({"event": "error", "message": "Analysis failed. Please try again."})

        else:
            # Conversational reply — save history and return message
            await save_history(req.session_id, updated_history)
            yield sse({"event": "message", "text": str(router_result)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
