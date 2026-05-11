import os
import time
import uuid
import logging
from datetime import datetime
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from backend.config.settings import config
from backend.chatbot.intent_classifier import intent_classifier
from backend.chatbot.rag_engine import rag_engine
from backend.chatbot.llm_client import llm_client
from backend.api.schemas import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("anavrin")

MAX_SESSIONS = 100
sessions: Dict[str, List[dict]] = {}
_startup_time_ms: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Anavrin AI Backend...")
    start = time.time()

    import joblib
    tfidf = None
    for ext in ["joblib", "pkl"]:
        tfidf_path = os.path.join(config.MODELS_DIR, f"tfidf.{ext}")
        if os.path.exists(tfidf_path):
            tfidf = joblib.load(tfidf_path)
            break

    intent_classifier.load(tfidf=tfidf)
    rag_engine.load(tfidf=tfidf)

    elapsed = time.time() - start
    global _startup_time_ms
    _startup_time_ms = round(elapsed * 1000, 1)
    logger.info(f"Backend ready in {elapsed:.2f}s")
    yield


app = FastAPI(
    title="Anavrin AI",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if intent_classifier.is_loaded else "degraded",
        model_loaded=intent_classifier.is_loaded,
        model_name=intent_classifier.model_name,
        intents_count=len(intent_classifier.classes),
        load_time_ms=_startup_time_ms,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not intent_classifier.is_loaded:
        raise HTTPException(status_code=503, detail="System initializing")

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = next(iter(sessions))
            del sessions[oldest]
        sessions[session_id] = []

    sessions[session_id].append({"role": "user", "content": req.message})

    intent, confidence, all_probs = intent_classifier.predict(req.message)
    category = rag_engine.intent_category.get(intent, "unknown")
    context, fallback = rag_engine.get_combined_context(
        intent=intent,
        query=req.message,
        classifier_confidence=confidence,
    )

    history = sessions[session_id][-10:]
    response_text = await llm_client.generate(
        user_message=req.message,
        intent=intent,
        confidence=confidence,
        context=context,
        history=history,
        fallback_response=fallback,
    )

    sessions[session_id].append({"role": "assistant", "content": response_text})

    return ChatResponse(
        user_message=req.message,
        intent=intent,
        category=category,
        confidence=round(confidence, 4),
        response=response_text,
        session_id=session_id,
        model_used=intent_classifier.model_name,
        timestamp=datetime.now().isoformat(),
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared"}


@app.get("/api/metrics")
async def get_metrics():
    try:
        import json
        metrics_path = os.path.join(config.EVAL_DIR, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


if os.path.exists(config.FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        index_path = os.path.join(config.FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse("<h1>Frontend not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
