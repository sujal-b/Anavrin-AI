"""
ML Customer Support Chatbot - FastAPI Backend
Production-quality implementation with proper error handling,
LLM integration, and streaming support.
"""

import re
import json
import os
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chatbot")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
    FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

    # LLM Configuration
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")  # "nvidia", "openrouter", or "mock"
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")

    # Model priority (best first)
    MODEL_PRIORITY = ["naive_bayes", "knn", "random_forest"]

    # Text preprocessing
    MAX_TEXT_LENGTH = 500
    CLEAN_PATTERN = re.compile(r"[^\w\s]")


config = Config()


# ─────────────────────────────────────────────────────────────
# STATE (thread-safe for single-worker, sufficient for demo)
# ─────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.classifier = None
        self.tfidf = None
        self.intent_to_response: dict[str, str] = {}
        self.intent_to_category: dict[str, str] = {}
        self.metrics: dict = {}
        self.model_name: str = "none"
        self.loaded: bool = False
        self.load_time: float = 0.0

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "model": self.model_name,
            "intents": len(self.intent_to_response),
            "load_time_ms": round(self.load_time * 1000, 1),
        }


state = AppState()


# ─────────────────────────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalize user input for TF-IDF vectorization."""
    text = text.lower().strip()
    text = config.CLEAN_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text)  # collapse whitespace
    return text[: config.MAX_TEXT_LENGTH]  # cap length


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_models() -> None:
    """Load trained ML models and intent mappings."""
    start = time.time()

    # Load TF-IDF
    for ext in ["joblib", "pkl"]:
        path = os.path.join(config.MODELS_DIR, f"tfidf.{ext}")
        if os.path.exists(path):
            state.tfidf = joblib.load(path)
            logger.info(f"Loaded TF-IDF vectorizer ({ext})")
            break

    if state.tfidf is None:
        raise FileNotFoundError("TF-IDF vectorizer not found in models/")

    # Load classifier (best first)
    for name in config.MODEL_PRIORITY:
        for ext in ["joblib", "pkl"]:
            path = os.path.join(config.MODELS_DIR, f"{name}.{ext}")
            if os.path.exists(path):
                state.classifier = joblib.load(path)
                state.model_name = name
                logger.info(f"Loaded classifier: {name} ({ext})")
                break
        if state.classifier is not None:
            break

    if state.classifier is None:
        raise FileNotFoundError("No classifier found in models/")

    # Load intent → response mapping
    dataset_path = os.path.join(config.DATA_DIR, "processed_dataset.pkl")
    if os.path.exists(dataset_path):
        df = pd.read_pickle(dataset_path)
        # Use the most common response per intent
        state.intent_to_response = (
            df.groupby("intent")["response"]
            .agg(lambda x: x.value_counts().index[0])
            .to_dict()
        )
        state.intent_to_category = df.drop_duplicates("intent").set_index("intent")["category"].to_dict()
        logger.info(f"Loaded {len(state.intent_to_response)} intent-response mappings")

    # Load evaluation metrics
    metrics_path = os.path.join(config.EVAL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            state.metrics = json.load(f)
        logger.info("Loaded evaluation metrics")

    state.load_time = time.time() - start
    state.loaded = True
    logger.info(f"All models loaded in {state.load_time:.2f}s")


# ─────────────────────────────────────────────────────────────
# LLM CLIENT
# �────────────────────────────────────────────────────────────
async def generate_llm_response(
    user_message: str,
    intent: str,
    context: str = "",
    system_prompt: str = "",
) -> str:
    """Generate response using LLM API with fallback to template."""

    if config.LLM_PROVIDER == "mock" or not config.LLM_API_KEY:
        return state.intent_to_response.get(
            intent,
            "I'd be happy to help you with that. Could you provide more details?",
        )

    default_system = f"""You are a helpful e-commerce customer support agent.
The user's intent has been classified as: {intent}.
Relevant context from knowledge base: {context}
Provide a helpful, concise response. Be specific and actionable."""

    messages = [
        {"role": "system", "content": system_prompt or default_system},
        {"role": "user", "content": user_message},
    ]

    try:
        import httpx

        if config.LLM_PROVIDER == "nvidia":
            url = "https://integrate.api.nvidia.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
            payload = {
                "model": config.LLM_MODEL,
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }
        elif config.LLM_PROVIDER == "openrouter":
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {config.LLM_API_KEY}",
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "ML Chatbot",
            }
            payload = {
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": messages,
                "max_tokens": 512,
                "temperature": 0.3,
            }
        else:
            return state.intent_to_response.get(intent, "I can help with that.")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    except Exception as e:
        logger.warning(f"LLM call failed: {e}. Using template response.")
        return state.intent_to_response.get(
            intent, "I'd be happy to help. Could you provide more details?"
        )


# ─────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500, description="User message")

    @validator("message")
    def validate_message(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        return v


class ChatResponse(BaseModel):
    user_message: str
    intent: str
    category: str
    confidence: float
    response: str
    model_used: str
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    intents_count: int
    load_time_ms: float
    timestamp: str


# ─────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    try:
        load_models()
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
    yield


app = FastAPI(
    title="ML Customer Support Chatbot",
    description="E-commerce support chatbot with ML intent classification and RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if state.loaded else "degraded",
        model_loaded=state.loaded,
        model_name=state.model_name,
        intents_count=len(state.intent_to_response),
        load_time_ms=round(state.load_time * 1000, 1),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user message through ML pipeline."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    start = time.time()

    # 1. Preprocess
    cleaned = preprocess_text(request.message)

    # 2. Vectorize
    vectorized = state.tfidf.transform([cleaned])

    # 3. Predict intent
    intent = state.classifier.predict(vectorized)[0]

    # 4. Get confidence
    confidence = 0.0
    if hasattr(state.classifier, "predict_proba"):
        proba = state.classifier.predict_proba(vectorized)
        confidence = float(proba.max())

    # 5. Get category
    category = state.intent_to_category.get(intent, "unknown")

    # 6. Generate response (LLM or template)
    context = state.intent_to_response.get(intent, "")
    response_text = await generate_llm_response(
        user_message=request.message,
        intent=intent,
        context=context,
    )

    elapsed = time.time() - start
    logger.info(
        f"Chat processed: intent={intent}, confidence={confidence:.3f}, "
        f"time={elapsed:.3f}s"
    )

    return ChatResponse(
        user_message=request.message,
        intent=intent,
        category=category,
        confidence=round(confidence, 4),
        response=response_text,
        model_used=state.model_name,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/api/metrics")
async def get_metrics():
    """Return model evaluation metrics."""
    if not state.metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    return state.metrics


@app.get("/api/intents")
async def get_intents():
    """Return list of supported intents."""
    return {
        "intents": list(state.intent_to_response.keys()),
        "count": len(state.intent_to_response),
    }


# Serve frontend
if os.path.exists(config.FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_frontend():
        index_path = os.path.join(config.FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return HTMLResponse(content=f.read())
        return HTMLResponse(content="<h1>Frontend not found</h1>")


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
