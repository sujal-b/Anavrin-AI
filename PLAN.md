# ML Customer Support Chatbot — Consolidated Implementation Plan

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [File Structure](#file-structure)
4. [Dataset](#dataset)
5. [ML Pipeline](#ml-pipeline)
6. [Backend](#backend)
7. [Session Management](#session-management)
8. [Frontend](#frontend)
9. [Evaluation](#evaluation)
10. [Execution Timeline](#execution-timeline)

---

## Project Overview

ML-powered e-commerce customer support chatbot that:
- Classifies user intents using trained ML models (Naive Bayes, KNN, Random Forest)
- Retrieves relevant knowledge from vector database (RAG via FAISS)
- Generates responses using LLM API (Nvidia NIM / OpenRouter)
- Maintains conversation context per session (last 10 messages)
- Displays evaluation metrics and graphs

**Domain:** E-commerce (orders, refunds, shipping, payments)
**Dataset:** bitext/Bitext-customer-support-llm-chatbot-training-dataset (26.9k rows)

---

## Architecture

```
User Input
    ↓
[Intent Classifier] → Naive Bayes / KNN (trained on bitext)
    ↓
[RAG Retrieval] → FAISS vector search (relevant Q&A pairs)
    ↓
[LLM Synthesis] → Nvidia NIM / OpenRouter API + session history
    ↓
Response + Confidence Score + Session Context
```

**Tech Stack:**
- Backend: FastAPI (Python)
- Frontend: HTML/CSS/JS (no React)
- ML: scikit-learn (Naive Bayes, KNN, Random Forest)
- Vector DB: FAISS (local)
- LLM: Nvidia NIM API or OpenRouter API
- Session: In-memory dict (no database)
- Graphs: matplotlib, seaborn

---

## File Structure

```
ml-chatbot/
├── kaggle/
│   └── train_kaggle.py              # Upload to Kaggle notebook
├── data/
│   └── processed_dataset.pkl        # Download from Kaggle
├── models/
│   ├── naive_bayes.joblib           # Download from Kaggle
│   ├── knn.joblib                   # Download from Kaggle
│   ├── random_forest.joblib         # Download from Kaggle
│   └── tfidf.joblib                 # Download from Kaggle
├── backend/
│   ├── main.py                      # FastAPI app, routes, session store
│   ├── config/
│   │   └── settings.py              # Config class (dirs, LLM keys, model priority)
│   ├── api/
│   │   └── schemas.py               # Pydantic models (ChatRequest, ChatResponse)
│   └── chatbot/
│       ├── __init__.py
│       ├── intent_classifier.py     # ML intent classification + confidence
│       ├── rag_engine.py            # FAISS vector search + fallback
│       └── llm_client.py            # LLM API calls + retry + fallback
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── evaluation/
│   ├── metrics.json                 # Download from Kaggle
│   ├── individual_scores.png        # Download from Kaggle
│   ├── correlation_matrix.png       # Download from Kaggle
│   ├── confusion_matrix.png         # Download from Kaggle
│   └── flow_diagram.png             # Create locally
├── requirements.txt
└── PLAN.md                          # This file
```

**Key:** `backend/chatbot/` modules are the canonical implementations. `backend/core/` is DELETED.

---

## Dataset

**Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`

**Columns:**
- `instruction` — user question
- `response` — expected answer
- `intent` — 27 intent labels
- `category` — 10 categories (ORDER, REFUND, PAYMENT, etc.)

**Preprocessing (Kaggle):**
1. Load CSV from HuggingFace
2. Clean text (lowercase, remove special chars)
3. Split into train/test (80/20)
4. Vectorize instructions using TF-IDF (max_features=5000, ngram_range=(1,2))
5. Store response chunks in FAISS index
6. Export: models/*.pkl, data/processed_dataset.pkl, evaluation/metrics.json

---

## ML Pipeline

### Step 1: Data Loading
```python
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(dataset['train'])
```

### Step 2: Preprocessing
```python
df['instruction_clean'] = df['instruction'].str.lower()
df['instruction_clean'] = df['instruction_clean'].str.replace(r'[^\w\s]', '', regex=True)
```

### Step 3: Feature Extraction
- TF-IDF vectorization on `instruction` column
- Max features: 5000
- N-grams: (1,2)

### Step 4: Model Training
Train and compare:
1. **Naive Bayes** (MultinomialNB)
2. **K-Nearest Neighbors** (KNeighborsClassifier, n_neighbors=5)
3. **Random Forest** (RandomForestClassifier, n_estimators=100)

### Step 5: Evaluation
- Accuracy, Precision, Recall, F1 Score (macro + weighted)
- Confusion matrix
- Classification report

### Step 6: Intent Prediction (at runtime)
```python
def predict_intent(text):
    cleaned = preprocess(text)          # lowercase, remove special chars
    vectorized = tfidf.transform([cleaned])
    intent = classifier.predict(vectorized)[0]
    confidence = classifier.predict_proba(vectorized).max()
    return intent, confidence
```

---

## Backend

### Module: `backend/chatbot/intent_classifier.py`

**Class:** `IntentClassifier`
- `load()` → loads TF-IDF + classifier from models/
- `preprocess(text)` → clean + normalize
- `predict(text)` → (intent, confidence, all_probabilities)
- `predict_batch(texts)` → [(intent, confidence), ...]
- `get_info()` → model metadata

**Singleton:** `intent_classifier = IntentClassifier()`

### Module: `backend/chatbot/rag_engine.py`

**Class:** `RAGEngine`
- `load()` → loads dataset + builds FAISS index
- `search(query, top_k=5)` → [{instruction, response, intent, score}]
- `get_context(intent, max_entries=3)` → context string for LLM
- `get_sample_response(intent)` → most common response

**Fallback:** Keyword search when FAISS unavailable.

**Singleton:** `rag_engine = RAGEngine()`

### Module: `backend/chatbot/llm_client.py`

**Class:** `LLMClient`
- `generate(user_message, intent, context, system_prompt, fallback_response)` → str
- Retry logic (max_retries=2, exponential backoff)
- Timeout handling (30s)
- Providers: nvidia, openrouter, mock
- Fallback to template responses on failure

**Singleton:** `llm_client = LLMClient(provider, api_key, model)`

### Module: `backend/config/settings.py`

```python
class Config:
    BASE_DIR = ...                    # Project root
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    EVAL_DIR = BASE_DIR / "evaluation"
    FRONTEND_DIR = BASE_DIR / "frontend"
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-8b-instruct")
    MODEL_PRIORITY = ["naive_bayes", "knn", "random_forest"]
    MAX_TEXT_LENGTH = 500

config = Config()
```

### Module: `backend/api/schemas.py`

```python
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    user_message: str
    intent: str
    category: str
    confidence: float
    response: str
    session_id: str
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    intents_count: int
    load_time_ms: float
    timestamp: str
```

### Module: `backend/main.py`

**Endpoints:**
- `POST /api/chat` — process message → classify → RAG → LLM → return
- `GET /api/health` — system status
- `GET /api/metrics` — model evaluation metrics
- `DELETE /api/session/{session_id}` — clear session
- `GET /` — serve frontend

**Startup:** Load all modules via `lifespan` context manager.

---

## Session Management

### Purpose
Retain conversation context across messages. When user says "my order ID is 12345" then "can you refund that order", AI understands "that order" = 12345.

### Flow
```
Frontend generates UUID → localStorage as "session_id"
    ↓
Each POST /api/chat includes { message, session_id }
    ↓
Backend resolves session (create new if missing)
    ↓
Append user message to session
    ↓
Classify intent + RAG retrieve
    ↓
Build LLM messages: system prompt + last 10 messages from session
    ↓
Call LLM with context array (not just current message)
    ↓
Append assistant response to session
    ↓
Return response + session_id
```

### Backend Implementation

**In `backend/main.py`:**

```python
# Global session store
sessions: Dict[str, List[dict]] = {}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Resolve session
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    # 2. Append user message
    sessions[session_id].append({"role": "user", "content": req.message})

    # 3. Classify intent + RAG
    intent, confidence, category = intent_classifier.predict(req.message)
    context = rag_engine.get_context(intent)

    # 4. Build messages for LLM with history
    history = sessions[session_id][-10:]  # Last 10 messages
    response_text = await llm_client.generate(
        user_message=req.message,
        intent=intent,
        context=context,
        system_prompt=f"You are Anavrin AI. Intent: {intent}. Context: {context}",
        fallback_response=rag_engine.get_sample_response(intent)
    )

    # 5. Append assistant response
    sessions[session_id].append({"role": "assistant", "content": response_text})

    # 6. Return with session_id
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
```

**Session cleanup:**
```python
MAX_SESSIONS = 100

# In chat endpoint, before creating new session:
if len(sessions) >= MAX_SESSIONS:
    oldest = next(iter(sessions))
    del sessions[oldest]
```

### Frontend Implementation

**In `frontend/script.js`:**

```javascript
// Session management
function getSessionId() {
    let sid = localStorage.getItem('chat_session_id');
    if (!sid) {
        sid = crypto.randomUUID();
        localStorage.setItem('chat_session_id', sid);
    }
    return sid;
}

let currentSessionId = getSessionId();

// In sendMessage:
const res = await fetch(`${API}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        message,
        session_id: currentSessionId
    })
});

const data = await res.json();
if (data.session_id) {
    currentSessionId = data.session_id;
    localStorage.setItem('chat_session_id', currentSessionId);
}
```

### Context Window Behavior

| Scenario | What happens |
|----------|-------------|
| Message 1-10 | All messages sent to LLM |
| Message 11+ | Only last 10 sent, oldest dropped |
| Page refresh | Session preserved via localStorage |
| Server restart | Session lost, new UUID generated |
| Multiple tabs | Each tab gets own session |

---

## Frontend

### HTML Chat Interface (`frontend/index.html`)

Features:
- Chat bubble UI
- Typing indicator
- Intent display (shows classified intent)
- Confidence score display
- Chat history sidebar

**No React. No build tools. Just HTML/CSS/JS.**

---

## Evaluation

### Required Graphs

1. **Individual Bar Chart** — per-class F1 scores
2. **Flow Diagram** — pipeline architecture (matplotlib diagram)
3. **Correlation Matrix** — feature correlations (seaborn heatmap)

### Required Metrics
- **F1 Score** (recommended — balances precision/recall)

### Model Comparison Table
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Naive Bayes | ? | ? | ? | ? |
| KNN | ? | ? | ? | ? |
| Random Forest | ? | ? | ? | ? |

---

## Kaggle GPU Training Workflow

### Why Kaggle?
- Free Tesla T4/P100 GPU
- Pre-installed libraries (sklearn, pandas, matplotlib, seaborn)
- 30 hours/week GPU runtime
- Jupyter notebook environment

### Workflow
```
KAGGLE NOTEBOOK                    LOCAL MACHINE
┌─────────────────┐               ┌─────────────────┐
│ 1. Load data    │               │                 │
│ 2. Train models │  download     │ 4. FastAPI      │
│ 3. Export .pkl  │ ───────────►  │ 5. FAISS index  │
│ 4. Save graphs  │   .pkl files  │ 6. HTML frontend│
└─────────────────┘               └─────────────────┘
```

### Files to Download from Kaggle
```
models/
├── naive_bayes.joblib
├── knn.joblib
├── random_forest.joblib
└── tfidf.joblib

evaluation/
├── metrics.json
├── individual_scores.png
├── correlation_matrix.png
└── confusion_matrix.png

data/
└── processed_dataset.pkl
```

---

## Execution Timeline

### Day 1-2: Kaggle GPU Training
- [ ] Create Kaggle notebook
- [ ] Enable GPU (T4)
- [ ] Upload train_kaggle.py
- [ ] Run training pipeline
- [ ] Generate all graphs
- [ ] Download .pkl files + graphs
- [ ] Verify model performance

### Day 3: Local Backend Setup
- [ ] Set up FastAPI project structure
- [ ] Implement `backend/config/settings.py`
- [ ] Implement `backend/api/schemas.py`
- [ ] Implement `backend/chatbot/intent_classifier.py`
- [ ] Implement `backend/chatbot/rag_engine.py`
- [ ] Implement `backend/chatbot/llm_client.py`
- [ ] Implement `backend/main.py` with session management
- [ ] Test all endpoints

### Day 4: Frontend
- [ ] Create HTML chat interface
- [ ] Style with CSS
- [ ] Add JavaScript for API calls + session management
- [ ] Display intent + confidence

### Day 5: Integration + Testing
- [ ] Connect frontend to backend
- [ ] Test full pipeline
- [ ] Test session persistence across page refresh
- [ ] Fix bugs
- [ ] Create Flow diagram graph

### Day 6: Polish + Submit
- [ ] Final testing
- [ ] Generate all required graphs
- [ ] Write README
- [ ] Package for submission

---

## Requirements

### Local Backend (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0
faiss-cpu==1.7.4
httpx==0.25.0
joblib==1.3.2
numpy==1.26.2
pandas==2.1.4
scikit-learn==1.3.2
pydantic==2.5.0
matplotlib==3.8.2
seaborn==0.13.0
```

### Kaggle Notebook (pre-installed)
- scikit-learn
- pandas
- matplotlib
- seaborn
- datasets (HuggingFace)
- faiss-cpu (install with `!pip install`)

---

## Implementation Rules

1. **Single source of truth:** `backend/chatbot/` modules only. No `backend/core/`.
2. **main.py imports from chatbot/:** `from backend.chatbot.intent_classifier import intent_classifier`
3. **Session management lives in main.py:** Not in chatbot/ modules.
4. **All LLM calls go through llm_client.py:** No direct httpx calls elsewhere.
5. **Config from settings.py:** No hardcoded paths or env reads scattered.

---

**Created:** May 2026
**Version:** 3.0 (Consolidated from v2.0 chatbot plan + session management plan)
