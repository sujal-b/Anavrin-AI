# ML Customer Support Chatbot — Simplified Implementation Plan

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [ML Pipeline](#ml-pipeline)
5. [Backend](#backend)
6. [Frontend](#frontend)
7. [Evaluation](#evaluation)
8. [Execution Timeline](#execution-timeline)
9. [File Structure](#file-structure)

---

## Project Overview

ML-powered e-commerce customer support chatbot that:
- Classifies user intents using trained ML models (Naive Bayes, KNN)
- Retrieves relevant knowledge from vector database (RAG)
- Generates responses using LLM API (Nvidia NIM / OpenRouter)
- Displays evaluation metrics and graphs

**Domain:** E-commerce (orders, refunds, shipping, payments)
**Deadline:** 1 week
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
[LLM Synthesis] → Nvidia NIM / OpenRouter API
    ↓
Response + Confidence Score
```

**Tech Stack:**
- Backend: FastAPI (Python)
- Frontend: HTML/CSS/JS (no React)
- ML: scikit-learn (Naive Bayes, KNN, Random Forest)
- Vector DB: FAISS (local)
- LLM: Nvidia NIM API or OpenRouter API
- Database: SQLite (chat history)
- Graphs: matplotlib, seaborn

---

## Dataset

**Source:** `bitext/Bitext-customer-support-llm-chatbot-training-dataset`

**Columns:**
- `instruction` — user question
- `response` — expected answer
- `intent` — 27 intent labels
- `category` — 10 categories (ORDER, REFUND, PAYMENT, etc.)

**Preprocessing:**
1. Load CSV from HuggingFace
2. Clean text (lowercase, remove special chars)
3. Split into train/test (80/20)
4. Vectorize instructions using TF-IDF
5. Store response chunks in FAISS index

---

## ML Pipeline

### Step 1: Data Loading
```python
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
```

### Step 2: Feature Extraction
- TF-IDF vectorization on `instruction` column
- Max features: 5000
- N-grams: (1,2)

### Step 3: Model Training
Train and compare:
1. **Naive Bayes** (MultinomialNB)
2. **K-Nearest Neighbors** (KNeighborsClassifier)
3. **Random Forest** (RandomForestClassifier) — bonus comparison

### Step 4: Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix
- Classification report

### Step 5: Intent Prediction
```python
def predict_intent(user_message):
    vectorized = tfidf.transform([user_message])
    intent = classifier.predict(vectorized)
    confidence = classifier.predict_proba(vectorized).max()
    return intent, confidence
```

---

## Backend

### File: `backend/main.py`

**Endpoints:**
- `POST /api/chat` — process user message, return response
- `GET /api/history` — get chat history
- `GET /api/metrics` — return model evaluation metrics
- `GET /api/health` — health check

**Pipeline Flow:**
1. Receive user message
2. Classify intent (ML model)
3. Retrieve relevant docs (FAISS)
4. Call LLM API with context
5. Save to SQLite
6. Return response

### LLM Integration
```python
import httpx

async def call_llm(prompt, context):
    # Option A: Nvidia NIM
    # Option B: OpenRouter
    response = await httpx.post(
        API_URL,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"messages": [{"role": "user", "content": prompt}]}
    )
    return response.json()["choices"][0]["message"]["content"]
```

---

## Frontend

### Simple HTML Chat Interface

**File: `frontend/index.html`**

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

### Required Metrics (pick one)
- **F1 Score** (recommended — balances precision/recall)
- OR **Recall**

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

### Kaggle Notebook Setup
1. Create new notebook on kaggle.com
2. Enable GPU: Settings → Accelerator → GPU T4 × 2
3. Add dataset: `bitext/Bitext-customer-support-llm-chatbot-training-dataset`
4. Upload `train_kaggle.py` to notebook
5. Run all cells
6. Download generated files:
   - `models/naive_bayes.pkl`
   - `models/knn.pkl`
   - `models/random_forest.pkl`
   - `evaluation/*.png`
   - `evaluation/metrics.json`
   - `data/processed_dataset.pkl`

### Kaggle Notebook Code
```python
# Cell 1: Install dependencies
!pip install faiss-cpu datasets

# Cell 2: Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import os

# Cell 3: Load dataset
from datasets import load_dataset
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
df = pd.DataFrame(dataset['train'])

# Cell 4: Preprocess
df['instruction_clean'] = df['instruction'].str.lower()
df['instruction_clean'] = df['instruction_clean'].str.replace(r'[^\w\s]', '', regex=True)

# Cell 5: Feature extraction
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['instruction_clean'])
y = df['intent']

# Cell 6: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Cell 7: Train models
models = {
    'naive_bayes': MultinomialNB(),
    'knn': KNeighborsClassifier(n_neighbors=5),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'accuracy': model.score(X_test, y_test),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"{name} F1 (macro): {results[name]['f1_macro']:.4f}")

# Cell 8: Save models
os.makedirs('models', exist_ok=True)
for name, model in models.items():
    with open(f'models/{name}.pkl', 'wb') as f:
        pickle.dump(model, f)
with open('models/tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Cell 9: Generate graphs
os.makedirs('evaluation', exist_ok=True)

# Individual F1 scores
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
f1_scores = [results[m]['f1_macro'] for m in model_names]
plt.bar(model_names, f1_scores, color=['#3498db', '#2ecc71', '#e74c3c'])
plt.title('Model Comparison - F1 Score (Macro)')
plt.ylabel('F1 Score')
plt.savefig('evaluation/individual_scores.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
intent_counts = df['intent'].value_counts()
top_intents = intent_counts.head(10).index
df_top = df[df['intent'].isin(top_intents)]
correlation_data = pd.crosstab(df_top['category'], df_top['intent'])
sns.heatmap(correlation_data, annot=True, fmt='d', cmap='YlOrRd')
plt.title('Intent-Category Correlation Matrix')
plt.savefig('evaluation/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix for best model
best_model_name = max(results, key=lambda x: results[x]['f1_macro'])
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=best_model.classes_[:10], 
            yticklabels=best_model.classes_[:10])
plt.title(f'Confusion Matrix - {best_model_name}')
plt.savefig('evaluation/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Cell 10: Save metrics
os.makedirs('evaluation', exist_ok=True)
with open('evaluation/metrics.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

# Cell 11: Save processed dataset
df.to_pickle('data/processed_dataset.pkl')

print("Done! Download models/ and evaluation/ folders.")
```

### Files to Download from Kaggle
```
models/
├── naive_bayes.pkl
├── knn.pkl
├── random_forest.pkl
└── tfidf.pkl

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
- [ ] Load trained models (.pkl files)
- [ ] Implement intent prediction endpoint
- [ ] Set up FAISS index with response chunks
- [ ] Implement RAG retrieval
- [ ] Integrate LLM API (Nvidia NIM / OpenRouter)
- [ ] Add SQLite for chat history

### Day 4: Frontend
- [ ] Create HTML chat interface
- [ ] Style with CSS
- [ ] Add JavaScript for API calls
- [ ] Display intent + confidence

### Day 5: Integration + Testing
- [ ] Connect frontend to backend
- [ ] Test full pipeline
- [ ] Fix bugs
- [ ] Create Flow diagram graph

### Day 6: Polish + Submit
- [ ] Final testing
- [ ] Generate all required graphs
- [ ] Write README
- [ ] Package for submission

---

## File Structure

```
ml-chatbot/
├── kaggle/
│   └── train_kaggle.py          # Upload to Kaggle notebook
├── data/
│   └── processed_dataset.pkl    # Download from Kaggle
├── models/
│   ├── naive_bayes.pkl          # Download from Kaggle
│   ├── knn.pkl                  # Download from Kaggle
│   ├── random_forest.pkl        # Download from Kaggle
│   └── tfidf.pkl                # Download from Kaggle
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── chatbot/
│       ├── intent_classifier.py
│       ├── rag_engine.py
│       └── llm_client.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
├── evaluation/
│   ├── metrics.json             # Download from Kaggle
│   ├── individual_scores.png    # Download from Kaggle
│   ├── correlation_matrix.png   # Download from Kaggle
│   ├── confusion_matrix.png     # Download from Kaggle
│   └── flow_diagram.png         # Create locally
├── requirements.txt
└── README.md
```

---

## Requirements

### Local Backend (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0
faiss-cpu==1.7.4
httpx==0.25.0
matplotlib==3.8.2
seaborn==0.13.0
sqlalchemy==2.0.23
pydantic==2.5.0
python-jose==3.3.0
```

### Kaggle Notebook (pre-installed)
- scikit-learn
- pandas
- matplotlib
- seaborn
- datasets (HuggingFace)
- faiss-cpu (install with !pip install)

---

## What Was Cut (and Why)

| Original Item | Status | Reason |
|---------------|--------|--------|
| React frontend | ❌ Removed | Overkill for demo. HTML simpler. |
| Docker | ❌ Removed | Not needed for local demo. |
| CI/CD | ❌ Removed | No deployment pipeline needed. |
| Kubernetes | ❌ Removed | No scaling needed. |
| Prometheus | ❌ Removed | No monitoring needed. |
| A/B testing | ❌ Removed | Single model comparison enough. |
| Human handoff | ❌ Removed | Not required for ML project. |
| JWT auth | ❌ Removed | Not core to ML demo. |
| Multiple vector DBs | ❌ Removed | FAISS only. Simple. |
| Pinecone | ❌ Removed | Local FAISS sufficient. |

---

## Evaluation Checklist

- [ ] Intent classifier trained (Naive Bayes + KNN)
- [ ] F1 score calculated
- [ ] Individual bar chart (per-class F1)
- [ ] Correlation matrix (feature correlations)
- [ ] Flow diagram (pipeline architecture)
- [ ] Model comparison table
- [ ] Working chat interface
- [ ] LLM integration working
- [ ] README written

---

**Last Updated:** May 2026
**Version:** 2.0 (Simplified for 1-week ML Mini Project)
