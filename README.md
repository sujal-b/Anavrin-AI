# Anavrin AI — ML Customer Support Chatbot

ML-powered e-commerce customer support chatbot with intent classification, RAG retrieval, and LLM response generation.

## Project Structure

```
├── backend/
│   ├── main.py                 # FastAPI app
│   └── chatbot/
│       ├── intent_classifier.py
│       ├── rag_engine.py
│       └── llm_client.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   └── assets/
├── kaggle/
│   └── train_kaggle.py         # Kaggle GPU training script
├── models/                     # ⚠️ NOT in git — regenerate via Kaggle
│   ├── naive_bayes.joblib
│   ├── knn.joblib
│   ├── random_forest.joblib
│   └── tfidf.joblib
├── data/                       # ⚠️ NOT in git — regenerate via Kaggle
│   └── processed_dataset.pkl
├── evaluation/                 # ⚠️ NOT in git — regenerate via Kaggle
│   ├── *.png
│   └── metrics.json
├── requirements.txt
├── chatml.bat                  # Shortcut: run `chatml` to start server
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
# Or use: chatml
```

Open http://localhost:8000

## Regenerating Models & Data

The `models/`, `data/`, and `evaluation/` folders are not in git (too large). To regenerate:

1. Go to [Kaggle](https://kaggle.com)
2. Create new notebook
3. Enable GPU (Settings → Accelerator → GPU T4)
4. Upload `kaggle/train_kaggle.py`
5. Run all cells
6. Download output files to project root

## Features

- Intent classification (Naive Bayes, KNN, Random Forest)
- FAISS vector similarity search (RAG)
- LLM integration (OpenRouter / Nvidia NIM API)
- Light theme responsive UI
- Real-time confidence scoring
