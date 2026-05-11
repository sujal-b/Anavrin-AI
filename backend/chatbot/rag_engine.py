"""
RAG Engine - Retrieval-Augmented Generation with FAISS vector search.
Retrieves relevant knowledge base entries for context-augmented responses.
"""

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("chatbot.rag")


class RAGEngine:
    """Retrieval-Augmented Generation engine using FAISS for vector search."""

    def __init__(self, data_dir: str = "data", models_dir: str = "models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.dataset: Optional[pd.DataFrame] = None
        self.intent_responses: dict[str, list[str]] = {}
        self.intent_category: dict[str, str] = {}
        self.index = None
        self.tfidf = None
        self.is_loaded = False

    def load(self, tfidf=None) -> bool:
        """Load dataset and build FAISS index.
        
        Args:
            tfidf: Pre-loaded TF-IDF vectorizer. If None, loads from disk.
        """
        try:
            import joblib

            # Load processed dataset
            dataset_path = os.path.join(self.data_dir, "processed_dataset.pkl")
            if not os.path.exists(dataset_path):
                logger.warning(f"Dataset not found: {dataset_path}")
                return False

            self.dataset = pd.read_pickle(dataset_path)
            logger.info(f"Loaded dataset: {len(self.dataset)} rows")

            # Build intent → responses mapping (multiple responses per intent)
            self.intent_responses = (
                self.dataset.groupby("intent")["response"]
                .apply(list)
                .to_dict()
            )

            # Build intent → category mapping
            if "category" in self.dataset.columns:
                self.intent_category = (
                    self.dataset.drop_duplicates("intent")
                    .set_index("intent")["category"]
                    .to_dict()
                )

            # Use shared TF-IDF or load from disk
            if tfidf is not None:
                self.tfidf = tfidf
                logger.info("Using shared TF-IDF vectorizer")
            else:
                for ext in ["joblib", "pkl"]:
                    tfidf_path = os.path.join(self.models_dir, f"tfidf.{ext}")
                    if os.path.exists(tfidf_path):
                        self.tfidf = joblib.load(tfidf_path)
                        break

                if self.tfidf is None:
                    logger.warning("TF-IDF not found, vector search disabled")
                    return True  # Still usable without FAISS

            # Build FAISS index
            self._build_index()

            self.is_loaded = True
            logger.info("RAG engine loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load RAG engine: {e}")
            return False

    def _build_index(self) -> None:
        """Build FAISS index from instruction embeddings."""
        try:
            import faiss

            # Vectorize all instructions
            instructions = self.dataset["instruction"].tolist()
            vectors = self.tfidf.transform(instructions)

            # Convert to dense (FAISS requires dense vectors)
            if hasattr(vectors, "toarray"):
                dense_vectors = vectors.toarray().astype("float32")
            else:
                dense_vectors = np.array(vectors, dtype="float32")

            # Normalize for cosine similarity
            faiss.normalize_L2(dense_vectors)

            # Build index
            dimension = dense_vectors.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine after normalization
            self.index.add(dense_vectors)

            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={dimension}")

        except ImportError:
            logger.warning("FAISS not installed. Install with: pip install faiss-cpu")
            self.index = None
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {e}")
            self.index = None

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for relevant knowledge base entries.
        Returns list of {instruction, response, intent, score}.
        """
        if not self.is_loaded or self.index is None or self.tfidf is None:
            return self._fallback_search(query, top_k)

        try:
            # Vectorize query
            vectorized = self.tfidf.transform([query])

            if hasattr(vectorized, "toarray"):
                dense_vector = vectorized.toarray().astype("float32")
            else:
                dense_vector = np.array(vectorized, dtype="float32")

            # Normalize
            import faiss
            faiss.normalize_L2(dense_vector)

            # Search
            scores, indices = self.index.search(dense_vector, min(top_k, self.index.ntotal))

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:  # FAISS returns -1 for missing results
                    continue
                row = self.dataset.iloc[idx]
                results.append({
                    "instruction": row["instruction"],
                    "response": row["response"],
                    "intent": row["intent"],
                    "score": float(score),
                })

            return results

        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> list[dict]:
        """Fallback keyword search when FAISS is unavailable."""
        if self.dataset is None:
            return []

        query_lower = query.lower()
        results = []

        for _, row in self.dataset.iterrows():
            instruction = row["instruction"].lower()
            # Simple word overlap scoring
            query_words = set(query_lower.split())
            inst_words = set(instruction.split())
            overlap = len(query_words & inst_words)

            if overlap > 0:
                results.append({
                    "instruction": row["instruction"],
                    "response": row["response"],
                    "intent": row["intent"],
                    "score": overlap / len(query_words) if query_words else 0,
                })

        # Sort by score, return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_intent_context(self, intent: str, max_entries: int = 3) -> list[dict]:
        """Get intent-matched Q&A pairs from dataset."""
        if self.dataset is None or intent not in self.intent_responses:
            return []

        intent_rows = self.dataset[self.dataset["intent"] == intent].head(max_entries)
        return [
            {"instruction": row["instruction"], "response": row["response"]}
            for _, row in intent_rows.iterrows()
        ]

    def get_first_response(self, intent: str) -> str:
        """Get the first response for an intent (most frequent in dataset order)."""
        if intent in self.intent_responses and self.intent_responses[intent]:
            return self.intent_responses[intent][0]
        return ""

    def get_combined_context(
        self,
        intent: str,
        query: str,
        classifier_confidence: float,
        intent_max: int = 3,
        faiss_top_k: int = 5,
        confidence_threshold: float = 0.7,
        high_score_threshold: float = 0.9,
    ) -> tuple[str, str]:
        """
        Get combined context for LLM prompt + fallback response.

        Returns:
            (context_string, fallback_response)
        """
        intent_entries = self.get_intent_context(intent, intent_max)
        faiss_results = self.search(query, faiss_top_k) if self.is_loaded else []

        best_faiss_score = 0.0
        best_faiss_response = ""

        if classifier_confidence >= confidence_threshold:
            entries = intent_entries
        else:
            seen = {(e["instruction"], e["response"]) for e in intent_entries}
            for r in faiss_results:
                key = (r["instruction"], r["response"])
                if key not in seen:
                    intent_entries.append({"instruction": r["instruction"], "response": r["response"]})
                    seen.add(key)
            entries = intent_entries

        for r in faiss_results:
            if r["score"] > best_faiss_score:
                best_faiss_score = r["score"]
                best_faiss_response = r["response"]

        if best_faiss_score >= high_score_threshold and best_faiss_response:
            fallback = best_faiss_response
        else:
            fallback = self.get_first_response(intent)

        lines = []
        for i, e in enumerate(entries[: intent_max + faiss_top_k], 1):
            lines.append(f"{i}. [intent: {intent}] Q: {e['instruction']} -> A: {e['response']}")
        context = "\n".join(lines)

        return context, fallback


# Singleton instance
rag_engine = RAGEngine()
