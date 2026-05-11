"""
Intent Classifier - ML-based intent classification with confidence scoring.
Supports multiple model formats (joblib, pkl) with fallback handling.
"""

import os
import re
import logging
from typing import Optional, Tuple

import joblib
import numpy as np

from backend.config.settings import config

logger = logging.getLogger("chatbot.classifier")


class IntentClassifier:
    """ML intent classifier with preprocessing and confidence scoring."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.classifier = None
        self.tfidf = None
        self.model_name: str = "none"
        self.is_loaded = False
        self.classes: list[str] = []

        # Preprocessing pattern
        self._clean_pattern = re.compile(r"[^\w\s]")
        self._whitespace_pattern = re.compile(r"\s+")

    def load(self, tfidf=None) -> bool:
        """Load TF-IDF vectorizer and classifier.
        
        Args:
            tfidf: Pre-loaded TF-IDF vectorizer. If None, loads from disk.
        """
        try:
            if tfidf is not None:
                self.tfidf = tfidf
                logger.info("Using shared TF-IDF vectorizer")
            else:
                # Load TF-IDF
                for ext in ["joblib", "pkl"]:
                    path = os.path.join(self.models_dir, f"tfidf.{ext}")
                    if os.path.exists(path):
                        self.tfidf = joblib.load(path)
                        logger.info(f"Loaded TF-IDF ({ext})")
                        break

                if self.tfidf is None:
                    raise FileNotFoundError("TF-IDF vectorizer not found")

            # Load classifier (best first)
            for name in config.MODEL_PRIORITY:
                for ext in ["joblib", "pkl"]:
                    path = os.path.join(self.models_dir, f"{name}.{ext}")
                    if os.path.exists(path):
                        self.classifier = joblib.load(path)
                        self.model_name = name
                        logger.info(f"Loaded classifier: {name} ({ext})")
                        break
                if self.classifier is not None:
                    break

            if self.classifier is None:
                raise FileNotFoundError("No classifier found")

            # Get class labels
            if hasattr(self.classifier, "classes_"):
                self.classes = list(self.classifier.classes_)

            self.is_loaded = True
            logger.info(f"Intent classifier ready: {self.model_name}, {len(self.classes)} classes")
            return True

        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            return False

    def preprocess(self, text: str) -> str:
        """Clean and normalize text for TF-IDF."""
        text = text.lower().strip()
        text = self._clean_pattern.sub("", text)
        text = self._whitespace_pattern.sub(" ", text)
        return text[:500]  # Cap length

    def predict(self, text: str) -> Tuple[str, float, dict[str, float]]:
        """
        Predict intent from text.
        Returns: (intent, confidence, all_probabilities)
        """
        if not self.is_loaded:
            return ("unknown", 0.0, {})

        cleaned = self.preprocess(text)
        vectorized = self.tfidf.transform([cleaned])

        # Predict intent
        intent = self.classifier.predict(vectorized)[0]

        # Get confidence
        confidence = 0.0
        all_probs = {}

        if hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(vectorized)
            confidence = float(proba.max())

            # Get top 5 probabilities
            if hasattr(self.classifier, "classes_"):
                prob_values = proba[0]
                top_indices = np.argsort(prob_values)[::-1][:5]
                for idx in top_indices:
                    class_name = self.classifier.classes_[idx]
                    all_probs[class_name] = round(float(prob_values[idx]), 4)

        return (intent, round(confidence, 4), all_probs)

    def predict_batch(self, texts: list[str]) -> list[Tuple[str, float]]:
        """Predict intents for multiple texts."""
        if not self.is_loaded:
            return [("unknown", 0.0) for _ in texts]

        cleaned = [self.preprocess(t) for t in texts]
        vectorized = self.tfidf.transform(cleaned)

        intents = self.classifier.predict(vectorized)

        results = []
        if hasattr(self.classifier, "predict_proba"):
            probas = self.classifier.predict_proba(vectorized)
            for intent, proba in zip(intents, probas):
                results.append((intent, round(float(proba.max()), 4)))
        else:
            results = [(intent, 0.0) for intent in intents]

        return results

    def get_info(self) -> dict:
        """Get classifier information."""
        return {
            "model": self.model_name,
            "classes": len(self.classes),
            "loaded": self.is_loaded,
        }


# Singleton instance
intent_classifier = IntentClassifier()
