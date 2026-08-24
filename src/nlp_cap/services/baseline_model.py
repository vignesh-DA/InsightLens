"""
baseline_model.py
Loads the trained TF-IDF vectorizer and sentiment classifier.
Provides a predict function for fast, deterministic overall sentiment.
"""

import os
import joblib
import numpy as np

# Paths to saved model artifacts
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")

# Load at module level (once at startup)
_vectorizer = None
_model = None


def _load_models():
    """Load the trained vectorizer and model from disk."""
    global _vectorizer, _model

    if not os.path.exists(VECTORIZER_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model files not found in {MODELS_DIR}.\n"
            "Run 'python src/nlp_cap/models/train_baseline.py' first to train the baseline model."
        )

    _vectorizer = joblib.load(VECTORIZER_PATH)
    _model = joblib.load(MODEL_PATH)
    print(f"[+] Baseline model loaded from {MODELS_DIR}")


def predict_sentiment(review_text: str) -> tuple[str, float]:
    """
    Predict overall sentiment for a single review.

    Args:
        review_text: The product review text to analyze.

    Returns:
        Tuple of (sentiment_label, confidence_score).
        sentiment_label: "Positive" or "Negative"
        confidence_score: float between 0.0 and 1.0
    """
    global _vectorizer, _model

    if _vectorizer is None or _model is None:
        _load_models()

    # Vectorize the input text
    X = _vectorizer.transform([review_text])

    # Predict label
    label = _model.predict(X)[0]

    # Get confidence (probability of the predicted class)
    probabilities = _model.predict_proba(X)[0]
    class_index = list(_model.classes_).index(label)
    confidence = float(probabilities[class_index])

    return label, round(confidence, 4)
