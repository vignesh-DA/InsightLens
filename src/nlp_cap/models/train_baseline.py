"""
train_baseline.py
Trains TF-IDF + Logistic Regression and Multinomial Naive Bayes models
on the prepared Amazon Reviews dataset. Evaluates both and saves the
best performer as the production model.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Paths
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(MODELS_DIR), "data")
DATASET_PATH = os.path.join(DATA_DIR, "reviews_dataset.csv")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "sentiment_model.pkl")


def load_data():
    """Load and validate the reviews dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Run 'python src/nlp_cap/data/prepare_data.py' first."
        )

    df = pd.read_csv(DATASET_PATH)
    print(f"[i] Loaded {len(df):,} reviews")
    print(f"    Columns: {list(df.columns)}")
    print(f"    Sentiment distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    # Drop any rows with missing text
    df = df.dropna(subset=["review_text", "sentiment"])
    return df


def evaluate_model(name, model, X_test, y_test):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"\n{'='*50}")
    print(f"[+] {name} Results")
    print(f"{'='*50}")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")
    print(f"\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))
    print(f"    Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    {cm}")

    return acc, f1


def train():
    """Train, evaluate, and save the best baseline model."""
    # Load data
    df = load_data()
    X = df["review_text"].tolist()
    y = df["sentiment"].tolist()

    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[i] Split: {len(X_train):,} train / {len(X_test):,} test")

    # TF-IDF Vectorization
    print("\n[*] Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Unigrams + bigrams
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"    Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # --- Train Logistic Regression ---
    print("\n[*] Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=42,
    )
    lr_model.fit(X_train_tfidf, y_train)
    lr_acc, lr_f1 = evaluate_model("Logistic Regression", lr_model, X_test_tfidf, y_test)

    # --- Train Multinomial Naive Bayes ---
    print("\n[*] Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)
    nb_acc, nb_f1 = evaluate_model("Multinomial Naive Bayes", nb_model, X_test_tfidf, y_test)

    # --- Select and save best model ---
    if lr_f1 >= nb_f1:
        best_model = lr_model
        best_name = "Logistic Regression"
    else:
        best_model = nb_model
        best_name = "Multinomial Naive Bayes"

    print(f"\n{'='*50}")
    print(f"[+] Best model: {best_name} (F1={max(lr_f1, nb_f1):.4f})")
    print(f"{'='*50}")

    # Save vectorizer and model
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    print(f"\n[+] Saved vectorizer -> {VECTORIZER_PATH}")
    print(f"[+] Saved model     -> {MODEL_PATH}")
    print("\n[+] Training complete!")


if __name__ == "__main__":
    train()
